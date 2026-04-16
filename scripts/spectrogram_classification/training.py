from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, Dataset

from scripts.spectrogram_classification.data import (
    AudioExample,
    fit_waveform_length,
    load_audio_waveform,
    split_audio_examples,
)
from scripts.spectrogram_classification.features import (
    compute_cqt,
    compute_linear_spectrogram,
    compute_mel_spectrogram,
    compute_mfcc_feature_vector,
    compute_mfcc_map,
    pitch_shift_waveform,
)
from scripts.spectrogram_classification.models import MlpAudioClassifier, SpectrogramCnnClassifier
from utils.io_helpers import ensure_dir, save_json, write_csv_rows
from utils.project_paths import SPECTROGRAM_METRICS_DIR, SPECTROGRAM_MODEL_DIR, SPECTROGRAM_PREDICTION_DIR


@dataclass(frozen=True)
class ExperimentConfig:
    output_name: str
    feature_kind: str
    model_kind: str
    label_mode: str = "binary"
    sample_rate: int = 16000
    clip_seconds: float = 4.0
    batch_size: int = 32
    num_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    random_seed: int = 42
    num_classes: int = 2
    max_examples_per_class: int | None = None
    augment_pitch_shift: bool = False
    augment_steps: tuple[float, ...] = (-2.0, 2.0)


class AudioFeatureDataset(Dataset):
    def __init__(self, inputs: list[torch.Tensor], targets: list[int], examples: list[AudioExample]) -> None:
        self.inputs = inputs
        self.targets = targets
        self.examples = examples

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], torch.tensor(self.targets[index], dtype=torch.long)


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _select_examples(examples: list[AudioExample], config: ExperimentConfig) -> list[AudioExample]:
    grouped: dict[str, list[AudioExample]] = {}
    for example in examples:
        grouped.setdefault(example.label_for_mode(config.label_mode), []).append(example)

    selected_labels = sorted(grouped.keys())[: config.num_classes]
    selected: list[AudioExample] = []
    for label in selected_labels:
        label_examples = grouped[label]
        if config.max_examples_per_class is not None:
            label_examples = label_examples[: config.max_examples_per_class]
        selected.extend(label_examples)
    return selected


def _build_label_mapping(examples: list[AudioExample], label_mode: str) -> dict[str, int]:
    labels = {example.label_for_mode(label_mode) for example in examples}
    return {label: index for index, label in enumerate(sorted(labels))}


def _target_num_samples(config: ExperimentConfig) -> int:
    return int(config.sample_rate * config.clip_seconds)


def _compute_feature_tensor(waveform: torch.Tensor, sample_rate: int, feature_kind: str) -> torch.Tensor:
    if feature_kind == "mfcc_vector":
        return compute_mfcc_feature_vector(waveform, sample_rate)
    if feature_kind == "mfcc_map":
        return compute_mfcc_map(waveform, sample_rate).unsqueeze(0)
    if feature_kind == "spectrogram":
        return compute_linear_spectrogram(waveform, sample_rate).unsqueeze(0)
    if feature_kind == "mel_spectrogram":
        return compute_mel_spectrogram(waveform, sample_rate).unsqueeze(0)
    if feature_kind == "cqt":
        return compute_cqt(waveform, sample_rate).unsqueeze(0)
    raise ValueError(f"Unsupported feature kind: {feature_kind}")


def _build_dataset(
    examples: list[AudioExample],
    label_to_index: dict[str, int],
    config: ExperimentConfig,
    augment: bool = False,
) -> AudioFeatureDataset:
    inputs: list[torch.Tensor] = []
    targets: list[int] = []
    expanded_examples: list[AudioExample] = []
    target_num_samples = _target_num_samples(config)

    for example in examples:
        waveform, sample_rate = load_audio_waveform(example.file_path, target_sample_rate=config.sample_rate)
        waveform = fit_waveform_length(waveform, target_num_samples)
        feature_tensor = _compute_feature_tensor(waveform, sample_rate, config.feature_kind).to(torch.float32)
        inputs.append(feature_tensor)
        label = example.label_for_mode(config.label_mode)
        targets.append(label_to_index[label])
        expanded_examples.append(example)

        if augment and config.augment_pitch_shift:
            for semitone_steps in config.augment_steps:
                shifted = pitch_shift_waveform(waveform, sample_rate, semitone_steps)
                shifted_feature = _compute_feature_tensor(shifted, sample_rate, config.feature_kind).to(torch.float32)
                inputs.append(shifted_feature)
                targets.append(label_to_index[label])
                expanded_examples.append(example)

    return AudioFeatureDataset(inputs, targets, expanded_examples)


def _build_model(dataset: AudioFeatureDataset, config: ExperimentConfig) -> nn.Module:
    sample_input = dataset.inputs[0]
    if config.model_kind == "mlp":
        return MlpAudioClassifier(input_dim=int(sample_input.numel()), num_classes=config.num_classes)
    if config.model_kind == "cnn":
        return SpectrogramCnnClassifier(num_classes=config.num_classes, in_channels=int(sample_input.shape[0]))
    raise ValueError(f"Unsupported model kind: {config.model_kind}")


def _evaluate_model(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, list[int], list[int]]:
    model.eval()
    predictions: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            truths.extend(targets.cpu().tolist())
    accuracy = float(accuracy_score(truths, predictions)) if truths else 0.0
    return accuracy, truths, predictions


def _train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: ExperimentConfig,
    device: torch.device,
    model_path: Path,
) -> list[dict]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict] = []
    best_val_accuracy = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        batch_losses: list[float] = []
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            batch_losses.append(float(loss.item()))

        train_accuracy, _, _ = _evaluate_model(model, train_loader, device)
        val_accuracy, _, _ = _evaluate_model(model, val_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(batch_losses)) if batch_losses else 0.0,
                "train_accuracy": train_accuracy,
                "val_accuracy": val_accuracy,
            }
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            epochs_without_improvement = 0
            ensure_dir(model_path.parent)
            cpu_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
            torch.save(cpu_state, model_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.patience:
                break

    return history


def run_audio_experiment(
    examples: list[AudioExample],
    config: ExperimentConfig,
    output_model_dir: Path = SPECTROGRAM_MODEL_DIR,
    output_metrics_dir: Path = SPECTROGRAM_METRICS_DIR,
    output_prediction_dir: Path = SPECTROGRAM_PREDICTION_DIR,
) -> dict:
    selected_examples = _select_examples(examples, config)
    selected_labels = {example.label_for_mode(config.label_mode) for example in selected_examples}
    if len(selected_labels) < config.num_classes:
        raise ValueError(f"Need at least {config.num_classes} classes for {config.output_name}.")
    if len(selected_examples) < config.num_classes * 3:
        raise ValueError(f"Not enough audio files to run {config.output_name}.")

    set_random_seed(config.random_seed)
    label_to_index = _build_label_mapping(selected_examples, config.label_mode)
    index_to_label = {index: label for label, index in label_to_index.items()}
    train_examples, val_examples, test_examples = split_audio_examples(
        selected_examples,
        seed=config.random_seed,
        label_mode=config.label_mode,
    )
    train_dataset = _build_dataset(train_examples, label_to_index, config, augment=config.augment_pitch_shift)
    val_dataset = _build_dataset(val_examples, label_to_index, config, augment=False)
    test_dataset = _build_dataset(test_examples, label_to_index, config, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _build_model(train_dataset, config).to(device)
    model_path = output_model_dir / f"{config.output_name}_best.pt"
    history = _train_model(model, train_loader, val_loader, config, device, model_path)

    model.load_state_dict(torch.load(model_path, map_location=device))
    test_accuracy, truths, predictions = _evaluate_model(model, test_loader, device)
    metrics = {
        "output_name": config.output_name,
        "feature_kind": config.feature_kind,
        "model_kind": config.model_kind,
        "sample_rate": config.sample_rate,
        "clip_seconds": config.clip_seconds,
        "label_mode": config.label_mode,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "num_classes": config.num_classes,
        "labels": index_to_label,
        "test_accuracy": test_accuracy,
        "confusion_matrix": confusion_matrix(truths, predictions).tolist(),
        "classification_report": classification_report(
            truths,
            predictions,
            target_names=[index_to_label[index] for index in range(config.num_classes)],
            output_dict=True,
            zero_division=0,
        ),
        "history": history,
        "model_path": str(model_path),
    }

    prediction_rows = []
    for example, truth, prediction in zip(test_dataset.examples, truths, predictions):
        selected_label = example.label_for_mode(config.label_mode)
        prediction_rows.append(
            {
                "file_path": example.file_path,
                "label": selected_label,
                "instrument_label": example.instrument_label,
                "family_label": example.family_label,
                "y_true": truth,
                "y_pred": prediction,
                "predicted_label": index_to_label[prediction],
            }
        )

    metrics_path = output_metrics_dir / f"{config.output_name}_metrics.json"
    prediction_path = output_prediction_dir / f"{config.output_name}_predictions.csv"
    save_json(metrics_path, metrics)
    write_csv_rows(prediction_path, prediction_rows)
    return {
        "metrics": metrics,
        "metrics_path": metrics_path,
        "prediction_path": prediction_path,
        "model_path": model_path,
    }


def run_experiment_suite(
    examples: list[AudioExample],
    configs: list[ExperimentConfig],
    output_metrics_dir: Path = SPECTROGRAM_METRICS_DIR,
) -> dict:
    results = [run_audio_experiment(examples, config) for config in configs]
    summary = {
        "experiments": [
            {
                "output_name": result["metrics"]["output_name"],
                "feature_kind": result["metrics"]["feature_kind"],
                "model_kind": result["metrics"]["model_kind"],
                "test_accuracy": result["metrics"]["test_accuracy"],
                "model_path": str(result["model_path"]),
                "metrics_path": str(result["metrics_path"]),
                "prediction_path": str(result["prediction_path"]),
            }
            for result in results
        ]
    }
    summary_path = output_metrics_dir / "experiment_summary.json"
    save_json(summary_path, summary)
    return {"results": results, "summary_path": summary_path}
