#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import SPECTROGRAM_EVALUATION_DIR, SPECTROGRAM_WEIGHT_DIR, ensure_dir, save_json, write_csv_rows


BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001


@dataclass(frozen=True)
class ExperimentSpec:
    output_name: str
    weight_name: str
    waveforms: list
    labels: list[int]
    label_names: list[str]
    feature_func: object
    classifier_factory: object


def _load_notebook_module(module_name: str):
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Could not import converted notebook module '{module_name}'. "
            "Pass --module-name with the Python module name created from the notebook export."
        ) from exc


def _ordered_label_names(label_map: dict[str, int]) -> list[str]:
    return [name for name, _index in sorted(label_map.items(), key=lambda item: item[1])]


def build_experiment_specs(module_name: str = "spectrogram_classification") -> list[ExperimentSpec]:
    notebook = _load_notebook_module(module_name)
    binary_label_names = _ordered_label_names(notebook.INSTRUMENT_MAP)
    four_class_label_names = _ordered_label_names(notebook.INSTRUMENT_MAP_7)
    return [
        ExperimentSpec(
            output_name="mfcc_mlp",
            weight_name="best_mlp_model.weights",
            waveforms=notebook.waveforms,
            labels=notebook.labels,
            label_names=binary_label_names,
            feature_func=notebook.extract_mfcc,
            classifier_factory=notebook.MLPClassifier,
        ),
        ExperimentSpec(
            output_name="spectrogram_cnn",
            weight_name="best_spec_model.weights",
            waveforms=notebook.waveforms,
            labels=notebook.labels,
            label_names=binary_label_names,
            feature_func=notebook.extract_spec,
            classifier_factory=notebook.SimpleCNN,
        ),
        ExperimentSpec(
            output_name="mel_spectrogram_cnn",
            weight_name="best_mel_model.weights",
            waveforms=notebook.waveforms,
            labels=notebook.labels,
            label_names=binary_label_names,
            feature_func=notebook.extract_mel,
            classifier_factory=notebook.SimpleCNN,
        ),
        ExperimentSpec(
            output_name="cqt_cnn",
            weight_name="best_q_model.weights",
            waveforms=notebook.waveforms,
            labels=notebook.labels,
            label_names=binary_label_names,
            feature_func=notebook.extract_q,
            classifier_factory=notebook.SimpleCNN,
        ),
        ExperimentSpec(
            output_name="augmented_cqt_cnn",
            weight_name="best_augmented_model.weights",
            waveforms=notebook.augmented_waveforms,
            labels=notebook.augmented_labels,
            label_names=binary_label_names,
            feature_func=notebook.extract_q,
            classifier_factory=notebook.SimpleCNN,
        ),
        ExperimentSpec(
            output_name="four_class_cnn",
            weight_name="best_model_7.weights",
            waveforms=notebook.waveforms,
            labels=notebook.labels_7,
            label_names=four_class_label_names,
            feature_func=notebook.feature_func_7,
            classifier_factory=notebook.ImprovedCNN4Classes,
        ),
    ]


def split_data(waveforms, labels, train_ratio: float = 0.7, valid_ratio: float = 0.15):
    assert train_ratio + valid_ratio < 1
    test_ratio = 1 - (train_ratio + valid_ratio)
    n_total = len(waveforms)
    n_train = int(n_total * train_ratio)
    n_valid = int(n_total * valid_ratio)
    n_test = int(n_total * test_ratio)
    w_train = waveforms[:n_train]
    w_valid = waveforms[n_train:n_train + n_valid]
    w_test = waveforms[n_train + n_valid:n_train + n_valid + n_test]
    y_train = labels[:n_train]
    y_valid = labels[n_train:n_train + n_valid]
    y_test = labels[n_train + n_valid:n_train + n_valid + n_test]
    return w_train, w_valid, w_test, y_train, y_valid, y_test


def process_data(waveforms, feature_function):
    return [feature_function(waveform) for waveform in waveforms]


class InstrumentDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        features = self.features[idx]
        label = self.labels[idx]
        return features, torch.tensor(label, dtype=torch.long)


class Loaders:
    def __init__(self, waveforms, labels, feature_function, seed: int = 0):
        torch.manual_seed(seed)
        random.seed(seed)
        self.w_train, self.w_valid, self.w_test, self.y_train, self.y_valid, self.y_test = split_data(waveforms, labels)

        self.x_train = process_data(self.w_train, feature_function)
        self.x_valid = process_data(self.w_valid, feature_function)
        self.x_test = process_data(self.w_test, feature_function)

        self.data_train = InstrumentDataset(self.x_train, self.y_train)
        self.data_valid = InstrumentDataset(self.x_valid, self.y_valid)
        self.data_test = InstrumentDataset(self.x_test, self.y_test)

        self.loader_train = DataLoader(self.data_train, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loader_valid = DataLoader(self.data_valid, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        self.loader_test = DataLoader(self.data_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


class Pipeline:
    def __init__(self, module, learning_rate: float, seed: int = 0, device: str = "cpu"):
        torch.manual_seed(seed)
        random.seed(seed)
        requested_device = device
        if requested_device == "cuda" and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.criterion = nn.CrossEntropyLoss()
        self.model = module.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def evaluate(self, loader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        return correct / total if total else 0.0

    def predict(self, loader) -> tuple[list[int], list[int]]:
        self.model.eval()
        truths: list[int] = []
        predictions: list[int] = []
        with torch.no_grad():
            for inputs, labels in loader:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                truths.extend(labels.cpu().tolist())
                predictions.extend(predicted.cpu().tolist())
        return truths, predictions

    def train(self, loaders: Loaders, num_epochs: int = 1, model_path: Path | None = None):
        val_acc = 0.0
        best_val_acc = -1.0
        history: list[dict] = []
        for epoch in range(num_epochs):
            self.model.train()
            losses: list[float] = []
            for inputs, labels in loaders.loader_train:
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()
                losses.append(float(loss))

            val_acc = self.evaluate(loaders.loader_valid)
            history.append(
                {
                    "epoch": epoch,
                    "loss": float(sum(losses) / len(losses)) if losses else 0.0,
                    "valid_accuracy": val_acc,
                }
            )
            print(
                "Epoch "
                + str(epoch)
                + ", loss = "
                + str(history[-1]["loss"])
                + ", validation accuracy = "
                + str(val_acc)
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if model_path is not None:
                    cpu_state = {key: value.detach().cpu() for key, value in self.model.state_dict().items()}
                    torch.save(cpu_state, model_path)

        print("Final validation accuracy = " + str(val_acc) + ", best = " + str(best_val_acc))
        return val_acc, best_val_acc, history

    def load(self, path: Path) -> None:
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError:
            state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)


def _label_names(spec: ExperimentSpec) -> list[str]:
    unique_labels = sorted(set(spec.labels))
    return [spec.label_names[label] for label in unique_labels]


def evaluate_saved_experiment(spec: ExperimentSpec, weight_dir: Path, seed: int = 0, device: str = "cpu") -> dict:
    loaders = Loaders(spec.waveforms, spec.labels, spec.feature_func, seed=seed)
    pipeline = Pipeline(spec.classifier_factory(), LEARNING_RATE, seed=seed, device=device)
    weight_path = weight_dir / spec.weight_name
    pipeline.load(weight_path)

    train_accuracy = pipeline.evaluate(loaders.loader_train)
    valid_accuracy = pipeline.evaluate(loaders.loader_valid)
    test_accuracy = pipeline.evaluate(loaders.loader_test)
    truths, predictions = pipeline.predict(loaders.loader_test)
    labels = _label_names(spec)

    return {
        "output_name": spec.output_name,
        "weight_name": spec.weight_name,
        "weight_path": str(weight_path),
        "train_size": len(loaders.data_train),
        "valid_size": len(loaders.data_valid),
        "test_size": len(loaders.data_test),
        "train_accuracy": train_accuracy,
        "valid_accuracy": valid_accuracy,
        "test_accuracy": test_accuracy,
        "labels": labels,
        "confusion_matrix": confusion_matrix(truths, predictions).tolist(),
    }


def run_experiment(
    spec: ExperimentSpec,
    weight_dir: Path,
    evaluation_dir: Path,
    seed: int = 0,
    device: str = "cpu",
) -> dict:
    ensure_dir(weight_dir)
    ensure_dir(evaluation_dir)
    weight_path = weight_dir / spec.weight_name
    loaders = Loaders(spec.waveforms, spec.labels, spec.feature_func, seed=seed)
    pipeline = Pipeline(spec.classifier_factory(), LEARNING_RATE, seed=seed, device=device)
    print(f"[Train] {spec.output_name} -> {weight_path} on {pipeline.device}")
    final_val_acc, best_val_acc, history = pipeline.train(loaders, NUM_EPOCHS, weight_path)

    metrics = evaluate_saved_experiment(spec, weight_dir, seed=seed, device=device)
    metrics["final_valid_accuracy"] = final_val_acc
    metrics["best_valid_accuracy"] = best_val_acc
    metrics["history"] = history

    save_json(evaluation_dir / f"{spec.output_name}_metrics.json", metrics)
    return metrics


def write_summary(results: list[dict], evaluation_dir: Path) -> None:
    save_json(evaluation_dir / "weight_summary.json", {"experiments": results})
    rows = []
    for result in results:
        rows.append(
            {
                "output_name": result["output_name"],
                "weight_name": result["weight_name"],
                "train_size": result["train_size"],
                "valid_size": result["valid_size"],
                "test_size": result["test_size"],
                "final_valid_accuracy": f"{result['final_valid_accuracy']:.4f}",
                "best_valid_accuracy": f"{result['best_valid_accuracy']:.4f}",
                "train_accuracy": f"{result['train_accuracy']:.4f}",
                "valid_accuracy": f"{result['valid_accuracy']:.4f}",
                "test_accuracy": f"{result['test_accuracy']:.4f}",
            }
        )
    write_csv_rows(evaluation_dir / "weight_summary.csv", rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train notebook-defined spectrogram models and save .weights files.")
    parser.add_argument("--module-name", type=str, default="spectrogram_classification")
    parser.add_argument("--weight-dir", type=Path, default=SPECTROGRAM_WEIGHT_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=SPECTROGRAM_EVALUATION_DIR)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()

    results = []
    for spec in build_experiment_specs(args.module_name):
        results.append(run_experiment(spec, args.weight_dir, args.evaluation_dir, seed=args.seed, device=args.device))
    write_summary(results, args.evaluation_dir)
    print(f"[Train] Wrote weights under {args.weight_dir}")
    print(f"[Train] Wrote evaluation under {args.evaluation_dir}")


if __name__ == "__main__":
    main()
