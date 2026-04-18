#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import (  # noqa: E402
    INSTRUMENT_LABELS,
    N_CLASSES,
    TIME_STEPS_PER_BEAT,
    collect_split_files,
)
from scripts.automatic_music_instrumentation.core.metrics import normalize_confusion_matrix  # noqa: E402
from scripts.automatic_music_instrumentation.core.models import build_model  # noqa: E402
from scripts.automatic_music_instrumentation.core.pitch_zones import pitch_zone_predict_events  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import (  # noqa: E402
    AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT,
    AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR,
    AUTOMATIC_INSTRUMENTATION_VISUAL_DIR,
)


PANEL_FACE = "#f5efe4"
PANEL_INNER = "#fffaf2"
INK = "#17212b"
MUTED = "#5b6470"
GRID = "#cabca8"

INSTRUMENT_COLORS = {
    "piano": "#2563ad",
    "guitar": "#c46a11",
    "bass": "#2c7a4b",
    "strings": "#a23b68",
    "brass": "#7454b3",
}
LABEL_COLORS = [INSTRUMENT_COLORS[label] for label in INSTRUMENT_LABELS]

MODEL_ORDER = [
    "pitch_zones",
    "note_mlp",
    "sequence_lstm",
    "bidirectional_lstm",
    "compact_transformer",
    "causal_transformer",
    "full_transformer",
]
MODEL_TITLES = {
    "ground_truth": "Ground truth",
    "pitch_zones": "Pitch-zone rule",
    "note_mlp": "Note MLP",
    "sequence_lstm": "Online LSTM",
    "bidirectional_lstm": "Offline BiLSTM",
    "compact_transformer": "Compact Transformer",
    "causal_transformer": "Causal Transformer",
    "full_transformer": "Full Transformer",
}
MODEL_FAMILIES = {
    "pitch_zones": "Rule baseline",
    "note_mlp": "Independent note classifier",
    "sequence_lstm": "Recurrent online",
    "bidirectional_lstm": "Recurrent offline",
    "compact_transformer": "Attention offline",
    "causal_transformer": "Attention online",
    "full_transformer": "Attention offline",
}
MODEL_COLORS = {
    "pitch_zones": "#8b6f47",
    "note_mlp": "#2f6fbb",
    "sequence_lstm": "#2f855a",
    "bidirectional_lstm": "#0f766e",
    "compact_transformer": "#9f3a64",
    "causal_transformer": "#b45309",
    "full_transformer": "#6d5bd0",
}


@dataclass(frozen=True)
class ModelMetric:
    model: str
    title: str
    score: float
    best_val_loss: float | None
    final_val_loss: float | None
    run_dir: Path | None


def _style_axis(ax: plt.Axes, grid: bool = True) -> None:
    ax.set_facecolor(PANEL_INNER)
    if grid:
        ax.grid(True, alpha=0.22, color=GRID, linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#d9ccb8")
        spine.set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=9)


def _instrument_legend_handles() -> list[Line2D]:
    return [Line2D([0], [0], color=color, linewidth=3.0, label=label) for label, color in zip(INSTRUMENT_LABELS, LABEL_COLORS)]


def _figure_to_palette_frame(fig: plt.Figure, dpi: int = 105) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert("P", palette=Image.Palette.ADAPTIVE, colors=224)


def _write_gif(frames: list[Image.Image], output_path: Path, duration_ms: int = 110) -> None:
    if not frames:
        return
    ensure_dir(output_path.parent)
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _sample_indices(num_src: int, num_dst: int) -> list[int]:
    if num_src <= 1:
        return [0] * max(1, num_dst)
    if num_dst <= 1:
        return [0]
    return [round(index * (num_src - 1) / (num_dst - 1)) for index in range(num_dst)]


def _load_arrays(files: list[Path], max_files: int) -> list[np.ndarray]:
    arrays: list[np.ndarray] = []
    for file_path in files[:max_files]:
        array = np.load(file_path)
        if array.ndim == 2 and array.shape[1] == 4 and array.size:
            arrays.append(array)
    return arrays


def _merged_label_counts(arrays: list[np.ndarray]) -> np.ndarray:
    counts = np.zeros(N_CLASSES, dtype=np.int64)
    for array in arrays:
        labels, label_counts = np.unique(array[:, 3], return_counts=True)
        for label, count in zip(labels, label_counts):
            if 0 <= int(label) < N_CLASSES:
                counts[int(label)] += int(count)
    return counts


def _pitch_label_matrix(arrays: list[np.ndarray]) -> np.ndarray:
    matrix = np.zeros((N_CLASSES, 128), dtype=np.int64)
    for array in arrays:
        for pitch, label in zip(array[:, 1].astype(int), array[:, 3].astype(int)):
            if 0 <= label < N_CLASSES and 0 <= pitch <= 127:
                matrix[label, pitch] += 1
    return matrix


def _label_share(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels.astype(int), minlength=N_CLASSES).astype(float)
    total = max(float(counts.sum()), 1.0)
    return counts / total


def _agreement(predicted: np.ndarray, truth: np.ndarray) -> float:
    length = min(len(predicted), len(truth))
    if length == 0:
        return 0.0
    return float(np.mean(predicted[:length] == truth[:length]))


def _roll_axis(
    ax: plt.Axes,
    events: np.ndarray,
    labels: np.ndarray,
    title: str,
    subtitle: str | None = None,
    progress: float = 1.0,
    show_labels: bool = True,
) -> None:
    _style_axis(ax)
    if events.size == 0:
        ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
        ax.text(0.5, 0.5, "No note events", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        return

    starts_all = events[:, 0].astype(float) / TIME_STEPS_PER_BEAT
    durations_all = np.maximum(events[:, 2].astype(float) / TIME_STEPS_PER_BEAT, 0.05)
    pitches_all = events[:, 1].astype(float)
    x_max = max(float(np.max(starts_all + durations_all)), 1.0)
    y_min = max(0.0, float(pitches_all.min()) - 4)
    y_max = min(127.0, float(pitches_all.max()) + 4)

    reveal_x = x_max * max(0.0, min(progress, 1.0))
    visible = starts_all <= reveal_x if progress < 0.999 else np.ones_like(starts_all, dtype=bool)
    starts = starts_all[visible]
    durations = durations_all[visible]
    pitches = pitches_all[visible]
    visible_labels = labels[: len(events)].astype(int)[visible]

    for label_index, label_name in enumerate(INSTRUMENT_LABELS):
        mask = visible_labels == label_index
        if np.any(mask):
            ax.hlines(
                pitches[mask],
                starts[mask],
                starts[mask] + durations[mask],
                color=LABEL_COLORS[label_index],
                linewidth=2.05,
                alpha=0.88,
                label=label_name,
            )
    if progress < 0.999:
        ax.axvline(reveal_x, color=INK, linewidth=1.5, alpha=0.45)

    ax.set_title(title, loc="left", fontsize=11.2, fontweight="bold", color=INK, pad=7)
    if subtitle:
        ax.text(0.0, 1.01, subtitle, transform=ax.transAxes, ha="left", va="bottom", fontsize=8.2, color=MUTED)
    ax.set_xlabel("Beat", color=MUTED)
    ax.set_ylabel("Pitch", color=MUTED)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(y_min, y_max)
    if not show_labels:
        ax.set_xlabel("")
        ax.set_ylabel("")


def _event_roll(
    array: np.ndarray,
    output_path: Path,
    title: str,
    label_column: np.ndarray | None = None,
    max_notes: int = 2200,
) -> None:
    ensure_dir(output_path.parent)
    display = array[:max_notes]
    labels = display[:, 3].astype(int) if label_column is None else label_column[: len(display)].astype(int)
    fig, ax = plt.subplots(figsize=(13.5, 4.8), facecolor=PANEL_FACE)
    _roll_axis(ax, display, labels, title)
    ax.legend(ncol=len(INSTRUMENT_LABELS), loc="upper right", frameon=True, fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_label_distribution(arrays: list[np.ndarray], output_path: Path) -> None:
    counts = _merged_label_counts(arrays)
    total = int(counts.sum())
    percentages = counts / max(total, 1)
    fig, ax = plt.subplots(figsize=(9.8, 4.7), facecolor=PANEL_FACE)
    _style_axis(ax)
    bars = ax.bar(INSTRUMENT_LABELS, counts, color=LABEL_COLORS, edgecolor="#1f1f1f", linewidth=0.4)
    ax.set_title("Instrument Label Distribution", loc="left", fontweight="bold", color=INK)
    ax.set_ylabel("Note events")
    ax.tick_params(axis="x", labelrotation=12)
    for bar, count, pct in zip(bars, counts, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, count, f"{count:,}\n{pct:.1%}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_pitch_label_heatmap(arrays: list[np.ndarray], output_path: Path) -> None:
    matrix = _pitch_label_matrix(arrays).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)
    fig, ax = plt.subplots(figsize=(12, 4.8), facecolor=PANEL_FACE)
    image = ax.imshow(normalized, aspect="auto", cmap="magma", origin="lower", interpolation="nearest")
    ax.set_title("Pitch Usage by Instrument Label", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("MIDI pitch")
    ax.set_ylabel("Instrument label")
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(INSTRUMENT_LABELS)
    ax.set_xticks(range(0, 128, 12))
    fig.colorbar(image, ax=ax, fraction=0.028, pad=0.02, label="Within-label share")
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_pitch_zone_map(output_path: Path) -> None:
    pitches = np.arange(128)
    labels = pitch_zone_predict_events(np.column_stack([np.zeros(128), pitches, np.ones(128), np.zeros(128)]))
    fig, ax = plt.subplots(figsize=(12, 2.4), facecolor=PANEL_FACE)
    _style_axis(ax)
    for label_index, label_name in enumerate(INSTRUMENT_LABELS):
        mask = labels == label_index
        ax.scatter(pitches[mask], np.full(np.count_nonzero(mask), label_index), s=28, color=LABEL_COLORS[label_index], label=label_name)
    ax.set_title("Fixed Pitch-Zone Rule", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("MIDI pitch")
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(INSTRUMENT_LABELS)
    ax.set_xlim(-1, 128)
    ax.legend(ncol=len(INSTRUMENT_LABELS), loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _load_losses(run_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.load(run_dir / "train_losses.npy") if (run_dir / "train_losses.npy").exists() else np.asarray([])
    valid = np.load(run_dir / "val_losses.npy") if (run_dir / "val_losses.npy").exists() else np.asarray([])
    acc = np.load(run_dir / "val_accs.npy") if (run_dir / "val_accs.npy").exists() else np.asarray([])
    return train, valid, acc


def _save_training_curves(run_dir: Path, output_path: Path, val_steps: int = 500) -> bool:
    train, valid, acc = _load_losses(run_dir)
    if train.size == 0 and valid.size == 0 and acc.size == 0:
        return False
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 7.0), facecolor=PANEL_FACE, sharex=True)
    for ax in axes:
        _style_axis(ax)
    if train.size:
        axes[0].plot(np.arange(train.size), train, color=MODEL_COLORS["note_mlp"], alpha=0.26, label="Training loss")
        if train.size >= 20:
            window = min(100, max(5, train.size // 8))
            moving = np.convolve(train, np.ones(window), "valid") / window
            axes[0].plot(np.arange(moving.size) + window / 2, moving, color=MODEL_COLORS["note_mlp"], linewidth=2.0, label="Training loss MA")
    if valid.size:
        axes[0].plot(np.arange(valid.size) * val_steps, valid, color=MODEL_COLORS["causal_transformer"], marker="o", label="Validation loss")
    axes[0].set_title("Training and Validation Loss", loc="left", fontweight="bold", color=INK)
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="best")
    if acc.size:
        axes[1].plot(np.arange(acc.size) * val_steps, acc, color=MODEL_COLORS["sequence_lstm"], marker="o", linewidth=2.0)
    axes[1].set_title("Validation Accuracy", loc="left", fontweight="bold", color=INK)
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_ylim(0.0, 1.0)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _find_confusion_matrix(run_dir: Path | None, evaluation_dir: Path | None) -> Path | None:
    candidates: list[Path] = []
    if evaluation_dir is not None and evaluation_dir.exists():
        candidates.extend(sorted(evaluation_dir.glob("*_confusion_matrix.npy")))
    if run_dir is not None and run_dir.exists():
        candidates.extend(sorted(run_dir.glob("*confusion_matrix.npy")))
        candidates.extend(sorted((run_dir / "evaluation").glob("*_confusion_matrix.npy")))
    for candidate in candidates:
        if "normalized" not in candidate.name:
            return candidate
    return None


def _save_confusion_matrix(matrix_path: Path, output_path: Path) -> bool:
    matrix = np.load(matrix_path)
    normalized = normalize_confusion_matrix(matrix)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.1), facecolor=PANEL_FACE)
    for ax, data, title in [
        (axes[0], matrix, "Raw Confusion Matrix"),
        (axes[1], normalized, "Normalized Confusion Matrix"),
    ]:
        image = ax.imshow(data, cmap="Blues")
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.set_xlabel("Prediction")
        ax.set_ylabel("Ground truth")
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(INSTRUMENT_LABELS, rotation=30, ha="right")
        ax.set_yticklabels(INSTRUMENT_LABELS)
        for row in range(N_CLASSES):
            for col in range(N_CLASSES):
                text = f"{data[row, col]:.2f}" if data.dtype.kind == "f" else str(int(data[row, col]))
                ax.text(col, row, text, ha="center", va="center", fontsize=8)
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def _build_overview_panel(image_paths: list[tuple[Path, str]], output_path: Path) -> None:
    existing = [(path, title) for path, title in image_paths if path.exists()]
    if not existing:
        return
    cols = 2
    rows = int(np.ceil(len(existing) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5.6 * rows), facecolor=PANEL_FACE)
    axes_array = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_array.flat:
        ax.axis("off")
    for ax, (path, title) in zip(axes_array.flat, existing):
        ax.imshow(plt.imread(path))
        ax.set_title(title, loc="left", fontweight="bold", color=INK)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _read_summary(suite_root: Path) -> dict[str, ModelMetric]:
    summary_path = suite_root / "model_suite" / "model_suite_summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing model suite summary: {summary_path}")
    metrics: dict[str, ModelMetric] = {}
    with summary_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            model = row["model"]
            run_dir = Path(row["run_dir"]) if row.get("run_dir") else None
            score_text = row.get("final_val_accuracy") or row.get("test_or_rule_accuracy") or "0"
            metrics[model] = ModelMetric(
                model=model,
                title=MODEL_TITLES.get(model, model),
                score=float(score_text),
                best_val_loss=float(row["best_val_loss"]) if row.get("best_val_loss") else None,
                final_val_loss=float(row["final_val_loss"]) if row.get("final_val_loss") else None,
                run_dir=run_dir,
            )
    return metrics


def _select_sample(sample_dir: Path, max_beats: int) -> Path:
    candidates = sorted(sample_dir.glob("*.npy"))
    if not candidates:
        raise FileNotFoundError(f"No sample .npy files found in {sample_dir}")
    best_path = candidates[0]
    best_score = -1.0
    cutoff = max_beats * TIME_STEPS_PER_BEAT
    for path in candidates:
        array = np.load(path)
        if array.ndim != 2 or array.shape[1] != 4:
            continue
        cropped = array[array[:, 0] < cutoff]
        if cropped.size == 0:
            continue
        label_count = len(np.unique(cropped[:, 3]))
        note_balance = min(len(cropped), 900) / 900.0
        score = label_count * 10.0 + note_balance
        if score > best_score:
            best_score = score
            best_path = path
    return best_path


def _load_sample(sample_path: Path, max_beats: int, max_notes: int) -> np.ndarray:
    array = np.load(sample_path)
    if array.ndim != 2 or array.shape[1] != 4 or array.size == 0:
        raise ValueError(f"Expected a non-empty event array shaped (n, 4): {sample_path}")
    cutoff = max_beats * TIME_STEPS_PER_BEAT
    cropped = array[array[:, 0] < cutoff].copy()
    if cropped.size == 0:
        cropped = array.copy()
    order = np.lexsort((cropped[:, 1], cropped[:, 0]))
    cropped = cropped[order][:max_notes]
    if cropped.size:
        cropped[:, 0] -= int(cropped[:, 0].min())
    return cropped.astype(np.int64, copy=False)


def _load_checkpoint(checkpoint: Path, model_name: str, device: torch.device) -> torch.nn.Module:
    try:
        payload = torch.load(checkpoint, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(checkpoint, map_location=device)
    resolved_name = str(payload.get("config", {}).get("model_name", model_name))
    model = build_model(resolved_name, n_classes=N_CLASSES).to(device)
    state = payload["model_state"] if "model_state" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model


def _predict_checkpoint(events: np.ndarray, checkpoint: Path, model_name: str, device: torch.device) -> np.ndarray:
    inputs = torch.as_tensor(events[np.newaxis, :, :3], dtype=torch.long, device=device)
    model = _load_checkpoint(checkpoint, model_name, device)
    with torch.inference_mode():
        labels = torch.argmax(model(inputs)[0], dim=-1).detach().cpu().numpy().astype(np.int64)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return labels


def _predict_suite(events: np.ndarray, metrics: dict[str, ModelMetric], device: torch.device) -> dict[str, np.ndarray]:
    predictions: dict[str, np.ndarray] = {
        "ground_truth": events[:, 3].astype(np.int64),
        "pitch_zones": pitch_zone_predict_events(events).astype(np.int64),
    }
    for model_name in MODEL_ORDER:
        if model_name == "pitch_zones":
            continue
        metric = metrics.get(model_name)
        if metric is None or metric.run_dir is None:
            continue
        checkpoint = metric.run_dir / "checkpoints" / "best_model.pt"
        if not checkpoint.exists():
            continue
        predictions[model_name] = _predict_checkpoint(events, checkpoint, model_name, device)
    return predictions


def _best_model_name(metrics: dict[str, ModelMetric], predictions: dict[str, np.ndarray]) -> str:
    trained = [name for name in MODEL_ORDER if name != "pitch_zones" and name in metrics and name in predictions]
    if trained:
        return max(trained, key=lambda name: metrics[name].score)
    available = [name for name in MODEL_ORDER if name in predictions]
    return available[-1] if available else "ground_truth"


def _best_transformer_name(metrics: dict[str, ModelMetric]) -> str | None:
    candidates = [name for name in ("compact_transformer", "causal_transformer", "full_transformer") if name in metrics]
    if not candidates:
        return None
    return max(candidates, key=lambda name: metrics[name].score)


def _checkpoint_step(path: Path) -> int:
    stem = path.stem
    if stem.startswith("model_"):
        try:
            return int(stem.rsplit("_", 1)[1])
        except ValueError:
            return 0
    if stem == "final_model":
        return 10**12
    if stem == "best_model":
        return 10**12 - 1
    return 0


def _checkpoint_sequence(run_dir: Path, frame_count: int) -> list[Path]:
    checkpoints_dir = run_dir / "checkpoints"
    checkpoints = sorted(checkpoints_dir.glob("model_*.pt"), key=_checkpoint_step)
    if not checkpoints:
        for name in ("latest_model.pt", "best_model.pt", "final_model.pt"):
            candidate = checkpoints_dir / name
            if candidate.exists():
                checkpoints = [candidate]
                break
    if not checkpoints:
        return []
    return [checkpoints[index] for index in _sample_indices(len(checkpoints), frame_count)]


def _gif_model_names(predictions: dict[str, np.ndarray]) -> list[str]:
    preferred = ["sequence_lstm", "bidirectional_lstm", "full_transformer"]
    names = [name for name in preferred if name in predictions]
    if "full_transformer" not in names and "compact_transformer" in predictions:
        names.append("compact_transformer")
    return names[:4]


def _predict_training_evolution(
    events: np.ndarray,
    metrics: dict[str, ModelMetric],
    final_predictions: dict[str, np.ndarray],
    device: torch.device,
    frame_count: int,
) -> tuple[list[dict[str, np.ndarray]], list[str]]:
    model_names = _gif_model_names(final_predictions)
    histories: dict[str, list[np.ndarray]] = {}
    step_history: dict[str, list[int]] = {}
    for model_name in model_names:
        metric = metrics.get(model_name)
        if metric is None or metric.run_dir is None:
            histories[model_name] = [final_predictions[model_name]] * frame_count
            step_history[model_name] = [0] * frame_count
            continue
        checkpoints = _checkpoint_sequence(metric.run_dir, frame_count)
        if not checkpoints:
            histories[model_name] = [final_predictions[model_name]] * frame_count
            step_history[model_name] = [0] * frame_count
            continue
        model_frames: list[np.ndarray] = []
        model_steps: list[int] = []
        for checkpoint in checkpoints:
            model_frames.append(_predict_checkpoint(events, checkpoint, model_name, device))
            model_steps.append(_checkpoint_step(checkpoint))
        histories[model_name] = model_frames
        step_history[model_name] = model_steps

    frame_predictions: list[dict[str, np.ndarray]] = []
    frame_labels: list[str] = []
    for frame_index in range(frame_count):
        current = dict(final_predictions)
        steps = []
        for model_name in model_names:
            model_frames = histories[model_name]
            frame_slot = min(frame_index, len(model_frames) - 1)
            current[model_name] = model_frames[frame_slot]
            step_value = step_history[model_name][frame_slot]
            if 0 < step_value < 10**11:
                steps.append(step_value)
        frame_predictions.append(current)
        frame_labels.append(f"training step {max(steps):,}" if steps else f"frame {frame_index + 1}/{frame_count}")
    return frame_predictions, frame_labels


def _model_ranking_axis(ax: plt.Axes, metrics: dict[str, ModelMetric], progress: float = 1.0) -> None:
    models = [name for name in MODEL_ORDER if name in metrics]
    if not models:
        ax.axis("off")
        return
    _style_axis(ax)
    scores = [metrics[name].score for name in models]
    y = np.arange(len(models))
    animated_scores = []
    for index, score in enumerate(scores):
        local_progress = float(np.clip(progress * len(models) - index, 0.0, 1.0))
        animated_scores.append(0.5 + (score - 0.5) * local_progress)
    ax.barh(y, animated_scores, color=[MODEL_COLORS[name] for name in models], edgecolor="#1f2933", linewidth=0.45)
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_TITLES[name] for name in models], fontsize=7.7)
    ax.invert_yaxis()
    ax.set_xlim(0.50, max(scores) + 0.055)
    ax.set_xlabel("Score", color=MUTED)
    ax.set_title("Suite Ranking", loc="left", fontsize=10.5, fontweight="bold", color=INK)
    for yi, score, animated in zip(y, scores, animated_scores):
        if animated > 0.505:
            ax.text(animated + 0.004, yi, f"{score:.3f}", va="center", fontsize=7.4, color=INK)


def _training_axis(
    ax: plt.Axes,
    metrics: dict[str, ModelMetric],
    progress: float = 1.0,
    val_steps: int = 500,
    loss: bool = False,
) -> None:
    _style_axis(ax)
    max_x = 1
    plotted = False
    arrays: list[tuple[str, np.ndarray]] = []
    for name in MODEL_ORDER:
        metric = metrics.get(name)
        if metric is None or metric.run_dir is None:
            continue
        array_path = metric.run_dir / ("val_losses.npy" if loss else "val_accs.npy")
        if not array_path.exists():
            continue
        values = np.load(array_path)
        if values.size == 0:
            continue
        arrays.append((name, values))
        max_x = max(max_x, int((values.size - 1) * val_steps))
    for name, values in arrays:
        upto = min(values.size, max(1, int(np.ceil(values.size * progress))))
        x = np.arange(upto) * val_steps
        ax.plot(x, values[:upto], color=MODEL_COLORS[name], linewidth=1.7, label=MODEL_TITLES[name])
        plotted = True
    ax.set_xlim(0, max_x)
    if loss:
        ax.set_title("Validation Loss", loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_ylabel("Loss", color=MUTED)
    else:
        ax.set_title("Validation Accuracy", loc="left", fontsize=10.5, fontweight="bold", color=INK)
        ax.set_ylabel("Accuracy", color=MUTED)
        if arrays:
            all_values = np.concatenate([values for _name, values in arrays])
            ax.set_ylim(max(0.40, float(all_values.min()) - 0.02), min(0.75, float(all_values.max()) + 0.025))
    ax.set_xlabel("Step", color=MUTED)
    if plotted:
        ax.legend(fontsize=6.1, ncol=2, loc="lower right" if not loss else "upper right", frameon=True)
    else:
        ax.text(0.5, 0.5, "No training history", ha="center", va="center", transform=ax.transAxes, color=MUTED)


def _confusion_for_model(name: str, metrics: dict[str, ModelMetric]) -> np.ndarray | None:
    metric = metrics.get(name)
    if metric is None or metric.run_dir is None:
        return None
    path = metric.run_dir / "val_confusion_matrix.npy"
    if not path.exists():
        return None
    return normalize_confusion_matrix(np.load(path))


def _confusion_axis(ax: plt.Axes, matrix: np.ndarray | None, title: str, progress: float = 1.0) -> None:
    if matrix is None:
        ax.axis("off")
        ax.text(0.5, 0.5, "No confusion matrix", ha="center", va="center", transform=ax.transAxes, color=MUTED)
        return
    display = matrix * max(0.0, min(progress, 1.0))
    ax.set_facecolor(PANEL_INNER)
    ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(INSTRUMENT_LABELS, rotation=32, ha="right", fontsize=7)
    ax.set_yticklabels(INSTRUMENT_LABELS, fontsize=7)
    ax.set_xlabel("Prediction", color=MUTED, fontsize=8)
    ax.set_ylabel("Truth", color=MUTED, fontsize=8)
    for row in range(N_CLASSES):
        for col in range(N_CLASSES):
            if progress > 0.30:
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7, color=INK)


def _mini_confusion_axis(ax: plt.Axes, matrix: np.ndarray | None, title: str, progress: float = 1.0, show_y: bool = False) -> None:
    if matrix is None:
        ax.axis("off")
        return
    display = matrix * (0.18 + 0.82 * max(0.0, min(progress, 1.0)))
    ax.set_facecolor(PANEL_INNER)
    ax.imshow(display, cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_title(title, loc="left", fontsize=8.5, fontweight="bold", color=INK, pad=4)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(INSTRUMENT_LABELS, rotation=45, ha="right", fontsize=5.8)
    ax.set_yticklabels(INSTRUMENT_LABELS if show_y else [""] * N_CLASSES, fontsize=5.8)
    ax.tick_params(length=0, pad=1)
    for row in range(N_CLASSES):
        for col in range(N_CLASSES):
            if progress > 0.45:
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=5.2, color=INK)


def _label_share_axis(ax: plt.Axes, labels: np.ndarray, title: str, progress: float = 1.0) -> None:
    shares = _label_share(labels) * max(0.0, min(progress, 1.0))
    _style_axis(ax, grid=False)
    bars = ax.barh(np.arange(N_CLASSES), shares, color=LABEL_COLORS, edgecolor="#1f2933", linewidth=0.35)
    ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(INSTRUMENT_LABELS, fontsize=7.5)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Share", color=MUTED, fontsize=8)
    for bar, share in zip(bars, _label_share(labels)):
        if progress > 0.50:
            ax.text(min(share + 0.012, 0.96), bar.get_y() + bar.get_height() / 2, f"{share:.0%}", va="center", fontsize=7.2, color=INK)


def _paper_prediction_rows(predictions: dict[str, np.ndarray]) -> list[tuple[str, str | None, np.ndarray | None]]:
    rows: list[tuple[str, str | None, np.ndarray | None]] = [
        ("Mixture\ninput", None, None),
        ("Ground\ntruth", "ground_truth", predictions["ground_truth"]),
    ]
    model_rows = ["sequence_lstm", "bidirectional_lstm", "full_transformer"]
    if "full_transformer" not in predictions and "compact_transformer" in predictions:
        model_rows[-1] = "compact_transformer"
    for model_name in model_rows:
        if model_name in predictions:
            rows.append((MODEL_TITLES[model_name].replace(" ", "\n", 1), model_name, predictions[model_name]))
    return rows[:6]


def _stacked_arrangement_axis(
    ax: plt.Axes,
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, ModelMetric],
    progress: float = 1.0,
    title: str = "Aligned Instrumentation Predictions",
    reveal_notes: bool = False,
) -> None:
    rows = _paper_prediction_rows(predictions)
    starts = events[:, 0].astype(float) / TIME_STEPS_PER_BEAT
    durations = np.maximum(events[:, 2].astype(float) / TIME_STEPS_PER_BEAT, 0.05)
    pitches = events[:, 1].astype(float)
    x_max = max(float(np.max(starts + durations)), 1.0)
    reveal_x = x_max * max(0.0, min(progress, 1.0))
    visible = starts <= reveal_x if reveal_notes and progress < 0.999 else np.ones_like(starts, dtype=bool)
    pitch_min = float(pitches.min())
    pitch_span = max(float(pitches.max() - pitches.min()), 1.0)
    row_height = 0.76
    row_pad = 0.12

    ax.set_facecolor(PANEL_INNER)
    for spine in ax.spines.values():
        spine.set_color("#d9ccb8")
        spine.set_linewidth(0.9)
    ax.tick_params(colors=MUTED, labelsize=8)

    for row_index, (row_label, model_name, row_labels) in enumerate(rows):
        y_base = len(rows) - row_index - 1
        y_values = y_base + row_pad + ((pitches - pitch_min) / pitch_span) * row_height
        row_visible = visible
        if row_labels is None:
            ax.hlines(
                y_values[row_visible],
                starts[row_visible],
                starts[row_visible] + durations[row_visible],
                color="#333333",
                linewidth=1.05,
                alpha=0.62,
            )
            ax.hlines(
                y_values[row_visible][::4],
                starts[row_visible][::4],
                starts[row_visible][::4] + durations[row_visible][::4],
                color="#777777",
                linewidth=0.85,
                alpha=0.48,
                linestyles=(0, (2.0, 2.0)),
            )
        else:
            labels = row_labels[: len(events)].astype(int)
            for label_index in range(N_CLASSES):
                mask = (labels == label_index) & row_visible
                if np.any(mask):
                    ax.hlines(
                        y_values[mask],
                        starts[mask],
                        starts[mask] + durations[mask],
                        color=LABEL_COLORS[label_index],
                        linewidth=1.65,
                        alpha=0.86,
                    )
        if row_index < len(rows) - 1:
            ax.axhline(y_base - 0.06, color="#d3c6b4", linewidth=0.8)

    if reveal_notes and progress < 0.999:
        ax.axvline(reveal_x, color=INK, linewidth=1.25, alpha=0.42)

    ax.set_title(title, loc="left", fontsize=11.4, fontweight="bold", color=INK, pad=7)
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(-0.18, len(rows) - 0.05)
    ax.set_xlabel("Beat", color=MUTED)
    ax.set_yticks([len(rows) - index - 1 + 0.50 for index in range(len(rows))])
    ax.set_yticklabels([row[0] for row in rows], fontsize=8.5)
    ax.grid(axis="x", color=GRID, alpha=0.17, linewidth=0.8)


def _save_paper_arrangement_panel(
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, ModelMetric],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(15.4, 7.2), facecolor=PANEL_FACE)
    _stacked_arrangement_axis(ax, events, predictions, metrics, progress=1.0, title="Paper-Style Prediction Comparison")
    fig.legend(handles=_instrument_legend_handles(), ncol=N_CLASSES, loc="lower center", bbox_to_anchor=(0.55, 0.01), frameon=True, fontsize=8)
    fig.tight_layout(rect=[0.02, 0.06, 0.98, 0.98])
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_prediction_grid(events: np.ndarray, predictions: dict[str, np.ndarray], metrics: dict[str, ModelMetric], output_path: Path) -> None:
    order = ["ground_truth"] + MODEL_ORDER
    fig, axes = plt.subplots(4, 2, figsize=(17, 18.4), facecolor=PANEL_FACE)
    truth = predictions["ground_truth"]
    for ax, name in zip(axes.flat, order):
        labels = predictions.get(name)
        if labels is None:
            ax.axis("off")
            continue
        if name == "ground_truth":
            subtitle = "Reference labels from the multitrack arrangement"
        else:
            metric = metrics.get(name)
            score = metric.score if metric is not None else _agreement(labels, truth)
            subtitle = f"{MODEL_FAMILIES.get(name, '')} | suite score {score:.3f} | sample agreement {_agreement(labels, truth):.3f}"
        _roll_axis(ax, events, labels, MODEL_TITLES.get(name, name), subtitle)
    fig.legend(handles=_instrument_legend_handles(), ncol=N_CLASSES, loc="upper center", bbox_to_anchor=(0.58, 0.958), frameon=True)
    fig.suptitle(
        "Automatic Instrumentation: One Piano-Roll, Many Arrangers",
        x=0.04,
        y=0.992,
        ha="left",
        fontsize=20,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.925])
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_accuracy_dashboard(metrics: dict[str, ModelMetric], output_path: Path, val_steps: int = 500) -> None:
    models = [name for name in MODEL_ORDER if name in metrics]
    fig = plt.figure(figsize=(16, 10), facecolor=PANEL_FACE, constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.35], height_ratios=[1.0, 1.0])
    ax_bar = fig.add_subplot(gs[:, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_loss = fig.add_subplot(gs[1, 1])
    _model_ranking_axis(ax_bar, metrics)
    ax_bar.set_title("Model Suite Ranking", loc="left", fontsize=16, fontweight="bold", color=INK)
    _training_axis(ax_acc, metrics, progress=1.0, val_steps=val_steps, loss=False)
    ax_acc.set_title("Validation Accuracy Over Training", loc="left", fontsize=16, fontweight="bold", color=INK)
    _training_axis(ax_loss, metrics, progress=1.0, val_steps=val_steps, loss=True)
    ax_loss.set_title("Validation Loss Over Training", loc="left", fontsize=16, fontweight="bold", color=INK)
    fig.suptitle("Automatic Instrumentation Training Dashboard", x=0.02, y=0.995, ha="left", fontsize=24, fontweight="bold", color=INK)
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _save_confusion_wall(metrics: dict[str, ModelMetric], output_path: Path) -> None:
    models = [name for name in MODEL_ORDER if name != "pitch_zones" and name in metrics and metrics[name].run_dir is not None]
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.2), facecolor=PANEL_FACE)
    last_image = None
    for ax, name in zip(axes.flat, models):
        matrix = _confusion_for_model(name, metrics)
        if matrix is None:
            ax.axis("off")
            continue
        last_image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(f"{MODEL_TITLES[name]} | acc {metrics[name].score:.3f}", loc="left", fontsize=12, fontweight="bold", color=INK)
        ax.set_xticks(range(N_CLASSES))
        ax.set_yticks(range(N_CLASSES))
        ax.set_xticklabels(INSTRUMENT_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_yticklabels(INSTRUMENT_LABELS, fontsize=8)
        ax.set_xlabel("Prediction", color=MUTED)
        ax.set_ylabel("Truth", color=MUTED)
        for row in range(N_CLASSES):
            for col in range(N_CLASSES):
                ax.text(col, row, f"{matrix[row, col]:.2f}", ha="center", va="center", fontsize=7, color=INK)
    for ax in axes.flat[len(models) :]:
        ax.axis("off")
    if last_image is not None:
        fig.colorbar(last_image, ax=axes.ravel().tolist(), fraction=0.018, pad=0.012, label="Row-normalized share")
    fig.suptitle("Confusion Matrix Wall: Where Each Model Swaps Instruments", x=0.02, y=0.995, ha="left", fontsize=22, fontweight="bold", color=INK)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _save_cover_panel(events: np.ndarray, predictions: dict[str, np.ndarray], metrics: dict[str, ModelMetric], output_path: Path) -> None:
    fig = plt.figure(figsize=(18, 10.5), facecolor=PANEL_FACE, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[0.38, 1.0, 1.0], width_ratios=[1.15, 1.15, 0.9])
    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.82, "Automatic Instrumentation", fontsize=30, fontweight="bold", color=INK, va="top")
    title_ax.text(0.0, 0.24, "From one symbolic note stream to instrument-aware arrangements, with model-suite evidence.", fontsize=13, color=MUTED)
    ax_gt = fig.add_subplot(gs[1, :2])
    _roll_axis(ax_gt, events, predictions["ground_truth"], "Ground truth")
    best_name = _best_model_name(metrics, predictions)
    ax_best = fig.add_subplot(gs[2, :2])
    _roll_axis(ax_best, events, predictions[best_name], f"Best trained model: {MODEL_TITLES.get(best_name, best_name)}")
    ax_rank = fig.add_subplot(gs[1:, 2])
    _model_ranking_axis(ax_rank, metrics)
    ax_rank.set_title("Suite score", loc="left", fontsize=14, fontweight="bold", color=INK)
    fig.legend(handles=_instrument_legend_handles(), ncol=N_CLASSES, loc="lower center", bbox_to_anchor=(0.42, -0.01), frameon=True)
    fig.savefig(output_path, dpi=170, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def _readme_panel_figure(
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, ModelMetric],
    progress: float = 1.0,
    val_steps: int = 500,
    display_predictions: dict[str, np.ndarray] | None = None,
    frame_label: str | None = None,
) -> plt.Figure:
    display_predictions = display_predictions or predictions
    truth = predictions["ground_truth"]
    best_name = _best_model_name(metrics, predictions)
    best_labels = display_predictions.get(best_name, predictions[best_name])
    best_metric = metrics.get(best_name)
    best_score = best_metric.score if best_metric is not None else _agreement(best_labels, truth)
    transformer_name = _best_transformer_name(metrics)
    transformer_text = "n/a" if transformer_name is None else f"{MODEL_TITLES[transformer_name]} {metrics[transformer_name].score:.3f}"

    fig = plt.figure(figsize=(15.0, 8.8), facecolor=PANEL_FACE, constrained_layout=True)
    grid = fig.add_gridspec(
        4,
        5,
        height_ratios=[0.30, 1.24, 1.24, 0.92],
        width_ratios=[1.18, 1.18, 1.18, 0.95, 0.95],
        hspace=0.18,
        wspace=0.12,
    )

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.70, "Automatic Instrumentation", fontsize=21.5, fontweight="bold", color=INK)
    title_ax.text(
        0.0,
        0.18,
        "Ground-truth tracks, learned arrangers, validation curves, and confusion structure in one generated panel.",
        fontsize=9.5,
        color=MUTED,
    )

    arrangement_ax = fig.add_subplot(grid[1:3, 0:3])
    _stacked_arrangement_axis(
        arrangement_ax,
        events,
        display_predictions,
        metrics,
        progress=progress,
        title="Same Note Stream, Different Arrangers",
        reveal_notes=False,
    )

    ranking_ax = fig.add_subplot(grid[1, 3:5])
    _model_ranking_axis(ranking_ax, metrics, progress=progress)

    training_ax = fig.add_subplot(grid[2, 3:5])
    _training_axis(training_ax, metrics, progress=progress, val_steps=val_steps, loss=False)

    confusion_names = [name for name in ("sequence_lstm", "bidirectional_lstm", "full_transformer") if _confusion_for_model(name, metrics) is not None]
    if "full_transformer" not in confusion_names and _confusion_for_model("compact_transformer", metrics) is not None:
        confusion_names.append("compact_transformer")
    if not confusion_names and _confusion_for_model(best_name, metrics) is not None:
        confusion_names = [best_name]
    confusion_names = confusion_names[:3]
    confusion_grid = grid[3, 0:3].subgridspec(1, max(1, len(confusion_names)), wspace=0.18)
    if confusion_names:
        for index, model_name in enumerate(confusion_names):
            confusion_ax = fig.add_subplot(confusion_grid[0, index])
            _mini_confusion_axis(
                confusion_ax,
                _confusion_for_model(model_name, metrics),
                MODEL_TITLES.get(model_name, model_name),
                progress=progress,
                show_y=index == 0,
            )
    else:
        confusion_ax = fig.add_subplot(confusion_grid[0, 0])
        confusion_ax.axis("off")
        confusion_ax.text(0.5, 0.5, "No confusion matrices", ha="center", va="center", color=MUTED, transform=confusion_ax.transAxes)

    summary_ax = fig.add_subplot(grid[3, 3:5])
    summary_ax.set_facecolor(PANEL_INNER)
    summary_ax.set_xticks([])
    summary_ax.set_yticks([])
    for spine in summary_ax.spines.values():
        spine.set_color("#d9ccb8")
        spine.set_linewidth(0.9)
    summary_ax.text(0.04, 0.86, "Suite Summary", fontsize=11, fontweight="bold", color=INK, transform=summary_ax.transAxes)
    summary_ax.text(0.04, 0.63, f"Rendered notes: {len(events):,}", fontsize=8.5, color=MUTED, transform=summary_ax.transAxes)
    summary_ax.text(0.04, 0.44, f"Best trained model: {MODEL_TITLES.get(best_name, best_name)} ({best_score:.3f})", fontsize=8.5, color=MUTED, transform=summary_ax.transAxes)
    summary_ax.text(0.04, 0.25, f"Best transformer: {transformer_text}", fontsize=8.5, color=MUTED, transform=summary_ax.transAxes)
    progress_text = frame_label or f"panel progress {progress:.0%}"
    summary_ax.text(0.04, 0.08, progress_text, fontsize=8.0, color=MUTED, transform=summary_ax.transAxes)
    return fig


def _save_readme_static_panel(
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, ModelMetric],
    output_path: Path,
    val_steps: int = 500,
) -> None:
    fig = _readme_panel_figure(events, predictions, metrics, progress=1.0, val_steps=val_steps, frame_label="final checkpoint view")
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_readme_animated_panel(
    events: np.ndarray,
    predictions: dict[str, np.ndarray],
    metrics: dict[str, ModelMetric],
    output_path: Path,
    val_steps: int = 500,
    evolution_predictions: list[dict[str, np.ndarray]] | None = None,
    evolution_labels: list[str] | None = None,
) -> None:
    frame_count = len(evolution_predictions) if evolution_predictions is not None else 18
    frames: list[Image.Image] = []
    for frame_index in range(frame_count):
        progress = (frame_index + 1) / frame_count
        display_predictions = evolution_predictions[frame_index] if evolution_predictions is not None else predictions
        frame_label = evolution_labels[frame_index] if evolution_labels is not None else None
        fig = _readme_panel_figure(
            events,
            predictions,
            metrics,
            progress=progress,
            val_steps=val_steps,
            display_predictions=display_predictions,
            frame_label=frame_label,
        )
        frames.append(_figure_to_palette_frame(fig, dpi=103))
    frames.extend([frames[-1].copy(), frames[-1].copy(), frames[-1].copy()])
    _write_gif(frames, output_path, duration_ms=170)


def _write_readme_snippet(output_dir: Path, manifest: dict) -> None:
    lines = [
        "# Automatic Instrumentation Visuals",
        "",
        "Use these generated assets in the project README:",
        "",
        "```markdown",
        "![Automatic Instrumentation Animated Panel](visual/readme_automatic_instrumentation_animated_panel.gif)",
        "",
        "![Automatic Instrumentation Static Panel](visual/readme_automatic_instrumentation_static_panel.png)",
        "",
        "![Automatic Instrumentation Model Comparison](visual/model_prediction_comparison.png)",
        "```",
        "",
        "Generated files:",
    ]
    for key, value in manifest["paths"].items():
        lines.append(f"- `{key}`: `{value}`")
    (output_dir / "README_visuals_snippet.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def render_basic_gallery(
    data_dir: Path | None,
    split: str,
    output_dir: Path,
    run_dir: Path | None,
    evaluation_dir: Path | None,
    max_files: int,
    max_notes: int,
    val_steps: int,
) -> dict:
    ensure_dir(output_dir)
    files = getattr(collect_split_files(data_dir), split)
    arrays = _load_arrays(files, max_files=max_files)
    if not arrays:
        raise FileNotFoundError("No event arrays found for rendering.")

    sample = arrays[0]
    paths = {
        "ground_truth_roll": output_dir / "event_roll_ground_truth.png",
        "pitch_zone_roll": output_dir / "event_roll_pitch_zones.png",
        "label_distribution": output_dir / "label_distribution.png",
        "pitch_label_heatmap": output_dir / "pitch_label_heatmap.png",
        "pitch_zone_map": output_dir / "pitch_zone_map.png",
        "training_curves": output_dir / "training_curves.png",
        "confusion_matrices": output_dir / "confusion_matrices.png",
        "overview": output_dir / "automatic_instrumentation_overview.png",
    }

    _event_roll(sample, paths["ground_truth_roll"], "Ground-Truth Instrumentation", max_notes=max_notes)
    predicted_labels = pitch_zone_predict_events(sample)
    _event_roll(sample, paths["pitch_zone_roll"], "Fixed Pitch-Zone Instrumentation", predicted_labels, max_notes=max_notes)
    _save_label_distribution(arrays, paths["label_distribution"])
    _save_pitch_label_heatmap(arrays, paths["pitch_label_heatmap"])
    _save_pitch_zone_map(paths["pitch_zone_map"])

    training_curves_written = False
    if run_dir is not None:
        training_curves_written = _save_training_curves(run_dir, paths["training_curves"], val_steps=val_steps)

    matrix_path = _find_confusion_matrix(run_dir, evaluation_dir)
    confusion_written = False
    if matrix_path is not None:
        confusion_written = _save_confusion_matrix(matrix_path, paths["confusion_matrices"])

    _build_overview_panel(
        [
            (paths["ground_truth_roll"], "Ground Truth"),
            (paths["pitch_zone_roll"], "Pitch-Zone Rule"),
            (paths["label_distribution"], "Label Distribution"),
            (paths["pitch_label_heatmap"], "Pitch Usage"),
            (paths["training_curves"], "Training Curves"),
            (paths["confusion_matrices"], "Confusion Matrices"),
        ],
        paths["overview"],
    )

    return {
        "output_dir": str(output_dir),
        "split": split,
        "files_used": min(len(files), max_files),
        "training_curves_written": training_curves_written,
        "confusion_written": confusion_written,
        "paths": {key: str(value) for key, value in paths.items() if value.exists()},
    }


def render_suite_visuals(
    suite_root: Path,
    output_dir: Path,
    sample_file: Path | None,
    sample_dir: Path,
    max_beats: int,
    max_notes: int,
    device: torch.device,
    data_dir: Path | None,
    split: str,
    max_files: int,
    val_steps: int,
    gif_frames: int,
    include_basic_gallery: bool = True,
) -> dict:
    ensure_dir(output_dir)
    metrics = _read_summary(suite_root)
    sample_path = sample_file or _select_sample(sample_dir, max_beats=max_beats)
    events = _load_sample(sample_path, max_beats=max_beats, max_notes=max_notes)
    predictions = _predict_suite(events, metrics, device=device)
    evolution_predictions, evolution_labels = _predict_training_evolution(
        events,
        metrics,
        predictions,
        device=device,
        frame_count=gif_frames,
    )

    paths = {
        "readme_static_panel": output_dir / "readme_automatic_instrumentation_static_panel.png",
        "readme_animated_panel": output_dir / "readme_automatic_instrumentation_animated_panel.gif",
        "cover_panel": output_dir / "cover_panel.png",
        "paper_arrangement_panel": output_dir / "paper_arrangement_panel.png",
        "model_prediction_comparison": output_dir / "model_prediction_comparison.png",
        "training_dashboard": output_dir / "training_dashboard.png",
        "confusion_matrix_wall": output_dir / "confusion_matrix_wall.png",
        "manifest": output_dir / "visual_manifest.json",
    }

    _save_readme_static_panel(events, predictions, metrics, paths["readme_static_panel"], val_steps=val_steps)
    _save_readme_animated_panel(
        events,
        predictions,
        metrics,
        paths["readme_animated_panel"],
        val_steps=val_steps,
        evolution_predictions=evolution_predictions,
        evolution_labels=evolution_labels,
    )
    _save_cover_panel(events, predictions, metrics, paths["cover_panel"])
    _save_paper_arrangement_panel(events, predictions, metrics, paths["paper_arrangement_panel"])
    _save_prediction_grid(events, predictions, metrics, paths["model_prediction_comparison"])
    _save_accuracy_dashboard(metrics, paths["training_dashboard"], val_steps=val_steps)
    _save_confusion_wall(metrics, paths["confusion_matrix_wall"])

    basic_summary = None
    if include_basic_gallery:
        best_name = _best_model_name(metrics, predictions)
        run_dir = metrics[best_name].run_dir if best_name in metrics else None
        basic_summary = render_basic_gallery(
            data_dir=data_dir,
            split=split,
            output_dir=output_dir,
            run_dir=run_dir,
            evaluation_dir=None,
            max_files=max_files,
            max_notes=max_notes,
            val_steps=val_steps,
        )

    manifest = {
        "suite_root": str(suite_root),
        "output_dir": str(output_dir),
        "sample_file": str(sample_path),
        "events_rendered": int(len(events)),
        "max_beats": max_beats,
        "device": str(device),
        "paths": {key: str(path) for key, path in paths.items() if key != "manifest" and path.exists()},
        "scores": {name: metric.score for name, metric in metrics.items()},
        "sample_agreement": {
            name: _agreement(labels, predictions["ground_truth"])
            for name, labels in predictions.items()
            if name != "ground_truth"
        },
        "gif_frames": gif_frames,
        "basic_gallery": basic_summary,
    }
    save_json(paths["manifest"], manifest)
    _write_readme_snippet(output_dir, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Render automatic instrumentation visuals and README-ready panels.")
    parser.add_argument("--suite-root", type=Path, default=None, help="Completed model-suite output root. Enables checkpoint-based README panels.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-dir", type=Path, default=None)
    parser.add_argument("--evaluation-dir", type=Path, default=AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT / "evaluation")
    parser.add_argument("--sample-file", type=Path, default=None)
    parser.add_argument("--sample-dir", type=Path, default=AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR / "samples" / "processed")
    parser.add_argument("--max-files", type=int, default=250)
    parser.add_argument("--max-notes", type=int, default=900)
    parser.add_argument("--max-beats", type=int, default=128)
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--gif-frames", type=int, default=16)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--skip-basic-gallery", action="store_true")
    args = parser.parse_args()

    if args.suite_root is not None:
        suite_root = args.suite_root.resolve()
        output_dir = (args.output_dir or suite_root / "visual").resolve()
        device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
        summary = render_suite_visuals(
            suite_root=suite_root,
            output_dir=output_dir,
            sample_file=args.sample_file.resolve() if args.sample_file else None,
            sample_dir=args.sample_dir.resolve(),
            max_beats=args.max_beats,
            max_notes=args.max_notes,
            device=device,
            data_dir=args.data_dir,
            split=args.split,
            max_files=args.max_files,
            val_steps=args.val_steps,
            gif_frames=args.gif_frames,
            include_basic_gallery=not args.skip_basic_gallery,
        )
    else:
        output_dir = (args.output_dir or AUTOMATIC_INSTRUMENTATION_VISUAL_DIR).resolve()
        summary = render_basic_gallery(
            data_dir=args.data_dir,
            split=args.split,
            output_dir=output_dir,
            run_dir=args.run_dir,
            evaluation_dir=args.evaluation_dir,
            max_files=args.max_files,
            max_notes=args.max_notes,
            val_steps=args.val_steps,
        )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
