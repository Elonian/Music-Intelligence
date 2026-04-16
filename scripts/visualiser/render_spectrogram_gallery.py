#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import sys
from collections import Counter
from pathlib import Path

import imageio.v2 as imageio
import librosa
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spectrogram_classification import build_feature_bundle, discover_audio_examples, load_audio_waveform
from scripts.visualiser.visualiser import (
    save_confusion_matrix,
    save_feature_cycle_gif,
    save_feature_heatmap,
    save_training_curve,
    save_waveform_plot,
)
from utils import (
    SPECTROGRAM_DATA_ROOT,
    SPECTROGRAM_EVALUATION_DIR,
    VISUAL_SPECTROGRAM_FEATURE_DIR,
    VISUAL_SPECTROGRAM_MODEL_DIR,
    VISUAL_SPECTROGRAM_README_DIR,
    ensure_dir,
    load_json,
)


CLASS_ORDER = ["guitar_acoustic", "guitar_electronic", "vocal_acoustic", "vocal_synthetic"]
CLASS_COLORS = {
    "guitar_acoustic": "#2f6fbb",
    "guitar_electronic": "#d97706",
    "vocal_acoustic": "#2f855a",
    "vocal_synthetic": "#9f3a64",
}
MODEL_LABELS = {
    "mfcc_mlp": "MFCC MLP",
    "spectrogram_cnn": "STFT CNN",
    "mel_spectrogram_cnn": "Mel CNN",
    "cqt_cnn": "CQT CNN",
    "augmented_cqt_cnn": "Aug. CQT CNN",
    "four_class_cnn": "Four-Class MLP",
}


def _short_label(label: str) -> str:
    return label.replace("_", " ").title()


def _resolve_audio_root(data_dir: Path) -> Path:
    direct = Path(data_dir)
    candidates = [
        direct,
        direct / "nsynth_subset",
    ]
    for candidate in candidates:
        if candidate.exists() and any(candidate.rglob("*.wav")):
            return candidate
    return direct


def _load_experiment_metrics(metrics_dir: Path) -> list[dict]:
    metrics_dir = Path(metrics_dir)
    metrics = []
    for path in sorted(metrics_dir.glob("*_metrics.json")):
        metrics.append(load_json(path))
    return metrics


def _labels_from_metrics(metrics: dict) -> list[str]:
    labels = metrics.get("labels", [])
    if isinstance(labels, dict):
        return [labels[str(index)] for index in range(len(labels))]
    return list(labels)


def _history_value(history: list[dict], key: str, fallback: str, frame_index: int) -> float:
    if not history:
        return 0.0
    row = history[min(frame_index, len(history) - 1)]
    return float(row.get(key, row.get(fallback, 0.0)))


def _load_waveform(file_path: str | Path, sample_rate: int) -> tuple[np.ndarray, int]:
    waveform, loaded_sample_rate = librosa.load(str(file_path), sr=sample_rate, mono=True)
    waveform = np.asarray(waveform, dtype=np.float32)
    peak = float(np.max(np.abs(waveform))) if waveform.size else 0.0
    if peak > 1e-8:
        waveform = waveform / peak
    return waveform, loaded_sample_rate


def _feature_views(waveform: np.ndarray, sample_rate: int) -> dict[str, np.ndarray]:
    stft = librosa.stft(waveform, n_fft=1024, hop_length=256)
    stft_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    mel = librosa.feature.melspectrogram(y=waveform, sr=sample_rate, n_mels=96, hop_length=256)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    cqt = librosa.cqt(waveform, sr=sample_rate, hop_length=256, bins_per_octave=12, n_bins=72)
    cqt_db = librosa.amplitude_to_db(np.abs(cqt), ref=np.max)
    mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=32)
    return {
        "Waveform": waveform,
        "STFT Power": stft_db,
        "Mel-Spectrogram": mel_db,
        "Constant-Q": cqt_db,
        "MFCC": mfcc,
    }


def _normalise_map(feature_map: np.ndarray) -> np.ndarray:
    feature_map = np.asarray(feature_map, dtype=float)
    if feature_map.size == 0:
        return feature_map
    low = float(np.nanpercentile(feature_map, 2))
    high = float(np.nanpercentile(feature_map, 98))
    if high <= low:
        return np.zeros_like(feature_map)
    return np.clip((feature_map - low) / (high - low), 0.0, 1.0)


def _plot_heat(ax: plt.Axes, feature_map: np.ndarray, title: str, cmap: str = "magma") -> None:
    ax.imshow(feature_map, aspect="auto", origin="lower", cmap=cmap)
    ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _plot_waveform(ax: plt.Axes, waveform: np.ndarray, sample_rate: int, title: str, color: str) -> None:
    max_points = 2200
    stride = max(1, waveform.shape[0] // max_points)
    indices = np.arange(0, waveform.shape[0], stride)
    time_axis = indices / float(sample_rate)
    ax.plot(time_axis, waveform[indices], color=color, linewidth=0.9)
    ax.fill_between(time_axis, waveform[indices], 0.0, color=color, alpha=0.12)
    ax.axhline(0.0, color="#555555", linewidth=0.6, alpha=0.6)
    ax.set_title(title, fontsize=10, loc="left", pad=4)
    ax.set_xlim(0.0, waveform.shape[0] / float(sample_rate))
    ax.set_ylim(-1.05, 1.05)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _select_class_examples(data_dir: Path, sample_rate: int) -> list[tuple[str, str, np.ndarray, int]]:
    examples = discover_audio_examples(_resolve_audio_root(data_dir), label_mode="family")
    selected = {}
    for example in examples:
        if example.family_label in CLASS_ORDER and example.family_label not in selected:
            waveform, loaded_sample_rate = _load_waveform(example.file_path, sample_rate)
            selected[example.family_label] = (example.file_path, waveform, loaded_sample_rate)
    return [(label, *selected[label]) for label in CLASS_ORDER if label in selected]


def _render_sample_feature_views(data_dir: Path, visual_dir: Path, sample_rate: int) -> None:
    examples = discover_audio_examples(data_dir)
    if not examples:
        raise FileNotFoundError(f"No audio files found under {data_dir}.")

    sample_example = examples[0]
    waveform, loaded_sample_rate = load_audio_waveform(sample_example.file_path, target_sample_rate=sample_rate)
    feature_bundle = build_feature_bundle(waveform, loaded_sample_rate)

    save_waveform_plot(
        waveform.squeeze(0).cpu().numpy(),
        loaded_sample_rate,
        visual_dir / "sample_waveform.png",
        f"Sample Waveform: {sample_example.label}",
        color="#2f6fbb",
    )
    save_feature_heatmap(feature_bundle["mfcc_map"].cpu().numpy(), visual_dir / "sample_mfcc.png", "Sample MFCC")
    save_feature_heatmap(
        feature_bundle["spectrogram"].cpu().numpy(),
        visual_dir / "sample_spectrogram.png",
        "Sample Spectrogram",
    )
    save_feature_heatmap(
        feature_bundle["mel_spectrogram"].cpu().numpy(),
        visual_dir / "sample_mel_spectrogram.png",
        "Sample Mel-Spectrogram",
    )
    save_feature_heatmap(feature_bundle["cqt"].cpu().numpy(), visual_dir / "sample_cqt.png", "Sample CQT")
    save_feature_cycle_gif(
        [
            ("MFCC", feature_bundle["mfcc_map"].cpu().numpy()),
            ("Spectrogram", feature_bundle["spectrogram"].cpu().numpy()),
            ("Mel-Spectrogram", feature_bundle["mel_spectrogram"].cpu().numpy()),
            ("CQT", feature_bundle["cqt"].cpu().numpy()),
        ],
        visual_dir / "sample_feature_cycle.gif",
    )


def _render_class_gallery(data_dir: Path, visual_dir: Path, sample_rate: int) -> None:
    selected = _select_class_examples(data_dir, sample_rate)
    if not selected:
        return

    fig = plt.figure(figsize=(13.5, 8.2), facecolor="#f7f3ea", constrained_layout=True)
    grid = fig.add_gridspec(len(selected), 3, width_ratios=[1.2, 1.75, 1.75], hspace=0.15, wspace=0.08)
    for row, (label, _path, waveform, loaded_sample_rate) in enumerate(selected):
        views = _feature_views(waveform, loaded_sample_rate)
        color = CLASS_COLORS[label]
        _plot_waveform(fig.add_subplot(grid[row, 0]), views["Waveform"], loaded_sample_rate, _short_label(label), color)
        _plot_heat(fig.add_subplot(grid[row, 1]), _normalise_map(views["Mel-Spectrogram"]), "Mel Energy")
        _plot_heat(fig.add_subplot(grid[row, 2]), _normalise_map(views["Constant-Q"]), "Constant-Q", cmap="viridis")
    fig.suptitle("Four-Class Audio Signatures", fontsize=18, fontweight="bold", x=0.02, ha="left")
    ensure_dir(visual_dir)
    fig.savefig(visual_dir / "class_signature_gallery.png", dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_experiment_metrics(metrics_dir: Path, visual_dir: Path) -> list[dict]:
    ensure_dir(visual_dir)
    metrics = _load_experiment_metrics(metrics_dir)
    if not metrics:
        return []

    metrics = sorted(metrics, key=lambda row: row["test_accuracy"])
    labels = [MODEL_LABELS.get(row["output_name"], row["output_name"]) for row in metrics]
    scores = [float(row["test_accuracy"]) for row in metrics]
    colors = ["#8c8c8c", "#2f6fbb", "#2f855a", "#805ad5", "#d97706", "#9f3a64"]

    fig, ax = plt.subplots(figsize=(10.8, 4.7), facecolor="#f7f3ea")
    bars = ax.bar(labels, scores, color=colors[: len(scores)], edgecolor="#1f1f1f", linewidth=0.4)
    ax.set_ylim(0.78, 1.01)
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Spectrogram Classification Model Accuracy", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", labelrotation=15)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, score + 0.006, f"{score:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(visual_dir / "experiment_accuracy.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)

    for row in metrics:
        print(f"[Spectrogram Visualiser] Rendering metric plots: {row['output_name']}", flush=True)
        labels_for_matrix = _labels_from_metrics(row)
        save_confusion_matrix(
            np.asarray(row["confusion_matrix"], dtype=int),
            labels_for_matrix,
            visual_dir / f"{row['output_name']}_confusion.png",
            f"{MODEL_LABELS.get(row['output_name'], row['output_name'])} Confusion Matrix",
        )
        if row.get("history"):
            save_training_curve(
                row["history"],
                visual_dir / f"{row['output_name']}_training_curve.png",
                f"{MODEL_LABELS.get(row['output_name'], row['output_name'])} Training Curve",
            )
    return metrics


def _class_counts(data_dir: Path) -> Counter:
    root = _resolve_audio_root(data_dir)
    counts: Counter = Counter()
    for path in root.rglob("*.wav"):
        counts["_".join(path.name.split("_")[:2])] += 1
    return counts


def _render_static_readme_panel(data_dir: Path, metrics: list[dict], output_path: Path, sample_rate: int) -> None:
    selected = _select_class_examples(data_dir, sample_rate)
    if not selected or not metrics:
        return

    counts = _class_counts(data_dir)
    metrics_by_name = {row["output_name"]: row for row in metrics}
    best_binary = metrics_by_name.get("augmented_cqt_cnn", max(metrics, key=lambda row: row["test_accuracy"]))
    four_class = metrics_by_name.get("four_class_cnn", metrics[-1])
    sample_label, _path, waveform, loaded_sample_rate = selected[0]
    views = _feature_views(waveform, loaded_sample_rate)

    fig = plt.figure(figsize=(15, 9.2), facecolor="#f7f3ea", constrained_layout=True)
    grid = fig.add_gridspec(4, 5, height_ratios=[0.42, 1, 1, 1], hspace=0.20, wspace=0.12)
    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.68, "Spectrogram Classification", fontsize=25, fontweight="bold", color="#1f2933")
    title_ax.text(
        0.0,
        0.14,
        "Waveforms become time-frequency maps; compact spectral descriptors and CNN features separate guitar and vocal families.",
        fontsize=11,
        color="#4a5568",
    )

    _plot_waveform(fig.add_subplot(grid[1, 0]), views["Waveform"], loaded_sample_rate, f"Waveform: {_short_label(sample_label)}", CLASS_COLORS[sample_label])
    _plot_heat(fig.add_subplot(grid[1, 1]), _normalise_map(views["STFT Power"]), "STFT Power")
    _plot_heat(fig.add_subplot(grid[1, 2]), _normalise_map(views["Mel-Spectrogram"]), "Mel-Spectrogram")
    _plot_heat(fig.add_subplot(grid[1, 3]), _normalise_map(views["Constant-Q"]), "Constant-Q", cmap="viridis")
    _plot_heat(fig.add_subplot(grid[1, 4]), _normalise_map(views["MFCC"]), "MFCC", cmap="cividis")

    for idx, (label, _file_path, class_waveform, class_sr) in enumerate(selected):
        ax = fig.add_subplot(grid[2, idx])
        class_views = _feature_views(class_waveform, class_sr)
        _plot_heat(ax, _normalise_map(class_views["Mel-Spectrogram"]), _short_label(label))
        ax.text(
            0.02,
            0.08,
            f"{counts[label]} clips",
            transform=ax.transAxes,
            color="#ffffff",
            fontsize=9,
            bbox={"facecolor": CLASS_COLORS[label], "edgecolor": "none", "alpha": 0.88, "pad": 3},
        )

    cm_ax = fig.add_subplot(grid[2, 4])
    matrix = np.asarray(four_class["confusion_matrix"], dtype=int)
    cm_ax.imshow(matrix, cmap="Blues")
    cm_labels = [_short_label(label).replace(" ", "\n") for label in _labels_from_metrics(four_class)]
    cm_ax.set_xticks(range(len(cm_labels)))
    cm_ax.set_yticks(range(len(cm_labels)))
    cm_ax.set_xticklabels(cm_labels, fontsize=7, rotation=35, ha="right")
    cm_ax.set_yticklabels(cm_labels, fontsize=7)
    cm_ax.set_title("4-Class Confusion", fontsize=10, loc="left")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            cm_ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=9, color="#111111")

    acc_ax = fig.add_subplot(grid[3, 0:3])
    ordered = sorted(metrics, key=lambda row: row["test_accuracy"])
    names = [MODEL_LABELS.get(row["output_name"], row["output_name"]) for row in ordered]
    values = [float(row["test_accuracy"]) for row in ordered]
    acc_ax.barh(names, values, color=["#8c8c8c", "#2f6fbb", "#2f855a", "#805ad5", "#d97706", "#9f3a64"][: len(values)])
    acc_ax.set_xlim(0.78, 1.01)
    acc_ax.set_title("Saved Model Test Accuracy", fontsize=11, loc="left")
    acc_ax.grid(axis="x", alpha=0.22)
    for y_index, value in enumerate(values):
        acc_ax.text(value + 0.004, y_index, f"{value:.3f}", va="center", fontsize=8.5)

    summary_ax = fig.add_subplot(grid[3, 3:5])
    summary_ax.axis("off")
    summary_ax.text(0.0, 0.92, "Evaluation", fontsize=12, fontweight="bold", color="#1f2933")
    summary_ax.text(0.0, 0.68, f"Binary best: {best_binary['test_accuracy']:.3f}", fontsize=11, color="#2f6fbb")
    summary_ax.text(0.0, 0.48, f"Four-class: {four_class['test_accuracy']:.3f}", fontsize=11, color="#9f3a64")
    summary_ax.text(0.0, 0.28, f"Train / valid / test: {four_class['train_size']} / {four_class['valid_size']} / {four_class['test_size']}", fontsize=9, color="#4a5568")
    summary_ax.text(0.0, 0.08, "Weights are saved as CPU-loadable state dictionaries.", fontsize=8.5, color="#4a5568")

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=145, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_animated_readme_panel(data_dir: Path, metrics: list[dict], output_path: Path, sample_rate: int) -> None:
    selected = _select_class_examples(data_dir, sample_rate)
    if not selected or not metrics:
        return

    sample_label, _path, waveform, loaded_sample_rate = selected[0]
    views = _feature_views(waveform, loaded_sample_rate)
    ordered_metrics = sorted(metrics, key=lambda row: row["test_accuracy"])
    best_four = next((row for row in metrics if row["output_name"] == "four_class_cnn"), metrics[-1])
    history = best_four.get("history", [])
    class_mels = []
    for class_label, _class_path, class_waveform, class_sr in selected:
        class_mels.append((class_label, _normalise_map(_feature_views(class_waveform, class_sr)["Mel-Spectrogram"])))
    feature_specs = [
        ("STFT Power", "magma"),
        ("Mel-Spectrogram", "magma"),
        ("Constant-Q", "viridis"),
        ("MFCC", "cividis"),
    ]
    names = [MODEL_LABELS.get(row["output_name"], row["output_name"]) for row in ordered_metrics]
    final_values = [float(row["test_accuracy"]) for row in ordered_metrics]
    best_binary_value = max(final_values)
    colors = ["#8c8c8c", "#2f6fbb", "#2f855a", "#805ad5", "#d97706", "#9f3a64"]
    matrix = np.asarray(best_four["confusion_matrix"], dtype=int)

    def reveal_map(feature_map: np.ndarray, progress: float) -> np.ndarray:
        feature_map = np.asarray(feature_map, dtype=float)
        if feature_map.ndim != 2 or feature_map.shape[1] <= 1:
            return feature_map
        revealed = feature_map * 0.18
        end_col = max(1, int(feature_map.shape[1] * progress))
        revealed[:, :end_col] = feature_map[:, :end_col]
        return revealed

    frame_count = 60
    ensure_dir(output_path.parent)
    print("[Spectrogram Visualiser] Writing animated panel", flush=True)
    with imageio.get_writer(output_path, mode="I", duration=50, loop=0) as writer:
        for frame_index in range(frame_count):
            progress = (frame_index + 1) / frame_count
            print(f"[Spectrogram Visualiser] Rendering composite GIF frame {frame_index + 1}/{frame_count}", flush=True)

            fig = plt.figure(figsize=(14.2, 8.2), facecolor="#f7f3ea", constrained_layout=True)
            grid = fig.add_gridspec(4, 5, height_ratios=[0.35, 1.0, 1.0, 0.92], hspace=0.18, wspace=0.10)

            title_ax = fig.add_subplot(grid[0, :])
            title_ax.axis("off")
            title_ax.text(0.0, 0.66, "Spectrogram Classification", fontsize=20, fontweight="bold", color="#1f2933")
            title_ax.text(
                0.0,
                0.15,
                "One panel, separated tiles: waveform, time-frequency feature views, class signatures, and saved-model evaluation.",
                fontsize=9.5,
                color="#4a5568",
            )

            current_samples = max(1, int(waveform.shape[0] * progress))
            waveform_ax = fig.add_subplot(grid[1, 0])
            _plot_waveform(waveform_ax, waveform[:current_samples], loaded_sample_rate, f"Waveform: {_short_label(sample_label)}", CLASS_COLORS[sample_label])
            waveform_ax.set_xlim(0.0, waveform.shape[0] / float(loaded_sample_rate))

            for idx, (view_name, cmap) in enumerate(feature_specs, start=1):
                feature_ax = fig.add_subplot(grid[1, idx])
                feature_map = reveal_map(_normalise_map(views[view_name]), progress)
                _plot_heat(feature_ax, feature_map, view_name, cmap=cmap)

            for idx, (class_label, class_mel) in enumerate(class_mels):
                class_ax = fig.add_subplot(grid[2, idx])
                _plot_heat(class_ax, reveal_map(class_mel, progress), _short_label(class_label), cmap="magma")
                class_ax.text(
                    0.02,
                    0.08,
                    class_label,
                    transform=class_ax.transAxes,
                    color="#ffffff",
                    fontsize=8.0,
                    bbox={"facecolor": CLASS_COLORS[class_label], "edgecolor": "none", "alpha": 0.88, "pad": 2.5},
                )

            cm_ax = fig.add_subplot(grid[2, 4])
            matrix_progress = np.rint(matrix * progress).astype(int)
            cm_ax.imshow(matrix_progress, cmap="Blues", vmin=0, vmax=max(1, int(matrix.max())))
            cm_ax.set_xticks([])
            cm_ax.set_yticks([])
            cm_ax.set_title("4-Class Confusion", fontsize=9.5, loc="left")
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    cm_ax.text(j, i, str(matrix_progress[i, j]), ha="center", va="center", fontsize=7.5)

            bar_ax = fig.add_subplot(grid[3, 0:3])
            values = [0.78 + (value - 0.78) * progress for value in final_values]
            bar_ax.barh(names, values, color=colors[: len(values)])
            bar_ax.set_xlim(0.78, 1.01)
            bar_ax.set_title("Saved Weights: Test Accuracy", fontsize=10, loc="left")
            bar_ax.grid(axis="x", alpha=0.22)
            bar_ax.tick_params(axis="y", labelsize=8)
            for y_index, value in enumerate(values):
                bar_ax.text(value + 0.004, y_index, f"{final_values[y_index]:.3f}", va="center", fontsize=7.5)

            curve_ax = fig.add_subplot(grid[3, 3])
            upto = min(len(history), max(1, int(progress * len(history))))
            epochs = list(range(len(history)))
            valid_values = [float(row.get("valid_accuracy", row.get("val_accuracy", 0.0))) for row in history[:upto]]
            train_values = [float(row.get("train_accuracy", np.nan)) for row in history[:upto]]
            curve_ax.plot(epochs[:upto], valid_values, color="#9f3a64", linewidth=1.8, label="valid")
            if not np.all(np.isnan(train_values)):
                curve_ax.plot(epochs[:upto], train_values, color="#2f6fbb", linewidth=1.3, label="train")
            curve_ax.set_ylim(0.70, 1.01)
            curve_ax.set_xlim(0, max(1, len(history) - 1))
            curve_ax.set_title("Four-Class Learning", fontsize=9.5, loc="left")
            curve_ax.grid(True, alpha=0.22)
            curve_ax.legend(fontsize=6.5, loc="lower right")

            summary_ax = fig.add_subplot(grid[3, 4])
            summary_ax.axis("off")
            summary_ax.text(0.0, 0.82, "Evaluation", fontsize=10, fontweight="bold", color="#1f2933")
            summary_ax.text(0.0, 0.58, f"Binary best: {best_binary_value:.3f}", fontsize=8.8, color="#2f6fbb")
            summary_ax.text(0.0, 0.37, f"Four-class: {best_four['test_accuracy']:.3f}", fontsize=8.8, color="#9f3a64")
            summary_ax.text(0.0, 0.17, f"Frame progress: {progress:.0%}", fontsize=8.0, color="#4a5568")

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=110, facecolor=fig.get_facecolor())
            plt.close(fig)
            buffer.seek(0)
            writer.append_data(imageio.imread(buffer))
            plt.close("all")
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render spectrogram-classification visual assets.")
    parser.add_argument("--data-dir", type=Path, default=SPECTROGRAM_DATA_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=SPECTROGRAM_EVALUATION_DIR)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    ensure_dir(VISUAL_SPECTROGRAM_FEATURE_DIR)
    ensure_dir(VISUAL_SPECTROGRAM_MODEL_DIR)
    ensure_dir(VISUAL_SPECTROGRAM_README_DIR)
    print("[Spectrogram Visualiser] Rendering feature views", flush=True)
    _render_sample_feature_views(args.data_dir, VISUAL_SPECTROGRAM_FEATURE_DIR, args.sample_rate)
    print("[Spectrogram Visualiser] Rendering class gallery", flush=True)
    _render_class_gallery(args.data_dir, VISUAL_SPECTROGRAM_FEATURE_DIR, args.sample_rate)
    print("[Spectrogram Visualiser] Rendering metrics", flush=True)
    metrics = _render_experiment_metrics(args.metrics_dir, VISUAL_SPECTROGRAM_MODEL_DIR)
    print("[Spectrogram Visualiser] Rendering README static panel", flush=True)
    _render_static_readme_panel(
        args.data_dir,
        metrics,
        VISUAL_SPECTROGRAM_README_DIR / "readme_spectrogram_static_panel.png",
        args.sample_rate,
    )
    print("[Spectrogram Visualiser] Rendering README animated panel", flush=True)
    _render_animated_readme_panel(
        args.data_dir,
        metrics,
        VISUAL_SPECTROGRAM_README_DIR / "readme_spectrogram_animated_panel.gif",
        args.sample_rate,
    )
    print(f"[Spectrogram Visualiser] Wrote assets under {VISUAL_SPECTROGRAM_README_DIR.parent}")


if __name__ == "__main__":
    main()
