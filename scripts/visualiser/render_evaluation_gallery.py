#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.binary_classify.train_midi_classifier import build_classifier_outputs
from utils import (
    CLASSIFIER_OUTPUT_DIR,
    EVALUATION_OUTPUT_DIR,
    VISUAL_AUDIO_DIR,
    VISUAL_CLASSIFIER_DIR,
    ensure_dir,
    read_csv_rows,
)


BASELINE_KEYS = [
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
]

ENHANCED_KEYS = [
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
    "pitch_span",
    "log_beats",
    "log_note_density",
    "average_velocity_norm",
    "drum_channel_ratio",
]


def _build_image_panel(items: list[tuple[Path, str]], output_path: Path, title: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle(title, fontsize=16)
    for ax, (image_path, label) in zip(axes.flat, items):
        image = plt.imread(image_path)
        ax.imshow(image)
        ax.set_title(label, fontsize=11)
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def _train_subset(rows: list[dict], feature_keys: list[str], limit_per_class: int) -> tuple[float, np.ndarray]:
    piano = [row for row in rows if row["label"] == "piano"][:limit_per_class]
    drums = [row for row in rows if row["label"] == "drums"][:limit_per_class]
    subset = piano + drums

    X = [[float(row[key]) for key in feature_keys] for row in subset]
    y = [int(row["target"]) for row in subset]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, random_state=42))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = float(np.mean(np.asarray(y_pred) == np.asarray(y_test)))
    matrix = confusion_matrix(y_test, y_pred, labels=[0, 1])
    return accuracy, matrix


def _render_progress_gif(rows: list[dict], output_path: Path) -> None:
    sample_sizes = [20, 40, 60, 80, 100, 120]
    baseline_scores = []
    enhanced_scores = []
    baseline_mats = []
    enhanced_mats = []

    for size in sample_sizes:
        b_score, b_mat = _train_subset(rows, BASELINE_KEYS, size)
        e_score, e_mat = _train_subset(rows, ENHANCED_KEYS, size)
        baseline_scores.append(b_score)
        enhanced_scores.append(e_score)
        baseline_mats.append(b_mat)
        enhanced_mats.append(e_mat)

    frames = []
    for idx, size in enumerate(sample_sizes):
        fig = plt.figure(figsize=(10.5, 7.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0])

        ax_curve = fig.add_subplot(gs[0, :])
        ax_curve.plot(sample_sizes[: idx + 1], baseline_scores[: idx + 1], color="#5B6C8F", linewidth=2.4, marker="o", label="Baseline")
        ax_curve.plot(sample_sizes[: idx + 1], enhanced_scores[: idx + 1], color="#0F9D58", linewidth=2.4, marker="o", label="Enhanced")
        ax_curve.set_ylim(0.0, 1.05)
        ax_curve.set_xlim(sample_sizes[0], sample_sizes[-1])
        ax_curve.set_title(f"Evaluation Progress ({size} files per class)")
        ax_curve.set_xlabel("Files Per Class")
        ax_curve.set_ylabel("Accuracy")
        ax_curve.grid(True, alpha=0.25)
        ax_curve.legend(loc="lower right")
        ax_curve.text(
            0.02,
            0.92,
            f"Baseline: {baseline_scores[idx]:.3f}\nEnhanced: {enhanced_scores[idx]:.3f}",
            transform=ax_curve.transAxes,
            va="top",
            ha="left",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
        )

        for ax, mat, title in [
            (fig.add_subplot(gs[1, 0]), baseline_mats[idx], "Baseline Confusion"),
            (fig.add_subplot(gs[1, 1]), enhanced_mats[idx], "Enhanced Confusion"),
        ]:
            im = ax.imshow(mat, cmap="Blues", vmin=0, vmax=max(int(mat.max()), 1))
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(["drums", "piano"])
            ax.set_yticklabels(["drums", "piano"])
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            ax.set_title(title)
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(int(mat[i, j])), ha="center", va="center")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    ensure_dir(output_path.parent)
    imageio.mimsave(output_path, frames, duration=0.9, loop=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render evaluation panels and progress animation.")
    parser.add_argument("--classifier-dir", type=Path, default=CLASSIFIER_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=120)
    args = parser.parse_args()

    build_classifier_outputs(args.classifier_dir, max_files=args.max_files)
    rows = read_csv_rows(args.classifier_dir / "feature_rows.csv")

    output_dir = ensure_dir(args.output_dir)
    audio_panel = output_dir / "audio_overview.png"
    classifier_panel = output_dir / "classifier_overview.png"
    progress_gif = output_dir / "evaluation_progress.gif"

    _build_image_panel(
        [
            (VISUAL_AUDIO_DIR / "melody_sine_waveform.png", "Sine Melody"),
            (VISUAL_AUDIO_DIR / "sine_spectrogram.png", "Sine Spectrogram"),
            (VISUAL_AUDIO_DIR / "fade_comparison.png", "Linear Fade"),
            (VISUAL_AUDIO_DIR / "delay_comparison.png", "Delay Effect"),
        ],
        audio_panel,
        "Audio Synthesis Overview",
    )
    _build_image_panel(
        [
            (VISUAL_CLASSIFIER_DIR / "accuracy_comparison.png", "Accuracy"),
            (VISUAL_CLASSIFIER_DIR / "baseline_confusion_matrix.png", "Baseline Confusion"),
            (VISUAL_CLASSIFIER_DIR / "enhanced_confusion_matrix.png", "Enhanced Confusion"),
            (VISUAL_CLASSIFIER_DIR / "unique_vs_pitch_span.png", "Feature Separation"),
        ],
        classifier_panel,
        "Symbolic Classification Overview",
    )
    _render_progress_gif(rows, progress_gif)

    print(f"[Visualiser Evaluation] Wrote {audio_panel}")
    print(f"[Visualiser Evaluation] Wrote {classifier_panel}")
    print(f"[Visualiser Evaluation] Wrote {progress_gif}")


if __name__ == "__main__":
    main()
