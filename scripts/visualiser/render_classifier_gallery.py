#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.binary_classify.train_midi_classifier import build_classifier_outputs
from scripts.visualiser.visualiser import save_accuracy_bar, save_confusion_matrix, save_feature_scatter
from utils import CLASSIFIER_OUTPUT_DIR, VISUAL_CLASSIFIER_DIR, ensure_dir, load_json, read_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Render classifier visuals.")
    parser.add_argument("--classifier-dir", type=Path, default=CLASSIFIER_OUTPUT_DIR)
    parser.add_argument("--visual-dir", type=Path, default=VISUAL_CLASSIFIER_DIR)
    parser.add_argument("--max-files", type=int, default=200)
    args = parser.parse_args()

    build_classifier_outputs(args.classifier_dir, max_files=args.max_files)
    visual_dir = ensure_dir(args.visual_dir)

    baseline_metrics = load_json(args.classifier_dir / "baseline_metrics.json")
    enhanced_metrics = load_json(args.classifier_dir / "enhanced_metrics.json")
    summary = load_json(args.classifier_dir / "classifier_summary.json")
    rows = read_csv_rows(args.classifier_dir / "feature_rows.csv")

    save_confusion_matrix(
        np.asarray(baseline_metrics["confusion_matrix"]),
        ["drums", "piano"],
        visual_dir / "baseline_confusion_matrix.png",
        "Baseline Confusion Matrix",
    )
    save_confusion_matrix(
        np.asarray(enhanced_metrics["confusion_matrix"]),
        ["drums", "piano"],
        visual_dir / "enhanced_confusion_matrix.png",
        "Enhanced Confusion Matrix",
    )
    save_feature_scatter(
        rows,
        visual_dir / "unique_vs_pitch_span.png",
        "Unique Pitches vs Pitch Span",
        "unique_pitch_num",
        "pitch_span",
    )
    save_feature_scatter(
        rows,
        visual_dir / "average_pitch_vs_drum_ratio.png",
        "Average Pitch vs Drum Channel Ratio",
        "average_pitch_value",
        "drum_channel_ratio",
    )
    save_accuracy_bar(
        {
            "Baseline": summary["baseline_accuracy"],
            "Enhanced": summary["enhanced_accuracy"],
        },
        visual_dir / "accuracy_comparison.png",
        "Classifier Accuracy Comparison",
    )

    print(f"[Visualiser Classifier] Wrote assets under {visual_dir}")


if __name__ == "__main__":
    main()
