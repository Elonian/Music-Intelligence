#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    AUDIO_OUTPUT_DIR,
    CLASSIFIER_OUTPUT_DIR,
    EVALUATION_DIR,
    ensure_dir,
    load_json,
    read_csv_rows,
    save_json,
)

BASELINE_FIELDS = (
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
)

ENHANCED_FIELDS = (
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
    "pitch_span",
    "log_beats",
    "log_note_density",
    "average_velocity_norm",
    "drum_channel_ratio",
)

SWEEP_SEEDS = (0, 1, 2, 3, 7, 11, 42, 99)


def _matrix_from_rows(rows: list[dict], fields: tuple[str, ...]) -> list[list[float]]:
    return [[float(row[field]) for field in fields] for row in rows]


def _run_seed_sweep(rows: list[dict]) -> dict:
    y = [int(row["target"]) for row in rows]
    baseline_X = _matrix_from_rows(rows, BASELINE_FIELDS)
    enhanced_X = _matrix_from_rows(rows, ENHANCED_FIELDS)
    baseline_scores: list[float] = []
    enhanced_scores: list[float] = []

    for seed in SWEEP_SEEDS:
        Xb_train, Xb_test, y_train, y_test = train_test_split(
            baseline_X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        Xe_train, Xe_test, _, _ = train_test_split(
            enhanced_X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )

        baseline_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )
        enhanced_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )

        baseline_model.fit(Xb_train, y_train)
        enhanced_model.fit(Xe_train, y_train)

        baseline_scores.append(float(accuracy_score(y_test, baseline_model.predict(Xb_test))))
        enhanced_scores.append(float(accuracy_score(y_test, enhanced_model.predict(Xe_test))))

    return {
        "seeds": list(SWEEP_SEEDS),
        "baseline_scores": baseline_scores,
        "enhanced_scores": enhanced_scores,
        "baseline_mean": float(sum(baseline_scores) / len(baseline_scores)),
        "enhanced_mean": float(sum(enhanced_scores) / len(enhanced_scores)),
        "baseline_min": float(min(baseline_scores)),
        "enhanced_min": float(min(enhanced_scores)),
        "baseline_max": float(max(baseline_scores)),
        "enhanced_max": float(max(enhanced_scores)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate project metrics into summary files.")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_OUTPUT_DIR)
    parser.add_argument("--classifier-dir", type=Path, default=CLASSIFIER_OUTPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    audio_summary = load_json(args.audio_dir / "audio_summary.json")
    classifier_summary = load_json(args.classifier_dir / "classifier_summary.json")
    classifier_rows = read_csv_rows(args.classifier_dir / "feature_rows.csv")
    sweep = _run_seed_sweep(classifier_rows)

    summary = {
        "audio_clip_count": len(audio_summary["audio_files"]),
        "lead_note_count": len(audio_summary["lead_notes"]),
        "sample_rate": audio_summary["sample_rate"],
        "delay_tail_seconds": audio_summary["delay_tail_seconds"],
        "baseline_accuracy": classifier_summary["baseline_accuracy"],
        "enhanced_accuracy": classifier_summary["enhanced_accuracy"],
        "accuracy_gain": classifier_summary["accuracy_gain"],
        "row_count": classifier_summary["row_count"],
        "split_sweep_seed_count": len(sweep["seeds"]),
        "baseline_seed_sweep_mean": sweep["baseline_mean"],
        "enhanced_seed_sweep_mean": sweep["enhanced_mean"],
        "baseline_seed_sweep_min": sweep["baseline_min"],
        "enhanced_seed_sweep_min": sweep["enhanced_min"],
        "baseline_seed_sweep_max": sweep["baseline_max"],
        "enhanced_seed_sweep_max": sweep["enhanced_max"],
    }

    save_json(args.output_dir / "metrics_summary.json", summary)
    lines = [
        "| Metric | Value |",
        "| --- | ---: |",
    ]
    for key, value in summary.items():
        if isinstance(value, float):
            lines.append(f"| {key} | {value:.4f} |")
        else:
            lines.append(f"| {key} | {value} |")
    (args.output_dir / "metrics_summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[Eval] Wrote {args.output_dir / 'metrics_summary.json'}")
    print(f"[Eval] Wrote {args.output_dir / 'metrics_summary.md'}")


if __name__ == "__main__":
    main()
