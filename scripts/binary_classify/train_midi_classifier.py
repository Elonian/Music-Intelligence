#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    CLASSIFIER_OUTPUT_DIR,
    baseline_feature_vector,
    enhanced_feature_vector,
    ensure_dir,
    find_midi_files,
    save_json,
    summarize_midi_file,
    write_csv_rows,
)


def _collect_rows(max_files: int | None = None) -> list[dict]:
    piano_files, drum_files = find_midi_files()
    if max_files is not None:
        piano_files = piano_files[:max_files]
        drum_files = drum_files[:max_files]

    rows = []
    for label_name, numeric_label, files in [("piano", 1, piano_files), ("drums", 0, drum_files)]:
        for file_path in files:
            summary = summarize_midi_file(file_path)
            summary["label"] = label_name
            summary["target"] = numeric_label
            rows.append(summary)
    return rows


def _train(rows: list[dict], feature_name: str, feature_fn) -> dict:
    X = [feature_fn(row) for row in rows]
    y = [row["target"] for row in rows]
    paths = [row["file_path"] for row in rows]
    labels = [row["label"] for row in rows]

    X_train, X_test, y_train, y_test, _path_train, path_test, _label_train, label_test = train_test_split(
        X,
        y,
        paths,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=42),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    prediction_rows = []
    for file_path, label_name, truth, pred in zip(path_test, label_test, y_test, y_pred):
        prediction_rows.append(
            {
                "file_path": file_path,
                "label": label_name,
                "y_true": truth,
                "y_pred": int(pred),
            }
        )

    return {
        "metrics": {
            "feature_name": feature_name,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "classification_report": classification_report(
                y_test,
                y_pred,
                target_names=["drums", "piano"],
                output_dict=True,
                zero_division=0,
            ),
        },
        "prediction_rows": prediction_rows,
    }


def build_classifier_outputs(output_dir: Path = CLASSIFIER_OUTPUT_DIR, max_files: int | None = 200) -> dict:
    output_dir = ensure_dir(output_dir)
    rows = _collect_rows(max_files=max_files)

    baseline = _train(rows, "baseline", baseline_feature_vector)
    enhanced = _train(rows, "enhanced", enhanced_feature_vector)

    rows_path = output_dir / "feature_rows.csv"
    baseline_metrics_path = output_dir / "baseline_metrics.json"
    enhanced_metrics_path = output_dir / "enhanced_metrics.json"
    baseline_predictions_path = output_dir / "baseline_predictions.csv"
    enhanced_predictions_path = output_dir / "enhanced_predictions.csv"
    summary_path = output_dir / "classifier_summary.json"

    write_csv_rows(rows_path, rows)
    write_csv_rows(baseline_predictions_path, baseline["prediction_rows"])
    write_csv_rows(enhanced_predictions_path, enhanced["prediction_rows"])
    save_json(baseline_metrics_path, baseline["metrics"])
    save_json(enhanced_metrics_path, enhanced["metrics"])
    save_json(
        summary_path,
        {
            "max_files_per_class": max_files,
            "row_count": len(rows),
            "baseline_accuracy": baseline["metrics"]["accuracy"],
            "enhanced_accuracy": enhanced["metrics"]["accuracy"],
            "accuracy_gain": enhanced["metrics"]["accuracy"] - baseline["metrics"]["accuracy"],
            "feature_rows_csv": str(rows_path),
        },
    )

    return {
        "rows_path": rows_path,
        "baseline_metrics_path": baseline_metrics_path,
        "enhanced_metrics_path": enhanced_metrics_path,
        "baseline_predictions_path": baseline_predictions_path,
        "enhanced_predictions_path": enhanced_predictions_path,
        "summary_path": summary_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the MIDI baseline and enhanced classifiers.")
    parser.add_argument("--output-dir", type=Path, default=CLASSIFIER_OUTPUT_DIR)
    parser.add_argument("--max-files", type=int, default=200)
    args = parser.parse_args()

    artifacts = build_classifier_outputs(args.output_dir, max_files=args.max_files)
    print(f"[Classifier] Wrote {artifacts['summary_path']}")
    print(f"[Classifier] Wrote {artifacts['baseline_metrics_path']}")
    print(f"[Classifier] Wrote {artifacts['enhanced_metrics_path']}")


if __name__ == "__main__":
    main()
