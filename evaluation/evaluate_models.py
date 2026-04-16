#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spectrogram_classification.train_notebook_weights import (
    build_experiment_specs,
    evaluate_saved_experiment,
)
from utils import SPECTROGRAM_EVALUATION_DIR, SPECTROGRAM_WEIGHT_DIR, ensure_dir, save_json, write_csv_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate saved spectrogram-classification .weights files.")
    parser.add_argument("--module-name", type=str, default="spectrogram_classification")
    parser.add_argument("--weight-dir", type=Path, default=SPECTROGRAM_WEIGHT_DIR)
    parser.add_argument("--output-dir", type=Path, default=SPECTROGRAM_EVALUATION_DIR)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    ensure_dir(args.output_dir)
    results = []
    for spec in build_experiment_specs(args.module_name):
        metrics = evaluate_saved_experiment(spec, args.weight_dir, seed=args.seed)
        save_json(args.output_dir / f"{spec.output_name}_metrics.json", metrics)
        results.append(metrics)

    save_json(args.output_dir / "weight_summary.json", {"experiments": results})
    rows = []
    for result in results:
        rows.append(
            {
                "output_name": result["output_name"],
                "weight_name": result["weight_name"],
                "train_size": result["train_size"],
                "valid_size": result["valid_size"],
                "test_size": result["test_size"],
                "train_accuracy": f"{result['train_accuracy']:.4f}",
                "valid_accuracy": f"{result['valid_accuracy']:.4f}",
                "test_accuracy": f"{result['test_accuracy']:.4f}",
            }
        )
    write_csv_rows(args.output_dir / "weight_summary.csv", rows)
    print(f"[Eval] Wrote {args.output_dir / 'weight_summary.json'}")
    print(f"[Eval] Wrote {args.output_dir / 'weight_summary.csv'}")


if __name__ == "__main__":
    main()
