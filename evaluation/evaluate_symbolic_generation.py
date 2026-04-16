#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.symbolic_music_generation import note_extraction
from scripts.symbolic_music_generation.build_markov_outputs import build_symbolic_outputs
from utils import (
    SYMBOLIC_DATA_ROOT,
    SYMBOLIC_EVALUATION_DIR,
    SYMBOLIC_GENERATED_DIR,
    SYMBOLIC_METRICS_DIR,
    SYMBOLIC_TABLE_DIR,
    ensure_dir,
    load_json,
    save_json,
    write_csv_rows,
)


def evaluate_symbolic_generation(
    data_dir: Path = SYMBOLIC_DATA_ROOT,
    metrics_dir: Path = SYMBOLIC_METRICS_DIR,
    table_dir: Path = SYMBOLIC_TABLE_DIR,
    generated_dir: Path = SYMBOLIC_GENERATED_DIR,
    evaluation_dir: Path = SYMBOLIC_EVALUATION_DIR,
    generated_length: int = 500,
    rebuild: bool = False,
) -> dict:
    summary_path = metrics_dir / "markov_summary.json"
    generated_path = generated_dir / "q10.mid"
    if rebuild or not summary_path.exists() or not generated_path.exists():
        build_symbolic_outputs(
            data_dir=data_dir,
            metrics_dir=metrics_dir,
            table_dir=table_dir,
            generated_dir=generated_dir,
            generated_length=generated_length,
        )

    summary = load_json(summary_path)
    generated_note_count = len(note_extraction(generated_path)) if generated_path.exists() else 0
    evaluation = {
        "file_count": int(summary["file_count"]),
        "note_event_count": int(summary["note_event_count"]),
        "unique_pitch_count": int(summary["unique_pitch_count"]),
        "mean_note_bigram_perplexity": float(summary["mean_note_bigram_perplexity"]),
        "mean_note_trigram_perplexity": float(summary["mean_note_trigram_perplexity"]),
        "mean_beat_bigram_perplexity": float(summary["mean_beat_bigram_perplexity"]),
        "mean_beat_position_perplexity": float(summary["mean_beat_position_perplexity"]),
        "mean_beat_trigram_perplexity": float(summary["mean_beat_trigram_perplexity"]),
        "generated_note_count": generated_note_count,
        "generated_length_requested": int(summary["generated_length_requested"]),
        "generated_length_matches": generated_note_count == int(summary["generated_length_requested"]),
        "generated_midi_path": str(generated_path),
    }

    ensure_dir(evaluation_dir)
    save_json(evaluation_dir / "symbolic_generation_evaluation.json", evaluation)
    write_csv_rows(
        evaluation_dir / "symbolic_generation_evaluation.csv",
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in evaluation.items()
        ],
    )
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate symbolic Markov-chain generation outputs.")
    parser.add_argument("--data-dir", type=Path, default=SYMBOLIC_DATA_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=SYMBOLIC_METRICS_DIR)
    parser.add_argument("--table-dir", type=Path, default=SYMBOLIC_TABLE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=SYMBOLIC_GENERATED_DIR)
    parser.add_argument("--evaluation-dir", type=Path, default=SYMBOLIC_EVALUATION_DIR)
    parser.add_argument("--generated-length", type=int, default=500)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    evaluation = evaluate_symbolic_generation(
        data_dir=args.data_dir,
        metrics_dir=args.metrics_dir,
        table_dir=args.table_dir,
        generated_dir=args.generated_dir,
        evaluation_dir=args.evaluation_dir,
        generated_length=args.generated_length,
        rebuild=args.rebuild,
    )
    print(f"[Symbolic Eval] Generated note count: {evaluation['generated_note_count']}")
    print(f"[Symbolic Eval] Wrote {args.evaluation_dir / 'symbolic_generation_evaluation.json'}")


if __name__ == "__main__":
    main()
