#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.data import collect_split_files, resolve_processed_dir, summarize_packed_split, summarize_split  # noqa: E402
from scripts.multitrack_generation.constants import INSTRUMENT_LABELS  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import MULTITRACK_GENERATION_OUTPUT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect multitrack generation data.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=MULTITRACK_GENERATION_OUTPUT_ROOT)
    parser.add_argument("--max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = resolve_processed_dir(args.data_dir)
    splits = collect_split_files(processed_dir)
    def summarize(name: str, files: list[Path]) -> dict:
        if args.max_files is None:
            packed = summarize_packed_split(processed_dir, name)
            if packed is not None:
                return packed
        return summarize_split(files, max_files=args.max_files)

    summary = {
        "processed_dir": str(processed_dir),
        "instrument_labels": list(INSTRUMENT_LABELS),
        "splits": {
            "train": summarize("train", splits.train),
            "valid": summarize("valid", splits.valid),
            "test": summarize("test", splits.test),
        },
    }
    output_path = ensure_dir(args.output_root / "dataset") / "dataset_summary.json"
    save_json(output_path, summary)
    print(output_path)


if __name__ == "__main__":
    main()
