from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import (
    build_packed_split,
    build_packed_split_from_zip,
    collect_split_files,
    resolve_processed_dir,
)
from utils.io_helpers import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build packed memory-mapped dataset caches for automatic instrumentation.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed data dir or automatic_music_instrumentation dir.")
    parser.add_argument("--splits", nargs="+", default=["train", "valid", "test"], choices=["train", "valid", "test"])
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--from-files", action="store_true", help="Read extracted .npy files instead of split zip files.")
    parser.add_argument("--summary-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processed_dir = resolve_processed_dir(args.data_dir)
    split_files = collect_split_files(processed_dir)
    by_split = {
        "train": split_files.train,
        "valid": split_files.valid,
        "test": split_files.test,
    }
    summaries = []
    for split in args.splits:
        zip_path = processed_dir.parent / "processed_zips" / f"{split}.zip"
        if zip_path.exists() and not args.from_files:
            summary = build_packed_split_from_zip(
                zip_path,
                processed_dir,
                split,
                overwrite=args.overwrite,
                progress=lambda message: print(message, flush=True),
            )
        else:
            summary = build_packed_split(
                by_split[split],
                processed_dir,
                split,
                overwrite=args.overwrite,
                progress=lambda message: print(message, flush=True),
            )
        summaries.append(summary)
        print(summary, flush=True)
    if args.summary_path is not None:
        save_json(args.summary_path, {"processed_dir": str(processed_dir), "splits": summaries})


if __name__ == "__main__":
    main()
