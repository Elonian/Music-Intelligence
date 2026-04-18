from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import collect_split_files, inspect_event_files, summarize_dataset
from utils.io_helpers import save_json
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_METRICS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify the automatic instrumentation data layout.")
    parser.add_argument("--data-root", type=Path, default=None, help="automatic_music_instrumentation data root.")
    parser.add_argument("--max-check-files", type=int, default=25, help="Use null/negative for all files.")
    parser.add_argument("--out", type=Path, default=AUTOMATIC_INSTRUMENTATION_METRICS_DIR / "data_summary.json")
    args = parser.parse_args()

    summary = summarize_dataset(args.data_root)
    splits = collect_split_files(summary.processed_dir)
    max_check = None if args.max_check_files is not None and args.max_check_files < 0 else args.max_check_files
    payload = {
        "summary": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(summary).items()},
        "train_inspection": inspect_event_files(splits.train, max_files=max_check),
        "valid_inspection": inspect_event_files(splits.valid, max_files=max_check),
        "test_inspection": inspect_event_files(splits.test, max_files=max_check),
    }
    save_json(args.out, payload)
    print(payload)


if __name__ == "__main__":
    main()
