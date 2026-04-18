from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import collect_split_files
from scripts.automatic_music_instrumentation.core.pitch_zones import evaluate_pitch_zone_files
from utils.io_helpers import save_json
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_METRICS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the fixed pitch-zone rule baseline.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed data dir or automatic_music_instrumentation dir.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out", type=Path, default=AUTOMATIC_INSTRUMENTATION_METRICS_DIR / "pitch_zones_test_metrics.json")
    args = parser.parse_args()

    splits = collect_split_files(args.data_dir)
    files = getattr(splits, args.split)
    metrics = evaluate_pitch_zone_files(files, max_files=args.max_files)
    metrics["split"] = args.split
    save_json(args.out, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
