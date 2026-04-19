#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.data import build_dataloader, collect_split_files, resolve_processed_dir  # noqa: E402
from scripts.multitrack_generation.metrics import evaluate_model, save_confusion_matrices  # noqa: E402
from scripts.multitrack_generation.models import MODEL_SPECS, load_model_checkpoint  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import MULTITRACK_GENERATION_CHECKPOINT_DIR, MULTITRACK_GENERATION_EVALUATION_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a multitrack generation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, default=MULTITRACK_GENERATION_CHECKPOINT_DIR / "best_model_20260331.pt")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--positional-mode", choices=["sequence", "notebook"], default=None)
    parser.add_argument("--max-beats", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(args.checkpoint, model_name=args.model, device=device, positional_mode=args.positional_mode)
    processed_dir = resolve_processed_dir(args.data_dir)
    splits = collect_split_files(processed_dir)
    files = getattr(splits, args.split)
    if args.max_files:
        files = files[: args.max_files]
    loader = build_dataloader(
        files,
        batch_size=args.batch_size,
        max_beats=args.max_beats,
        max_seq_len=args.max_seq_len,
        augmentation=False,
        shuffle=False,
        num_workers=args.num_workers,
        use_packed=args.max_files is None,
        processed_dir=processed_dir,
        split_name=args.split,
    )
    metrics = evaluate_model(model, loader, device, max_batches=args.max_batches, collect_confusion=True)
    run_name = args.run_name or args.checkpoint.stem
    output_dir = ensure_dir(args.output_dir or (MULTITRACK_GENERATION_EVALUATION_DIR / run_name / args.split))
    save_confusion_matrices(metrics["confusion_matrices"], output_dir)
    payload = {
        key: value
        for key, value in metrics.items()
        if key != "confusion_matrices"
    }
    payload.update(
        {
            "checkpoint": str(args.checkpoint),
            "processed_dir": str(processed_dir),
            "split": args.split,
            "files": len(files),
            "device": str(device),
        }
    )
    save_json(output_dir / "metrics.json", payload)
    print(output_dir / "metrics.json")


if __name__ == "__main__":
    main()
