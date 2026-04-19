#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.models import MODEL_SPECS  # noqa: E402
from scripts.multitrack_generation.training import TrainConfig, train_model  # noqa: E402
from utils.project_paths import MULTITRACK_GENERATION_LOG_ROOT, MULTITRACK_GENERATION_OUTPUT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multitrack generation Transformer.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="full")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=MULTITRACK_GENERATION_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=MULTITRACK_GENERATION_LOG_ROOT)
    parser.add_argument("--positional-mode", choices=["sequence", "notebook"], default="sequence")
    parser.add_argument("--max-beats", type=int, default=32)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-valid-files", type=int, default=None)
    parser.add_argument("--max-val-batches", type=int, default=None)
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--no-packed-cache", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-steps", type=int, default=25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_name=args.model,
        run_name=args.run_name,
        data_dir=args.data_dir,
        output_root=args.output_root,
        log_root=args.log_root,
        positional_mode=args.positional_mode,
        max_beats=args.max_beats,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        val_steps=args.val_steps,
        num_workers=args.num_workers,
        random_seed=args.seed,
        max_train_files=args.max_train_files,
        max_valid_files=args.max_valid_files,
        preload=args.preload,
        use_packed_cache=not args.no_packed_cache,
        require_cuda=args.require_cuda,
        amp=args.amp,
        log_steps=args.log_steps,
        max_val_batches=args.max_val_batches,
    )
    print(train_model(config))


if __name__ == "__main__":
    main()
