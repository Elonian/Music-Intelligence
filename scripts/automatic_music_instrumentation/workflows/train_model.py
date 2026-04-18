from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.models import MODEL_SPECS
from scripts.automatic_music_instrumentation.core.training import TrainConfig, train_one_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an automatic instrumentation model.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default="full_transformer")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed data dir or automatic_music_instrumentation dir.")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--log-root", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-files", type=int, default=None)
    parser.add_argument("--max-valid-files", type=int, default=None)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--no-packed-cache", action="store_true")
    parser.add_argument("--build-packed-cache", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA mixed precision.")
    parser.add_argument("--log-steps", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = TrainConfig(
        model_name=args.model,
        run_name=args.run_name,
        data_dir=args.data_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        val_steps=args.val_steps,
        num_workers=args.num_workers,
        max_train_files=args.max_train_files,
        max_valid_files=args.max_valid_files,
        preload=not args.no_preload,
        use_packed_cache=not args.no_packed_cache,
        build_packed_cache=args.build_packed_cache,
        require_cuda=args.require_cuda,
        amp=args.amp,
        log_steps=args.log_steps,
    )
    if args.output_root is not None:
        config.output_root = args.output_root
    if args.log_root is not None:
        config.log_root = args.log_root
    metrics = train_one_model(config)
    print(metrics)


if __name__ == "__main__":
    main()
