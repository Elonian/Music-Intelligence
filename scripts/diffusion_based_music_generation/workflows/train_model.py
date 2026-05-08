#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diffusion_based_music_generation.paths import NSYNTH_VALID_AUDIO_DIR, PRETRAINED_KEYBOARD_CKPT, RUNS_DIR  # noqa: E402
from scripts.diffusion_based_music_generation.training import FineTuneConfig, fine_tune  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the diffusion flow model on NSynth audio.")
    parser.add_argument("--run-name", default="guitar_smoke")
    parser.add_argument("--audio-dir", type=Path, default=NSYNTH_VALID_AUDIO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=PRETRAINED_KEYBOARD_CKPT)
    parser.add_argument("--output-root", type=Path, default=RUNS_DIR)
    parser.add_argument("--instrument-filter", default="guitar")
    parser.add_argument("--max-files", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--p-uncond", type=float, default=0.1)
    parser.add_argument("--t-sample", choices=["uniform", "logit_normal"], default="logit_normal")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-steps", type=int, default=None)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_train_steps = None if args.max_train_steps is not None and args.max_train_steps <= 0 else args.max_train_steps
    config = FineTuneConfig(
        run_name=args.run_name,
        audio_dir=args.audio_dir,
        checkpoint=args.checkpoint,
        output_root=args.output_root,
        instrument_filter=None if args.instrument_filter == "" else args.instrument_filter,
        max_files=args.max_files,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        p_uncond=args.p_uncond,
        t_sample=args.t_sample,
        num_workers=args.num_workers,
        cache=not args.no_cache,
        seed=args.seed,
        max_train_steps=max_train_steps,
        require_cuda=args.require_cuda,
    )
    summary = fine_tune(config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
