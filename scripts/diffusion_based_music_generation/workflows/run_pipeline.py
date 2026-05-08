#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_diffusion_generation import evaluate_generation  # noqa: E402
from scripts.diffusion_based_music_generation.generation import GenerationConfig, generate_samples  # noqa: E402
from scripts.diffusion_based_music_generation.paths import GENERATED_DIR, NSYNTH_VALID_AUDIO_DIR, OUTPUT_ROOT, PRETRAINED_KEYBOARD_CKPT, RUNS_DIR, SMOKE_DIR  # noqa: E402
from scripts.diffusion_based_music_generation.training import FineTuneConfig, fine_tune  # noqa: E402
from scripts.diffusion_based_music_generation.workflows.smoke_test import run_smoke_test  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke, fine-tuning, generation, and evaluation.")
    parser.add_argument("--run-name", default="smoke_full_run")
    parser.add_argument("--audio-dir", type=Path, default=NSYNTH_VALID_AUDIO_DIR)
    parser.add_argument("--checkpoint", type=Path, default=PRETRAINED_KEYBOARD_CKPT)
    parser.add_argument("--instrument-filter", default="guitar")
    parser.add_argument("--max-files", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sampler", choices=["euler", "cfg", "heun", "rk4", "naive"], default="heun")
    parser.add_argument("--n-samples", type=int, default=8)
    parser.add_argument("--n-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_train_steps = None if args.max_train_steps is not None and args.max_train_steps <= 0 else args.max_train_steps
    pipeline_dir = ensure_dir(OUTPUT_ROOT / "pipelines" / args.run_name)
    smoke_summary = None
    if not args.skip_smoke:
        smoke_summary = run_smoke_test(
            output_dir=SMOKE_DIR / args.run_name,
            checkpoint=args.checkpoint,
            audio_dir=args.audio_dir,
            require_cuda=args.require_cuda,
        )

    train_summary = fine_tune(
        FineTuneConfig(
            run_name=args.run_name,
            audio_dir=args.audio_dir,
            checkpoint=args.checkpoint,
            output_root=RUNS_DIR,
            instrument_filter=args.instrument_filter,
            max_files=args.max_files,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
            max_train_steps=max_train_steps,
            require_cuda=args.require_cuda,
        )
    )
    generated_summary = generate_samples(
        GenerationConfig(
            checkpoint=Path(train_summary["checkpoint"]),
            output_dir=GENERATED_DIR / args.run_name,
            sampler=args.sampler,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
            guidance_scale=args.guidance_scale,
            batch_size=args.batch_size,
            seed=args.seed,
            max_wavs=min(args.n_samples, 8),
            require_cuda=args.require_cuda,
        )
    )
    evaluation = evaluate_generation(
        Path(generated_summary["npz_path"]),
        output_dir=OUTPUT_ROOT / "evaluation" / args.run_name,
        max_audio=min(args.n_samples, 8),
    )
    summary = {
        "pipeline_dir": str(pipeline_dir),
        "smoke": smoke_summary,
        "training": train_summary,
        "generation": generated_summary,
        "evaluation": evaluation,
    }
    save_json(pipeline_dir / "pipeline_summary.json", summary)
    print(json.dumps({
        "pipeline_summary": str(pipeline_dir / "pipeline_summary.json"),
        "checkpoint": train_summary["checkpoint"],
        "samples": generated_summary["npz_path"],
        "evaluation_sample_count": evaluation["sample_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
