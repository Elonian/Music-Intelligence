#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diffusion_based_music_generation.generation import GenerationConfig, generate_samples  # noqa: E402
from scripts.diffusion_based_music_generation.paths import GENERATED_DIR, PRETRAINED_KEYBOARD_CKPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate diffusion spectrogram and WAV samples.")
    parser.add_argument("--checkpoint", type=Path, default=PRETRAINED_KEYBOARD_CKPT)
    parser.add_argument("--output-dir", type=Path, default=GENERATED_DIR / "samples")
    parser.add_argument("--sampler", choices=["euler", "cfg", "heun", "rk4", "naive"], default="heun")
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=25)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--pitch-start", type=int, default=48)
    parser.add_argument("--pitch-span", type=int, default=36)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wavs", type=int, default=8)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GenerationConfig(
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        sampler=args.sampler,
        n_samples=args.n_samples,
        n_steps=args.n_steps,
        guidance_scale=args.guidance_scale,
        pitch_start=args.pitch_start,
        pitch_span=args.pitch_span,
        batch_size=args.batch_size,
        seed=args.seed,
        max_wavs=args.max_wavs,
        require_cuda=args.require_cuda,
    )
    summary = generate_samples(config)
    print(json.dumps({key: summary[key] for key in ("npz_path", "sampler", "n_samples", "n_steps")}, indent=2))


if __name__ == "__main__":
    main()
