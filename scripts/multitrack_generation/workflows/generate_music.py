#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.generation import (  # noqa: E402
    GenerationConfig,
    generate_sequence,
    prompt_by_name,
    save_generation_bundle,
)
from scripts.multitrack_generation.models import MODEL_SPECS, build_model, load_model_checkpoint  # noqa: E402
from utils.project_paths import MULTITRACK_GENERATION_CHECKPOINT_DIR, MULTITRACK_GENERATION_GENERATED_DIR  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate multitrack symbolic music.")
    parser.add_argument("--checkpoint", type=Path, default=MULTITRACK_GENERATION_CHECKPOINT_DIR / "best_model_20260331.pt")
    parser.add_argument("--allow-random-model", action="store_true")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--positional-mode", choices=["sequence", "notebook"], default=None)
    parser.add_argument("--output-dir", type=Path, default=MULTITRACK_GENERATION_GENERATED_DIR)
    parser.add_argument("--name", default="sample")
    parser.add_argument("--prompt", choices=["empty", "piano_guitar_bass", "all_instruments", "twinkle"], default="empty")
    parser.add_argument("--decoding", choices=["greedy", "random", "topk", "topp"], default="topk")
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tempo-bpm", type=int, default=120)
    parser.add_argument("--no-event-order", action="store_true")
    parser.add_argument("--no-monotonic-beats", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.checkpoint.exists():
        model = load_model_checkpoint(args.checkpoint, model_name=args.model, device=device, positional_mode=args.positional_mode)
    elif args.allow_random_model:
        model = build_model(args.model or "full", positional_mode=args.positional_mode or "sequence").to(device).eval()
    else:
        raise FileNotFoundError(args.checkpoint)
    config = GenerationConfig(
        decoding=args.decoding,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_seq_len=args.max_seq_len,
        seed=args.seed,
        enforce_event_order=not args.no_event_order,
        enforce_monotonic_beats=not args.no_monotonic_beats,
    )
    sequence = generate_sequence(model, prompt=prompt_by_name(args.prompt), config=config, device=device)
    summary = save_generation_bundle(sequence, output_dir=args.output_dir, name=args.name, config=config, tempo_bpm=args.tempo_bpm)
    print(summary)


if __name__ == "__main__":
    main()
