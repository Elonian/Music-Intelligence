#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_playlist_continuation.collaborative_filtering import train_wrmf  # noqa: E402
from scripts.automatic_playlist_continuation.data import build_interaction_samples  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.playlist_continuation import load_playlists, summarize_playlist_collection  # noqa: E402
from utils.project_paths import (  # noqa: E402
    AUTOMATIC_PLAYLIST_CONTINUATION_MODEL_DIR,
    AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the automatic playlist continuation WRMF model.")
    parser.add_argument("--train-json", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_TRAIN_JSON)
    parser.add_argument("--output-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_MODEL_DIR / "wrmf")
    parser.add_argument("--factors", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=40.0)
    parser.add_argument("--lambda-reg", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-playlists", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    playlists = load_playlists(args.train_json)
    if args.max_playlists is not None:
        selected_keys = sorted(playlists, key=lambda item: int(item))[: args.max_playlists]
        playlists = {key: playlists[key] for key in selected_keys}
    data, tid_to_idx, idx_to_tid, tid_to_meta = build_interaction_samples(playlists, random_seed=args.seed)
    result = train_wrmf(
        data,
        num_users=len(playlists),
        num_items=len(tid_to_idx),
        num_factors=args.factors,
        alpha=args.alpha,
        lambda_reg=args.lambda_reg,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
    )

    output_dir = ensure_dir(args.output_dir)
    torch.save(
        {
            "model_state": result.model.state_dict(),
            "config": result.config,
            "history": result.history,
            "tid_to_idx": tid_to_idx,
            "idx_to_tid": idx_to_tid,
        },
        output_dir / "wrmf_model.pt",
    )
    save_json(output_dir / "history.json", {"history": result.history, "config": result.config})
    save_json(
        output_dir / "data_maps.json",
        {
            "tid_to_idx": tid_to_idx,
            "idx_to_tid": {str(key): value for key, value in idx_to_tid.items()},
            "tid_to_meta": {key: list(value) for key, value in tid_to_meta.items()},
        },
    )
    save_json(
        output_dir / "training_summary.json",
        {
            "dataset": summarize_playlist_collection(playlists),
            "interaction_rows": len(data),
            "positive_rows": int(sum(row[2] for row in data)),
            "negative_rows": int(sum(1 for row in data if row[2] == 0)),
            "final_loss": result.history[-1]["loss"] if result.history else None,
            "checkpoint": str(output_dir / "wrmf_model.pt"),
        },
    )
    print(output_dir / "training_summary.json")


if __name__ == "__main__":
    main()
