from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import N_CLASSES, PAD_LABEL, build_dataloader, collect_split_files
from scripts.automatic_music_instrumentation.core.metrics import evaluate_model, save_confusion_matrix_plot
from scripts.automatic_music_instrumentation.core.training import load_checkpoint_model, select_device
from utils.io_helpers import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained automatic instrumentation checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None, help="Override model name if checkpoint has no config.")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    device = select_device(False)
    model = load_checkpoint_model(args.checkpoint, model_name=args.model, device=device)
    splits = collect_split_files(args.data_dir)
    files = getattr(splits, args.split)
    if args.max_files is not None:
        files = files[: args.max_files]
    loader = build_dataloader(files, batch_size=args.batch_size, augmentation=False, shuffle=False, num_workers=args.num_workers)
    metrics = evaluate_model(model, loader, nn.CrossEntropyLoss(ignore_index=PAD_LABEL), device)

    out_dir = args.out_dir or args.checkpoint.parent.parent / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{args.split}_confusion_matrix.npy", metrics["confusion_matrix"])
    save_confusion_matrix_plot(metrics["confusion_matrix"], out_dir / f"{args.split}_confusion_matrix.png")
    save_confusion_matrix_plot(metrics["confusion_matrix"], out_dir / f"{args.split}_confusion_matrix_normalized.png", normalized=True)
    save_json(
        out_dir / f"{args.split}_metrics.json",
        {
            "split": args.split,
            "files": len(files),
            "loss": float(metrics["loss"]),
            "accuracy": float(metrics["accuracy"]),
            "num_predictions": int(metrics["num_predictions"]),
            "confusion_matrix": metrics["confusion_matrix"].tolist(),
            "normalized_confusion_matrix": metrics["normalized_confusion_matrix"].tolist(),
            "n_classes": N_CLASSES,
        },
    )
    print({"loss": metrics["loss"], "accuracy": metrics["accuracy"], "num_predictions": metrics["num_predictions"]})


if __name__ == "__main__":
    main()
