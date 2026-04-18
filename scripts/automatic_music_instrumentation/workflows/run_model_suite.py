from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import collect_split_files, has_packed_split, resolve_processed_dir
from scripts.automatic_music_instrumentation.core.models import MODEL_SPECS
from scripts.automatic_music_instrumentation.core.pitch_zones import evaluate_pitch_zone_files, evaluate_pitch_zone_packed
from scripts.automatic_music_instrumentation.core.training import TrainConfig, train_one_model
from utils.io_helpers import ensure_dir, save_json, write_csv_rows
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_LOG_ROOT, AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT


DEFAULT_MODELS = (
    "pitch_zones",
    "note_mlp",
    "sequence_lstm",
    "bidirectional_lstm",
    "compact_transformer",
    "causal_transformer",
    "full_transformer",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a model suite for automatic instrumentation.")
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS), help=f"Choices: pitch_zones plus {sorted(MODEL_SPECS)}")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT)
    parser.add_argument("--log-root", type=Path, default=AUTOMATIC_INSTRUMENTATION_LOG_ROOT)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--val-steps", type=int, default=500)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-train-files", type=int, default=None, help="Use a small number for quick checks.")
    parser.add_argument("--max-valid-files", type=int, default=None)
    parser.add_argument("--max-pitch-zone-files", type=int, default=None)
    parser.add_argument("--no-preload", action="store_true")
    parser.add_argument("--no-packed-cache", action="store_true")
    parser.add_argument("--build-packed-cache", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--log-steps", type=int, default=50)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_root / "model_suite")
    rows: list[dict] = []
    if "pitch_zones" in args.models:
        processed_dir = resolve_processed_dir(args.data_dir)
        if args.max_pitch_zone_files is None and has_packed_split(processed_dir, "test"):
            pitch_zone_metrics = evaluate_pitch_zone_packed(processed_dir, split="test")
        else:
            splits = collect_split_files(processed_dir)
            pitch_zone_metrics = evaluate_pitch_zone_files(splits.test, max_files=args.max_pitch_zone_files)
        save_json(output_dir / "pitch_zones_metrics.json", pitch_zone_metrics)
        rows.append(
            {
                "model": "pitch_zones",
                "run_dir": "",
                "log_dir": "",
                "best_val_loss": "",
                "final_val_loss": "",
                "final_val_accuracy": "",
                "test_or_rule_accuracy": pitch_zone_metrics["accuracy"],
            }
        )

    for model_name in [name for name in args.models if name != "pitch_zones"]:
        if model_name not in MODEL_SPECS:
            raise ValueError(f"Unknown model '{model_name}'.")
        config = TrainConfig(
            model_name=model_name,
            run_name=f"suite_{model_name}",
            data_dir=args.data_dir,
            output_root=args.output_root,
            log_root=args.log_root,
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
        metrics = train_one_model(config)
        rows.append(
            {
                "model": model_name,
                "run_dir": metrics["run_dir"],
                "log_dir": metrics["log_dir"],
                "best_val_loss": metrics["best_val_loss"],
                "final_val_loss": metrics["final_val_loss"],
                "final_val_accuracy": metrics["final_val_accuracy"],
                "test_or_rule_accuracy": "",
            }
        )

    write_csv_rows(output_dir / "model_suite_summary.csv", rows)
    save_json(output_dir / "model_suite_summary.json", {"rows": rows})
    print(rows)


if __name__ == "__main__":
    main()
