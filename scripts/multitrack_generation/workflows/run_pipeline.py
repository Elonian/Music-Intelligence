#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.project_paths import MULTITRACK_GENERATION_CHECKPOINT_DIR, MULTITRACK_GENERATION_OUTPUT_ROOT  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Multitrack data inspection, optional training, generation, and visuals.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny settings for a quick end-to-end check.")
    parser.add_argument("--skip-train", action="store_true", help="Use the downloaded checkpoint instead of training.")
    parser.add_argument("--run-name", default="multitrack_pipeline")
    parser.add_argument("--checkpoint", type=Path, default=MULTITRACK_GENERATION_CHECKPOINT_DIR / "best_model_20260331.pt")
    return parser.parse_args()


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    inspect_cmd = [python, "-m", "scripts.multitrack_generation.workflows.inspect_data"]
    if args.smoke:
        inspect_cmd.extend(["--max-files", "64"])
    _run(inspect_cmd)

    checkpoint = args.checkpoint
    model = "full"
    positional_mode = "sequence"
    if not args.skip_train:
        model = "tiny" if args.smoke else "full"
        train_cmd = [
            python,
            "-m",
            "scripts.multitrack_generation.workflows.train_model",
            "--model",
            model,
            "--run-name",
            args.run_name,
        ]
        if args.smoke:
            train_cmd.extend(
                [
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--max-seq-len",
                    "128",
                    "--max-train-files",
                    "8",
                    "--max-valid-files",
                    "4",
                    "--max-val-batches",
                    "1",
                    "--val-steps",
                    "1",
                    "--log-steps",
                    "1",
                ]
            )
        else:
            train_cmd.extend(
                [
                    "--epochs",
                    "50",
                    "--batch-size",
                    "16",
                    "--val-steps",
                    "1000",
                    "--log-steps",
                    "25",
                    "--num-workers",
                    "1",
                    "--require-cuda",
                    "--amp",
                ]
            )
        _run(train_cmd)
        checkpoint = MULTITRACK_GENERATION_OUTPUT_ROOT / "runs" / args.run_name / "checkpoints" / "best_model.pt"
    else:
        positional_mode = "notebook"

    gen_cmd = [
        python,
        "-m",
        "scripts.multitrack_generation.workflows.generate_music",
        "--checkpoint",
        str(checkpoint),
        "--model",
        model,
        "--name",
        args.run_name,
        "--prompt",
        "twinkle" if args.smoke else "piano_guitar_bass",
        "--max-seq-len",
        "96" if args.smoke else "256",
    ]
    if positional_mode:
        gen_cmd.extend(["--positional-mode", positional_mode])
    _run(gen_cmd)

    _run([python, "-m", "scripts.visualiser.render_multitrack_generation_gallery", "--run-name", args.run_name])


if __name__ == "__main__":
    main()
