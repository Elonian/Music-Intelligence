#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run automatic playlist continuation evaluation and synthesis demos.")
    parser.add_argument("--smoke", action="store_true", help="Use tiny settings for a quick functional check.")
    parser.add_argument("--skip-audio", action="store_true")
    parser.add_argument("--skip-synthesis", action="store_true")
    parser.add_argument("--extract-embeddings", action="store_true")
    return parser.parse_args()


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    args = parse_args()
    python = sys.executable
    eval_cmd = [
        python,
        "-m",
        "evaluation.evaluate_playlist_continuation",
        "--epochs",
        "1" if args.smoke else "10",
        "--batch-size",
        "512" if args.smoke else "1024",
    ]
    if args.smoke:
        eval_cmd.extend(["--max-train-playlists", "250", "--max-test-playlists", "25"])
    if args.skip_audio:
        eval_cmd.append("--skip-audio")
    if args.extract_embeddings:
        eval_cmd.append("--extract-embeddings")
    _run(eval_cmd)

    if not args.skip_synthesis:
        _run([python, "-m", "scripts.automatic_playlist_continuation.workflows.run_synthesis_demo"])


if __name__ == "__main__":
    main()
