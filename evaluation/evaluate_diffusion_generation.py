#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diffusion_based_music_generation.audio_io import tensor_to_audio_array  # noqa: E402
from scripts.diffusion_based_music_generation.dataset import FREQ_BINS, TIME_FRAMES, spec_to_audio  # noqa: E402
from scripts.diffusion_based_music_generation.paths import EVALUATION_DIR  # noqa: E402
from utils.io_helpers import ensure_dir, save_json, write_csv_rows  # noqa: E402


def _array_stats(array: np.ndarray) -> dict:
    values = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(values.shape),
        "finite": bool(np.isfinite(values).all()),
        "mean": float(values.mean()) if values.size else 0.0,
        "std": float(values.std()) if values.size else 0.0,
        "min": float(values.min()) if values.size else 0.0,
        "max": float(values.max()) if values.size else 0.0,
    }


def evaluate_generation(npz_path: Path, output_dir: Path = EVALUATION_DIR, max_audio: int = 8) -> dict:
    payload = np.load(npz_path, allow_pickle=False)
    samples = payload["samples"].astype(np.float32, copy=False)
    noises = payload["noises"].astype(np.float32, copy=False) if "noises" in payload.files else None
    pitches = payload["pitches"].astype(np.int64, copy=False) if "pitches" in payload.files else np.array([], dtype=np.int64)

    shape_ok = samples.ndim == 4 and samples.shape[1:] == (2, FREQ_BINS, TIME_FRAMES)
    per_sample_l2 = np.linalg.norm(samples.reshape(samples.shape[0], -1), axis=1) if samples.ndim == 4 else np.array([])
    distance_from_noise = None
    if noises is not None and noises.shape == samples.shape:
        distance_from_noise = float(np.mean((samples - noises) ** 2))

    audio_rows: list[dict] = []
    for index in range(min(int(max_audio), samples.shape[0] if samples.ndim else 0)):
        spec = torch.from_numpy(samples[index])
        audio = tensor_to_audio_array(spec_to_audio(spec), normalize=True)
        audio_rows.append(
            {
                "index": index,
                "pitch": int(pitches[index]) if index < len(pitches) else "",
                "audio_peak": float(np.max(np.abs(audio))) if audio.size else 0.0,
                "audio_rms": float(np.sqrt(np.mean(audio**2))) if audio.size else 0.0,
                "audio_mean": float(np.mean(audio)) if audio.size else 0.0,
            }
        )

    unique_pitches, pitch_counts = np.unique(pitches, return_counts=True) if len(pitches) else (np.array([]), np.array([]))
    summary = {
        "npz_path": str(npz_path),
        "shape_ok": bool(shape_ok),
        "sample_count": int(samples.shape[0]) if samples.ndim else 0,
        "samples": _array_stats(samples),
        "noises": None if noises is None else _array_stats(noises),
        "distance_from_noise_mse": distance_from_noise,
        "per_sample_l2_mean": float(per_sample_l2.mean()) if per_sample_l2.size else 0.0,
        "per_sample_l2_min": float(per_sample_l2.min()) if per_sample_l2.size else 0.0,
        "per_sample_l2_max": float(per_sample_l2.max()) if per_sample_l2.size else 0.0,
        "pitch_min": int(pitches.min()) if len(pitches) else None,
        "pitch_max": int(pitches.max()) if len(pitches) else None,
        "pitch_counts": {str(int(k)): int(v) for k, v in zip(unique_pitches, pitch_counts)},
        "sampler": str(payload["sampler"]) if "sampler" in payload.files else None,
        "n_steps": int(payload["n_steps"]) if "n_steps" in payload.files else None,
        "guidance_scale": float(payload["guidance_scale"]) if "guidance_scale" in payload.files else None,
        "audio_rows": audio_rows,
    }

    output_dir = ensure_dir(output_dir)
    save_json(output_dir / "diffusion_generation_evaluation.json", summary)
    write_csv_rows(output_dir / "diffusion_audio_stats.csv", audio_rows)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate generated diffusion spectrogram samples.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=EVALUATION_DIR)
    parser.add_argument("--max-audio", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = evaluate_generation(args.npz, output_dir=args.output_dir, max_audio=args.max_audio)
    print(json.dumps({key: summary[key] for key in ("sample_count", "shape_ok", "distance_from_noise_mse")}, indent=2))
    print(args.output_dir / "diffusion_generation_evaluation.json")


if __name__ == "__main__":
    main()
