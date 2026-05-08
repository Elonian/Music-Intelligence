from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import torch

from scripts.diffusion_based_music_generation.dataset import SR, spec_to_audio


def tensor_to_audio_array(audio: torch.Tensor, normalize: bool = True) -> np.ndarray:
    values = audio.detach().cpu().to(torch.float32).numpy()
    if normalize and values.size:
        peak = float(np.max(np.abs(values)))
        if peak > 1e-8:
            values = values / peak
    return np.clip(values, -1.0, 1.0).astype(np.float32, copy=False)


def write_wav(path: Path, audio: np.ndarray, sample_rate: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(np.clip(audio, -1.0, 1.0) * 32767.0, dtype="<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def write_sample_wavs(
    specs: torch.Tensor,
    output_dir: Path,
    pitches: torch.Tensor | None = None,
    max_wavs: int | None = None,
    sample_rate: int = SR,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = specs.shape[0] if max_wavs is None else min(int(max_wavs), specs.shape[0])
    paths: list[str] = []
    for index in range(count):
        pitch_suffix = "" if pitches is None else f"_pitch{int(pitches[index])}"
        path = output_dir / f"sample_{index:03d}{pitch_suffix}.wav"
        audio = tensor_to_audio_array(spec_to_audio(specs[index].detach().cpu()), normalize=True)
        write_wav(path, audio, sample_rate=sample_rate)
        paths.append(str(path))
    return paths


def spectrogram_stats(specs: torch.Tensor) -> dict:
    values = specs.detach().cpu().to(torch.float32)
    return {
        "shape": list(values.shape),
        "finite": bool(torch.isfinite(values).all()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "max": float(values.max()),
        "l2_mean": float(values.flatten(1).norm(dim=1).mean()) if values.ndim >= 2 else float(values.norm()),
    }
