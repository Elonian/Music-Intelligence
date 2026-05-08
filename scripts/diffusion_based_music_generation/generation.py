from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from scripts.diffusion_based_music_generation.audio_io import spectrogram_stats, write_sample_wavs
from scripts.diffusion_based_music_generation.dataset import FREQ_BINS, TIME_FRAMES
from scripts.diffusion_based_music_generation.model import load_flow_model
from scripts.diffusion_based_music_generation.paths import GENERATED_DIR, PRETRAINED_KEYBOARD_CKPT
from scripts.diffusion_based_music_generation.samplers import cfg_sample, euler_sample, heun_sample, naive_scale_sample, rk4_sample
from scripts.diffusion_based_music_generation.training import select_device, set_random_seed
from utils.io_helpers import ensure_dir, save_json


@dataclass
class GenerationConfig:
    checkpoint: Path = PRETRAINED_KEYBOARD_CKPT
    output_dir: Path = GENERATED_DIR / "samples"
    sampler: str = "heun"
    n_samples: int = 16
    n_steps: int = 25
    guidance_scale: float = 6.0
    pitch_start: int = 48
    pitch_span: int = 36
    batch_size: int = 16
    seed: int = 0
    max_wavs: int = 8
    require_cuda: bool = False


def _serializable_config(config: GenerationConfig) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def make_pitch_grid(n_samples: int, pitch_start: int = 48, pitch_span: int = 36, device=None) -> torch.Tensor:
    values = [int(pitch_start + index % pitch_span) for index in range(int(n_samples))]
    return torch.tensor(values, dtype=torch.long, device=device)


def sample_batch(
    model,
    noise: torch.Tensor,
    pitches: torch.Tensor,
    sampler: str,
    n_steps: int,
    guidance_scale: float,
) -> torch.Tensor:
    if sampler == "euler":
        return euler_sample(model, noise, pitches, n_steps=n_steps)
    if sampler == "cfg":
        return cfg_sample(model, noise, pitches, n_steps=n_steps, guidance_scale=guidance_scale)
    if sampler == "heun":
        return heun_sample(model, noise, pitches, n_steps=n_steps, guidance_scale=guidance_scale)
    if sampler == "rk4":
        return rk4_sample(model, noise, pitches, n_steps=n_steps, guidance_scale=guidance_scale)
    if sampler == "naive":
        return naive_scale_sample(model, noise, pitches, n_steps=n_steps, scale=guidance_scale)
    raise ValueError("sampler must be one of euler, cfg, heun, rk4, naive")


def generate_samples(config: GenerationConfig) -> dict:
    set_random_seed(config.seed)
    device = select_device(config.require_cuda)
    output_dir = ensure_dir(config.output_dir)
    model, _ckpt = load_flow_model(str(config.checkpoint), device=str(device))
    model.eval()

    pitches = make_pitch_grid(config.n_samples, config.pitch_start, config.pitch_span, device=device)
    noises = torch.randn(config.n_samples, 2, FREQ_BINS, TIME_FRAMES, device=device)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, config.n_samples, config.batch_size):
            stop = min(start + config.batch_size, config.n_samples)
            batch = sample_batch(
                model,
                noises[start:stop].clone(),
                pitches[start:stop],
                sampler=config.sampler,
                n_steps=config.n_steps,
                guidance_scale=config.guidance_scale,
            )
            outputs.append(batch.detach().cpu())

    samples = torch.cat(outputs, dim=0)
    npz_path = output_dir / "samples.npz"
    np.savez_compressed(
        npz_path,
        samples=samples.numpy().astype(np.float32),
        noises=noises.detach().cpu().numpy().astype(np.float32),
        pitches=pitches.detach().cpu().numpy().astype(np.int64),
        guidance_scale=np.array(config.guidance_scale, dtype=np.float32),
        n_steps=np.array(config.n_steps, dtype=np.int32),
        sampler=np.array(config.sampler),
        checkpoint=np.array(str(config.checkpoint)),
    )
    wav_paths = write_sample_wavs(samples, output_dir / "audio", pitches.cpu(), max_wavs=config.max_wavs)
    summary = {
        "output_dir": str(output_dir),
        "npz_path": str(npz_path),
        "checkpoint": str(config.checkpoint),
        "sampler": config.sampler,
        "n_samples": config.n_samples,
        "n_steps": config.n_steps,
        "guidance_scale": config.guidance_scale,
        "seed": config.seed,
        "pitch_min": int(pitches.min()),
        "pitch_max": int(pitches.max()),
        "stats": spectrogram_stats(samples),
        "wav_files": wav_paths,
        "config": _serializable_config(config),
    }
    save_json(output_dir / "generation_summary.json", summary)
    return summary
