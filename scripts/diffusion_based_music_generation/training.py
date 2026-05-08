from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.diffusion_based_music_generation.dataset import NSynthSpecDataset
from scripts.diffusion_based_music_generation.model import count_params, load_flow_model, save_flow_model
from scripts.diffusion_based_music_generation.paths import NSYNTH_VALID_AUDIO_DIR, PRETRAINED_KEYBOARD_CKPT, RUNS_DIR
from scripts.diffusion_based_music_generation.samplers import flow_loss, sample_timesteps
from utils.io_helpers import ensure_dir, save_json


@dataclass
class FineTuneConfig:
    run_name: str = "guitar_smoke"
    audio_dir: Path = NSYNTH_VALID_AUDIO_DIR
    checkpoint: Path = PRETRAINED_KEYBOARD_CKPT
    output_root: Path = RUNS_DIR
    instrument_filter: str | None = "guitar"
    max_files: int | None = 128
    epochs: int = 1
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    p_uncond: float = 0.1
    t_sample: str = "logit_normal"
    num_workers: int = 0
    cache: bool = True
    seed: int = 42
    require_cuda: bool = False
    max_train_steps: int | None = None
    grad_clip: float = 1.0
    log_every_epochs: int = 30


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA was required but is not available.")
    return torch.device("cpu")


def _serializable_config(config: FineTuneConfig) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def run_dir_for(config: FineTuneConfig) -> Path:
    return config.output_root / config.run_name


def fine_tune(config: FineTuneConfig) -> dict:
    set_random_seed(config.seed)
    device = select_device(config.require_cuda)
    run_dir = ensure_dir(run_dir_for(config))
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")

    save_json(run_dir / "config.json", _serializable_config(config))
    model, ckpt = load_flow_model(str(config.checkpoint), device=str(device))
    dataset = NSynthSpecDataset(
        str(config.audio_dir),
        max_files=config.max_files,
        instrument_filter=config.instrument_filter,
        cache=config.cache,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    if len(dataset) == 0 or len(loader) == 0:
        raise FileNotFoundError(f"No usable WAV files found in {config.audio_dir}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict] = []
    losses: list[float] = []
    step = 0
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_losses: list[float] = []
        for x_data, pitch in loader:
            x_data = x_data.to(device, non_blocking=True)
            pitch = pitch.to(device, non_blocking=True)
            t = sample_timesteps(x_data.shape[0], x_data.device, config.t_sample)

            optimizer.zero_grad(set_to_none=True)
            loss = flow_loss(model, x_data, pitch, t, p_uncond=config.p_uncond)
            loss.backward()
            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            value = float(loss.detach().cpu())
            losses.append(value)
            epoch_losses.append(value)
            step += 1
            if config.max_train_steps is not None and step >= config.max_train_steps:
                break

        row = {
            "epoch": epoch,
            "step": step,
            "loss": float(np.mean(epoch_losses)) if epoch_losses else None,
            "elapsed_seconds": time.time() - started,
        }
        history.append(row)
        save_json(run_dir / "history.json", {"history": history, "losses": losses})
        if epoch == 1 or epoch == config.epochs or (
            config.log_every_epochs > 0 and epoch % config.log_every_epochs == 0
        ):
            print(
                f"Epoch {epoch:4d}/{config.epochs} "
                f"loss={row['loss']:.6f} step={step} elapsed={row['elapsed_seconds']:.0f}s",
                flush=True,
            )
        if config.max_train_steps is not None and step >= config.max_train_steps:
            break

    model.eval()
    output_ckpt = checkpoint_dir / "model_ft.pt"
    saved_config = dict(ckpt.get("config", {}))
    save_flow_model(model, str(output_ckpt), saved_config, n_params=count_params(model))
    summary = {
        "run_dir": str(run_dir),
        "checkpoint": str(output_ckpt),
        "device": str(device),
        "dataset_files": len(dataset),
        "steps": step,
        "epochs_completed": len(history),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": float(np.mean(losses)) if losses else None,
        "source_checkpoint": str(config.checkpoint),
        "n_params": count_params(model),
    }
    save_json(run_dir / "train_summary.json", summary)
    return summary
