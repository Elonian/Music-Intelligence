#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diffusion_based_music_generation.dataset import FREQ_BINS, TIME_FRAMES, NSynthSpecDataset  # noqa: E402
from scripts.diffusion_based_music_generation.model import load_flow_model  # noqa: E402
from scripts.diffusion_based_music_generation.paths import NSYNTH_VALID_AUDIO_DIR, PRETRAINED_KEYBOARD_CKPT, SMOKE_DIR  # noqa: E402
from scripts.diffusion_based_music_generation.samplers import (  # noqa: E402
    cfg_sample,
    euler_sample,
    flow_loss,
    heun_sample,
    naive_scale_sample,
    rk4_sample,
    sample_timesteps,
)
from scripts.diffusion_based_music_generation.training import select_device, set_random_seed  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402


def run_smoke_test(
    output_dir: Path = SMOKE_DIR,
    checkpoint: Path = PRETRAINED_KEYBOARD_CKPT,
    audio_dir: Path = NSYNTH_VALID_AUDIO_DIR,
    require_cuda: bool = False,
) -> dict:
    set_random_seed(123)
    device = select_device(require_cuda)
    output_dir = ensure_dir(output_dir)
    model, ckpt = load_flow_model(str(checkpoint), device=str(device))
    model.eval()

    x1 = torch.randn(2, 2, FREQ_BINS, TIME_FRAMES, device=device)
    pitches = torch.tensor([60, 64], dtype=torch.long, device=device)
    out_euler = euler_sample(model, x1.clone(), pitches, n_steps=2)
    out_naive = naive_scale_sample(model, x1.clone(), pitches, n_steps=2, scale=1.0)
    out_cfg = cfg_sample(model, x1.clone(), pitches, n_steps=2, guidance_scale=2.0)
    out_heun = heun_sample(model, x1.clone(), pitches, n_steps=2, guidance_scale=1.0)
    out_rk4 = rk4_sample(model, x1.clone(), pitches, n_steps=1, guidance_scale=1.0)

    dataset = NSynthSpecDataset(str(audio_dir), max_files=2, instrument_filter="guitar", cache=True)
    spec, pitch = dataset[0]
    model_train = copy.deepcopy(model)
    model_train.train()
    t = sample_timesteps(1, device, "logit_normal")
    x_data = spec.unsqueeze(0).to(device)
    p_data = pitch.reshape(1).to(device)
    loss = flow_loss(model_train, x_data, p_data, t, p_uncond=0.1)
    loss.backward()

    summary = {
        "device": str(device),
        "checkpoint_model_type": ckpt.get("config", {}).get("model_type"),
        "dataset_len": len(dataset),
        "dataset_item_shape": list(spec.shape),
        "dataset_item_pitch": int(pitch),
        "euler_shape": list(out_euler.shape),
        "euler_finite": bool(torch.isfinite(out_euler).all()),
        "naive_scale_one_matches_euler": bool(torch.allclose(out_naive, out_euler, atol=1e-5)),
        "cfg_differs": bool(not torch.allclose(out_cfg, out_euler, atol=1e-4)),
        "heun_shape": list(out_heun.shape),
        "rk4_shape": list(out_rk4.shape),
        "flow_loss": float(loss.detach().cpu()),
        "flow_loss_has_grad": any(parameter.grad is not None for parameter in model_train.parameters()),
    }
    save_json(output_dir / "smoke_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run smoke checks for diffusion generation scripts.")
    parser.add_argument("--output-dir", type=Path, default=SMOKE_DIR)
    parser.add_argument("--checkpoint", type=Path, default=PRETRAINED_KEYBOARD_CKPT)
    parser.add_argument("--audio-dir", type=Path, default=NSYNTH_VALID_AUDIO_DIR)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_smoke_test(args.output_dir, args.checkpoint, args.audio_dir, require_cuda=args.require_cuda)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
