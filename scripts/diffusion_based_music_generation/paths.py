from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "diffusion_based_music_generation"
NSYNTH_VALID_AUDIO_DIR = DATA_ROOT / "nsynth" / "nsynth-valid" / "audio"
PRETRAINED_KEYBOARD_CKPT = DATA_ROOT / "pretrained_keyboard.pt"

OUTPUT_ROOT = ROOT / "outputs" / "diffusion_based_music_generation"
RUNS_DIR = OUTPUT_ROOT / "runs"
GENERATED_DIR = OUTPUT_ROOT / "generated"
EVALUATION_DIR = OUTPUT_ROOT / "evaluation"
SMOKE_DIR = OUTPUT_ROOT / "smoke"
