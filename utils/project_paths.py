from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data" / "sine_wave_binary_classification"
OUTPUT_ROOT = ROOT / "outputs" / "sine_wave_binary_classification"
AUDIO_OUTPUT_DIR = OUTPUT_ROOT / "sine_wave"
CLASSIFIER_OUTPUT_DIR = OUTPUT_ROOT / "binary_classification"
VISUAL_AUDIO_DIR = OUTPUT_ROOT / "visuals" / "audio"
VISUAL_CLASSIFIER_DIR = OUTPUT_ROOT / "visuals" / "classifier"
EVALUATION_OUTPUT_DIR = OUTPUT_ROOT / "evaluation"
EVALUATION_DIR = ROOT / "evaluation"
