#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.sine_wave.build_audio_gallery import build_audio_gallery
from scripts.visualiser.visualiser import (
    save_simple_waveform_gif,
    save_spectrogram_plot,
    save_waveform_comparison,
    save_waveform_plot,
)
from utils import AUDIO_OUTPUT_DIR, VISUAL_AUDIO_DIR, ensure_dir


def _load_audio(path: Path) -> tuple[int, np.ndarray]:
    sample_rate, audio = wavfile.read(path)
    audio = np.asarray(audio, dtype=float)
    peak = np.max(np.abs(audio)) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak
    return sample_rate, audio


def main() -> None:
    parser = argparse.ArgumentParser(description="Render the audio gallery visuals.")
    parser.add_argument("--audio-dir", type=Path, default=AUDIO_OUTPUT_DIR)
    parser.add_argument("--visual-dir", type=Path, default=VISUAL_AUDIO_DIR)
    args = parser.parse_args()

    artifacts = build_audio_gallery(args.audio_dir)
    visual_dir = ensure_dir(args.visual_dir)

    sr_sine, sine = _load_audio(artifacts["audio_files"]["lead_sine"])
    sr_saw, saw = _load_audio(artifacts["audio_files"]["lead_saw"])
    sr_faded, faded = _load_audio(artifacts["audio_files"]["faded"])
    sr_delayed, delayed = _load_audio(artifacts["audio_files"]["delayed"])
    sr_stacked, stacked = _load_audio(artifacts["audio_files"]["stacked"])

    save_waveform_plot(sine, sr_sine, visual_dir / "melody_sine_waveform.png", "Melody Waveform: Sine")
    save_waveform_plot(saw, sr_saw, visual_dir / "melody_sawtooth_waveform.png", "Melody Waveform: Sawtooth", color="tab:orange")
    save_waveform_plot(stacked, sr_stacked, visual_dir / "melody_stacked_waveform.png", "Melody Waveform: Stacked", color="tab:green")
    save_waveform_comparison(sine, faded, sr_sine, visual_dir / "fade_comparison.png", "Fade Comparison", "original", "faded")
    save_waveform_comparison(sine, delayed, sr_delayed, visual_dir / "delay_comparison.png", "Delay Comparison", "original", "delayed")
    save_spectrogram_plot(sine, sr_sine, visual_dir / "sine_spectrogram.png", "Sine Spectrogram")
    save_spectrogram_plot(saw, sr_saw, visual_dir / "sawtooth_spectrogram.png", "Sawtooth Spectrogram")
    save_simple_waveform_gif(sine, sr_sine, visual_dir / "melody_sine_evolution.gif", "Sine Melody Evolution", color="#2155a6")
    save_simple_waveform_gif(saw, sr_saw, visual_dir / "melody_sawtooth_evolution.gif", "Sawtooth Melody Evolution", color="#d97706")
    save_simple_waveform_gif(faded, sr_faded, visual_dir / "melody_fade_evolution.gif", "Fade-Out Evolution", color="#c2410c")
    save_simple_waveform_gif(delayed, sr_delayed, visual_dir / "melody_delay_evolution.gif", "Delay Echo Evolution", color="#0f766e")
    save_simple_waveform_gif(stacked, sr_stacked, visual_dir / "melody_stack_evolution.gif", "Layered Mix Evolution", color="#3f7d20")

    print(f"[Visualiser Audio] Wrote assets under {visual_dir}")


if __name__ == "__main__":
    main()
