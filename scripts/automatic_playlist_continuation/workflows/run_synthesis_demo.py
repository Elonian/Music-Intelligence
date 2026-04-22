#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_playlist_continuation.synthesis import adsr_envelope, get_lfo  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import AUTOMATIC_PLAYLIST_CONTINUATION_SYNTHESIS_DIR  # noqa: E402


def _int16(audio: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1e-8:
        audio = audio / peak
    return np.asarray(np.clip(audio, -1.0, 1.0) * 32767, dtype=np.int16)


def _write_wav(path: Path, sample_rate: int, audio: np.ndarray) -> None:
    samples = _int16(audio)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(samples.tobytes())


def _sawtooth(frequency: float, t: np.ndarray) -> np.ndarray:
    phase = frequency * t
    return 2.0 * (phase - np.floor(phase + 0.5))


def _one_pole_lowpass(audio: np.ndarray, cutoff_hz: np.ndarray, sample_rate: int) -> np.ndarray:
    output = np.zeros_like(audio)
    previous = 0.0
    for index, sample in enumerate(audio):
        cutoff = max(1.0, min(float(cutoff_hz[index]), sample_rate / 2 - 1.0))
        rc = 1.0 / (2.0 * np.pi * cutoff)
        alpha = (1.0 / sample_rate) / (rc + (1.0 / sample_rate))
        previous = previous + alpha * (float(sample) - previous)
        output[index] = previous
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render ADSR and LFO synthesis examples for automatic playlist continuation.")
    parser.add_argument("--output-dir", type=Path, default=AUTOMATIC_PLAYLIST_CONTINUATION_SYNTHESIS_DIR)
    parser.add_argument("--sample-rate", type=int, default=44100)
    parser.add_argument("--duration", type=float, default=3.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    sr = args.sample_rate
    t_adsr = np.linspace(0, 6.0, int(sr * 6.0), endpoint=False)
    wave_adsr = _sawtooth(440.0, t_adsr)
    envelope = adsr_envelope(5.0, 0.5, 0.5, 0.2, 1.0, sr)
    adsr_audio = wave_adsr[: envelope.size] * envelope
    adsr_path = output_dir / "adsr_sawtooth.wav"
    _write_wav(adsr_path, sr, adsr_audio)

    t_lfo = np.linspace(0, args.duration, int(sr * args.duration), endpoint=False)
    lfo = get_lfo(1000.0, 440.0, 0.5, t_lfo)
    saw = _sawtooth(220.0, t_lfo)
    filtered = _one_pole_lowpass(saw, lfo, sr)
    lfo_path = output_dir / "lfo_filtered_sawtooth.wav"
    _write_wav(lfo_path, sr, filtered)
    save_json(
        output_dir / "synthesis_summary.json",
        {
            "adsr_path": str(adsr_path),
            "lfo_path": str(lfo_path),
            "sample_rate": sr,
            "lfo_min_hz": float(np.min(lfo)),
            "lfo_max_hz": float(np.max(lfo)),
        },
    )
    print(output_dir / "synthesis_summary.json")


if __name__ == "__main__":
    main()
