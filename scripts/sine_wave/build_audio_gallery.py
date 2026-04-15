#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from scipy.io import wavfile


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    AUDIO_OUTPUT_DIR,
    SAMPLE_RATE,
    add_delay,
    audio_to_int16,
    ensure_dir,
    linear_fade_out,
    mix_audio,
    note_name_to_frequency,
    render_melody,
    save_json,
)


def build_audio_gallery(output_dir: Path = AUDIO_OUTPUT_DIR) -> dict:
    output_dir = ensure_dir(output_dir)

    lead_notes = ["C4", "E4", "G4", "C5", "A4", "G4", "E4", "D4", "C4"]
    lead_durations = [0.35, 0.35, 0.35, 0.6, 0.35, 0.35, 0.35, 0.35, 0.6]
    pad_notes = ["C3", "G3", "A3", "F3", "C3"]
    pad_durations = [0.7, 0.7, 0.7, 0.7, 1.2]

    lead_sine = render_melody(lead_notes, lead_durations, wave_kind="sine")
    lead_saw = render_melody(lead_notes, lead_durations, wave_kind="sawtooth")
    pad_sine = render_melody(pad_notes, pad_durations, wave_kind="sine")
    faded = linear_fade_out(lead_sine)
    delayed = add_delay(lead_sine)
    mix_len = min(lead_sine.shape[0], pad_sine.shape[0])
    stacked = mix_audio([lead_sine[:mix_len], pad_sine[:mix_len]], [0.7, 0.35])

    outputs = {
        "lead_sine": output_dir / "melody_sine.wav",
        "lead_saw": output_dir / "melody_sawtooth.wav",
        "faded": output_dir / "melody_faded.wav",
        "delayed": output_dir / "melody_delayed.wav",
        "stacked": output_dir / "melody_stacked.wav",
    }

    wavfile.write(outputs["lead_sine"], SAMPLE_RATE, audio_to_int16(lead_sine))
    wavfile.write(outputs["lead_saw"], SAMPLE_RATE, audio_to_int16(lead_saw))
    wavfile.write(outputs["faded"], SAMPLE_RATE, audio_to_int16(faded))
    wavfile.write(outputs["delayed"], SAMPLE_RATE, audio_to_int16(delayed))
    wavfile.write(outputs["stacked"], SAMPLE_RATE, audio_to_int16(stacked))

    summary = {
        "sample_rate": SAMPLE_RATE,
        "lead_notes": lead_notes,
        "lead_durations_seconds": lead_durations,
        "lead_frequencies_hz": [note_name_to_frequency(note) for note in lead_notes],
        "audio_files": {key: str(path) for key, path in outputs.items()},
        "clip_lengths_seconds": {
            "lead_sine": lead_sine.shape[0] / SAMPLE_RATE,
            "lead_saw": lead_saw.shape[0] / SAMPLE_RATE,
            "faded": faded.shape[0] / SAMPLE_RATE,
            "delayed": delayed.shape[0] / SAMPLE_RATE,
            "stacked": stacked.shape[0] / SAMPLE_RATE,
        },
        "delay_tail_seconds": 0.5,
    }
    summary_path = output_dir / "audio_summary.json"
    save_json(summary_path, summary)
    return {"summary_path": summary_path, "summary": summary, "audio_files": outputs}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the audio synthesis gallery.")
    parser.add_argument("--output-dir", type=Path, default=AUDIO_OUTPUT_DIR)
    args = parser.parse_args()

    artifacts = build_audio_gallery(args.output_dir)
    print(f"[Audio] Wrote {artifacts['summary_path']}")
    for path in artifacts["audio_files"].values():
        print(f"[Audio] Wrote {path}")


if __name__ == "__main__":
    main()
