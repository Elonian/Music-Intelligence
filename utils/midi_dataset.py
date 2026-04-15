from __future__ import annotations

import glob
import math
import zipfile
from pathlib import Path

import numpy as np
from mido import MidiFile

from utils.project_paths import DATA_ROOT, ROOT


def _candidate_data_roots() -> list[Path]:
    data_bundle_roots: list[Path] = []
    if DATA_ROOT.exists():
        data_bundle_roots.append(DATA_ROOT)
        data_bundle_roots.extend(sorted([path for path in DATA_ROOT.iterdir() if path.is_dir()]))

    cwd_data_root = Path.cwd() / "data" / "sine_wave_binary_classification"
    cwd_bundle_roots: list[Path] = []
    if cwd_data_root.exists():
        cwd_bundle_roots.append(cwd_data_root)
        cwd_bundle_roots.extend(sorted([path for path in cwd_data_root.iterdir() if path.is_dir()]))

    return [Path.cwd(), ROOT, *data_bundle_roots, *cwd_bundle_roots]


def _ensure_midi_dir(base_dir: Path, label: str) -> Path | None:
    midi_dir = base_dir / label
    zip_path = base_dir / f"{label}.zip"
    if midi_dir.exists():
        return midi_dir
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(base_dir)
        if midi_dir.exists():
            return midi_dir
    return None


def find_midi_files() -> tuple[list[str], list[str]]:
    for base_dir in _candidate_data_roots():
        piano_dir = _ensure_midi_dir(base_dir, "piano")
        drum_dir = _ensure_midi_dir(base_dir, "drums")
        piano_files = sorted(glob.glob(str(base_dir / "piano" / "*.mid")))
        drum_files = sorted(glob.glob(str(base_dir / "drums" / "*.mid")))
        if piano_dir is not None and drum_dir is not None and (piano_files or drum_files):
            return piano_files, drum_files
    return [], []


def summarize_midi_file(file_path: str) -> dict:
    mid = MidiFile(file_path)
    notes = []
    velocities = []
    channel_9_count = 0
    track_totals = []

    for track in mid.tracks:
        total_ticks = 0
        for msg in track:
            total_ticks += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                notes.append(msg.note)
                velocities.append(msg.velocity)
                if hasattr(msg, "channel") and msg.channel == 9:
                    channel_9_count += 1
        track_totals.append(total_ticks)

    unique_notes = sorted(set(notes))
    lowest = min(unique_notes) if unique_notes else 0
    highest = max(unique_notes) if unique_notes else 0
    beats = max(track_totals) / mid.ticks_per_beat if track_totals and mid.ticks_per_beat > 0 else 0.0
    note_density = len(notes) / beats if beats > 0 else 0.0

    return {
        "file_path": file_path,
        "lowest_pitch": lowest,
        "highest_pitch": highest,
        "unique_pitch_num": len(unique_notes),
        "average_pitch_value": float(np.average(unique_notes)) if unique_notes else 0.0,
        "pitch_span": highest - lowest if unique_notes else 0,
        "beat_count": beats,
        "log_beats": math.log1p(beats),
        "note_count": len(notes),
        "log_note_density": math.log1p(note_density),
        "average_velocity_norm": (float(np.average(velocities)) / 127.0) if velocities else 0.0,
        "drum_channel_ratio": (channel_9_count / len(notes)) if notes else 0.0,
    }


def baseline_feature_vector(summary: dict) -> list[float]:
    return [
        summary["lowest_pitch"],
        summary["highest_pitch"],
        summary["unique_pitch_num"],
        summary["average_pitch_value"],
    ]


def enhanced_feature_vector(summary: dict) -> list[float]:
    return [
        summary["lowest_pitch"],
        summary["highest_pitch"],
        summary["unique_pitch_num"],
        summary["average_pitch_value"],
        summary["pitch_span"],
        summary["log_beats"],
        summary["log_note_density"],
        summary["average_velocity_norm"],
        summary["drum_channel_ratio"],
    ]
