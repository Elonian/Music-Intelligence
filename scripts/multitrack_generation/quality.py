from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np

from scripts.multitrack_generation.constants import INSTRUMENT_LABELS, TIME_STEPS_PER_BEAT
from scripts.multitrack_generation.data import collect_split_files, has_packed_split, packed_split_paths, resolve_processed_dir
from scripts.multitrack_generation.events import crop_and_augment_notes, normalize_note_array, sequence_to_note_array


PAPER_OBJECTIVE_METRICS = ("pitch_class_entropy", "scale_consistency_percent", "groove_consistency_percent")
MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
NATURAL_MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)
SCALE_PATTERNS = {"major": MAJOR_SCALE, "minor": NATURAL_MINOR_SCALE}


def _safe_float(value: float | int | np.floating | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def pitch_class_entropy(notes: np.ndarray) -> float | None:
    notes = normalize_note_array(notes)
    if notes.size == 0:
        return None
    counts = np.bincount((notes[:, 1] % 12).astype(np.int64), minlength=12).astype(np.float64)
    total = float(counts.sum())
    if total <= 0:
        return None
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def scale_consistency(notes: np.ndarray) -> tuple[float | None, str | None]:
    notes = normalize_note_array(notes)
    if notes.size == 0:
        return None, None
    pitch_classes = (notes[:, 1] % 12).astype(np.int64)
    total = max(int(pitch_classes.size), 1)
    best_score = -1.0
    best_name: str | None = None
    for root in range(12):
        for mode, pattern in SCALE_PATTERNS.items():
            allowed = {(root + interval) % 12 for interval in pattern}
            score = float(np.mean([pc in allowed for pc in pitch_classes]))
            if score > best_score:
                best_score = score
                best_name = f"{root}:{mode}"
    return float(100.0 * best_score), best_name


def groove_consistency(notes: np.ndarray, beats_per_bar: int = 4) -> float | None:
    notes = normalize_note_array(notes)
    if notes.size == 0:
        return None
    bar_steps = int(beats_per_bar * TIME_STEPS_PER_BEAT)
    if bar_steps <= 0:
        return None
    max_step = int(np.max(notes[:, 0] + np.maximum(notes[:, 2], 1)))
    n_bars = max(1, int(math.ceil(max_step / bar_steps)))
    if n_bars < 2:
        return None
    patterns = np.zeros((n_bars, bar_steps), dtype=bool)
    for onset in notes[:, 0].astype(np.int64):
        if onset < 0:
            continue
        bar = int(onset // bar_steps)
        position = int(onset % bar_steps)
        if 0 <= bar < n_bars:
            patterns[bar, position] = True
    similarities = 1.0 - np.logical_xor(patterns[:-1], patterns[1:]).mean(axis=1)
    return float(100.0 * similarities.mean())


def _active_polyphony(notes: np.ndarray) -> tuple[float, int]:
    notes = normalize_note_array(notes)
    if notes.size == 0:
        return 0.0, 0
    end_steps = notes[:, 0] + np.maximum(notes[:, 2], 1)
    max_step = max(int(end_steps.max()), 1)
    deltas = np.zeros(max_step + 1, dtype=np.int32)
    for onset, _pitch, duration, _instrument in notes:
        start = max(0, int(onset))
        end = min(max_step, start + max(1, int(duration)))
        if end <= start:
            continue
        deltas[start] += 1
        deltas[end] -= 1
    active = np.cumsum(deltas[:-1])
    if active.size == 0:
        return 0.0, 0
    return float(active.mean()), int(active.max())


def _entropy_from_counts(counts: np.ndarray) -> float:
    counts = np.asarray(counts, dtype=np.float64)
    total = float(counts.sum())
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-np.sum(probabilities * np.log2(probabilities)))


def sequence_diagnostics(sequence: np.ndarray | None) -> dict:
    if sequence is None:
        return {}
    seq = np.asarray(sequence, dtype=np.int64)
    if seq.ndim != 2 or seq.shape[1] != 6:
        return {"sequence_valid": False}
    raw_notes = sequence_to_note_array(seq, deduplicate=False)
    unique_notes = sequence_to_note_array(seq, deduplicate=True)
    declared: list[int] = []
    for event in seq:
        event_type = int(event[0])
        if event_type == 1:
            instrument = int(event[5])
            if instrument not in declared:
                declared.append(instrument)
        elif event_type >= 2:
            break
    violations = 0
    note_events = 0
    if declared:
        allowed = set(declared)
        for event in seq:
            if int(event[0]) == 3:
                note_events += 1
                if int(event[5]) not in allowed:
                    violations += 1
    type_values = seq[:, 0].astype(np.int64)
    return {
        "sequence_valid": True,
        "sequence_len": int(seq.shape[0]),
        "raw_note_events": int(raw_notes.shape[0]),
        "unique_note_events": int(unique_notes.shape[0]),
        "duplicate_note_rate": float(1.0 - unique_notes.shape[0] / raw_notes.shape[0]) if raw_notes.shape[0] else 0.0,
        "has_start_song": bool(np.any(type_values == 0)),
        "has_start_notes": bool(np.any(type_values == 2)),
        "has_end_song": bool(np.any(type_values == 4)),
        "declared_instruments": [int(item) for item in declared],
        "note_instrument_violation_rate": float(violations / note_events) if note_events else 0.0,
    }


def note_quality_metrics(
    notes: np.ndarray,
    sequence: np.ndarray | None = None,
    tempo_bpm: int = 120,
) -> dict:
    notes = normalize_note_array(notes)
    scale_value, best_scale = scale_consistency(notes)
    pce = pitch_class_entropy(notes)
    groove = groove_consistency(notes)
    end_step = int(np.max(notes[:, 0] + np.maximum(notes[:, 2], 1))) if notes.size else 0
    length_beats = float(end_step / TIME_STEPS_PER_BEAT) if end_step else 0.0
    length_seconds = float(length_beats * 60.0 / max(tempo_bpm, 1))
    instrument_counts = np.bincount(notes[:, 3].astype(np.int64), minlength=len(INSTRUMENT_LABELS))[: len(INSTRUMENT_LABELS)] if notes.size else np.zeros(len(INSTRUMENT_LABELS), dtype=np.int64)
    average_polyphony, max_polyphony = _active_polyphony(notes)
    metrics = {
        "note_count": int(notes.shape[0]),
        "length_beats": length_beats,
        "length_seconds": length_seconds,
        "notes_per_beat": float(notes.shape[0] / length_beats) if length_beats > 0 else 0.0,
        "unique_pitch_count": int(np.unique(notes[:, 1]).size) if notes.size else 0,
        "unique_pitch_class_count": int(np.unique(notes[:, 1] % 12).size) if notes.size else 0,
        "active_instrument_count": int(np.count_nonzero(instrument_counts)),
        "instrument_entropy": _entropy_from_counts(instrument_counts),
        "average_polyphony": average_polyphony,
        "max_polyphony": int(max_polyphony),
        "mean_duration_steps": float(np.mean(notes[:, 2])) if notes.size else 0.0,
        "pitch_class_entropy": _safe_float(pce),
        "scale_consistency_percent": _safe_float(scale_value),
        "best_scale": best_scale,
        "groove_consistency_percent": _safe_float(groove),
        "instrument_counts": {label: int(instrument_counts[index]) for index, label in enumerate(INSTRUMENT_LABELS)},
    }
    metrics.update(sequence_diagnostics(sequence))
    return metrics


def summarize_metric_rows(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    numeric_keys = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
        }
    )
    summary: dict[str, object] = {
        "samples": len(rows),
        "nonempty_samples": int(sum(1 for row in rows if int(row.get("note_count", 0) or 0) > 0)),
    }
    for key in numeric_keys:
        values = np.asarray([float(row[key]) for row in rows if _safe_float(row.get(key)) is not None], dtype=np.float64)
        if values.size == 0:
            continue
        std = float(values.std(ddof=1)) if values.size > 1 else 0.0
        ci95 = float(1.96 * std / math.sqrt(values.size)) if values.size > 1 else 0.0
        summary[key] = {
            "mean": float(values.mean()),
            "std": std,
            "ci95": ci95,
            "min": float(values.min()),
            "max": float(values.max()),
            "count": int(values.size),
        }
    return summary


def paper_metric_distance(candidate: dict, reference_summary: dict) -> dict:
    components: dict[str, float] = {}
    total = 0.0
    for key in PAPER_OBJECTIVE_METRICS:
        value = _safe_float(candidate.get(key))
        reference = reference_summary.get(key)
        if value is None or not isinstance(reference, dict):
            continue
        ref_mean = _safe_float(reference.get("mean"))
        if ref_mean is None:
            continue
        ref_std = max(float(reference.get("std") or 0.0), abs(ref_mean) * 0.05, 1e-6)
        distance = abs(value - ref_mean) / ref_std
        components[key] = float(distance)
        total += float(distance)
    return {"paper_metric_distance": float(total), "paper_metric_distance_components": components}


def iter_reference_notes(
    processed_dir: Path | str | None,
    split: str = "test",
    max_files: int | None = None,
    max_beats: int = 32,
) -> Iterable[np.ndarray]:
    root = resolve_processed_dir(processed_dir)
    limit = None if max_files is None or max_files <= 0 else int(max_files)
    if has_packed_split(root, split):
        events_path, offsets_path = packed_split_paths(root, split)
        events = np.load(events_path, mmap_mode="r")
        offsets = np.load(offsets_path, mmap_mode="r")
        total = max(int(offsets.shape[0]) - 1, 0)
        count = total if limit is None else min(total, limit)
        for index in range(count):
            start = int(offsets[index])
            end = int(offsets[index + 1])
            yield crop_and_augment_notes(events[start:end], max_beats=max_beats, augmentation=False)
        return

    files = getattr(collect_split_files(root), split)
    if limit is not None:
        files = files[:limit]
    for filename in files:
        try:
            yield crop_and_augment_notes(np.load(filename, mmap_mode="r"), max_beats=max_beats, augmentation=False)
        except Exception:
            continue


def load_generated_notes(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    path = Path(path)
    if path.is_dir():
        notes_path = path / "notes.npy"
        sequence_path = path / "sequence.npy"
    else:
        notes_path = path
        sequence_path = path.with_name("sequence.npy")
    if not notes_path.exists():
        raise FileNotFoundError(notes_path)
    notes = normalize_note_array(np.load(notes_path))
    sequence = np.load(sequence_path) if sequence_path.exists() else None
    return notes, sequence


def find_generated_note_files(path: Path) -> list[Path]:
    path = Path(path)
    if path.is_file():
        return [path]
    if (path / "notes.npy").exists():
        return [path / "notes.npy"]
    return sorted(path.rglob("notes.npy"))
