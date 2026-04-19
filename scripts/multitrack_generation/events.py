from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.multitrack_generation.constants import (
    INSTRUMENT_LABELS,
    MAX_BEAT,
    MAX_DURATION_STEPS,
    TIME_STEPS_PER_BEAT,
    TYPE_END_SONG,
    TYPE_INSTRUMENT,
    TYPE_NOTE,
    TYPE_START_NOTES,
    TYPE_START_SONG,
)


def normalize_note_array(array: np.ndarray) -> np.ndarray:
    """Return a clean int64 note array shaped (n_notes, 4)."""
    notes = np.asarray(array)
    if notes.ndim != 2 or notes.shape[1] != 4:
        raise ValueError(f"expected note array shaped (num_notes, 4), got {notes.shape}")
    notes = notes.astype(np.int64, copy=True)
    if notes.size == 0:
        return notes.reshape(0, 4)
    notes[:, 0] = np.maximum(notes[:, 0], 0)
    notes[:, 1] = np.clip(notes[:, 1], 0, 127)
    notes[:, 2] = np.clip(notes[:, 2], 0, MAX_DURATION_STEPS)
    notes[:, 3] = np.clip(notes[:, 3], 0, len(INSTRUMENT_LABELS) - 1)
    order = np.lexsort((notes[:, 3], notes[:, 2], notes[:, 1], notes[:, 0]))
    return notes[order]


def crop_and_augment_notes(
    source: np.ndarray,
    max_beats: int = 32,
    augmentation: bool = False,
    transpose_low: int = -5,
    transpose_high: int = 6,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Crop multitrack note rows and optionally apply notebook-style augmentation.

    Training uses random crop plus random transposition. Validation/test use a
    deterministic crop from the beginning of the file.
    """
    notes = normalize_note_array(source)
    if notes.size == 0:
        return notes

    window = int(max_beats * TIME_STEPS_PER_BEAT)
    if not augmentation:
        return normalize_note_array(notes[notes[:, 0] < window])

    rng = rng or np.random.default_rng()
    n_beats = int(np.max(notes[:, 0]) // TIME_STEPS_PER_BEAT)
    if n_beats < max_beats + 1:
        start_time = 0
    else:
        start_beat = int(rng.integers(0, n_beats - max_beats + 1))
        start_time = start_beat * TIME_STEPS_PER_BEAT
    end_time = start_time + window
    cropped = notes[(notes[:, 0] >= start_time) & (notes[:, 0] < end_time)].copy()
    if cropped.size == 0:
        return cropped.reshape(0, 4)

    cropped[:, 0] -= start_time
    shift = int(rng.integers(transpose_low, transpose_high + 1))
    cropped[:, 1] += shift
    cropped[:, 1][cropped[:, 1] > 127] -= 12
    cropped[:, 1][cropped[:, 1] < 0] += 12
    return normalize_note_array(cropped)


def note_array_to_event_sequence(
    notes: np.ndarray,
    max_seq_len: int = 1024,
    include_instrument_events: bool = True,
) -> np.ndarray:
    """Convert rows [onset, pitch, duration, instrument] into multitrack events."""
    notes = normalize_note_array(notes)
    sequence: list[list[int]] = [[TYPE_START_SONG, 0, 0, 0, 0, 0]]

    if include_instrument_events and notes.size:
        for instrument in sorted(int(item) for item in np.unique(notes[:, 3])):
            sequence.append([TYPE_INSTRUMENT, 0, 0, 0, 0, instrument])

    sequence.append([TYPE_START_NOTES, 0, 0, 0, 0, 0])
    for onset, pitch, duration, instrument in notes:
        beat = int(onset // TIME_STEPS_PER_BEAT)
        position = int(onset % TIME_STEPS_PER_BEAT)
        sequence.append(
            [
                TYPE_NOTE,
                max(0, min(beat, MAX_BEAT)),
                max(0, min(position, TIME_STEPS_PER_BEAT - 1)),
                max(0, min(int(pitch), 127)),
                max(0, min(int(duration), MAX_DURATION_STEPS)),
                max(0, min(int(instrument), len(INSTRUMENT_LABELS) - 1)),
            ]
        )
    sequence.append([TYPE_END_SONG, 0, 0, 0, 0, 0])

    if max_seq_len > 0 and len(sequence) > max_seq_len:
        sequence = sequence[:max_seq_len]
        sequence[-1] = [TYPE_END_SONG, 0, 0, 0, 0, 0]
    return np.asarray(sequence, dtype=np.int64)


def sequence_to_note_array(sequence: np.ndarray, deduplicate: bool = True) -> np.ndarray:
    """Restore multitrack note events to rows [onset, pitch, duration, instrument]."""
    seq = np.asarray(sequence, dtype=np.int64)
    if seq.ndim != 2 or seq.shape[1] != 6:
        raise ValueError(f"expected event sequence shaped (seq_len, 6), got {seq.shape}")
    rows: list[tuple[int, int, int, int]] = []
    for event in seq:
        if int(event[0]) != TYPE_NOTE:
            continue
        beat, position, pitch, duration, instrument = (int(item) for item in event[1:])
        onset = max(0, beat) * TIME_STEPS_PER_BEAT + max(0, min(position, TIME_STEPS_PER_BEAT - 1))
        rows.append(
            (
                onset,
                max(0, min(pitch, 127)),
                max(1, min(duration, MAX_DURATION_STEPS)),
                max(0, min(instrument, len(INSTRUMENT_LABELS) - 1)),
            )
        )
    if deduplicate:
        rows = sorted(set(rows))
    else:
        rows = sorted(rows)
    if not rows:
        return np.empty((0, 4), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64)


def instrument_prompt(instruments: list[int] | tuple[int, ...]) -> np.ndarray:
    sequence = [[TYPE_START_SONG, 0, 0, 0, 0, 0]]
    for instrument in instruments:
        sequence.append([TYPE_INSTRUMENT, 0, 0, 0, 0, int(instrument)])
    sequence.append([TYPE_START_NOTES, 0, 0, 0, 0, 0])
    return np.asarray(sequence, dtype=np.int64)


def twinkle_prompt() -> np.ndarray:
    """Small N-beat continuation prompt used by multitrack-style generation."""
    notes = np.asarray(
        [
            [0, 60, 12, 0],
            [12, 60, 12, 0],
            [24, 67, 12, 0],
            [36, 67, 12, 0],
            [48, 69, 12, 0],
            [60, 69, 12, 0],
            [72, 67, 24, 0],
        ],
        dtype=np.int64,
    )
    return note_array_to_event_sequence(notes, max_seq_len=64)


def save_sequence_csv(sequence: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "type,beat,position,pitch,duration,instrument\n"
    rows = "\n".join(",".join(str(int(value)) for value in event) for event in sequence)
    output_path.write_text(header + rows + ("\n" if rows else ""), encoding="utf-8")


def save_note_csv(notes: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = "onset,pitch,duration,instrument\n"
    rows = "\n".join(",".join(str(int(value)) for value in note) for note in notes)
    output_path.write_text(header + rows + ("\n" if rows else ""), encoding="utf-8")
