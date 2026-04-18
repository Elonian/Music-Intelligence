from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.automatic_music_instrumentation.core.data import INSTRUMENT_LABELS, N_CLASSES, packed_split_paths
from scripts.automatic_music_instrumentation.core.metrics import confusion_matrix_numpy, normalize_confusion_matrix


def pitch_zone_predict_pitches(pitches: np.ndarray) -> np.ndarray:
    """Assign instruments with fixed pitch ranges."""
    labels = np.zeros_like(pitches, dtype=np.int64)
    labels[pitches >= 105] = 4
    labels[(pitches >= 83) & (pitches < 105)] = 3
    labels[(pitches >= 72) & (pitches < 83)] = 0
    labels[(pitches >= 44) & (pitches < 72)] = 1
    labels[pitches < 44] = 2
    return labels


def pitch_zone_predict_events(events: np.ndarray) -> np.ndarray:
    return pitch_zone_predict_pitches(events[:, 1])


def pitch_zone_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, files: int) -> dict:
    matrix = confusion_matrix_numpy(y_true, y_pred, n_classes=N_CLASSES)
    return {
        "model": "pitch_zones",
        "files": files,
        "num_predictions": int(len(y_true)),
        "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalize_confusion_matrix(matrix).tolist(),
        "labels": list(INSTRUMENT_LABELS),
    }


def evaluate_pitch_zone_files(files: list[Path], max_files: int | None = None) -> dict:
    selected = files if max_files is None else files[:max_files]
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for file_path in selected:
        array = np.load(file_path)
        if array.ndim != 2 or array.shape[1] != 4 or array.size == 0:
            continue
        y_true_parts.append(array[:, 3].astype(np.int64))
        y_pred_parts.append(pitch_zone_predict_events(array))
    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.int64)
    return pitch_zone_metrics_from_predictions(y_true, y_pred, files=len(selected))


def evaluate_pitch_zone_packed(processed_dir: Path | str | None, split: str = "test") -> dict:
    paths = packed_split_paths(processed_dir, split)
    events = np.load(paths.events, mmap_mode="r")
    offsets = np.load(paths.offsets, mmap_mode="r")
    if events.ndim != 2 or events.shape[1] != 4 or events.size == 0:
        return pitch_zone_metrics_from_predictions(np.array([], dtype=np.int64), np.array([], dtype=np.int64), files=0)
    y_true = np.asarray(events[:, 3], dtype=np.int64)
    y_pred = pitch_zone_predict_events(events)
    return pitch_zone_metrics_from_predictions(y_true, y_pred, files=max(int(offsets.shape[0]) - 1, 0))
