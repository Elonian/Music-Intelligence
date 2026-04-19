from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from scripts.multitrack_generation.constants import TYPE_PAD
from scripts.multitrack_generation.events import crop_and_augment_notes, note_array_to_event_sequence
from utils.project_paths import MULTITRACK_GENERATION_PROCESSED_DIR


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    valid: list[Path]
    test: list[Path]


def resolve_processed_dir(path: Path | str | None = None) -> Path:
    if path is None:
        return MULTITRACK_GENERATION_PROCESSED_DIR
    candidate = Path(path).expanduser().resolve()
    if (candidate / "processed" / "train").exists():
        return candidate / "processed"
    return candidate


def collect_split_files(processed_dir: Path | str | None = None, max_files: int | None = None) -> SplitFiles:
    root = resolve_processed_dir(processed_dir)
    splits: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        files = sorted((root / split).glob("*.npy"))
        splits[split] = files[:max_files] if max_files is not None else files
    return SplitFiles(train=splits["train"], valid=splits["valid"], test=splits["test"])


def packed_split_paths(processed_dir: Path | str | None, split: str) -> tuple[Path, Path]:
    root = resolve_processed_dir(processed_dir)
    packed = root / "packed"
    return packed / f"{split}_events.npy", packed / f"{split}_offsets.npy"


def has_packed_split(processed_dir: Path | str | None, split: str) -> bool:
    events_path, offsets_path = packed_split_paths(processed_dir, split)
    return events_path.exists() and offsets_path.exists()


def summarize_split(files: list[Path], max_files: int | None = None) -> dict:
    checked = files if max_files is None else files[:max_files]
    note_count = 0
    instrument_counts = np.zeros(5, dtype=np.int64)
    pitch_counts = np.zeros(128, dtype=np.int64)
    bad_files: list[str] = []
    for filename in checked:
        try:
            array = np.load(filename, mmap_mode="r")
            if array.ndim != 2 or array.shape[1] != 4:
                bad_files.append(str(filename))
                continue
            note_count += int(array.shape[0])
            labels, counts = np.unique(array[:, 3].astype(int), return_counts=True)
            for label, count in zip(labels, counts):
                if 0 <= int(label) < instrument_counts.size:
                    instrument_counts[int(label)] += int(count)
            pitches, counts = np.unique(array[:, 1].astype(int), return_counts=True)
            for pitch, count in zip(pitches, counts):
                if 0 <= int(pitch) < pitch_counts.size:
                    pitch_counts[int(pitch)] += int(count)
        except Exception:
            bad_files.append(str(filename))
    return {
        "files": len(checked),
        "notes": int(note_count),
        "instrument_counts": instrument_counts.tolist(),
        "pitch_counts": pitch_counts.tolist(),
        "bad_files": bad_files,
    }


def summarize_packed_split(processed_dir: Path | str | None, split: str, chunk_size: int = 1_000_000) -> dict | None:
    root = resolve_processed_dir(processed_dir)
    events_path = root / "packed" / f"{split}_events.npy"
    offsets_path = root / "packed" / f"{split}_offsets.npy"
    if not events_path.exists() or not offsets_path.exists():
        return None

    events = np.load(events_path, mmap_mode="r")
    offsets = np.load(offsets_path, mmap_mode="r")
    instrument_counts = np.zeros(5, dtype=np.int64)
    pitch_counts = np.zeros(128, dtype=np.int64)
    for start in range(0, int(events.shape[0]), chunk_size):
        chunk = events[start : start + chunk_size]
        labels = chunk[:, 3].astype(np.int64, copy=False)
        pitches = chunk[:, 1].astype(np.int64, copy=False)
        instrument_counts += np.bincount(np.clip(labels, 0, 4), minlength=5)[:5]
        pitch_counts += np.bincount(np.clip(pitches, 0, 127), minlength=128)[:128]
    return {
        "files": max(int(offsets.shape[0]) - 1, 0),
        "notes": int(events.shape[0]),
        "instrument_counts": instrument_counts.tolist(),
        "pitch_counts": pitch_counts.tolist(),
        "bad_files": [],
        "source": str(events_path),
    }


class MultitrackSequenceDataset(Dataset):
    """multitrack sequence dataset backed by .npy note-event files."""

    def __init__(
        self,
        filenames: list[Path],
        max_beats: int = 32,
        max_seq_len: int = 1024,
        augmentation: bool = False,
        preload: bool = False,
        transpose_low: int = -5,
        transpose_high: int = 6,
    ) -> None:
        super().__init__()
        self.filenames = list(filenames)
        self.max_beats = int(max_beats)
        self.max_seq_len = int(max_seq_len)
        self.augmentation = bool(augmentation)
        self.transpose_low = int(transpose_low)
        self.transpose_high = int(transpose_high)
        self.arrays: list[np.ndarray] | None = None
        if preload:
            arrays: list[np.ndarray] = []
            for filename in self.filenames:
                try:
                    arrays.append(np.load(filename).astype(np.int64, copy=False))
                except Exception:
                    continue
            self.arrays = arrays

    def __len__(self) -> int:
        return len(self.arrays) if self.arrays is not None else len(self.filenames)

    def _load_array(self, index: int) -> np.ndarray:
        if self.arrays is not None:
            return self.arrays[index]
        return np.load(self.filenames[index]).astype(np.int64, copy=False)

    def __getitem__(self, index: int) -> torch.Tensor:
        source = self._load_array(index)
        notes = crop_and_augment_notes(
            source,
            max_beats=self.max_beats,
            augmentation=self.augmentation,
            transpose_low=self.transpose_low,
            transpose_high=self.transpose_high,
        )
        sequence = note_array_to_event_sequence(notes, max_seq_len=self.max_seq_len)
        return torch.as_tensor(sequence, dtype=torch.long)


class PackedMultitrackSequenceDataset(Dataset):
    """Multitrack sequence dataset backed by packed memmap arrays."""

    def __init__(
        self,
        events_path: Path,
        offsets_path: Path,
        max_beats: int = 32,
        max_seq_len: int = 1024,
        augmentation: bool = False,
        transpose_low: int = -5,
        transpose_high: int = 6,
    ) -> None:
        super().__init__()
        self.events_path = Path(events_path)
        self.offsets_path = Path(offsets_path)
        self.max_beats = int(max_beats)
        self.max_seq_len = int(max_seq_len)
        self.augmentation = bool(augmentation)
        self.transpose_low = int(transpose_low)
        self.transpose_high = int(transpose_high)
        self._events: np.ndarray | None = None
        self._offsets: np.ndarray | None = None
        offsets = np.load(self.offsets_path, mmap_mode="r")
        self._length = max(int(offsets.shape[0]) - 1, 0)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_events"] = None
        state["_offsets"] = None
        return state

    @property
    def events(self) -> np.ndarray:
        if self._events is None:
            self._events = np.load(self.events_path, mmap_mode="r")
        return self._events

    @property
    def offsets(self) -> np.ndarray:
        if self._offsets is None:
            self._offsets = np.load(self.offsets_path, mmap_mode="r")
        return self._offsets

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> torch.Tensor:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        source = self.events[start:end]
        notes = crop_and_augment_notes(
            source,
            max_beats=self.max_beats,
            augmentation=self.augmentation,
            transpose_low=self.transpose_low,
            transpose_high=self.transpose_high,
        )
        sequence = note_array_to_event_sequence(notes, max_seq_len=self.max_seq_len)
        return torch.as_tensor(sequence, dtype=torch.long)


def pad_collate(sequences: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    if not sequences:
        return torch.empty((0, 0, 6), dtype=torch.long), torch.empty((0,), dtype=torch.long)
    lengths = torch.as_tensor([sequence.shape[0] for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    for row, length in enumerate(lengths.tolist()):
        if length < padded.shape[1]:
            padded[row, length:, 0] = TYPE_PAD
    return padded.long(), lengths


def build_dataloader(
    filenames: list[Path],
    batch_size: int,
    max_beats: int = 32,
    max_seq_len: int = 1024,
    augmentation: bool = False,
    shuffle: bool = False,
    num_workers: int = 0,
    preload: bool = False,
    drop_last: bool = False,
    use_packed: bool = False,
    processed_dir: Path | str | None = None,
    split_name: str | None = None,
) -> DataLoader:
    if use_packed and split_name is not None and has_packed_split(processed_dir, split_name):
        events_path, offsets_path = packed_split_paths(processed_dir, split_name)
        dataset: Dataset = PackedMultitrackSequenceDataset(
            events_path=events_path,
            offsets_path=offsets_path,
            max_beats=max_beats,
            max_seq_len=max_seq_len,
            augmentation=augmentation,
        )
    else:
        dataset = MultitrackSequenceDataset(
            filenames=filenames,
            max_beats=max_beats,
            max_seq_len=max_seq_len,
            augmentation=augmentation,
            preload=preload,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=pad_collate,
        pin_memory=torch.cuda.is_available(),
        drop_last=drop_last,
    )
