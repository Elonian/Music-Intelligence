from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Callable
from zipfile import ZipFile

import numpy as np
from numpy.lib import format as np_format
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from utils.project_paths import AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR


INSTRUMENT_LABELS = ("piano", "guitar", "bass", "strings", "brass")
N_CLASSES = len(INSTRUMENT_LABELS)
PAD_LABEL = N_CLASSES + 1
TIME_STEPS_PER_BEAT = 24
MAX_DURATION_STEPS = 192
PACKED_DIR_NAME = "packed"
STANDARD_NPY_HEADER_BYTES = 128


@dataclass(frozen=True)
class SplitFiles:
    train: list[Path]
    valid: list[Path]
    test: list[Path]


@dataclass(frozen=True)
class PackedSplitPaths:
    events: Path
    offsets: Path
    filenames: Path


@dataclass(frozen=True)
class DatasetSummary:
    processed_dir: Path
    notebook_path: Path | None
    raw_archive_path: Path | None
    clean_midi_dir: Path | None
    sample_file_count: int
    sample_json_count: int
    train_count: int
    valid_count: int
    test_count: int
    clean_midi_count: int | None


def resolve_processed_dir(path: Path | str | None = None) -> Path:
    """Return the directory containing train/valid/test/samples."""
    if path is None:
        return AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR
    candidate = Path(path).expanduser().resolve()
    if (candidate / "processed" / "train").exists():
        return candidate / "processed"
    return candidate


def collect_split_files(processed_dir: Path | str | None = None, max_files: int | None = None) -> SplitFiles:
    processed = resolve_processed_dir(processed_dir)
    splits: dict[str, list[Path]] = {}
    for split in ("train", "valid", "test"):
        files = sorted((processed / split).rglob("*.npy"))
        if max_files is not None:
            files = files[:max_files]
        splits[split] = files
    return SplitFiles(train=splits["train"], valid=splits["valid"], test=splits["test"])


def packed_split_paths(processed_dir: Path | str | None, split: str) -> PackedSplitPaths:
    processed = resolve_processed_dir(processed_dir)
    packed_dir = processed / PACKED_DIR_NAME
    return PackedSplitPaths(
        events=packed_dir / f"{split}_events.npy",
        offsets=packed_dir / f"{split}_offsets.npy",
        filenames=packed_dir / f"{split}_filenames.txt",
    )


def has_packed_split(processed_dir: Path | str | None, split: str) -> bool:
    paths = packed_split_paths(processed_dir, split)
    return paths.events.exists() and paths.offsets.exists()


def _read_npy_event_header(filename: Path) -> tuple[tuple[int, ...], np.dtype] | None:
    try:
        with filename.open("rb") as handle:
            version = np_format.read_magic(handle)
            if version == (1, 0):
                shape, fortran_order, dtype = np_format.read_array_header_1_0(handle)
            elif version == (2, 0):
                shape, fortran_order, dtype = np_format.read_array_header_2_0(handle)
            else:
                return None
        if fortran_order:
            return None
        return tuple(int(item) for item in shape), np.dtype(dtype)
    except Exception:
        return None


def _infer_standard_event_header_from_size(file_size: int) -> tuple[tuple[int, int], np.dtype] | None:
    data_bytes = file_size - STANDARD_NPY_HEADER_BYTES
    row_bytes = np.dtype(np.int64).itemsize * 4
    if data_bytes < 0 or data_bytes % row_bytes:
        return None
    return (int(data_bytes // row_bytes), 4), np.dtype(np.int64)


def _infer_standard_event_header(filename: Path) -> tuple[tuple[int, int], np.dtype] | None:
    try:
        return _infer_standard_event_header_from_size(filename.stat().st_size)
    except OSError:
        return None


def _load_standard_event_array(filename: Path, length: int) -> np.ndarray | None:
    try:
        flat = np.fromfile(
            filename,
            dtype=np.int64,
            count=length * 4,
            offset=STANDARD_NPY_HEADER_BYTES,
        )
    except Exception:
        return None
    if flat.size != length * 4:
        return None
    return flat.reshape(length, 4)


def _event_array_from_zip(zip_file: ZipFile, name: str, length: int) -> np.ndarray:
    with zip_file.open(name) as handle:
        handle.read(STANDARD_NPY_HEADER_BYTES)
        payload = handle.read(length * np.dtype(np.int64).itemsize * 4)
    flat = np.frombuffer(payload, dtype=np.int64)
    if flat.size == length * 4:
        return flat.reshape(length, 4)
    return np.load(BytesIO(zip_file.read(name)))


def build_packed_split(
    files: list[Path],
    processed_dir: Path | str | None,
    split: str,
    dtype: np.dtype | type = np.int32,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 1000,
) -> dict:
    """Pack many small note-event files into one memory-mapped split cache.

    Training can then slice one large array instead of opening thousands of
    individual files during startup or batch loading.
    """
    paths = packed_split_paths(processed_dir, split)
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and paths.events.exists() and paths.offsets.exists():
        offsets = np.load(paths.offsets, mmap_mode="r")
        return {
            "split": split,
            "events_path": str(paths.events),
            "offsets_path": str(paths.offsets),
            "file_count": max(int(offsets.shape[0]) - 1, 0),
            "event_count": int(offsets[-1]) if offsets.shape[0] else 0,
            "rebuilt": False,
        }

    valid_files: list[Path] = []
    lengths: list[int] = []
    total_rows = 0
    for file_index, filename in enumerate(files, start=1):
        header = _infer_standard_event_header(filename) or _read_npy_event_header(filename)
        if header is None:
            continue
        shape, _dtype = header
        if len(shape) != 2 or shape[1] != 4:
            continue
        valid_files.append(filename)
        length = int(shape[0])
        lengths.append(length)
        total_rows += length
        if progress is not None and progress_every > 0 and file_index % progress_every == 0:
            progress(f"{split}: scanned {file_index}/{len(files)} files")

    tmp_events = paths.events.with_name(f"{paths.events.stem}.tmp.npy")
    tmp_offsets = paths.offsets.with_name(f"{paths.offsets.stem}.tmp.npy")
    tmp_filenames = paths.filenames.with_suffix(paths.filenames.suffix + ".tmp")
    for tmp_path in (tmp_events, tmp_offsets, tmp_filenames):
        if tmp_path.exists():
            tmp_path.unlink()

    events = np.lib.format.open_memmap(tmp_events, mode="w+", dtype=dtype, shape=(total_rows, 4))
    offsets = np.zeros(len(valid_files) + 1, dtype=np.int64)
    cursor = 0
    for file_index, (filename, length) in enumerate(zip(valid_files, lengths), start=1):
        if length:
            array = _load_standard_event_array(filename, length)
            if array is None:
                array = np.load(filename, mmap_mode="r")
            events[cursor : cursor + length] = array.astype(dtype, copy=False)
        cursor += length
        offsets[file_index] = cursor
        if progress is not None and progress_every > 0 and file_index % progress_every == 0:
            progress(f"{split}: packed {file_index}/{len(valid_files)} files")
    del events

    np.save(tmp_offsets, offsets)
    with tmp_filenames.open("w", encoding="utf-8") as handle:
        for filename in valid_files:
            handle.write(f"{filename}\n")

    tmp_events.replace(paths.events)
    tmp_offsets.replace(paths.offsets)
    tmp_filenames.replace(paths.filenames)
    return {
        "split": split,
        "events_path": str(paths.events),
        "offsets_path": str(paths.offsets),
        "file_count": len(valid_files),
        "event_count": int(total_rows),
        "rebuilt": True,
    }


def build_packed_split_from_zip(
    zip_path: Path,
    processed_dir: Path | str | None,
    split: str,
    dtype: np.dtype | type = np.int32,
    overwrite: bool = False,
    progress: Callable[[str], None] | None = None,
    progress_every: int = 1000,
) -> dict:
    paths = packed_split_paths(processed_dir, split)
    paths.events.parent.mkdir(parents=True, exist_ok=True)
    if not overwrite and paths.events.exists() and paths.offsets.exists():
        offsets = np.load(paths.offsets, mmap_mode="r")
        return {
            "split": split,
            "events_path": str(paths.events),
            "offsets_path": str(paths.offsets),
            "file_count": max(int(offsets.shape[0]) - 1, 0),
            "event_count": int(offsets[-1]) if offsets.shape[0] else 0,
            "source_zip": str(zip_path),
            "rebuilt": False,
        }

    with ZipFile(zip_path) as zip_file:
        infos = sorted(
            (info for info in zip_file.infolist() if not info.is_dir() and info.filename.endswith(".npy")),
            key=lambda info: info.filename,
        )
        valid_names: list[str] = []
        lengths: list[int] = []
        total_rows = 0
        for file_index, info in enumerate(infos, start=1):
            header = _infer_standard_event_header_from_size(info.file_size)
            if header is None:
                array = np.load(BytesIO(zip_file.read(info.filename)))
                if array.ndim != 2 or array.shape[1] != 4:
                    continue
                length = int(array.shape[0])
            else:
                shape, _dtype = header
                length = int(shape[0])
            valid_names.append(info.filename)
            lengths.append(length)
            total_rows += length
            if progress is not None and progress_every > 0 and file_index % progress_every == 0:
                progress(f"{split}: scanned {file_index}/{len(infos)} zip members")

        tmp_events = paths.events.with_name(f"{paths.events.stem}.tmp.npy")
        tmp_offsets = paths.offsets.with_name(f"{paths.offsets.stem}.tmp.npy")
        tmp_filenames = paths.filenames.with_suffix(paths.filenames.suffix + ".tmp")
        for tmp_path in (tmp_events, tmp_offsets, tmp_filenames):
            if tmp_path.exists():
                tmp_path.unlink()

        events = np.lib.format.open_memmap(tmp_events, mode="w+", dtype=dtype, shape=(total_rows, 4))
        offsets = np.zeros(len(valid_names) + 1, dtype=np.int64)
        cursor = 0
        for file_index, (name, length) in enumerate(zip(valid_names, lengths), start=1):
            if length:
                array = _event_array_from_zip(zip_file, name, length)
                events[cursor : cursor + length] = array.astype(dtype, copy=False)
            cursor += length
            offsets[file_index] = cursor
            if progress is not None and progress_every > 0 and file_index % progress_every == 0:
                progress(f"{split}: packed {file_index}/{len(valid_names)} zip members")
        del events

    np.save(tmp_offsets, offsets)
    with tmp_filenames.open("w", encoding="utf-8") as handle:
        for name in valid_names:
            handle.write(f"{name}\n")

    tmp_events.replace(paths.events)
    tmp_offsets.replace(paths.offsets)
    tmp_filenames.replace(paths.filenames)
    return {
        "split": split,
        "events_path": str(paths.events),
        "offsets_path": str(paths.offsets),
        "file_count": len(valid_names),
        "event_count": int(total_rows),
        "source_zip": str(zip_path),
        "rebuilt": True,
    }


def summarize_dataset(data_root: Path | str | None = None, count_clean_midi: bool = True) -> DatasetSummary:
    if data_root is None:
        root = AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR.parent
    else:
        root = Path(data_root).expanduser().resolve()
    processed = resolve_processed_dir(root)
    clean_midi_dir = root / "lmd_clean_midi"
    raw_archive = root / "raw" / "clean_midi.tar.gz"
    notebook_candidates = sorted((root / "notebooks").glob("*.ipynb"))
    notebook = notebook_candidates[0] if notebook_candidates else None
    return DatasetSummary(
        processed_dir=processed,
        notebook_path=notebook,
        raw_archive_path=raw_archive if raw_archive.exists() else None,
        clean_midi_dir=clean_midi_dir if clean_midi_dir.exists() else None,
        sample_file_count=len(list((processed / "samples").rglob("*.*"))),
        sample_json_count=len(list((processed / "samples" / "converted").glob("*.json"))),
        train_count=len(list((processed / "train").glob("*.npy"))),
        valid_count=len(list((processed / "valid").glob("*.npy"))),
        test_count=len(list((processed / "test").glob("*.npy"))),
        clean_midi_count=len(list(clean_midi_dir.rglob("*.mid"))) if count_clean_midi and clean_midi_dir.exists() else None,
    )


def validate_event_array(array: np.ndarray) -> list[str]:
    issues: list[str] = []
    if array.ndim != 2 or array.shape[1] != 4:
        issues.append(f"expected shape (num_notes, 4), got {array.shape}")
        return issues
    if not np.issubdtype(array.dtype, np.integer):
        issues.append(f"expected integer dtype, got {array.dtype}")
    if array.size == 0:
        issues.append("array is empty")
        return issues
    if int(array[:, 0].min()) < 0:
        issues.append("negative onset time found")
    if int(array[:, 1].min()) < 0 or int(array[:, 1].max()) > 127:
        issues.append("pitch outside MIDI range 0..127 found")
    if int(array[:, 2].min()) < 0 or int(array[:, 2].max()) > MAX_DURATION_STEPS:
        issues.append("duration outside expected 0..192 range found")
    labels = set(int(label) for label in np.unique(array[:, 3]))
    unexpected = labels.difference(range(N_CLASSES))
    if unexpected:
        issues.append(f"unexpected instrument labels found: {sorted(unexpected)}")
    return issues


def inspect_event_files(files: list[Path], max_files: int | None = None) -> dict:
    checked = files if max_files is None else files[:max_files]
    bad: list[dict] = []
    total_notes = 0
    label_counts = {label: 0 for label in range(N_CLASSES)}
    for file_path in checked:
        try:
            array = np.load(file_path, mmap_mode="r")
            issues = validate_event_array(array)
            if issues:
                bad.append({"file": str(file_path), "issues": issues})
                continue
            total_notes += int(array.shape[0])
            labels, counts = np.unique(array[:, 3], return_counts=True)
            for label, count in zip(labels, counts):
                label_counts[int(label)] += int(count)
        except Exception as exc:  # pragma: no cover - diagnostics path
            bad.append({"file": str(file_path), "issues": [str(exc)]})
    return {
        "checked_files": len(checked),
        "total_notes": total_notes,
        "label_counts": {INSTRUMENT_LABELS[key]: value for key, value in label_counts.items()},
        "bad_files": bad,
    }


class MusicEventDataset(Dataset):
    """Note-event dataset.

    Each source .npy file has rows of
    [onset_time, pitch, duration, instrument_label].
    """

    def __init__(
        self,
        filenames: list[Path],
        max_beats: int = 32,
        max_seq_len: int = 1024,
        augmentation: bool = False,
        preload: bool = True,
        transpose_low: int = -5,
        transpose_high: int = 6,
    ) -> None:
        super().__init__()
        self.filenames = list(filenames)
        self.max_beats = max_beats
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation
        self.preload = preload
        self.transpose_low = transpose_low
        self.transpose_high = transpose_high
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

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self._load_array(index)
        if source.size == 0:
            empty_x = torch.empty((0, 3), dtype=torch.long)
            empty_y = torch.empty((0,), dtype=torch.long)
            return empty_x, empty_y

        if not self.augmentation:
            array = source[source[:, 0] < self.max_beats * TIME_STEPS_PER_BEAT].copy()
        else:
            n_beats = int(np.max(source[:, 0]) // TIME_STEPS_PER_BEAT)
            if n_beats < self.max_beats + 1:
                start_time = 0
            else:
                start_time = int(np.random.randint(n_beats - self.max_beats) * TIME_STEPS_PER_BEAT)
            end_time = start_time + self.max_beats * TIME_STEPS_PER_BEAT
            array = source[(source[:, 0] >= start_time) & (source[:, 0] < end_time)].copy()
            if array.size:
                array[:, 0] -= start_time
                shift = int(np.random.randint(self.transpose_low, self.transpose_high + 1))
                array[:, 1] += shift
                array[:, 1][array[:, 1] > 127] -= 12
                array[:, 1][array[:, 1] < 0] += 12

        array = array[: self.max_seq_len].copy()
        if array.size:
            array[:, 1] = np.clip(array[:, 1], 0, 127)
            array[:, 2] = np.clip(array[:, 2], 0, MAX_DURATION_STEPS)
        inputs = torch.as_tensor(array[:, :3], dtype=torch.long)
        labels = torch.as_tensor(array[:, 3], dtype=torch.long)
        return inputs, labels


class PackedMusicEventDataset(Dataset):
    """Memory-mapped version of MusicEventDataset backed by packed split files."""

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
        self.max_beats = max_beats
        self.max_seq_len = max_seq_len
        self.augmentation = augmentation
        self.transpose_low = transpose_low
        self.transpose_high = transpose_high
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

    def _load_array(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.events[start:end]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        source = self._load_array(index)
        if source.size == 0:
            empty_x = torch.empty((0, 3), dtype=torch.long)
            empty_y = torch.empty((0,), dtype=torch.long)
            return empty_x, empty_y

        if not self.augmentation:
            array = source[source[:, 0] < self.max_beats * TIME_STEPS_PER_BEAT].copy()
        else:
            n_beats = int(np.max(source[:, 0]) // TIME_STEPS_PER_BEAT)
            if n_beats < self.max_beats + 1:
                start_time = 0
            else:
                start_time = int(np.random.randint(n_beats - self.max_beats) * TIME_STEPS_PER_BEAT)
            end_time = start_time + self.max_beats * TIME_STEPS_PER_BEAT
            array = source[(source[:, 0] >= start_time) & (source[:, 0] < end_time)].copy()
            if array.size:
                array[:, 0] -= start_time
                shift = int(np.random.randint(self.transpose_low, self.transpose_high + 1))
                array[:, 1] += shift
                array[:, 1][array[:, 1] > 127] -= 12
                array[:, 1][array[:, 1] < 0] += 12

        array = array[: self.max_seq_len].copy()
        if array.size:
            array[:, 1] = np.clip(array[:, 1], 0, 127)
            array[:, 2] = np.clip(array[:, 2], 0, MAX_DURATION_STEPS)
        inputs = torch.as_tensor(array[:, :3], dtype=torch.long)
        labels = torch.as_tensor(array[:, 3], dtype=torch.long)
        return inputs, labels


def make_pad_collate(padding_label: int = PAD_LABEL) -> Callable:
    def pad_collate(batch: list[tuple[torch.Tensor, torch.Tensor]]) -> tuple[torch.Tensor, torch.Tensor]:
        samples = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        return (
            pad_sequence(samples, batch_first=True, padding_value=0),
            pad_sequence(labels, batch_first=True, padding_value=padding_label),
        )

    return pad_collate


def build_dataloader(
    files: list[Path],
    batch_size: int,
    max_beats: int = 32,
    max_seq_len: int = 1024,
    augmentation: bool = False,
    shuffle: bool = False,
    num_workers: int = 4,
    preload: bool = True,
    use_packed: bool = False,
    processed_dir: Path | str | None = None,
    split_name: str | None = None,
) -> DataLoader:
    if use_packed and processed_dir is not None and split_name is not None and has_packed_split(processed_dir, split_name):
        paths = packed_split_paths(processed_dir, split_name)
        dataset: Dataset = PackedMusicEventDataset(
            paths.events,
            paths.offsets,
            max_beats=max_beats,
            max_seq_len=max_seq_len,
            augmentation=augmentation,
        )
    else:
        dataset = MusicEventDataset(
            files,
            max_beats=max_beats,
            max_seq_len=max_seq_len,
            augmentation=augmentation,
            preload=preload,
        )
    loader_kwargs = {}
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=make_pad_collate(),
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
        **loader_kwargs,
    )
