from __future__ import annotations

import tarfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import torch
from sklearn.model_selection import train_test_split


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}


@dataclass(frozen=True)
class AudioExample:
    file_path: str
    instrument_label: str
    family_label: str

    @property
    def label(self) -> str:
        return self.instrument_label

    def label_for_mode(self, label_mode: str) -> str:
        if label_mode == "binary":
            return self.instrument_label
        if label_mode == "family":
            return self.family_label
        raise ValueError(f"Unsupported label mode: {label_mode}")


def resolve_audio_root(data_dir: Path) -> Path:
    data_dir = Path(data_dir)
    direct_candidates = [
        data_dir,
        data_dir / "nsynth_subset",
    ]
    for candidate in direct_candidates:
        if candidate.exists() and any(path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS for path in candidate.rglob("*")):
            return candidate

    for archive_path in sorted(data_dir.rglob("nsynth_subset.tar.gz")):
        if archive_path.exists():
            extract_root = archive_path.parent
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(extract_root)
            extracted_dir = extract_root / "nsynth_subset"
            if extracted_dir.exists():
                return extracted_dir
            return extract_root

    return data_dir


def _parse_nsynth_labels(path: Path) -> tuple[str, str]:
    name_parts = path.stem.split("_")
    instrument_label = name_parts[0] if name_parts else path.stem
    family_label = "_".join(name_parts[:2]) if len(name_parts) >= 2 else instrument_label
    return instrument_label, family_label


def discover_audio_examples(
    data_dir: Path,
    class_names: list[str] | None = None,
    label_mode: str = "binary",
) -> list[AudioExample]:
    data_dir = resolve_audio_root(Path(data_dir))
    if not data_dir.exists():
        return []

    allowed_classes = set(class_names) if class_names else None
    examples: list[AudioExample] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        instrument_label, family_label = _parse_nsynth_labels(path)
        example = AudioExample(str(path), instrument_label=instrument_label, family_label=family_label)
        if allowed_classes is not None and example.label_for_mode(label_mode) not in allowed_classes:
            continue
        examples.append(example)
    return examples


def load_audio_waveform(
    file_path: str | Path,
    target_sample_rate: int = 16000,
    mono: bool = True,
) -> tuple[torch.Tensor, int]:
    waveform_np, sample_rate = librosa.load(
        str(file_path),
        sr=target_sample_rate,
        mono=mono,
    )
    waveform = torch.from_numpy(waveform_np).to(torch.float32).unsqueeze(0)
    peak = float(waveform.abs().max()) if waveform.numel() else 0.0
    if peak > 1e-8:
        waveform = waveform / peak
    return waveform, sample_rate


def fit_waveform_length(waveform: torch.Tensor, target_num_samples: int) -> torch.Tensor:
    waveform = waveform.to(torch.float32)
    current_num_samples = waveform.shape[-1]
    if current_num_samples == target_num_samples:
        return waveform
    if current_num_samples > target_num_samples:
        return waveform[..., :target_num_samples]

    padded = torch.zeros((*waveform.shape[:-1], target_num_samples), dtype=waveform.dtype)
    padded[..., :current_num_samples] = waveform
    return padded


def split_audio_examples(
    examples: list[AudioExample],
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 0.15,
    label_mode: str = "binary",
) -> tuple[list[AudioExample], list[AudioExample], list[AudioExample]]:
    if not examples:
        return [], [], []

    labels = [example.label_for_mode(label_mode) for example in examples]
    train_examples, temp_examples = train_test_split(
        examples,
        test_size=1.0 - train_fraction,
        random_state=seed,
        stratify=labels,
    )
    if not temp_examples:
        return train_examples, [], []

    temp_labels = [example.label_for_mode(label_mode) for example in temp_examples]
    val_ratio_in_temp = val_fraction / max(1.0 - train_fraction, 1e-8)
    val_examples, test_examples = train_test_split(
        temp_examples,
        test_size=max(0.0, 1.0 - val_ratio_in_temp),
        random_state=seed,
        stratify=temp_labels,
    )
    return train_examples, val_examples, test_examples
