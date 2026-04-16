from __future__ import annotations

import numpy as np
import torch
import torchaudio

import librosa


def _safe_log_scale(feature_map: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return torch.log(feature_map.clamp_min(eps))


def _normalize_feature_map(feature_map: torch.Tensor) -> torch.Tensor:
    mean = feature_map.mean()
    std = feature_map.std()
    if float(std) <= 1e-8:
        return feature_map - mean
    return (feature_map - mean) / std


def compute_mfcc_map(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 40,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 64,
) -> torch.Tensor:
    transform = torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs={
            "n_fft": n_fft,
            "hop_length": hop_length,
            "n_mels": n_mels,
            "center": True,
            "power": 2.0,
        },
    )
    return transform(waveform).squeeze(0).to(torch.float32)


def compute_mfcc_feature_vector(
    waveform: torch.Tensor,
    sample_rate: int,
    n_mfcc: int = 40,
) -> torch.Tensor:
    mfcc_map = compute_mfcc_map(waveform, sample_rate, n_mfcc=n_mfcc)
    mean = mfcc_map.mean(dim=1)
    std = mfcc_map.std(dim=1)
    return torch.cat([mean, std], dim=0)


def compute_linear_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
) -> torch.Tensor:
    del sample_rate
    transform = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0,
    )
    spectrogram = transform(waveform).squeeze(0).to(torch.float32)
    return _normalize_feature_map(_safe_log_scale(spectrogram))


def compute_mel_spectrogram(
    waveform: torch.Tensor,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 128,
) -> torch.Tensor:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        power=2.0,
    )
    spectrogram = transform(waveform).squeeze(0).to(torch.float32)
    return _normalize_feature_map(_safe_log_scale(spectrogram))


def compute_cqt(
    waveform: torch.Tensor,
    sample_rate: int,
    bins_per_octave: int = 12,
    n_bins: int = 84,
    hop_length: int = 256,
) -> torch.Tensor:
    waveform_np = waveform.squeeze(0).detach().cpu().numpy()
    cqt = librosa.cqt(
        waveform_np,
        sr=sample_rate,
        hop_length=hop_length,
        bins_per_octave=bins_per_octave,
        n_bins=n_bins,
    )
    magnitude = np.abs(cqt).astype(np.float32)
    cqt_tensor = torch.from_numpy(np.log(np.maximum(magnitude, 1e-6)))
    return _normalize_feature_map(cqt_tensor)


def pitch_shift_waveform(
    waveform: torch.Tensor,
    sample_rate: int,
    semitone_steps: float,
) -> torch.Tensor:
    transform = torchaudio.transforms.PitchShift(sample_rate=sample_rate, n_steps=semitone_steps)
    shifted = transform(waveform)
    peak = float(shifted.abs().max()) if shifted.numel() else 0.0
    if peak > 1e-8:
        shifted = shifted / peak
    return shifted


def build_feature_bundle(
    waveform: torch.Tensor,
    sample_rate: int,
) -> dict[str, torch.Tensor]:
    return {
        "mfcc_vector": compute_mfcc_feature_vector(waveform, sample_rate),
        "mfcc_map": compute_mfcc_map(waveform, sample_rate),
        "spectrogram": compute_linear_spectrogram(waveform, sample_rate),
        "mel_spectrogram": compute_mel_spectrogram(waveform, sample_rate),
        "cqt": compute_cqt(waveform, sample_rate),
    }
