from __future__ import annotations

from scripts.spectrogram_classification.data import (
    AudioExample,
    discover_audio_examples,
    fit_waveform_length,
    load_audio_waveform,
    split_audio_examples,
)
from scripts.spectrogram_classification.features import (
    build_feature_bundle,
    compute_cqt,
    compute_linear_spectrogram,
    compute_mel_spectrogram,
    compute_mfcc_feature_vector,
    compute_mfcc_map,
    pitch_shift_waveform,
)
from scripts.spectrogram_classification.models import MlpAudioClassifier, SpectrogramCnnClassifier
from scripts.spectrogram_classification.training import (
    AudioFeatureDataset,
    ExperimentConfig,
    run_audio_experiment,
    run_experiment_suite,
    set_random_seed,
)

__all__ = [
    "AudioExample",
    "AudioFeatureDataset",
    "ExperimentConfig",
    "MlpAudioClassifier",
    "SpectrogramCnnClassifier",
    "build_feature_bundle",
    "compute_cqt",
    "compute_linear_spectrogram",
    "compute_mel_spectrogram",
    "compute_mfcc_feature_vector",
    "compute_mfcc_map",
    "discover_audio_examples",
    "fit_waveform_length",
    "load_audio_waveform",
    "pitch_shift_waveform",
    "run_audio_experiment",
    "run_experiment_suite",
    "set_random_seed",
    "split_audio_examples",
]
