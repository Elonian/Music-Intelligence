#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.spectrogram_classification import (
    ExperimentConfig,
    discover_audio_examples,
    run_experiment_suite,
)
from utils import SPECTROGRAM_DATA_ROOT, SPECTROGRAM_METRICS_DIR, ensure_dir


def _default_configs(max_examples_per_class: int | None, seed: int) -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            output_name="mfcc_mlp",
            feature_kind="mfcc_vector",
            model_kind="mlp",
            num_classes=2,
            max_examples_per_class=max_examples_per_class,
            augment_pitch_shift=False,
            random_seed=seed,
        ),
        ExperimentConfig(
            output_name="spectrogram_cnn",
            feature_kind="spectrogram",
            model_kind="cnn",
            num_classes=2,
            max_examples_per_class=max_examples_per_class,
            augment_pitch_shift=False,
            random_seed=seed,
        ),
        ExperimentConfig(
            output_name="mel_spectrogram_cnn",
            feature_kind="mel_spectrogram",
            model_kind="cnn",
            num_classes=2,
            max_examples_per_class=max_examples_per_class,
            augment_pitch_shift=False,
            random_seed=seed,
        ),
        ExperimentConfig(
            output_name="cqt_cnn",
            feature_kind="cqt",
            model_kind="cnn",
            num_classes=2,
            max_examples_per_class=max_examples_per_class,
            augment_pitch_shift=False,
            random_seed=seed,
        ),
        ExperimentConfig(
            output_name="mel_spectrogram_four_class",
            feature_kind="mel_spectrogram",
            model_kind="cnn",
            label_mode="family",
            num_classes=4,
            max_examples_per_class=max_examples_per_class,
            augment_pitch_shift=True,
            random_seed=seed,
            num_epochs=24,
            learning_rate=8e-4,
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Train spectrogram-based classification experiments.")
    parser.add_argument("--data-dir", type=Path, default=SPECTROGRAM_DATA_ROOT)
    parser.add_argument("--max-examples-per-class", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    examples = discover_audio_examples(args.data_dir)
    if not examples:
        raise FileNotFoundError(
            f"No audio files found under {args.data_dir}. "
            "Populate data/spectrogram_classification before running the training pipeline."
        )

    ensure_dir(SPECTROGRAM_METRICS_DIR)
    configs = _default_configs(args.max_examples_per_class, args.seed)
    runnable_configs: list[ExperimentConfig] = []
    label_count = len({example.label for example in examples})
    for config in configs:
        if config.num_classes <= label_count:
            runnable_configs.append(config)

    artifacts = run_experiment_suite(examples, runnable_configs)
    print(f"[Spectrogram Train] Wrote {artifacts['summary_path']}")


if __name__ == "__main__":
    main()
