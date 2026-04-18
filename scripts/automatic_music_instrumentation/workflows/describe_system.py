from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import INSTRUMENT_LABELS, summarize_dataset
from scripts.automatic_music_instrumentation.core.models import MODEL_SPECS


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the automatic instrumentation system summary.")
    parser.add_argument("--data-root", default=None)
    args = parser.parse_args()
    summary = summarize_dataset(args.data_root, count_clean_midi=False)
    print("Automatic music instrumentation pipeline")
    print("1. Data rows are [onset_time, pitch, duration, instrument_label].")
    print(f"2. Instrument labels are: {dict(enumerate(INSTRUMENT_LABELS))}.")
    print("3. Rule baseline: fixed pitch-zone assignment.")
    print("4. Main model: encoder-only Transformer with pitch/duration/beat/position embeddings.")
    print("5. Training: random 32-beat crops, random transposition -5..+6 semitones, CrossEntropy ignore padding, Adam, linear warmup.")
    print("6. Evaluation: test loss, note-level accuracy, raw and row-normalized confusion matrices.")
    print("7. Inference: predict labels for each note, then write notes back into piano/guitar/bass/strings/brass tracks.")
    print("")
    print("Available trainable models:")
    for spec in MODEL_SPECS.values():
        mode = "online" if spec.online else "offline"
        print(f"- {spec.name} ({mode}): {spec.description}")
    print("")
    print("Downloaded data summary:")
    print(f"- processed_dir: {summary.processed_dir}")
    print(f"- raw_archive_path: {summary.raw_archive_path}")
    print(f"- clean_midi_dir: {summary.clean_midi_dir}")
    print(f"- sample_file_count: {summary.sample_file_count}")
    print(f"- train_count: {summary.train_count}")
    print(f"- valid_count: {summary.valid_count}")
    print(f"- test_count: {summary.test_count}")
    if summary.clean_midi_count is not None:
        print(f"- clean_midi_count: {summary.clean_midi_count}")


if __name__ == "__main__":
    main()
