from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import INSTRUMENT_LABELS
from scripts.automatic_music_instrumentation.core.training import load_checkpoint_model, select_device


PROGRAM_LABEL_MAP = {
    0: 0,
    24: 1,
    32: 2,
    48: 3,
    61: 4,
}


def preprocess_music(music) -> tuple[np.ndarray, list]:
    rows: list[list[int]] = []
    notes: list = []
    for track in music:
        if getattr(track, "is_drum", False):
            continue
        if track.program not in PROGRAM_LABEL_MAP:
            continue
        for note in track:
            rows.append([note.time, note.pitch, min(note.duration, 192), PROGRAM_LABEL_MAP[track.program]])
            notes.append(note)
    if not rows:
        return np.zeros((0, 4), dtype=np.int64), []
    array = np.asarray(rows, dtype=np.int64)
    order = np.argsort(array[:, 0], kind="stable")
    return array[order], [notes[index] for index in order]


def assign_notes_to_tracks(music, notes: list, labels: np.ndarray):
    output = music.deepcopy()
    grouped = {index: [] for index in range(len(INSTRUMENT_LABELS))}
    for note, label in zip(notes, labels):
        grouped[int(label)].append(note)
    for index in range(min(len(output.tracks), len(INSTRUMENT_LABELS))):
        output.tracks[index].notes = grouped[index]
    return output


def save_pianoroll(music, path: Path) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    muspy = __import__("muspy")
    fig, ax = plt.subplots(figsize=(12, 3))
    muspy.visualization.show_pianoroll(music, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run automatic instrumentation inference on a MusPy-readable file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--input", type=Path, required=True, help="MusPy JSON/MIDI input.")
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--write-audio", action="store_true")
    args = parser.parse_args()

    muspy = __import__("muspy")
    device = select_device(False)
    model = load_checkpoint_model(args.checkpoint, model_name=args.model, device=device)
    music = muspy.load(args.input)
    array, notes = preprocess_music(music)
    if len(array) == 0:
        raise ValueError("No supported non-drum notes found. Expected programs 0, 24, 32, 48, or 61.")
    inputs = torch.as_tensor(array[np.newaxis, :, :3], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inputs)
    labels = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()
    output_music = assign_notes_to_tracks(music, notes, labels)
    output_json = args.out_prefix.with_suffix(".json")
    output_music.save(output_json)
    save_pianoroll(output_music, args.out_prefix.with_suffix(".png"))
    if args.write_audio:
        output_music.write_audio(str(args.out_prefix.with_suffix(".wav")))
    print({"output_json": str(output_json), "labels": dict(zip(INSTRUMENT_LABELS, np.bincount(labels, minlength=len(INSTRUMENT_LABELS)).tolist()))})


if __name__ == "__main__":
    main()
