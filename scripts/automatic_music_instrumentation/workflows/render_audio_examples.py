#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import INSTRUMENT_LABELS  # noqa: E402
from scripts.automatic_music_instrumentation.core.pitch_zones import pitch_zone_predict_events  # noqa: E402
from scripts.automatic_music_instrumentation.core.training import load_checkpoint_model, select_device  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR  # noqa: E402


PROGRAM_LABEL_MAP = {
    0: 0,
    24: 1,
    32: 2,
    48: 3,
    61: 4,
}
DEFAULT_SOUNDFONT = Path("/usr/share/sounds/sf2/FluidR3_GM.sf2")


def find_sample_file(sample_index: int) -> Path:
    sample_dir = AUTOMATIC_INSTRUMENTATION_PROCESSED_DIR / "samples" / "converted"
    files = sorted(sample_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No sample JSON files found in {sample_dir}.")
    return files[sample_index % len(files)]


def trim_music(music, max_beats: int):
    output = music.deepcopy()
    output.trim(output.resolution * max_beats)
    return output


def all_non_drum_notes(music) -> list:
    notes = []
    for track in music.tracks[: len(INSTRUMENT_LABELS)]:
        if getattr(track, "is_drum", False):
            continue
        notes.extend(track.notes)
    notes.sort(key=lambda note: (note.time, note.pitch, note.duration))
    return notes


def make_pitch_zone_music(music):
    output = music.deepcopy()
    grouped = {index: [] for index in range(len(INSTRUMENT_LABELS))}
    notes = all_non_drum_notes(music)
    if notes:
        events = np.asarray([[note.time, note.pitch, min(note.duration, 192), 0] for note in notes], dtype=np.int64)
        labels = pitch_zone_predict_events(events)
        for note, label in zip(notes, labels):
            grouped[int(label)].append(note)
    for index in range(min(len(output.tracks), len(INSTRUMENT_LABELS))):
        output.tracks[index].notes = grouped[index]
    return output


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


def make_model_music(music, checkpoint: Path, model_name: str | None = None):
    device = select_device(False)
    model = load_checkpoint_model(checkpoint, model_name=model_name, device=device)
    array, notes = preprocess_music(music)
    if len(array) == 0:
        raise ValueError("No supported non-drum notes found for model inference.")
    inputs = torch.as_tensor(array[np.newaxis, :, :3], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(inputs)
    labels = torch.argmax(logits[0], dim=-1).detach().cpu().numpy()
    output = music.deepcopy()
    grouped = {index: [] for index in range(len(INSTRUMENT_LABELS))}
    for note, label in zip(notes, labels):
        grouped[int(label)].append(note)
    for index in range(min(len(output.tracks), len(INSTRUMENT_LABELS))):
        output.tracks[index].notes = grouped[index]
    return output


def write_music_outputs(name: str, music, output_dir: Path, soundfont: Path | None, write_json: bool = True) -> dict:
    ensure_dir(output_dir)
    wav_path = output_dir / f"{name}.wav"
    kwargs = {"soundfont_path": soundfont} if soundfont is not None and soundfont.exists() else {}
    music.write_audio(wav_path, **kwargs)
    payload = {"wav": str(wav_path)}
    if write_json:
        json_path = output_dir / f"{name}.json"
        music.save(json_path)
        payload["json"] = str(json_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Render WAV audio examples for automatic instrumentation.")
    parser.add_argument("--input", type=Path, default=None, help="MusPy JSON sample. Defaults to a downloaded converted sample.")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-beats", type=int, default=32)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--soundfont", type=Path, default=DEFAULT_SOUNDFONT)
    args = parser.parse_args()

    muspy = __import__("muspy")
    sample_path = args.input or find_sample_file(args.sample_index)
    source_music = trim_music(muspy.load(sample_path), max_beats=args.max_beats)
    output_dir = ensure_dir(args.output_dir)

    outputs = {
        "source": str(sample_path),
        "max_beats": args.max_beats,
        "ground_truth": write_music_outputs("ground_truth", source_music, output_dir, args.soundfont),
        "pitch_zones": write_music_outputs("pitch_zones", make_pitch_zone_music(source_music), output_dir, args.soundfont),
    }
    if args.checkpoint is not None:
        outputs["model_inference"] = write_music_outputs(
            "model_inference",
            make_model_music(source_music, args.checkpoint, model_name=args.model),
            output_dir,
            args.soundfont,
        )
    save_json(output_dir / "audio_manifest.json", outputs)
    print(outputs)


if __name__ == "__main__":
    main()
