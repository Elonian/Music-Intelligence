from __future__ import annotations

import json
import math
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from scripts.multitrack_generation.constants import (
    EVENT_TYPE_LABELS,
    INSTRUMENT_LABELS,
    INSTRUMENT_PROGRAMS,
    TIME_STEPS_PER_BEAT,
    TYPE_END_SONG,
    TYPE_INSTRUMENT,
    TYPE_NOTE,
    TYPE_PAD,
    TYPE_START_NOTES,
    TYPE_START_SONG,
)
from scripts.multitrack_generation.events import (
    instrument_prompt,
    save_note_csv,
    save_sequence_csv,
    sequence_to_note_array,
    twinkle_prompt,
)
from utils.io_helpers import ensure_dir, save_json
from utils.project_paths import MULTITRACK_GENERATION_GENERATED_DIR


@dataclass
class GenerationConfig:
    decoding: str = "topk"
    temperature: float = 0.9
    top_k: int = 8
    top_p: float = 0.9
    max_seq_len: int = 256
    seed: int = 42
    prevent_pad: bool = True
    enforce_event_order: bool = True
    enforce_monotonic_beats: bool = True
    sample_valid_beat_logits: bool = True
    enforce_positive_duration: bool = True
    restrict_note_instruments_to_prompt: bool = True
    avoid_duplicate_notes: bool = True
    duplicate_retry_count: int = 8


def prompt_by_name(name: str) -> np.ndarray:
    if name == "empty":
        return np.asarray([[TYPE_START_SONG, 0, 0, 0, 0, 0]], dtype=np.int64)
    if name == "piano_guitar_bass":
        return instrument_prompt([0, 1, 2])
    if name == "all_instruments":
        return instrument_prompt([0, 1, 2, 3, 4])
    if name == "twinkle":
        return twinkle_prompt()[:-1]
    raise ValueError("unknown prompt; expected empty, piano_guitar_bass, all_instruments, or twinkle")


def _sample_logits(logits: torch.Tensor, config: GenerationConfig) -> int:
    temperature = max(float(config.temperature), 1e-6)
    if config.decoding == "greedy":
        return int(torch.argmax(logits).item())
    scaled = logits / temperature
    if config.decoding == "random":
        probs = torch.softmax(scaled, dim=-1)
        return int(torch.multinomial(probs, 1).item())
    if config.decoding == "topk":
        k = max(1, min(int(config.top_k), int(scaled.numel())))
        values, indices = torch.topk(scaled, k)
        probs = torch.softmax(values, dim=-1)
        return int(indices[torch.multinomial(probs, 1)].item())
    if config.decoding == "topp":
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(config.top_p)
        keep[0] = True
        filtered_probs = sorted_probs[keep]
        filtered_indices = sorted_indices[keep]
        filtered_probs = filtered_probs / filtered_probs.sum()
        return int(filtered_indices[torch.multinomial(filtered_probs, 1)].item())
    raise ValueError("decoding must be greedy, random, topk, or topp")


def _has_start_notes(sequence: list[list[int]]) -> bool:
    return any(event[0] == TYPE_START_NOTES for event in sequence)


def _last_note_beat(sequence: list[list[int]]) -> int:
    beats = [event[1] for event in sequence if event[0] == TYPE_NOTE]
    return max(beats) if beats else 0


def _declared_instruments(sequence: list[list[int]]) -> list[int]:
    instruments: list[int] = []
    for event in sequence:
        event_type = int(event[0])
        if event_type == TYPE_INSTRUMENT:
            instrument = int(event[5])
            if instrument not in instruments:
                instruments.append(instrument)
        elif event_type >= TYPE_START_NOTES:
            break
    return instruments


def _seen_note_events(sequence: list[list[int]]) -> set[tuple[int, int, int, int]]:
    seen: set[tuple[int, int, int, int]] = set()
    for event in sequence:
        if int(event[0]) != TYPE_NOTE:
            continue
        onset = int(event[1]) * TIME_STEPS_PER_BEAT + int(event[2])
        seen.add((onset, int(event[3]), max(1, int(event[4])), int(event[5])))
    return seen


def _constrain_type_logits(logits: torch.Tensor, sequence: list[list[int]], config: GenerationConfig) -> torch.Tensor:
    constrained = logits.clone()
    if config.prevent_pad:
        constrained[TYPE_PAD] = -torch.inf
    constrained[TYPE_START_SONG] = -torch.inf
    if not config.enforce_event_order:
        return constrained

    has_son = _has_start_notes(sequence)
    if has_son:
        constrained[TYPE_INSTRUMENT] = -torch.inf
        constrained[TYPE_START_NOTES] = -torch.inf
    else:
        constrained[TYPE_NOTE] = -torch.inf
        if len(sequence) <= 1:
            constrained[TYPE_END_SONG] = -torch.inf
    return constrained


def _safe_sample_logits(logits: torch.Tensor, config: GenerationConfig, fallback: int = 0) -> int:
    if torch.isfinite(logits).any():
        return _sample_logits(logits, config)
    return int(fallback)


@torch.no_grad()
def generate_sequence(
    model: nn.Module,
    prompt: np.ndarray | None = None,
    config: GenerationConfig | None = None,
    device: torch.device | str | None = None,
) -> np.ndarray:
    config = config or GenerationConfig()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    resolved_device = torch.device(device or next(model.parameters()).device)
    model = model.to(resolved_device)
    model.eval()
    sequence = (prompt if prompt is not None else prompt_by_name("empty")).astype(np.int64).tolist()

    while len(sequence) < config.max_seq_len:
        inputs = torch.as_tensor(sequence, dtype=torch.long, device=resolved_device).unsqueeze(0)
        outputs = model(inputs)
        logits = [head[0, -1, :].detach() for head in outputs]
        event_type = _sample_logits(_constrain_type_logits(logits[0], sequence, config), config)

        if event_type == TYPE_END_SONG:
            sequence.append([TYPE_END_SONG, 0, 0, 0, 0, 0])
            break
        if event_type == TYPE_INSTRUMENT:
            instrument = _sample_logits(logits[5], config)
            sequence.append([TYPE_INSTRUMENT, 0, 0, 0, 0, int(instrument)])
            continue
        if event_type == TYPE_START_NOTES:
            sequence.append([TYPE_START_NOTES, 0, 0, 0, 0, 0])
            continue

        beat_logits = logits[1].clone()
        last_beat = _last_note_beat(sequence)
        if config.enforce_monotonic_beats and config.sample_valid_beat_logits:
            beat_logits[:last_beat] = -torch.inf

        duration_logits = logits[4].clone()
        if config.enforce_positive_duration:
            duration_logits[0] = -torch.inf

        instrument_logits = logits[5].clone()
        declared_instruments = _declared_instruments(sequence)
        if config.restrict_note_instruments_to_prompt and declared_instruments:
            allowed = torch.full_like(instrument_logits, -torch.inf)
            for instrument in declared_instruments:
                if 0 <= instrument < allowed.numel():
                    allowed[instrument] = instrument_logits[instrument]
            instrument_logits = allowed

        seen_notes = _seen_note_events(sequence)
        event: list[int] | None = None
        retries = max(1, int(config.duplicate_retry_count) if config.avoid_duplicate_notes else 1)
        for _ in range(retries):
            beat = _safe_sample_logits(beat_logits, config, fallback=last_beat)
            if config.enforce_monotonic_beats and not config.sample_valid_beat_logits:
                beat = max(beat, last_beat)
            position = int(_sample_logits(logits[2], config))
            pitch = int(_sample_logits(logits[3], config))
            duration = int(_safe_sample_logits(duration_logits, config, fallback=1))
            instrument = int(_safe_sample_logits(instrument_logits, config, fallback=declared_instruments[0] if declared_instruments else 0))
            candidate = [
                TYPE_NOTE,
                int(min(beat, 63)),
                position,
                pitch,
                duration,
                instrument,
            ]
            note_key = (candidate[1] * TIME_STEPS_PER_BEAT + candidate[2], candidate[3], max(1, candidate[4]), candidate[5])
            event = candidate
            if not config.avoid_duplicate_notes or note_key not in seen_notes:
                break
        sequence.append(event)

    if sequence[-1][0] != TYPE_END_SONG:
        sequence.append([TYPE_END_SONG, 0, 0, 0, 0, 0])
    return np.asarray(sequence, dtype=np.int64)


def _varlen(value: int) -> bytes:
    value = max(0, int(value))
    chunks = [value & 0x7F]
    value >>= 7
    while value:
        chunks.insert(0, (value & 0x7F) | 0x80)
        value >>= 7
    return bytes(chunks)


def _midi_track(events: list[tuple[int, bytes]]) -> bytes:
    events = sorted(events, key=lambda item: (item[0], item[1][0] if item[1] else 0))
    payload = bytearray()
    last_tick = 0
    for tick, data in events:
        tick = max(int(tick), last_tick)
        payload.extend(_varlen(tick - last_tick))
        payload.extend(data)
        last_tick = tick
    payload.extend(b"\x00\xff\x2f\x00")
    return b"MTrk" + len(payload).to_bytes(4, "big") + bytes(payload)


def save_midi(notes: np.ndarray, output_path: Path, tempo_bpm: int = 120, ticks_per_quarter: int = 480) -> None:
    ensure_dir(output_path.parent)
    ticks_per_step = ticks_per_quarter // TIME_STEPS_PER_BEAT
    microseconds_per_quarter = int(60_000_000 / max(tempo_bpm, 1))
    meta_events = [
        (0, b"\xff\x51\x03" + microseconds_per_quarter.to_bytes(3, "big")),
        (0, b"\xff\x58\x04\x04\x02\x18\x08"),
    ]
    tracks = [_midi_track(meta_events)]

    for instrument_index, label in enumerate(INSTRUMENT_LABELS):
        channel = instrument_index
        program = INSTRUMENT_PROGRAMS[instrument_index]
        track_events: list[tuple[int, bytes]] = [
            (0, b"\xff\x03" + bytes([len(label)]) + label.encode("ascii")),
            (0, bytes([0xC0 | channel, program])),
        ]
        instrument_notes = notes[notes[:, 3] == instrument_index] if notes.size else np.empty((0, 4), dtype=np.int64)
        for onset, pitch, duration, _instrument in instrument_notes:
            start = int(onset) * ticks_per_step
            end = start + max(1, int(duration)) * ticks_per_step
            velocity = 76 if instrument_index != 2 else 90
            track_events.append((start, bytes([0x90 | channel, int(pitch), velocity])))
            track_events.append((end, bytes([0x80 | channel, int(pitch), 0])))
        tracks.append(_midi_track(track_events))

    header = b"MThd" + (6).to_bytes(4, "big") + (1).to_bytes(2, "big") + len(tracks).to_bytes(2, "big") + ticks_per_quarter.to_bytes(2, "big")
    output_path.write_bytes(header + b"".join(tracks))


def _instrument_harmonics(instrument: int) -> list[tuple[float, float]]:
    if instrument == 0:
        return [(1.0, 1.0), (2.0, 0.28), (3.0, 0.12), (4.0, 0.04)]
    if instrument == 1:
        return [(1.0, 1.0), (2.0, 0.46), (3.0, 0.24), (4.0, 0.14), (5.0, 0.08)]
    if instrument == 2:
        return [(0.5, 0.38), (1.0, 1.0), (2.0, 0.22), (3.0, 0.08)]
    if instrument == 3:
        return [(1.0, 0.95), (2.0, 0.22), (3.0, 0.13), (4.0, 0.04)]
    return [(1.0, 1.0), (2.0, 0.48), (3.0, 0.26), (5.0, 0.12)]


def _instrument_synth_params(instrument: int) -> dict[str, float]:
    params = [
        {"gain": 0.070, "attack": 0.006, "decay": 0.100, "sustain": 0.45, "release": 0.180, "pan": -0.28, "pluck": 0.55},
        {"gain": 0.060, "attack": 0.004, "decay": 0.080, "sustain": 0.34, "release": 0.120, "pan": 0.24, "pluck": 0.80},
        {"gain": 0.090, "attack": 0.010, "decay": 0.090, "sustain": 0.62, "release": 0.100, "pan": 0.02, "pluck": 0.36},
        {"gain": 0.052, "attack": 0.090, "decay": 0.220, "sustain": 0.72, "release": 0.340, "pan": -0.48, "pluck": 0.12},
        {"gain": 0.055, "attack": 0.030, "decay": 0.130, "sustain": 0.58, "release": 0.210, "pan": 0.46, "pluck": 0.30},
    ]
    return params[max(0, min(int(instrument), len(params) - 1))]


def _adsr_envelope(length: int, sample_rate: int, params: dict[str, float]) -> np.ndarray:
    if length <= 0:
        return np.zeros(0, dtype=np.float32)
    attack = min(length, max(1, int(params["attack"] * sample_rate)))
    decay = min(max(0, length - attack), max(1, int(params["decay"] * sample_rate)))
    release = min(max(0, length - attack - decay), max(1, int(params["release"] * sample_rate)))
    sustain_len = max(0, length - attack - decay - release)
    envelope = np.empty(length, dtype=np.float32)
    cursor = 0
    envelope[cursor : cursor + attack] = np.linspace(0.0, 1.0, attack, dtype=np.float32)
    cursor += attack
    if decay:
        envelope[cursor : cursor + decay] = np.linspace(1.0, params["sustain"], decay, dtype=np.float32)
        cursor += decay
    if sustain_len:
        envelope[cursor : cursor + sustain_len] = params["sustain"]
        cursor += sustain_len
    if release:
        start = envelope[cursor - 1] if cursor else params["sustain"]
        envelope[cursor : cursor + release] = np.linspace(start, 0.0, release, dtype=np.float32)
        cursor += release
    if cursor < length:
        envelope[cursor:] = 0.0
    return envelope


def save_audio_preview(
    notes: np.ndarray,
    output_path: Path,
    tempo_bpm: int = 120,
    sample_rate: int = 44100,
) -> None:
    ensure_dir(output_path.parent)
    if notes.size == 0:
        audio = np.zeros((sample_rate, 2), dtype=np.float32)
    else:
        seconds_per_step = 60.0 / float(tempo_bpm) / TIME_STEPS_PER_BEAT
        end_seconds = float(np.max(notes[:, 0] + notes[:, 2]) * seconds_per_step + 2.0)
        audio = np.zeros((max(1, int(end_seconds * sample_rate)), 2), dtype=np.float32)
        for onset, pitch, duration, instrument in notes:
            start = int(onset * seconds_per_step * sample_rate)
            length = max(1, int(duration * seconds_per_step * sample_rate))
            stop = min(audio.shape[0], start + length)
            if start >= audio.shape[0] or stop <= start:
                continue
            t = np.arange(stop - start, dtype=np.float32) / float(sample_rate)
            frequency = 440.0 * (2.0 ** ((float(pitch) - 69.0) / 12.0))
            params = _instrument_synth_params(int(instrument))
            tone = np.zeros_like(t)
            for harmonic, gain in _instrument_harmonics(int(instrument)):
                phase = 0.13 * int(instrument) * harmonic
                tone += gain * np.sin(2.0 * math.pi * frequency * harmonic * t + phase)
            if int(instrument) in {1, 4}:
                tone += 0.12 * np.tanh(3.0 * np.sin(2.0 * math.pi * frequency * t))
            if int(instrument) == 3:
                vibrato = 1.0 + 0.0035 * np.sin(2.0 * math.pi * 5.2 * t)
                tone = np.zeros_like(t)
                for harmonic, gain in _instrument_harmonics(int(instrument)):
                    tone += gain * np.sin(2.0 * math.pi * frequency * harmonic * vibrato * t)
            envelope = _adsr_envelope(tone.size, sample_rate, params)
            pluck = np.exp(-params["pluck"] * np.linspace(0.0, 5.0, tone.size, dtype=np.float32))
            tone = tone * envelope * (0.72 + 0.28 * pluck) * params["gain"]
            pan = max(-1.0, min(1.0, params["pan"]))
            left_gain = math.sqrt(0.5 * (1.0 - pan))
            right_gain = math.sqrt(0.5 * (1.0 + pan))
            audio[start:stop, 0] += tone * left_gain
            audio[start:stop, 1] += tone * right_gain

        for delay_seconds, gain, cross in ((0.115, 0.18, False), (0.235, 0.12, True), (0.365, 0.07, True)):
            delay = int(delay_seconds * sample_rate)
            if delay <= 0 or delay >= audio.shape[0]:
                continue
            delayed = audio[:-delay].copy()
            if cross:
                delayed = delayed[:, ::-1]
            audio[delay:] += gain * delayed
        audio = np.tanh(1.35 * audio)
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio = audio / peak * 0.85

    pcm = np.clip(audio, -1.0, 1.0)
    pcm_i16 = (pcm * 32767).astype("<i2")
    with wave.open(str(output_path), "wb") as handle:
        handle.setnchannels(2)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm_i16.tobytes())


def save_generation_bundle(
    sequence: np.ndarray,
    output_dir: Path | None = None,
    name: str = "sample",
    config: GenerationConfig | None = None,
    tempo_bpm: int = 120,
) -> dict:
    root = ensure_dir((output_dir or MULTITRACK_GENERATION_GENERATED_DIR) / name)
    notes = sequence_to_note_array(sequence)
    np.save(root / "sequence.npy", sequence)
    np.save(root / "notes.npy", notes)
    save_sequence_csv(sequence, root / "sequence.csv")
    save_note_csv(notes, root / "notes.csv")
    save_midi(notes, root / f"{name}.mid", tempo_bpm=tempo_bpm)
    save_audio_preview(notes, root / f"{name}.wav", tempo_bpm=tempo_bpm)
    type_counts = {
        EVENT_TYPE_LABELS[index]: int(np.sum(sequence[:, 0] == index))
        for index in range(len(EVENT_TYPE_LABELS))
    }
    summary = {
        "name": name,
        "output_dir": str(root),
        "sequence_len": int(sequence.shape[0]),
        "note_count": int(notes.shape[0]),
        "type_counts": type_counts,
        "instrument_counts": {
            label: int(np.sum(notes[:, 3] == index)) if notes.size else 0
            for index, label in enumerate(INSTRUMENT_LABELS)
        },
        "config": asdict(config) if config is not None else None,
    }
    save_json(root / "summary.json", summary)
    return summary
