from __future__ import annotations

import numpy as np


SAMPLE_RATE = 44100


def note_name_to_frequency(note_name: str) -> float:
    note_offsets = {
        "C": -9,
        "C#": -8,
        "D": -7,
        "D#": -6,
        "E": -5,
        "F": -4,
        "F#": -3,
        "G": -2,
        "G#": -1,
        "A": 0,
        "A#": 1,
        "B": 2,
    }
    note_name = note_name.strip()
    if len(note_name) >= 3 and note_name[1] == "#":
        note = note_name[:2]
        octave = int(note_name[2:])
    else:
        note = note_name[0]
        octave = int(note_name[1:])
    semitone_distance = note_offsets[note] + 12 * (octave - 4)
    return 440.0 * (2 ** (semitone_distance / 12))


def create_sine_wave(frequency: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)


def create_sawtooth_wave(frequency: float, duration: float, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
    wave = np.zeros_like(t)
    for harmonic in range(1, 20):
        wave += ((-1) ** (harmonic + 1)) * np.sin(2 * np.pi * harmonic * frequency * t) / harmonic
    return (2 / np.pi) * wave


def linear_fade_out(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        return audio
    return audio * np.linspace(1.0, 0.0, audio.shape[0])


def add_delay(audio: np.ndarray, delay_seconds: float = 0.5, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    audio = np.asarray(audio, dtype=float)
    delay_samples = int(delay_seconds * sample_rate)
    delayed = np.zeros(audio.shape[0] + delay_samples)
    delayed[: audio.shape[0]] += 0.7 * audio
    delayed[delay_samples : delay_samples + audio.shape[0]] += 0.3 * audio
    return delayed


def concatenate_audio(clips: list[np.ndarray]) -> np.ndarray:
    if not clips:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(clip, dtype=float) for clip in clips])


def mix_audio(clips: list[np.ndarray], amplitudes: list[float]) -> np.ndarray:
    if not clips:
        return np.array([], dtype=float)
    mixed = np.zeros_like(np.asarray(clips[0], dtype=float), dtype=float)
    for clip, amplitude in zip(clips, amplitudes):
        mixed += amplitude * np.asarray(clip, dtype=float)
    return mixed


def render_melody(note_names: list[str], durations: list[float], wave_kind: str = "sine", sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    clips = []
    for note_name, duration in zip(note_names, durations):
        frequency = note_name_to_frequency(note_name)
        if wave_kind == "sawtooth":
            clips.append(create_sawtooth_wave(frequency, duration, sample_rate))
        else:
            clips.append(create_sine_wave(frequency, duration, sample_rate))
    return concatenate_audio(clips)


def audio_to_int16(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=float)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 1e-12:
        return np.zeros(audio.shape, dtype=np.int16)
    scaled = np.clip(audio / peak, -1.0, 1.0)
    return np.asarray(np.round(scaled * 32767.0), dtype=np.int16)
