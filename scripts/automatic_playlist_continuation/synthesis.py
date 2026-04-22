from __future__ import annotations

import numpy as np


def adsr_envelope(
    duration: float,
    attack_time: float,
    decay_time: float,
    sustain_level: float,
    release_time: float,
    sr: int = 44100,
) -> np.ndarray:
    duration = max(float(duration), 0.0)
    attack_time = max(float(attack_time), 0.0)
    decay_time = max(float(decay_time), 0.0)
    sustain_level = float(np.clip(sustain_level, 0.0, 1.0))
    release_time = max(float(release_time), 0.0)
    sr = int(sr)

    held_samples = int(round(duration * sr))
    release_samples = int(round(release_time * sr))
    total_samples = held_samples + release_samples
    if total_samples <= 0:
        return np.zeros(0, dtype=np.float32)

    attack_samples = min(int(round(attack_time * sr)), held_samples)
    remaining = held_samples - attack_samples
    decay_samples = min(int(round(decay_time * sr)), remaining)
    sustain_samples = max(held_samples - attack_samples - decay_samples, 0)

    segments: list[np.ndarray] = []
    if attack_samples > 0:
        segments.append(np.linspace(0.0, 1.0, attack_samples, endpoint=False, dtype=np.float32))
    if decay_samples > 0:
        segments.append(np.linspace(1.0, sustain_level, decay_samples, endpoint=False, dtype=np.float32))
    if sustain_samples > 0:
        segments.append(np.full(sustain_samples, sustain_level, dtype=np.float32))
    held = np.concatenate(segments) if segments else np.zeros(0, dtype=np.float32)
    if held.size < held_samples:
        fill_value = float(held[-1]) if held.size else sustain_level
        held = np.pad(held, (0, held_samples - held.size), constant_values=fill_value)
    elif held.size > held_samples:
        held = held[:held_samples]

    release_start = float(held[-1]) if held.size else sustain_level
    if release_samples > 0:
        release = np.linspace(release_start, 0.0, release_samples, endpoint=True, dtype=np.float32)
        envelope = np.concatenate([held, release])
    else:
        envelope = held
    return np.clip(envelope, 0.0, 1.0).astype(np.float32, copy=False)


def get_lfo(base_cutoff: float, lfo_depth: float, lfo_freq: float, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float32)
    base_cutoff = float(base_cutoff)
    lfo_depth = float(lfo_depth)
    lfo_freq = float(lfo_freq)
    lfo = base_cutoff + lfo_depth * np.sin(2.0 * np.pi * lfo_freq * t)
    return np.maximum(lfo, 0.0).astype(np.float32, copy=False)
