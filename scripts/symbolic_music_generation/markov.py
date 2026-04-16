from __future__ import annotations

import math
import random
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import mido


POSITIONS_PER_BAR = 32
LENGTH_UNITS_PER_BEAT = 8
DEFAULT_TICKS_PER_BEAT = 480
ALLOWED_BEAT_LENGTHS = (2, 4, 8, 16, 32)
EPSILON = 1e-12


@dataclass(frozen=True)
class NoteEvent:
    start_tick: int
    end_tick: int
    pitch: int

    @property
    def duration_ticks(self) -> int:
        return max(0, self.end_tick - self.start_tick)


def find_pdmx_files(data_dir: str | Path) -> list[str]:
    root = Path(data_dir)
    if root.is_file() and root.suffix == ".zip":
        with zipfile.ZipFile(root) as archive:
            archive.extractall(root.parent)
        root = root.parent

    candidates = [root]
    if root.exists():
        candidates.extend(path for path in root.rglob("PDMX_subset") if path.is_dir())

    for candidate in candidates:
        files = sorted(candidate.glob("*.mid"))
        if files:
            return [str(path) for path in files]

    if root.exists():
        for archive_path in sorted(root.rglob("PDMX_subset.zip")):
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(archive_path.parent)
            extracted = archive_path.parent / "PDMX_subset"
            files = sorted(extracted.glob("*.mid"))
            if files:
                return [str(path) for path in files]

    return []


@lru_cache(maxsize=None)
def _read_note_events_cached(midi_file: str) -> tuple[NoteEvent, ...]:
    midi = mido.MidiFile(str(midi_file))
    active: dict[int, list[int]] = defaultdict(list)
    events: list[NoteEvent] = []
    absolute_tick = 0

    for message in mido.merge_tracks(midi.tracks):
        absolute_tick += int(message.time)
        if message.type == "note_on" and message.velocity > 0:
            active[message.note].append(absolute_tick)
        elif message.type == "note_off" or (message.type == "note_on" and message.velocity == 0):
            starts = active.get(message.note)
            if starts:
                start_tick = starts.pop(0)
                events.append(NoteEvent(start_tick=start_tick, end_tick=absolute_tick, pitch=int(message.note)))

    events.sort(key=lambda event: (event.start_tick, event.end_tick, event.pitch))
    return tuple(events)


def read_note_events(midi_file: str | Path) -> list[NoteEvent]:
    return list(_read_note_events_cached(str(midi_file)))


def note_extraction(midi_file: str | Path) -> list[int]:
    return [event.pitch for event in read_note_events(midi_file)]


def note_frequency(midi_files: Iterable[str | Path]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for midi_file in midi_files:
        counts.update(note_extraction(midi_file))
    return dict(sorted(counts.items()))


def _normalize_counts(counts: Counter[int] | dict[int, int]) -> dict[int, float]:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {}
    return {int(key): float(value) / total for key, value in sorted(counts.items())}


def note_unigram_probability(midi_files: Iterable[str | Path]) -> dict[int, float]:
    return _normalize_counts(note_frequency(midi_files))


def _transition_probability(sequences: Iterable[list[int]], order: int) -> tuple[defaultdict, defaultdict]:
    transition_counts: dict[object, Counter[int]] = defaultdict(Counter)
    for sequence in sequences:
        if len(sequence) <= order:
            continue
        for index in range(order, len(sequence)):
            if order == 1:
                key: object = sequence[index - 1]
            else:
                key = tuple(sequence[index - order : index])
            transition_counts[key][sequence[index]] += 1

    transitions = defaultdict(list)
    probabilities = defaultdict(list)
    for key in sorted(transition_counts.keys()):
        normalized = _normalize_counts(transition_counts[key])
        transitions[key] = list(normalized.keys())
        probabilities[key] = list(normalized.values())
    return transitions, probabilities


def note_bigram_probability(midi_files: Iterable[str | Path]) -> tuple[defaultdict, defaultdict]:
    sequences = [note_extraction(midi_file) for midi_file in midi_files]
    return _transition_probability(sequences, order=1)


def note_trigram_probability(midi_files: Iterable[str | Path]) -> tuple[defaultdict, defaultdict]:
    sequences = [note_extraction(midi_file) for midi_file in midi_files]
    return _transition_probability(sequences, order=2)


def probability_from_table(transitions: dict, probabilities: dict, key: object, value: int) -> float | None:
    values = transitions.get(key)
    weights = probabilities.get(key)
    if not values or not weights:
        return None
    try:
        index = values.index(value)
    except ValueError:
        return None
    return float(weights[index])


def sample_from_table(transitions: dict, probabilities: dict, key: object, fallback_probabilities: dict[int, float]) -> int:
    values = transitions.get(key)
    weights = probabilities.get(key)
    if values and weights:
        return int(random.choices(list(values), weights=list(weights), k=1)[0])
    if fallback_probabilities:
        return int(random.choices(list(fallback_probabilities.keys()), weights=list(fallback_probabilities.values()), k=1)[0])
    return 60


def sample_next_note(note: int, midi_files: Iterable[str | Path]) -> int:
    unigram_probabilities = note_unigram_probability(midi_files)
    bigram_transitions, bigram_probabilities = note_bigram_probability(midi_files)
    return sample_from_table(bigram_transitions, bigram_probabilities, int(note), unigram_probabilities)


def _safe_log(probability: float | None) -> float:
    return math.log(max(float(probability or 0.0), EPSILON))


def note_bigram_perplexity(midi_file: str | Path, midi_files: Iterable[str | Path]) -> float:
    notes = note_extraction(midi_file)
    if not notes:
        return float("inf")

    unigram_probabilities = note_unigram_probability(midi_files)
    bigram_transitions, bigram_probabilities = note_bigram_probability(midi_files)
    log_sum = _safe_log(unigram_probabilities.get(notes[0]))
    for previous_note, current_note in zip(notes[:-1], notes[1:]):
        probability = probability_from_table(bigram_transitions, bigram_probabilities, previous_note, current_note)
        if probability is None:
            probability = unigram_probabilities.get(current_note, EPSILON)
        log_sum += _safe_log(probability)
    return float(math.exp(-log_sum / len(notes)))


def note_trigram_perplexity(midi_file: str | Path, midi_files: Iterable[str | Path]) -> float:
    notes = note_extraction(midi_file)
    if not notes:
        return float("inf")

    unigram_probabilities = note_unigram_probability(midi_files)
    bigram_transitions, bigram_probabilities = note_bigram_probability(midi_files)
    trigram_transitions, trigram_probabilities = note_trigram_probability(midi_files)

    log_sum = _safe_log(unigram_probabilities.get(notes[0]))
    if len(notes) >= 2:
        probability = probability_from_table(bigram_transitions, bigram_probabilities, notes[0], notes[1])
        if probability is None:
            probability = unigram_probabilities.get(notes[1], EPSILON)
        log_sum += _safe_log(probability)

    for index in range(2, len(notes)):
        key = (notes[index - 2], notes[index - 1])
        probability = probability_from_table(trigram_transitions, trigram_probabilities, key, notes[index])
        if probability is None:
            probability = probability_from_table(bigram_transitions, bigram_probabilities, notes[index - 1], notes[index])
        if probability is None:
            probability = unigram_probabilities.get(notes[index], EPSILON)
        log_sum += _safe_log(probability)
    return float(math.exp(-log_sum / len(notes)))


def _nearest_allowed_length(raw_length: int) -> int:
    if raw_length <= 0:
        return ALLOWED_BEAT_LENGTHS[0]
    return min(ALLOWED_BEAT_LENGTHS, key=lambda length: (abs(length - raw_length), length))


def _beat_position(start_tick: int, ticks_per_beat: int) -> int:
    ticks_per_bar = ticks_per_beat * 4
    return int(round((start_tick % ticks_per_bar) * POSITIONS_PER_BAR / ticks_per_bar)) % POSITIONS_PER_BAR


def _beat_length(duration_ticks: int, ticks_per_beat: int) -> int:
    raw_length = int(round(duration_ticks * LENGTH_UNITS_PER_BEAT / ticks_per_beat))
    return _nearest_allowed_length(raw_length)


def beat_extraction(midi_file: str | Path) -> list[tuple[int, int]]:
    midi = mido.MidiFile(str(midi_file))
    ticks_per_beat = int(midi.ticks_per_beat or DEFAULT_TICKS_PER_BEAT)
    return [
        (_beat_position(event.start_tick, ticks_per_beat), _beat_length(event.duration_ticks, ticks_per_beat))
        for event in read_note_events(midi_file)
    ]


def beat_unigram_probability(midi_files: Iterable[str | Path]) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for midi_file in midi_files:
        counts.update(length for _position, length in beat_extraction(midi_file))
    return _normalize_counts(counts)


def beat_bigram_probability(midi_files: Iterable[str | Path]) -> tuple[defaultdict, defaultdict]:
    sequences = [[length for _position, length in beat_extraction(midi_file)] for midi_file in midi_files]
    return _transition_probability(sequences, order=1)


def beat_pos_bigram_probability(midi_files: Iterable[str | Path]) -> tuple[defaultdict, defaultdict]:
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for midi_file in midi_files:
        for position, length in beat_extraction(midi_file):
            transition_counts[position][length] += 1

    transitions = defaultdict(list)
    probabilities = defaultdict(list)
    for position in sorted(transition_counts.keys()):
        normalized = _normalize_counts(transition_counts[position])
        transitions[position] = list(normalized.keys())
        probabilities[position] = list(normalized.values())
    return transitions, probabilities


def beat_bigram_perplexity(midi_file: str | Path, midi_files: Iterable[str | Path]) -> tuple[float, float]:
    beats = beat_extraction(midi_file)
    if not beats:
        return float("inf"), float("inf")

    unigram_probabilities = beat_unigram_probability(midi_files)
    bigram_transitions, bigram_probabilities = beat_bigram_probability(midi_files)
    pos_transitions, pos_probabilities = beat_pos_bigram_probability(midi_files)

    lengths = [length for _position, length in beats]
    log_sum_q7 = _safe_log(unigram_probabilities.get(lengths[0]))
    for previous_length, current_length in zip(lengths[:-1], lengths[1:]):
        probability = probability_from_table(bigram_transitions, bigram_probabilities, previous_length, current_length)
        if probability is None:
            probability = unigram_probabilities.get(current_length, EPSILON)
        log_sum_q7 += _safe_log(probability)

    log_sum_q8 = 0.0
    for position, length in beats:
        probability = probability_from_table(pos_transitions, pos_probabilities, position, length)
        if probability is None:
            probability = unigram_probabilities.get(length, EPSILON)
        log_sum_q8 += _safe_log(probability)

    return float(math.exp(-log_sum_q7 / len(lengths))), float(math.exp(-log_sum_q8 / len(lengths)))


def beat_trigram_probability(midi_files: Iterable[str | Path]) -> tuple[defaultdict, defaultdict]:
    transition_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    for midi_file in midi_files:
        beats = beat_extraction(midi_file)
        for index in range(1, len(beats)):
            previous_length = beats[index - 1][1]
            position, current_length = beats[index]
            transition_counts[(previous_length, position)][current_length] += 1

    transitions = defaultdict(list)
    probabilities = defaultdict(list)
    for key in sorted(transition_counts.keys()):
        normalized = _normalize_counts(transition_counts[key])
        transitions[key] = list(normalized.keys())
        probabilities[key] = list(normalized.values())
    return transitions, probabilities


def beat_trigram_perplexity(midi_file: str | Path, midi_files: Iterable[str | Path]) -> float:
    beats = beat_extraction(midi_file)
    if not beats:
        return float("inf")

    unigram_probabilities = beat_unigram_probability(midi_files)
    pos_transitions, pos_probabilities = beat_pos_bigram_probability(midi_files)
    trigram_transitions, trigram_probabilities = beat_trigram_probability(midi_files)

    first_position, first_length = beats[0]
    probability = probability_from_table(pos_transitions, pos_probabilities, first_position, first_length)
    if probability is None:
        probability = unigram_probabilities.get(first_length, EPSILON)
    log_sum = _safe_log(probability)

    for index in range(1, len(beats)):
        previous_length = beats[index - 1][1]
        position, current_length = beats[index]
        probability = probability_from_table(trigram_transitions, trigram_probabilities, (previous_length, position), current_length)
        if probability is None:
            probability = probability_from_table(pos_transitions, pos_probabilities, position, current_length)
        if probability is None:
            probability = unigram_probabilities.get(current_length, EPSILON)
        log_sum += _safe_log(probability)
    return float(math.exp(-log_sum / len(beats)))


def generate_note_sequence(length: int, midi_files: Iterable[str | Path]) -> list[int]:
    midi_files = list(midi_files)
    unigram_probabilities = note_unigram_probability(midi_files)
    bigram_transitions, bigram_probabilities = note_bigram_probability(midi_files)
    trigram_transitions, trigram_probabilities = note_trigram_probability(midi_files)
    if length <= 0:
        return []

    notes = [sample_from_table({}, {}, None, unigram_probabilities)]
    if length >= 2:
        notes.append(sample_from_table(bigram_transitions, bigram_probabilities, notes[0], unigram_probabilities))
    while len(notes) < length:
        key = (notes[-2], notes[-1])
        next_note = sample_from_table(trigram_transitions, trigram_probabilities, key, unigram_probabilities)
        notes.append(next_note)
    return notes


def generate_beat_sequence(length: int, midi_files: Iterable[str | Path]) -> list[tuple[int, int]]:
    midi_files = list(midi_files)
    unigram_probabilities = beat_unigram_probability(midi_files)
    pos_transitions, pos_probabilities = beat_pos_bigram_probability(midi_files)
    position = 0
    beats: list[tuple[int, int]] = []
    for _index in range(max(0, length)):
        beat_length = sample_from_table(pos_transitions, pos_probabilities, position, unigram_probabilities)
        beats.append((position, beat_length))
        position = (position + beat_length) % POSITIONS_PER_BAR
    return beats


def save_midi(notes: list[int], beats: list[tuple[int, int]], output_path: str | Path = "q10.mid") -> str:
    midi = mido.MidiFile(ticks_per_beat=DEFAULT_TICKS_PER_BEAT)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(120), time=0))
    track.append(mido.Message("program_change", program=0, time=0))

    for pitch, (_position, beat_length) in zip(notes, beats):
        duration_ticks = max(1, int(round(beat_length * DEFAULT_TICKS_PER_BEAT / LENGTH_UNITS_PER_BEAT)))
        track.append(mido.Message("note_on", note=int(pitch), velocity=80, time=0))
        track.append(mido.Message("note_off", note=int(pitch), velocity=0, time=duration_ticks))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))
    return str(output_path)


def music_generate(length: int, midi_files: Iterable[str | Path], output_path: str | Path = "q10.mid", seed: int = 42) -> str:
    random.seed(seed)
    notes = generate_note_sequence(length, midi_files)
    beats = generate_beat_sequence(length, midi_files)
    return save_midi(notes, beats, output_path)
