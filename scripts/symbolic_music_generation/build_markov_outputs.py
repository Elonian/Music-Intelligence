#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
import statistics
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.symbolic_music_generation import (
    EPSILON,
    beat_extraction,
    find_pdmx_files,
    generate_beat_sequence,
    generate_note_sequence,
    note_extraction,
    probability_from_table,
    save_midi,
)
from utils import (
    SYMBOLIC_DATA_ROOT,
    SYMBOLIC_GENERATED_DIR,
    SYMBOLIC_METRICS_DIR,
    SYMBOLIC_TABLE_DIR,
    ensure_dir,
    save_json,
    write_csv_rows,
)


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


def _top_items(mapping: dict[int, int] | dict[int, float], limit: int = 12) -> list[dict]:
    return [
        {"value": int(key), "score": float(value)}
        for key, value in sorted(mapping.items(), key=lambda item: (-item[1], item[0]))[:limit]
    ]


def _transition_preview(transitions: dict, probabilities: dict, limit_keys: int = 12, limit_values: int = 8) -> dict:
    preview = {}
    for key in sorted(transitions.keys())[:limit_keys]:
        values = transitions[key]
        weights = probabilities[key]
        preview[str(key)] = [
            {"next": int(value), "probability": float(probability)}
            for value, probability in zip(values[:limit_values], weights[:limit_values])
        ]
    return preview


def _normalize_counter(counts: Counter[int]) -> dict[int, float]:
    total = float(sum(counts.values()))
    if total <= 0.0:
        return {}
    return {key: value / total for key, value in sorted(counts.items())}


def _transition_probability(sequences: list[list[int]], order: int) -> tuple[dict, dict]:
    transition_counts: dict[object, Counter[int]] = {}
    for sequence in sequences:
        for index in range(order, len(sequence)):
            key = sequence[index - 1] if order == 1 else tuple(sequence[index - order : index])
            transition_counts.setdefault(key, Counter())[sequence[index]] += 1

    transitions = {}
    probabilities = {}
    for key in sorted(transition_counts):
        normalized = _normalize_counter(transition_counts[key])
        transitions[key] = list(normalized.keys())
        probabilities[key] = list(normalized.values())
    return transitions, probabilities


def _position_probability(beat_sequences: list[list[tuple[int, int]]]) -> tuple[dict, dict]:
    transition_counts: dict[int, Counter[int]] = {}
    for beats in beat_sequences:
        for position, length in beats:
            transition_counts.setdefault(position, Counter())[length] += 1

    transitions = {}
    probabilities = {}
    for key in sorted(transition_counts):
        normalized = _normalize_counter(transition_counts[key])
        transitions[key] = list(normalized.keys())
        probabilities[key] = list(normalized.values())
    return transitions, probabilities


def _beat_trigram_probability(beat_sequences: list[list[tuple[int, int]]]) -> tuple[dict, dict]:
    transition_counts: dict[tuple[int, int], Counter[int]] = {}
    for beats in beat_sequences:
        for index in range(1, len(beats)):
            previous_length = beats[index - 1][1]
            position, current_length = beats[index]
            transition_counts.setdefault((previous_length, position), Counter())[current_length] += 1

    transitions = {}
    probabilities = {}
    for key in sorted(transition_counts):
        normalized = _normalize_counter(transition_counts[key])
        transitions[key] = list(normalized.keys())
        probabilities[key] = list(normalized.values())
    return transitions, probabilities


def _collect_sequences(midi_files: list[str]) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    note_sequences = []
    beat_sequences = []
    for index, midi_file in enumerate(midi_files, start=1):
        note_sequences.append(note_extraction(midi_file))
        beat_sequences.append(beat_extraction(midi_file))
        if index == 1 or index % 100 == 0 or index == len(midi_files):
            print(f"[Symbolic Markov] Parsed {index}/{len(midi_files)} files", flush=True)
    return note_sequences, beat_sequences


def _note_bigram_perplexity_from_tables(
    notes: list[int],
    unigram_probabilities: dict[int, float],
    bigram_transitions: dict,
    bigram_probabilities: dict,
) -> float:
    if not notes:
        return float("inf")
    log_sum = math.log(max(unigram_probabilities.get(notes[0], 0.0), EPSILON))
    for previous_note, current_note in zip(notes[:-1], notes[1:]):
        probability = probability_from_table(bigram_transitions, bigram_probabilities, previous_note, current_note)
        if probability is None:
            probability = unigram_probabilities.get(current_note, EPSILON)
        log_sum += math.log(max(probability, EPSILON))
    return float(math.exp(-log_sum / len(notes)))


def _note_trigram_perplexity_from_tables(
    notes: list[int],
    unigram_probabilities: dict[int, float],
    bigram_transitions: dict,
    bigram_probabilities: dict,
    trigram_transitions: dict,
    trigram_probabilities: dict,
) -> float:
    if not notes:
        return float("inf")
    log_sum = math.log(max(unigram_probabilities.get(notes[0], 0.0), EPSILON))
    if len(notes) >= 2:
        probability = probability_from_table(bigram_transitions, bigram_probabilities, notes[0], notes[1])
        if probability is None:
            probability = unigram_probabilities.get(notes[1], EPSILON)
        log_sum += math.log(max(probability, EPSILON))
    for index in range(2, len(notes)):
        key = (notes[index - 2], notes[index - 1])
        probability = probability_from_table(trigram_transitions, trigram_probabilities, key, notes[index])
        if probability is None:
            probability = probability_from_table(bigram_transitions, bigram_probabilities, notes[index - 1], notes[index])
        if probability is None:
            probability = unigram_probabilities.get(notes[index], EPSILON)
        log_sum += math.log(max(probability, EPSILON))
    return float(math.exp(-log_sum / len(notes)))


def _beat_perplexities_from_tables(
    beats: list[tuple[int, int]],
    unigram_probabilities: dict[int, float],
    bigram_transitions: dict,
    bigram_probabilities: dict,
    pos_transitions: dict,
    pos_probabilities: dict,
    trigram_transitions: dict,
    trigram_probabilities: dict,
) -> tuple[float, float, float]:
    if not beats:
        return float("inf"), float("inf"), float("inf")

    lengths = [length for _position, length in beats]
    log_bigram = math.log(max(unigram_probabilities.get(lengths[0], 0.0), EPSILON))
    for previous_length, current_length in zip(lengths[:-1], lengths[1:]):
        probability = probability_from_table(bigram_transitions, bigram_probabilities, previous_length, current_length)
        if probability is None:
            probability = unigram_probabilities.get(current_length, EPSILON)
        log_bigram += math.log(max(probability, EPSILON))

    log_position = 0.0
    for position, length in beats:
        probability = probability_from_table(pos_transitions, pos_probabilities, position, length)
        if probability is None:
            probability = unigram_probabilities.get(length, EPSILON)
        log_position += math.log(max(probability, EPSILON))

    first_position, first_length = beats[0]
    probability = probability_from_table(pos_transitions, pos_probabilities, first_position, first_length)
    if probability is None:
        probability = unigram_probabilities.get(first_length, EPSILON)
    log_trigram = math.log(max(probability, EPSILON))
    for index in range(1, len(beats)):
        previous_length = beats[index - 1][1]
        position, current_length = beats[index]
        probability = probability_from_table(trigram_transitions, trigram_probabilities, (previous_length, position), current_length)
        if probability is None:
            probability = probability_from_table(pos_transitions, pos_probabilities, position, current_length)
        if probability is None:
            probability = unigram_probabilities.get(current_length, EPSILON)
        log_trigram += math.log(max(probability, EPSILON))

    count = len(beats)
    return (
        float(math.exp(-log_bigram / count)),
        float(math.exp(-log_position / count)),
        float(math.exp(-log_trigram / count)),
    )


def _pitch_rows(counts: dict[int, int] | Counter[int], probabilities: dict[int, float]) -> list[dict]:
    return [
        {"pitch": pitch, "count": counts[pitch], "probability": f"{probabilities.get(pitch, 0.0):.8f}"}
        for pitch in sorted(counts)
    ]


def _beat_length_rows(beat_length_counts: Counter[int], probabilities: dict[int, float]) -> list[dict]:
    return [
        {"beat_length": length, "count": beat_length_counts[length], "probability": f"{probabilities.get(length, 0.0):.8f}"}
        for length in sorted(beat_length_counts)
    ]


def _perplexity_rows(
    midi_files: list[str],
    note_sequences: list[list[int]],
    beat_sequences: list[list[tuple[int, int]]],
    eval_count: int,
    note_unigram: dict[int, float],
    note_bigram_transitions: dict,
    note_bigram_probabilities: dict,
    note_trigram_transitions: dict,
    note_trigram_probabilities: dict,
    beat_unigram: dict[int, float],
    beat_bigram_transitions: dict,
    beat_bigram_probabilities: dict,
    beat_pos_transitions: dict,
    beat_pos_probabilities: dict,
    beat_trigram_transitions: dict,
    beat_trigram_probabilities: dict,
) -> list[dict]:
    rows = []
    for index in range(eval_count):
        beat_bigram_ppl, beat_position_ppl, beat_trigram_ppl = _beat_perplexities_from_tables(
            beat_sequences[index],
            beat_unigram,
            beat_bigram_transitions,
            beat_bigram_probabilities,
            beat_pos_transitions,
            beat_pos_probabilities,
            beat_trigram_transitions,
            beat_trigram_probabilities,
        )
        rows.append(
            {
                "file_name": Path(midi_files[index]).name,
                "note_bigram_perplexity": f"{_note_bigram_perplexity_from_tables(note_sequences[index], note_unigram, note_bigram_transitions, note_bigram_probabilities):.6f}",
                "note_trigram_perplexity": f"{_note_trigram_perplexity_from_tables(note_sequences[index], note_unigram, note_bigram_transitions, note_bigram_probabilities, note_trigram_transitions, note_trigram_probabilities):.6f}",
                "beat_bigram_perplexity": f"{beat_bigram_ppl:.6f}",
                "beat_position_perplexity": f"{beat_position_ppl:.6f}",
                "beat_trigram_perplexity": f"{beat_trigram_ppl:.6f}",
            }
        )
    return rows


def _generated_rows(notes: list[int], beats: list[tuple[int, int]]) -> list[dict]:
    return [
        {"index": index, "pitch": pitch, "beat_position": position, "beat_length": length}
        for index, (pitch, (position, length)) in enumerate(zip(notes, beats))
    ]


def build_symbolic_outputs(
    data_dir: Path = SYMBOLIC_DATA_ROOT,
    metrics_dir: Path = SYMBOLIC_METRICS_DIR,
    table_dir: Path = SYMBOLIC_TABLE_DIR,
    generated_dir: Path = SYMBOLIC_GENERATED_DIR,
    generated_length: int = 500,
    max_eval_files: int = 24,
) -> dict:
    midi_files = find_pdmx_files(data_dir)
    if not midi_files:
        raise FileNotFoundError(f"No MIDI files found under {data_dir}.")

    ensure_dir(metrics_dir)
    ensure_dir(table_dir)
    ensure_dir(generated_dir)

    note_sequences, beat_sequences = _collect_sequences(midi_files)
    note_counts: Counter[int] = Counter()
    beat_length_counts: Counter[int] = Counter()
    for notes in note_sequences:
        note_counts.update(notes)
    for beats_for_file in beat_sequences:
        beat_length_counts.update(length for _position, length in beats_for_file)

    print("[Symbolic Markov] Building transition tables", flush=True)
    note_probabilities = _normalize_counter(note_counts)
    beat_probabilities = _normalize_counter(beat_length_counts)
    note_bigram_transitions, note_bigram_probabilities = _transition_probability(note_sequences, order=1)
    note_trigram_transitions, note_trigram_probabilities = _transition_probability(note_sequences, order=2)
    beat_length_sequences = [[length for _position, length in beats] for beats in beat_sequences]
    beat_bigram_transitions, beat_bigram_probabilities = _transition_probability(beat_length_sequences, order=1)
    beat_pos_transitions, beat_pos_probabilities = _position_probability(beat_sequences)
    beat_trigram_transitions, beat_trigram_probabilities = _beat_trigram_probability(beat_sequences)

    eval_count = max(1, min(max_eval_files, len(midi_files)))
    perplexity_rows = _perplexity_rows(
        midi_files,
        note_sequences,
        beat_sequences,
        eval_count,
        note_probabilities,
        note_bigram_transitions,
        note_bigram_probabilities,
        note_trigram_transitions,
        note_trigram_probabilities,
        beat_probabilities,
        beat_bigram_transitions,
        beat_bigram_probabilities,
        beat_pos_transitions,
        beat_pos_probabilities,
        beat_trigram_transitions,
        beat_trigram_probabilities,
    )

    print("[Symbolic Markov] Generating q10.mid", flush=True)
    random.seed(42)
    notes = generate_note_sequence(generated_length, midi_files)
    beats = generate_beat_sequence(generated_length, midi_files)
    generated_path = Path(save_midi(notes, beats, generated_dir / "q10.mid"))

    print("[Symbolic Markov] Writing tables", flush=True)
    write_csv_rows(table_dir / "note_pitch_distribution.csv", _pitch_rows(note_counts, note_probabilities))
    write_csv_rows(table_dir / "beat_length_distribution.csv", _beat_length_rows(beat_length_counts, beat_probabilities))
    write_csv_rows(table_dir / "perplexity_by_file.csv", perplexity_rows)
    write_csv_rows(table_dir / "generated_sequence.csv", _generated_rows(notes, beats))

    note_event_count = sum(note_counts.values())
    beat_event_count = sum(len(beats_for_file) for beats_for_file in beat_sequences)
    summary = {
        "file_count": len(midi_files),
        "note_event_count": note_event_count,
        "beat_event_count": beat_event_count,
        "unique_pitch_count": len(note_counts),
        "top_pitches": _top_items(note_counts),
        "top_pitch_probabilities": _top_items(note_probabilities),
        "mean_note_bigram_perplexity": _mean([float(row["note_bigram_perplexity"]) for row in perplexity_rows]),
        "mean_note_trigram_perplexity": _mean([float(row["note_trigram_perplexity"]) for row in perplexity_rows]),
        "mean_beat_bigram_perplexity": _mean([float(row["beat_bigram_perplexity"]) for row in perplexity_rows]),
        "mean_beat_position_perplexity": _mean([float(row["beat_position_perplexity"]) for row in perplexity_rows]),
        "mean_beat_trigram_perplexity": _mean([float(row["beat_trigram_perplexity"]) for row in perplexity_rows]),
        "generated_midi_path": str(generated_path),
        "generated_note_count": len(note_extraction(generated_path)),
        "generated_length_requested": generated_length,
    }
    save_json(metrics_dir / "markov_summary.json", summary)
    save_json(
        metrics_dir / "transition_preview.json",
        {
            "note_bigram": _transition_preview(note_bigram_transitions, note_bigram_probabilities),
            "note_trigram": _transition_preview(note_trigram_transitions, note_trigram_probabilities),
            "beat_bigram": _transition_preview(beat_bigram_transitions, beat_bigram_probabilities),
            "beat_position": _transition_preview(beat_pos_transitions, beat_pos_probabilities),
            "beat_trigram": _transition_preview(beat_trigram_transitions, beat_trigram_probabilities),
        },
    )
    return {"summary": summary, "midi_files": midi_files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build symbolic Markov-chain metrics and generated MIDI.")
    parser.add_argument("--data-dir", type=Path, default=SYMBOLIC_DATA_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=SYMBOLIC_METRICS_DIR)
    parser.add_argument("--table-dir", type=Path, default=SYMBOLIC_TABLE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=SYMBOLIC_GENERATED_DIR)
    parser.add_argument("--generated-length", type=int, default=500)
    parser.add_argument("--max-eval-files", type=int, default=24)
    args = parser.parse_args()

    artifacts = build_symbolic_outputs(
        data_dir=args.data_dir,
        metrics_dir=args.metrics_dir,
        table_dir=args.table_dir,
        generated_dir=args.generated_dir,
        generated_length=args.generated_length,
        max_eval_files=args.max_eval_files,
    )
    print(f"[Symbolic Markov] Processed {len(artifacts['midi_files'])} MIDI files")
    print(f"[Symbolic Markov] Wrote {args.metrics_dir / 'markov_summary.json'}")


if __name__ == "__main__":
    main()
