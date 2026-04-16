#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.symbolic_music_generation import (
    ALLOWED_BEAT_LENGTHS,
    POSITIONS_PER_BAR,
    beat_pos_bigram_probability,
    beat_extraction,
    find_pdmx_files,
    note_bigram_probability,
    note_extraction,
    note_frequency,
    note_unigram_probability,
)
from scripts.symbolic_music_generation.build_markov_outputs import build_symbolic_outputs
from utils import (
    SYMBOLIC_DATA_ROOT,
    SYMBOLIC_GENERATED_DIR,
    SYMBOLIC_METRICS_DIR,
    SYMBOLIC_TABLE_DIR,
    VISUAL_SYMBOLIC_DIR,
    VISUAL_SYMBOLIC_README_DIR,
    ensure_dir,
    load_json,
    read_csv_rows,
)


PANEL_FACE = "#f7f3ea"
INK = "#1f2933"
MUTED = "#4a5568"
BLUE = "#2f6fbb"
GREEN = "#2f855a"
GOLD = "#d97706"
ROSE = "#9f3a64"
PURPLE = "#805ad5"
_NOTE_COUNTS_CACHE: dict[int, int] | None = None
_BIGRAM_MATRIX_CACHE: tuple[list[int], np.ndarray] | None = None
_BEAT_POSITION_MATRIX_CACHE: np.ndarray | None = None


def _load_generated_rows(table_dir: Path) -> list[dict]:
    path = table_dir / "generated_sequence.csv"
    if not path.exists():
        return []
    return read_csv_rows(path)


def _warm_midi_cache(midi_files: list[str]) -> None:
    for index, midi_file in enumerate(midi_files, start=1):
        note_extraction(midi_file)
        beat_extraction(midi_file)
        if index == 1 or index % 100 == 0 or index == len(midi_files):
            print(f"[Symbolic Visualiser] Loaded {index}/{len(midi_files)} MIDI files", flush=True)


def _cumulative_beats(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitches = np.asarray([int(row["pitch"]) for row in rows], dtype=float)
    lengths = np.asarray([int(row["beat_length"]) / 8.0 for row in rows], dtype=float)
    starts = np.concatenate([[0.0], np.cumsum(lengths[:-1])]) if lengths.size else np.asarray([], dtype=float)
    return starts, lengths, pitches


def _save_pitch_distribution(midi_files: list[str], visual_dir: Path) -> None:
    counts = _note_counts(midi_files)
    probabilities = note_unigram_probability(midi_files)
    pitches = list(counts.keys())
    fig, ax = plt.subplots(figsize=(11, 4.6), facecolor=PANEL_FACE)
    ax.bar(pitches, [counts[pitch] for pitch in pitches], color=BLUE, edgecolor="#1f1f1f", linewidth=0.35)
    ax.set_title("Pitch Event Distribution", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("MIDI pitch")
    ax.set_ylabel("Count")
    ax.grid(axis="y", alpha=0.22)
    top_pitch = max(probabilities, key=probabilities.get)
    ax.text(
        0.99,
        0.92,
        f"Most likely pitch: {top_pitch}\nProbability: {probabilities[top_pitch]:.3f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=INK,
        bbox={"facecolor": "#fffaf2", "edgecolor": "#d8cbb9", "pad": 6},
    )
    fig.tight_layout()
    ensure_dir(visual_dir)
    fig.savefig(visual_dir / "pitch_distribution.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _note_counts(midi_files: list[str]) -> dict[int, int]:
    global _NOTE_COUNTS_CACHE
    if _NOTE_COUNTS_CACHE is None:
        _NOTE_COUNTS_CACHE = note_frequency(midi_files)
    return _NOTE_COUNTS_CACHE


def _bigram_matrix(midi_files: list[str]) -> tuple[list[int], np.ndarray]:
    global _BIGRAM_MATRIX_CACHE
    if _BIGRAM_MATRIX_CACHE is not None:
        return _BIGRAM_MATRIX_CACHE

    counts = _note_counts(midi_files)
    pitches = list(counts.keys())
    index = {pitch: idx for idx, pitch in enumerate(pitches)}
    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for file_index, midi_file in enumerate(midi_files, start=1):
        notes = note_extraction(midi_file)
        for previous_note, next_note in zip(notes[:-1], notes[1:]):
            transition_counts[previous_note][next_note] += 1
        if file_index % 250 == 0 or file_index == len(midi_files):
            print(f"[Symbolic Visualiser] Bigram matrix {file_index}/{len(midi_files)}", flush=True)
    matrix = np.zeros((len(pitches), len(pitches)), dtype=float)
    for previous_note, counts_for_note in transition_counts.items():
        if previous_note not in index:
            continue
        row = index[previous_note]
        total = float(sum(counts_for_note.values()))
        for next_note, count in counts_for_note.items():
            if next_note in index:
                matrix[row, index[next_note]] = float(count) / total
    _BIGRAM_MATRIX_CACHE = (pitches, matrix)
    return _BIGRAM_MATRIX_CACHE


def _save_note_bigram_heatmap(midi_files: list[str], visual_dir: Path) -> None:
    pitches, matrix = _bigram_matrix(midi_files)
    fig, ax = plt.subplots(figsize=(8.8, 7.5), facecolor=PANEL_FACE)
    image = ax.imshow(matrix, cmap="magma", aspect="auto", origin="lower", vmin=0.0)
    ax.set_title("First-Order Pitch Transition Probabilities", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Next pitch")
    ax.set_ylabel("Previous pitch")
    tick_step = max(1, len(pitches) // 12)
    ticks = list(range(0, len(pitches), tick_step))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([pitches[idx] for idx in ticks], rotation=45, ha="right")
    ax.set_yticklabels([pitches[idx] for idx in ticks])
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="p(next | previous)")
    fig.tight_layout()
    fig.savefig(visual_dir / "note_bigram_heatmap.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _beat_position_matrix(midi_files: list[str]) -> np.ndarray:
    global _BEAT_POSITION_MATRIX_CACHE
    if _BEAT_POSITION_MATRIX_CACHE is not None:
        return _BEAT_POSITION_MATRIX_CACHE

    transition_counts: dict[int, Counter[int]] = defaultdict(Counter)
    for file_index, midi_file in enumerate(midi_files, start=1):
        for position, beat_length in beat_extraction(midi_file):
            transition_counts[position][beat_length] += 1
        if file_index % 250 == 0 or file_index == len(midi_files):
            print(f"[Symbolic Visualiser] Beat-position matrix {file_index}/{len(midi_files)}", flush=True)
    matrix = np.zeros((len(ALLOWED_BEAT_LENGTHS), POSITIONS_PER_BAR), dtype=float)
    length_index = {length: idx for idx, length in enumerate(ALLOWED_BEAT_LENGTHS)}
    for position, counts_for_position in transition_counts.items():
        total = float(sum(counts_for_position.values()))
        for beat_length, count in counts_for_position.items():
            if beat_length in length_index:
                matrix[length_index[beat_length], int(position)] = float(count) / total
    _BEAT_POSITION_MATRIX_CACHE = matrix
    return matrix


def _save_beat_position_heatmap(midi_files: list[str], visual_dir: Path) -> None:
    matrix = _beat_position_matrix(midi_files)
    fig, ax = plt.subplots(figsize=(11.5, 3.8), facecolor=PANEL_FACE)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto", origin="lower", vmin=0.0)
    ax.set_title("Beat Length Given Bar Position", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Beat position in 32-slot bar")
    ax.set_ylabel("Length")
    ax.set_yticks(range(len(ALLOWED_BEAT_LENGTHS)))
    ax.set_yticklabels(ALLOWED_BEAT_LENGTHS)
    ax.set_xticks(range(0, POSITIONS_PER_BAR, 4))
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.025, label="p(length | position)")
    fig.tight_layout()
    fig.savefig(visual_dir / "beat_position_heatmap.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_perplexity_plot(summary: dict, visual_dir: Path) -> None:
    labels = ["Note Bigram", "Note Trigram", "Beat Bigram", "Beat Position", "Beat Trigram"]
    values = [
        summary["mean_note_bigram_perplexity"],
        summary["mean_note_trigram_perplexity"],
        summary["mean_beat_bigram_perplexity"],
        summary["mean_beat_position_perplexity"],
        summary["mean_beat_trigram_perplexity"],
    ]
    colors = [BLUE, PURPLE, GOLD, GREEN, ROSE]
    fig, ax = plt.subplots(figsize=(10.8, 4.6), facecolor=PANEL_FACE)
    bars = ax.bar(labels, values, color=colors, edgecolor="#1f1f1f", linewidth=0.35)
    ax.set_title("Held-Out Sequence Perplexity", loc="left", fontweight="bold", color=INK)
    ax.set_ylabel("Lower is better")
    ax.grid(axis="y", alpha=0.22)
    ax.tick_params(axis="x", labelrotation=12)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.04, f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(visual_dir / "perplexity_comparison.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_generated_piano_roll(rows: list[dict], visual_dir: Path, max_notes: int = 180) -> None:
    rows = rows[:max_notes]
    starts, lengths, pitches = _cumulative_beats(rows)
    fig, ax = plt.subplots(figsize=(12, 4.6), facecolor=PANEL_FACE)
    for start, length, pitch in zip(starts, lengths, pitches):
        ax.hlines(pitch, start, start + length, color=BLUE, linewidth=3.0, alpha=0.84)
    ax.set_title("Generated Markov Melody", loc="left", fontweight="bold", color=INK)
    ax.set_xlabel("Beat")
    ax.set_ylabel("Pitch")
    ax.grid(True, alpha=0.20)
    if starts.size:
        ax.set_xlim(0.0, float(starts[-1] + lengths[-1]))
        ax.set_ylim(float(np.min(pitches) - 2), float(np.max(pitches) + 2))
    fig.tight_layout()
    fig.savefig(visual_dir / "generated_piano_roll.png", dpi=160, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_static_panel(midi_files: list[str], summary: dict, generated_rows: list[dict], output_path: Path) -> None:
    counts = _note_counts(midi_files)
    pitches = list(counts.keys())
    pitch_values = [counts[pitch] for pitch in pitches]
    _pitches, note_matrix = _bigram_matrix(midi_files)
    beat_matrix = _beat_position_matrix(midi_files)
    starts, lengths, generated_pitches = _cumulative_beats(generated_rows[:140])

    fig = plt.figure(figsize=(15.5, 9.2), facecolor=PANEL_FACE, constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[0.38, 1.0, 1.05], hspace=0.18, wspace=0.13)

    title_ax = fig.add_subplot(grid[0, :])
    title_ax.axis("off")
    title_ax.text(0.0, 0.68, "Symbolic Music Generation", fontsize=25, fontweight="bold", color=INK)
    title_ax.text(
        0.0,
        0.14,
        "Pitch and rhythm events are counted from MIDI, normalized into Markov transition tables, and sampled into a new melody.",
        fontsize=11,
        color=MUTED,
    )

    pitch_ax = fig.add_subplot(grid[1, 0])
    pitch_ax.bar(pitches, pitch_values, color=BLUE, linewidth=0.0)
    pitch_ax.set_title("Pitch Counts", fontsize=11, loc="left")
    pitch_ax.set_xlabel("Pitch")
    pitch_ax.set_ylabel("Count")
    pitch_ax.grid(axis="y", alpha=0.20)

    note_ax = fig.add_subplot(grid[1, 1])
    note_ax.imshow(note_matrix, cmap="magma", aspect="auto", origin="lower", vmin=0.0)
    note_ax.set_title("Pitch Bigram", fontsize=11, loc="left")
    note_ax.set_xticks([])
    note_ax.set_yticks([])

    beat_ax = fig.add_subplot(grid[1, 2])
    beat_ax.imshow(beat_matrix, cmap="viridis", aspect="auto", origin="lower", vmin=0.0)
    beat_ax.set_title("Beat Length | Position", fontsize=11, loc="left")
    beat_ax.set_xlabel("Position")
    beat_ax.set_yticks(range(len(ALLOWED_BEAT_LENGTHS)))
    beat_ax.set_yticklabels(ALLOWED_BEAT_LENGTHS, fontsize=8)

    summary_ax = fig.add_subplot(grid[1, 3])
    summary_ax.axis("off")
    summary_ax.text(0.0, 0.92, "Corpus", fontsize=12, fontweight="bold", color=INK)
    summary_ax.text(0.0, 0.72, f"Files: {summary['file_count']}", fontsize=10, color=MUTED)
    summary_ax.text(0.0, 0.55, f"Note events: {summary['note_event_count']}", fontsize=10, color=MUTED)
    summary_ax.text(0.0, 0.38, f"Unique pitches: {summary['unique_pitch_count']}", fontsize=10, color=MUTED)
    summary_ax.text(0.0, 0.21, f"Generated notes: {summary['generated_note_count']}", fontsize=10, color=MUTED)

    roll_ax = fig.add_subplot(grid[2, :2])
    for start, length, pitch in zip(starts, lengths, generated_pitches):
        roll_ax.hlines(pitch, start, start + length, color=BLUE, linewidth=2.2, alpha=0.86)
    roll_ax.set_title("Generated Melody", fontsize=11, loc="left")
    roll_ax.set_xlabel("Beat")
    roll_ax.set_ylabel("Pitch")
    roll_ax.grid(True, alpha=0.18)
    if starts.size:
        roll_ax.set_xlim(0.0, float(starts[-1] + lengths[-1]))
        roll_ax.set_ylim(float(np.min(generated_pitches) - 2), float(np.max(generated_pitches) + 2))

    ppl_ax = fig.add_subplot(grid[2, 2:])
    labels = ["Note\nBigram", "Note\nTrigram", "Beat\nBigram", "Beat\nPosition", "Beat\nTrigram"]
    values = [
        summary["mean_note_bigram_perplexity"],
        summary["mean_note_trigram_perplexity"],
        summary["mean_beat_bigram_perplexity"],
        summary["mean_beat_position_perplexity"],
        summary["mean_beat_trigram_perplexity"],
    ]
    ppl_ax.bar(labels, values, color=[BLUE, PURPLE, GOLD, GREEN, ROSE])
    ppl_ax.set_title("Perplexity", fontsize=11, loc="left")
    ppl_ax.grid(axis="y", alpha=0.20)
    for index, value in enumerate(values):
        ppl_ax.text(index, value + 0.04, f"{value:.2f}", ha="center", fontsize=8.5)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=155, facecolor=fig.get_facecolor())
    plt.close(fig)


def _render_animated_panel(
    midi_files: list[str],
    generated_rows: list[dict],
    summary: dict,
    output_path: Path,
    frame_count: int = 36,
) -> None:
    if not generated_rows:
        return

    full_rows = generated_rows[:180]
    visual_files = midi_files[: min(len(midi_files), 400)]
    note_sequences = [note_extraction(midi_file) for midi_file in visual_files]
    all_counts = Counter(note for sequence in note_sequences for note in sequence)
    top_pitches = [pitch for pitch, _count in all_counts.most_common(18)]
    top_pitches = sorted(top_pitches)
    pitch_index = {pitch: index for index, pitch in enumerate(top_pitches)}
    heldout_notes = note_sequences[-1] if note_sequences else []
    frame_targets = np.linspace(max(1, len(visual_files) // frame_count), len(visual_files), frame_count).astype(int)
    pitch_counts: Counter[int] = Counter()
    bigram_counts: dict[int, Counter[int]] = defaultdict(Counter)
    trigram_counts: dict[tuple[int, int], Counter[int]] = defaultdict(Counter)
    loaded_files = 0
    bigram_trace: list[float] = []
    trigram_trace: list[float] = []

    def probability(counter: Counter[int], value: int) -> float | None:
        total = sum(counter.values())
        if total <= 0 or counter.get(value, 0) <= 0:
            return None
        return counter[value] / total

    def transition_probability(table: dict, key: object, value: int) -> float | None:
        if key not in table:
            return None
        return probability(table[key], value)

    def perplexities(notes: list[int]) -> tuple[float, float]:
        if not notes or not pitch_counts:
            return 0.0, 0.0
        eps = 1e-12
        log_bigram = np.log(max(probability(pitch_counts, notes[0]) or eps, eps))
        for previous_note, current_note in zip(notes[:-1], notes[1:]):
            prob = transition_probability(bigram_counts, previous_note, current_note)
            if prob is None:
                prob = probability(pitch_counts, current_note)
            log_bigram += np.log(max(prob or eps, eps))

        log_trigram = np.log(max(probability(pitch_counts, notes[0]) or eps, eps))
        if len(notes) >= 2:
            prob = transition_probability(bigram_counts, notes[0], notes[1])
            if prob is None:
                prob = probability(pitch_counts, notes[1])
            log_trigram += np.log(max(prob or eps, eps))
        for index in range(2, len(notes)):
            prob = transition_probability(trigram_counts, (notes[index - 2], notes[index - 1]), notes[index])
            if prob is None:
                prob = transition_probability(bigram_counts, notes[index - 1], notes[index])
            if prob is None:
                prob = probability(pitch_counts, notes[index])
            log_trigram += np.log(max(prob or eps, eps))
        return float(np.exp(-log_bigram / len(notes))), float(np.exp(-log_trigram / len(notes)))

    ensure_dir(output_path.parent)
    with imageio.get_writer(output_path, mode="I", duration=85, loop=0) as writer:
        for frame_index, target_files in enumerate(frame_targets):
            print(f"[Symbolic Visualiser] Animated frame {frame_index + 1}/{frame_count}", flush=True)
            progress = (frame_index + 1) / frame_count
            while loaded_files < target_files:
                sequence = note_sequences[loaded_files]
                pitch_counts.update(sequence)
                for previous_note, next_note in zip(sequence[:-1], sequence[1:]):
                    bigram_counts[previous_note][next_note] += 1
                for index in range(2, len(sequence)):
                    trigram_counts[(sequence[index - 2], sequence[index - 1])][sequence[index]] += 1
                loaded_files += 1

            bigram_ppl, trigram_ppl = perplexities(heldout_notes)
            bigram_trace.append(bigram_ppl)
            trigram_trace.append(trigram_ppl)

            visible_count = max(1, int(len(full_rows) * progress))
            rows = full_rows[:visible_count]
            starts, lengths, pitches = _cumulative_beats(rows)
            all_starts, all_lengths, all_pitches = _cumulative_beats(full_rows)

            matrix = np.zeros((len(top_pitches), len(top_pitches)), dtype=float)
            for previous_note, next_counts in bigram_counts.items():
                if previous_note not in pitch_index:
                    continue
                total = float(sum(next_counts.values()))
                if total <= 0:
                    continue
                for next_note, count in next_counts.items():
                    if next_note in pitch_index:
                        matrix[pitch_index[previous_note], pitch_index[next_note]] = count / total
            pitch_total = max(1, sum(pitch_counts.values()))
            pitch_probs = [pitch_counts[pitch] / pitch_total for pitch in top_pitches]

            fig = plt.figure(figsize=(13.6, 8.0), facecolor=PANEL_FACE, constrained_layout=True)
            grid = fig.add_gridspec(3, 2, height_ratios=[0.32, 1.0, 1.0], hspace=0.16, wspace=0.14)
            title_ax = fig.add_subplot(grid[0, :])
            title_ax.axis("off")
            title_ax.text(0.0, 0.68, "Symbolic Markov Chain Learning", fontsize=22, fontweight="bold", color=INK)
            title_ax.text(
                0.0,
                0.18,
                f"Training files added: {loaded_files}/{len(visual_files)}. Pitch probabilities, transitions, perplexity, and generated melody update together.",
                fontsize=10,
                color=MUTED,
            )

            pitch_ax = fig.add_subplot(grid[1, 0])
            pitch_ax.bar(top_pitches, pitch_probs, color=BLUE, edgecolor="#1f1f1f", linewidth=0.25)
            pitch_ax.set_title("Evolving Pitch Distribution", loc="left", fontsize=11)
            pitch_ax.set_xlabel("Pitch")
            pitch_ax.set_ylabel("Probability")
            pitch_ax.set_ylim(0.0, max(0.18, max(pitch_probs) * 1.18 if pitch_probs else 0.18))
            pitch_ax.grid(axis="y", alpha=0.20)

            chain_ax = fig.add_subplot(grid[1, 1])
            chain_ax.imshow(matrix, cmap="magma", aspect="auto", origin="lower", vmin=0.0)
            chain_ax.set_title("First-Order Markov Transition Matrix", loc="left", fontsize=11)
            chain_ax.set_xlabel("Next pitch")
            chain_ax.set_ylabel("Previous pitch")
            tick_step = max(1, len(top_pitches) // 8)
            ticks = list(range(0, len(top_pitches), tick_step))
            chain_ax.set_xticks(ticks)
            chain_ax.set_yticks(ticks)
            chain_ax.set_xticklabels([top_pitches[index] for index in ticks], fontsize=8, rotation=45, ha="right")
            chain_ax.set_yticklabels([top_pitches[index] for index in ticks], fontsize=8)

            curve_ax = fig.add_subplot(grid[2, 0])
            x_values = np.arange(1, len(bigram_trace) + 1)
            curve_ax.plot(x_values, bigram_trace, color=BLUE, linewidth=1.8, label="bigram")
            curve_ax.plot(x_values, trigram_trace, color=PURPLE, linewidth=1.8, label="trigram")
            curve_ax.set_title("Perplexity as the Corpus Grows", loc="left", fontsize=11)
            curve_ax.set_xlabel("Animation frame")
            curve_ax.set_ylabel("Held-out perplexity")
            curve_ax.grid(True, alpha=0.20)
            curve_ax.legend(loc="upper right", fontsize=8)
            if len(bigram_trace) >= 2:
                ymax = max(max(bigram_trace), max(trigram_trace)) * 1.08
                curve_ax.set_ylim(0.0, ymax)

            roll_ax = fig.add_subplot(grid[2, 1])
            for start, length, pitch in zip(starts, lengths, pitches):
                roll_ax.hlines(pitch, start, start + length, color=BLUE, linewidth=2.7, alpha=0.88)
            roll_ax.set_title("Generated Melody Reveal", loc="left", fontsize=11)
            roll_ax.set_xlabel("Beat")
            roll_ax.set_ylabel("Pitch")
            roll_ax.grid(True, alpha=0.18)
            roll_ax.set_xlim(0.0, float(all_starts[-1] + all_lengths[-1]))
            roll_ax.set_ylim(float(np.min(all_pitches) - 2), float(np.max(all_pitches) + 2))
            roll_ax.text(
                0.98,
                0.92,
                f"generated notes {visible_count}/{len(full_rows)}\nfinal corpus trigram PPL {summary['mean_note_trigram_perplexity']:.2f}",
                transform=roll_ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                color=INK,
                bbox={"facecolor": "#fffaf2", "edgecolor": "#d8cbb9", "pad": 5},
            )

            buffer = io.BytesIO()
            fig.savefig(buffer, format="png", dpi=90, facecolor=fig.get_facecolor())
            plt.close(fig)
            buffer.seek(0)
            writer.append_data(imageio.imread(buffer))
            plt.close("all")
            gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(description="Render symbolic music-generation visual assets.")
    parser.add_argument("--data-dir", type=Path, default=SYMBOLIC_DATA_ROOT)
    parser.add_argument("--metrics-dir", type=Path, default=SYMBOLIC_METRICS_DIR)
    parser.add_argument("--table-dir", type=Path, default=SYMBOLIC_TABLE_DIR)
    parser.add_argument("--generated-dir", type=Path, default=SYMBOLIC_GENERATED_DIR)
    parser.add_argument("--visual-dir", type=Path, default=VISUAL_SYMBOLIC_DIR)
    parser.add_argument("--readme-dir", type=Path, default=VISUAL_SYMBOLIC_README_DIR)
    parser.add_argument("--generated-length", type=int, default=500)
    parser.add_argument("--max-files", type=int, default=400)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    if args.rebuild or not (args.metrics_dir / "markov_summary.json").exists() or not (args.table_dir / "generated_sequence.csv").exists():
        build_symbolic_outputs(
            data_dir=args.data_dir,
            metrics_dir=args.metrics_dir,
            table_dir=args.table_dir,
            generated_dir=args.generated_dir,
            generated_length=args.generated_length,
        )
    midi_files = find_pdmx_files(args.data_dir)
    if args.max_files is not None and args.max_files > 0:
        midi_files = midi_files[: args.max_files]
    _warm_midi_cache(midi_files)
    visual_dir = ensure_dir(args.visual_dir)
    readme_dir = ensure_dir(args.readme_dir)
    summary = load_json(args.metrics_dir / "markov_summary.json")
    generated_rows = _load_generated_rows(args.table_dir)

    print("[Symbolic Visualiser] Rendering pitch distribution", flush=True)
    _save_pitch_distribution(midi_files, visual_dir)
    print("[Symbolic Visualiser] Rendering note bigram heatmap", flush=True)
    _save_note_bigram_heatmap(midi_files, visual_dir)
    print("[Symbolic Visualiser] Rendering beat-position heatmap", flush=True)
    _save_beat_position_heatmap(midi_files, visual_dir)
    print("[Symbolic Visualiser] Rendering perplexity plot", flush=True)
    _save_perplexity_plot(summary, visual_dir)
    print("[Symbolic Visualiser] Rendering generated piano roll", flush=True)
    _save_generated_piano_roll(generated_rows, visual_dir)
    print("[Symbolic Visualiser] Rendering static README panel", flush=True)
    _render_static_panel(midi_files, summary, generated_rows, readme_dir / "readme_symbolic_static_panel.png")
    print("[Symbolic Visualiser] Rendering animated README panel", flush=True)
    _render_animated_panel(midi_files, generated_rows, summary, readme_dir / "readme_symbolic_animated_panel.gif")
    print(f"[Symbolic Visualiser] Wrote assets under {args.visual_dir.parent}")


if __name__ == "__main__":
    main()
