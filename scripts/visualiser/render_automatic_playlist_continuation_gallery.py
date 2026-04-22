#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT, ensure_dir, load_json, save_json  # noqa: E402


RUN_PREFIX = "full_run_"
K_VALUES = (5, 10, 20, 50, 100)
BG = "#f4f1ea"
PANEL = "#fffdf8"
INK = "#18212b"
MUTED = "#69727c"
CF = "#1f6f8b"
AUDIO = "#d0713f"
GREEN = "#2f855a"
GOLD = "#d6a23d"
PURPLE = "#7557a8"
GRID = "#d8d2c6"
HITMAP_CMAP = LinearSegmentedColormap.from_list("hitmap", ["#f1eee6", "#bee3c3", "#2f855a"])


def _latest_run_dir() -> Path:
    candidates = [
        path
        for path in AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT.glob(f"{RUN_PREFIX}*")
        if path.is_dir()
    ]
    if not candidates:
        return AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def _save_figure(fig: plt.Figure, output_path: Path) -> Path:
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def _time_axis(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.arange(audio.shape[0], dtype=float) / float(sample_rate)


def _downsample(audio: np.ndarray, sample_rate: int, max_points: int = 3200) -> tuple[np.ndarray, np.ndarray]:
    if audio.shape[0] <= max_points:
        return _time_axis(audio, sample_rate), audio
    stride = max(1, audio.shape[0] // max_points)
    indices = np.arange(0, audio.shape[0], stride)
    return indices / float(sample_rate), audio[indices]


def _save_waveform_plot(audio: np.ndarray, sample_rate: int, output_path: Path, title: str, color: str = "#2f6f9f") -> Path:
    audio = np.asarray(audio, dtype=float)
    t, y = _downsample(audio, sample_rate)
    fig, ax = plt.subplots(figsize=(10.5, 4.2), facecolor="#fbfaf7")
    ax.plot(t, y, color=color, linewidth=1.25)
    ax.fill_between(t, y, 0.0, color=color, alpha=0.12)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0.0, max(audio.shape[0] / float(sample_rate), 1e-6))
    ax.set_ylim(-1.1, 1.1)
    _set_axis_style(ax)
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _save_spectrogram_plot(audio: np.ndarray, sample_rate: int, output_path: Path, title: str) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 4.4), facecolor="#fbfaf7")
    ax.specgram(np.asarray(audio, dtype=float), NFFT=2048, Fs=sample_rate, noverlap=1536, cmap="magma")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()
    return _save_figure(fig, output_path)


def _save_simple_waveform_gif(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
    color: str = "#2f6f9f",
    frame_count: int = 32,
) -> Path:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install imageio to render GIFs, or omit --include-gifs.") from exc

    audio = np.asarray(audio, dtype=float)
    total_duration = max(audio.shape[0] / float(sample_rate), 1e-6)
    full_t, full_y = _downsample(audio, sample_rate)
    frames = []
    for frame_idx in range(1, frame_count + 1):
        end = max(1, int(frame_idx / frame_count * audio.shape[0]))
        t, y = _downsample(audio[:end], sample_rate)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#fbfaf7")
        ax.plot(full_t, full_y, color="#d8d0c5", linewidth=1.0, alpha=0.9)
        ax.plot(t, y, color=color, linewidth=1.5)
        ax.fill_between(t, y, 0.0, color=color, alpha=0.14)
        ax.axvline(end / float(sample_rate), color="#1f1f1f", linewidth=1.2, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, total_duration)
        ax.set_ylim(-1.1, 1.1)
        _set_axis_style(ax)
        fig.tight_layout()
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (4,))
        frames.append(frame[:, :, :3].copy())
        plt.close(fig)

    ensure_dir(output_path.parent)
    imageio.mimsave(output_path, frames, duration=0.08, loop=0)
    return output_path


def _set_axis_style(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.grid(True, alpha=0.42, linewidth=0.8, color=GRID)
    for spine in ax.spines.values():
        spine.set_color("#c8c0b4")
        spine.set_alpha(0.7)
    ax.tick_params(colors=INK, labelsize=9.5)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)
    ax.title.set_color(INK)


def _clean_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _card(ax: plt.Axes, title: str, value: str, subtitle: str, color: str) -> None:
    _clean_axis(ax)
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            linewidth=1.0,
            edgecolor="#ded6c9",
            facecolor=PANEL,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.add_patch(
        patches.FancyBboxPatch(
            (0.0,
             0.0),
            0.025,
            1.0,
            boxstyle="round,pad=0.0,rounding_size=0.035",
            linewidth=0.0,
            facecolor=color,
            transform=ax.transAxes,
            clip_on=False,
        )
    )
    ax.text(0.08, 0.78, title.upper(), transform=ax.transAxes, color=MUTED, fontsize=8.5, fontweight="bold")
    ax.text(0.08, 0.42, value, transform=ax.transAxes, color=INK, fontsize=21, fontweight="bold", va="center")
    ax.text(0.08, 0.14, subtitle, transform=ax.transAxes, color=MUTED, fontsize=8.8, va="bottom")


def _format_int(value: object) -> str:
    return f"{int(_as_float(value)):,}"


def _format_pct(value: object) -> str:
    return f"{_as_float(value) * 100.0:.1f}%"


def _metric(metrics: dict, key: str) -> float:
    return _as_float(metrics.get(key), np.nan)


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _curve_records(run_dir: Path) -> list[dict]:
    curve_path = run_dir / "metrics" / "training_validation_curve.json"
    if curve_path.exists():
        payload = load_json(curve_path)
        return list(payload.get("records", []))

    history_path = run_dir / "models" / "wrmf" / "history.json"
    if not history_path.exists():
        return []
    payload = load_json(history_path)
    records = []
    for row in payload.get("history", []):
        epoch = int(row.get("epoch", len(records))) + 1
        records.append({"epoch": epoch, "train_loss": _as_float(row.get("loss"))})
    return records


def _save_executive_dashboard(run_dir: Path, visual_dir: Path) -> Path | None:
    summary = _optional_json(run_dir / "metrics" / "playlist_continuation_summary.json")
    embedding = _optional_json(run_dir / "metrics" / "embedding_summary.json")
    cf = _optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    records = _curve_records(run_dir)
    if not any([summary, embedding, cf, audio, records]):
        return None

    fig = plt.figure(figsize=(18, 11.2), facecolor=BG)
    gs = fig.add_gridspec(5, 12, height_ratios=[0.9, 1.25, 2.25, 2.25, 2.45], hspace=0.62, wspace=0.45)

    title_ax = fig.add_subplot(gs[0, :])
    _clean_axis(title_ax)
    title_ax.text(0.0, 0.66, "Automatic Playlist Continuation", color=INK, fontsize=27, fontweight="bold", transform=title_ax.transAxes)
    title_ax.text(
        0.0,
        0.24,
        "Full-run dashboard: data coverage, WRMF training behavior, recommendation quality, and synthesis evidence.",
        color=MUTED,
        fontsize=12.5,
        transform=title_ax.transAxes,
    )
    title_ax.text(
        1.0,
        0.42,
        run_dir.name,
        ha="right",
        va="center",
        color=MUTED,
        fontsize=11,
        transform=title_ax.transAxes,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": PANEL, "edgecolor": "#ded6c9"},
    )

    train = summary.get("train", {})
    coverage = summary.get("query_target_train_coverage", {})
    best_record = max(records, key=lambda row: _as_float(row.get("validation_accuracy_at_10"))) if records else {}
    cards = [
        ("Training playlists", _format_int(train.get("playlists")), f"{_format_int(train.get('track_rows'))} track rows", CF),
        ("Embedding coverage", _format_pct(_as_float(embedding.get("selected_files_present")) / max(_as_float(embedding.get("requested_track_ids")), 1.0)), f"{_format_int(embedding.get('selected_files_present'))} files present", GREEN),
        ("CF Hit@10", _format_pct(cf.get("hit_rate_at_10")), f"MRR {_metric(cf, 'mrr'):.3f}", GOLD),
        ("Best epoch", str(int(best_record.get("epoch", 0))) if best_record else "n/a", f"validation accuracy {_format_pct(best_record.get('validation_accuracy_at_10', 0.0))}", PURPLE),
    ]
    for idx, (label, value, subtitle, color) in enumerate(cards):
        _card(fig.add_subplot(gs[1, idx * 3 : (idx + 1) * 3]), label, value, subtitle, color)

    ax_loss = fig.add_subplot(gs[2, :5])
    if records:
        epochs = [int(row["epoch"]) for row in records]
        losses = [_as_float(row.get("train_loss", row.get("loss"))) for row in records]
        ax_loss.plot(epochs, losses, color=CF, linewidth=2.8, marker="o", markersize=5.5)
        ax_loss.fill_between(epochs, losses, min(losses), color=CF, alpha=0.11)
        ax_loss.text(0.04, 0.88, f"loss drop {(1.0 - losses[-1] / max(losses[0], 1e-9)) * 100.0:.1f}%", transform=ax_loss.transAxes, fontsize=10, color=INK, bbox={"boxstyle": "round,pad=0.32", "facecolor": "#eef5f6", "edgecolor": "#c6d9df"})
    ax_loss.set_title("Training Convergence", loc="left", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("WRMF loss")
    ax_loss.set_ylim(bottom=0)
    _set_axis_style(ax_loss)

    ax_metrics = fig.add_subplot(gs[2, 5:9])
    metric_names = ["hit_rate_at_10", "target_precision_at_10", "ndcg_at_10", "mrr"]
    labels = ["Hit@10", "Target P@10", "NDCG@10", "MRR"]
    y = np.arange(len(labels))
    cf_values = [_metric(cf, key) for key in metric_names]
    audio_values = [_metric(audio, key) for key in metric_names]
    ax_metrics.barh(y + 0.18, cf_values, 0.34, color=CF, label="CF")
    ax_metrics.barh(y - 0.18, audio_values, 0.34, color=AUDIO, label="Audio")
    ax_metrics.set_yticks(y)
    ax_metrics.set_yticklabels(labels)
    ax_metrics.set_xlim(0.0, max(0.52, np.nanmax(cf_values + audio_values) * 1.18))
    ax_metrics.set_title("Model Comparison", loc="left", fontweight="bold")
    ax_metrics.legend(loc="lower right")
    _set_axis_style(ax_metrics)

    ax_cov = fig.add_subplot(gs[2, 9:])
    cov_labels = ["Query", "Target"]
    cov_values = [_metric(coverage, "query_known_rate"), _metric(coverage, "target_known_rate")]
    bars = ax_cov.bar(cov_labels, cov_values, color=[GREEN, PURPLE], width=0.58)
    ax_cov.set_ylim(0.0, 1.05)
    ax_cov.set_title("Known Track Coverage", loc="left", fontweight="bold")
    ax_cov.set_ylabel("Rate")
    for bar, value in zip(bars, cov_values):
        ax_cov.text(bar.get_x() + bar.get_width() / 2.0, value + 0.025, f"{value:.1%}", ha="center", va="bottom", fontsize=10, color=INK)
    _set_axis_style(ax_cov)

    ax_depth = fig.add_subplot(gs[3, :7])
    for label, metrics, color, marker in [("Collaborative filtering", cf, CF, "o"), ("Audio similarity", audio, AUDIO, "s")]:
        values = [_metric(metrics, f"hit_rate_at_{k}") for k in K_VALUES]
        ax_depth.plot(K_VALUES, values, marker=marker, linewidth=2.8, markersize=6, color=color, label=label)
    ax_depth.set_title("Recall Opportunity by Recommendation Depth", loc="left", fontweight="bold")
    ax_depth.set_xlabel("Recommendation depth k")
    ax_depth.set_ylabel("Hit rate")
    ax_depth.set_xticks(K_VALUES)
    ax_depth.set_ylim(0.0, 1.02)
    ax_depth.legend(loc="lower right")
    _set_axis_style(ax_depth)

    ax_sample = fig.add_subplot(gs[3, 7:])
    cf_rows = _preview_rows(run_dir / "rankings" / "collaborative_filtering_preview.csv", limit=18)
    matrix = _hit_matrix(cf_rows)
    if matrix.size:
        ax_sample.imshow(matrix, aspect="auto", interpolation="nearest", cmap=HITMAP_CMAP, vmin=0.0, vmax=1.0)
    ax_sample.set_title("CF Top-10 Hits, Preview Playlists", loc="left", fontweight="bold")
    ax_sample.set_xlabel("Rank position")
    ax_sample.set_ylabel("Playlist")
    ax_sample.set_xticks(range(10))
    ax_sample.set_xticklabels([str(i) for i in range(1, 11)])
    ax_sample.set_yticks(range(len(cf_rows)))
    ax_sample.set_yticklabels([row["playlist_id"] for row in cf_rows], fontsize=8)
    for spine in ax_sample.spines.values():
        spine.set_color("#c8c0b4")

    ax_dist = fig.add_subplot(gs[4, :5])
    bins = np.asarray([1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 20000], dtype=float)
    cf_ranks = _first_ranks(cf)
    audio_ranks = _first_ranks(audio)
    if cf_ranks:
        ax_dist.hist(cf_ranks, bins=bins, color=CF, alpha=0.68, label="CF")
    if audio_ranks:
        ax_dist.hist(audio_ranks, bins=bins, color=AUDIO, alpha=0.58, label="Audio")
    ax_dist.set_xscale("log")
    ax_dist.set_title("First Relevant Recommendation Rank", loc="left", fontweight="bold")
    ax_dist.set_xlabel("Rank, log scale")
    ax_dist.set_ylabel("Playlist count")
    ax_dist.legend(loc="best")
    _set_axis_style(ax_dist)

    ax_embed = fig.add_subplot(gs[4, 5:8])
    embed_values = [
        _as_float(embedding.get("requested_track_ids")),
        _as_float(embedding.get("selected_files_present")),
        _as_float(embedding.get("extracted_files")),
    ]
    ax_embed.bar(["Requested", "Present", "Extracted"], embed_values, color=[MUTED, GREEN, AUDIO])
    ax_embed.set_title("Embedding Files", loc="left", fontweight="bold")
    ax_embed.set_ylabel("Count")
    ax_embed.tick_params(axis="x", rotation=14)
    _set_axis_style(ax_embed)

    ax_note = fig.add_subplot(gs[4, 8:])
    _clean_axis(ax_note)
    final = records[-1] if records else {}
    note_lines = [
        "Run interpretation",
        f"Final loss: {_as_float(final.get('train_loss')):.4f}",
        f"Final CF MRR: {_metric(cf, 'mrr'):.4f}",
        f"Final CF Hit@10: {_format_pct(cf.get('hit_rate_at_10'))}",
        f"Audio baseline Hit@10: {_format_pct(audio.get('hit_rate_at_10'))}",
        f"Query known rate: {_format_pct(coverage.get('query_known_rate'))}",
    ]
    ax_note.add_patch(
        patches.FancyBboxPatch(
            (0.0, 0.0),
            1.0,
            1.0,
            boxstyle="round,pad=0.022,rounding_size=0.035",
            linewidth=1.0,
            edgecolor="#ded6c9",
            facecolor=PANEL,
            transform=ax_note.transAxes,
        )
    )
    ax_note.text(0.08, 0.84, note_lines[0], transform=ax_note.transAxes, fontsize=14, color=INK, fontweight="bold")
    for index, line in enumerate(note_lines[1:]):
        ax_note.text(0.08, 0.68 - index * 0.12, line, transform=ax_note.transAxes, fontsize=11, color=INK)

    return _save_figure(fig, visual_dir / "apc_full_run_dashboard.png")


def _save_training_diagnostics(run_dir: Path, visual_dir: Path) -> Path | None:
    records = _curve_records(run_dir)
    if not records:
        return None

    epochs = np.asarray([int(row["epoch"]) for row in records], dtype=float)
    losses = np.asarray([_as_float(row.get("train_loss", row.get("loss"))) for row in records], dtype=float)
    acc = np.asarray([_as_float(row.get("validation_accuracy_at_10"), np.nan) for row in records], dtype=float)
    precision = np.asarray([_as_float(row.get("validation_precision_at_10"), np.nan) for row in records], dtype=float)
    mrr = np.asarray([_as_float(row.get("validation_mrr"), np.nan) for row in records], dtype=float)
    ndcg = np.asarray([_as_float(row.get("validation_ndcg_at_10"), np.nan) for row in records], dtype=float)
    best_idx = int(np.nanargmax(acc)) if not np.all(np.isnan(acc)) else len(records) - 1

    fig = plt.figure(figsize=(15, 9), facecolor=BG)
    gs = fig.add_gridspec(3, 8, height_ratios=[0.55, 2.0, 1.8], hspace=0.46, wspace=0.45)
    title_ax = fig.add_subplot(gs[0, :])
    _clean_axis(title_ax)
    title_ax.text(0.0, 0.6, "WRMF Training Diagnostics", fontsize=23, fontweight="bold", color=INK, transform=title_ax.transAxes)
    title_ax.text(0.0, 0.18, "Loss converges quickly; validation quality is tracked separately to expose the useful epoch.", fontsize=11.5, color=MUTED, transform=title_ax.transAxes)

    ax_loss = fig.add_subplot(gs[1, :5])
    ax_loss.plot(epochs, losses, linewidth=3.0, marker="o", color=CF)
    ax_loss.fill_between(epochs, losses, losses.min(), color=CF, alpha=0.12)
    ax_loss.scatter([epochs[best_idx]], [losses[best_idx]], s=140, color=GOLD, edgecolor=INK, linewidth=0.9, zorder=5)
    ax_loss.annotate(
        f"best validation epoch {int(epochs[best_idx])}",
        xy=(epochs[best_idx], losses[best_idx]),
        xytext=(epochs[best_idx] + 0.5, losses[best_idx] + max(losses) * 0.12),
        arrowprops={"arrowstyle": "->", "color": INK, "linewidth": 1.2},
        fontsize=10,
        color=INK,
    )
    ax_loss.set_title("Training Loss Trajectory", loc="left", fontweight="bold")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.set_ylim(bottom=0)
    _set_axis_style(ax_loss)

    ax_cards = fig.add_subplot(gs[1, 5:])
    _clean_axis(ax_cards)
    card_values = [
        ("Start loss", f"{losses[0]:.3f}", CF),
        ("Final loss", f"{losses[-1]:.3f}", GREEN),
        ("Loss reduction", f"{(1.0 - losses[-1] / max(losses[0], 1e-9)) * 100.0:.1f}%", GOLD),
        ("Best Hit@10", f"{acc[best_idx] * 100.0:.1f}%", PURPLE),
    ]
    for idx, (label, value, color) in enumerate(card_values):
        y = 0.78 - idx * 0.22
        ax_cards.add_patch(patches.FancyBboxPatch((0.04, y - 0.11), 0.9, 0.16, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor=PANEL, edgecolor="#ded6c9", transform=ax_cards.transAxes))
        ax_cards.add_patch(patches.Rectangle((0.04, y - 0.11), 0.025, 0.16, color=color, transform=ax_cards.transAxes))
        ax_cards.text(0.10, y + 0.005, label, color=MUTED, fontsize=9.5, transform=ax_cards.transAxes)
        ax_cards.text(0.64, y + 0.005, value, color=INK, fontsize=15, fontweight="bold", ha="right", transform=ax_cards.transAxes)

    ax_val = fig.add_subplot(gs[2, :])
    for label, values, color, marker in [
        ("Hit@10", acc, GREEN, "o"),
        ("Target precision@10", precision, AUDIO, "s"),
        ("MRR", mrr, PURPLE, "^"),
        ("NDCG@10", ndcg, "#1697a6", "D"),
    ]:
        ax_val.plot(epochs, values, linewidth=2.6, marker=marker, markersize=5.8, label=label, color=color)
    ax_val.axvline(epochs[best_idx], color=GOLD, linewidth=2.0, alpha=0.75)
    ax_val.text(epochs[best_idx] + 0.08, max(0.05, acc[best_idx]), f"best Hit@10 {acc[best_idx]:.2f}", color=INK, fontsize=10, va="bottom")
    ax_val.set_title("Validation Metrics per Epoch", loc="left", fontweight="bold")
    ax_val.set_xlabel("Epoch")
    ax_val.set_ylabel("Score")
    ax_val.set_ylim(0.0, max(0.55, np.nanmax([acc, precision, mrr, ndcg]) * 1.2))
    ax_val.legend(loc="upper right", ncols=4)
    _set_axis_style(ax_val)
    return _save_figure(fig, visual_dir / "training_diagnostics_showcase.png")


def _save_training_validation_curve(run_dir: Path, visual_dir: Path) -> Path | None:
    records = _curve_records(run_dir)
    if not records:
        return None

    epochs = [int(row["epoch"]) for row in records]
    train_loss = [_as_float(row.get("train_loss", row.get("loss"))) for row in records]
    accuracy = [_as_float(row.get("validation_accuracy_at_10"), np.nan) for row in records]
    precision = [_as_float(row.get("validation_precision_at_10"), np.nan) for row in records]
    mrr = [_as_float(row.get("validation_mrr"), np.nan) for row in records]
    ndcg = [_as_float(row.get("validation_ndcg_at_10"), np.nan) for row in records]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True, facecolor="#fbfaf7")
    axes[0].plot(epochs, train_loss, color="#1f77b4", marker="o", linewidth=2.4)
    axes[0].set_title("WRMF Training Loss", loc="left", fontweight="bold")
    axes[0].set_ylabel("Loss")
    axes[0].set_ylim(bottom=0)
    _set_axis_style(axes[0])

    plotted_validation = False
    series = [
        ("Validation accuracy@10", accuracy, "#2ca02c", "o"),
        ("Target precision@10", precision, "#ff7f0e", "s"),
        ("MRR", mrr, "#9467bd", "^"),
        ("NDCG@10", ndcg, "#17becf", "D"),
    ]
    for label, values, color, marker in series:
        array = np.asarray(values, dtype=float)
        if not np.all(np.isnan(array)):
            axes[1].plot(epochs, values, color=color, marker=marker, linewidth=2.0, label=label)
            plotted_validation = True
    axes[1].set_title("Validation Recommendation Quality", loc="left", fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_ylim(0.0, 1.0 if not plotted_validation else max(0.5, float(np.nanmax([accuracy, precision, mrr, ndcg])) * 1.18))
    _set_axis_style(axes[1])
    if plotted_validation:
        axes[1].legend(loc="best")

    fig.suptitle("Automatic Playlist Continuation: Training and Validation Curves", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, visual_dir / "training_loss_validation_accuracy.png")


def _save_metric_comparison(run_dir: Path, visual_dir: Path) -> Path | None:
    cf = _optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    if not cf and not audio:
        return None

    metric_defs = [
        ("mrr", "MRR"),
        ("hit_rate_at_10", "Hit@10"),
        ("target_precision_at_10", "Target Precision@10"),
        ("ndcg_at_10", "NDCG@10"),
        ("map_at_10", "MAP@10"),
    ]
    labels = [label for _, label in metric_defs]
    cf_values = [_as_float(cf.get(key)) for key, _ in metric_defs]
    audio_values = [_as_float(audio.get(key)) for key, _ in metric_defs]
    x = np.arange(len(labels), dtype=float)
    width = 0.36

    fig, ax = plt.subplots(figsize=(10.4, 5.6), facecolor="#fbfaf7")
    ax.bar(x - width / 2.0, cf_values, width, label="Collaborative filtering", color="#2f6f9f")
    ax.bar(x + width / 2.0, audio_values, width, label="Audio similarity", color="#d97941")
    ax.set_title("Recommender Metric Comparison", loc="left", fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylim(0.0, max(0.25, max(cf_values + audio_values) * 1.25))
    ax.legend(loc="best")
    _set_axis_style(ax)
    for index, value in enumerate(cf_values):
        ax.text(index - width / 2.0, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    for index, value in enumerate(audio_values):
        ax.text(index + width / 2.0, value + 0.006, f"{value:.3f}", ha="center", va="bottom", fontsize=8.5)
    fig.tight_layout()
    return _save_figure(fig, visual_dir / "recommender_metric_comparison.png")


def _save_retrieval_depth_curves(run_dir: Path, visual_dir: Path) -> Path | None:
    cf = _optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    if not cf and not audio:
        return None

    panels = [
        ("hit_rate_at_{k}", "Hit Rate", "Higher means at least one target appears by depth k."),
        ("target_precision_at_{k}", "Target Precision", "Hits normalized by target count."),
        ("ndcg_at_{k}", "NDCG", "Ranking quality with earlier hits weighted more."),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), facecolor="#fbfaf7")
    for ax, (pattern, title, detail) in zip(axes, panels):
        cf_values = [_as_float(cf.get(pattern.format(k=k)), np.nan) for k in K_VALUES]
        audio_values = [_as_float(audio.get(pattern.format(k=k)), np.nan) for k in K_VALUES]
        ax.plot(K_VALUES, cf_values, marker="o", linewidth=2.2, color="#2f6f9f", label="CF")
        ax.plot(K_VALUES, audio_values, marker="s", linewidth=2.2, color="#d97941", label="Audio")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.0, -0.25, detail, transform=ax.transAxes, fontsize=8.5, color="#555555")
        ax.set_xlabel("Recommendation depth k")
        ax.set_ylabel("Score")
        ax.set_xticks(K_VALUES)
        ax.set_ylim(0.0, 1.02)
        _set_axis_style(ax)
        ax.legend(loc="best")
    fig.suptitle("Retrieval Quality as Recommendation Depth Increases", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save_figure(fig, visual_dir / "retrieval_depth_curves.png")


def _save_dataset_coverage_panel(run_dir: Path, visual_dir: Path) -> Path | None:
    summary = _optional_json(run_dir / "metrics" / "playlist_continuation_summary.json")
    embedding = _optional_json(run_dir / "metrics" / "embedding_summary.json")
    if not summary and not embedding:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), facecolor="#fbfaf7")
    train = summary.get("train", {})
    test = summary.get("test", {})
    collection_labels = ["playlists", "track_rows", "unique_tracks"]
    x = np.arange(len(collection_labels), dtype=float)
    axes[0, 0].bar(x - 0.18, [_as_float(train.get(key)) for key in collection_labels], 0.36, label="Train", color="#2f6f9f")
    axes[0, 0].bar(x + 0.18, [_as_float(test.get(key)) for key in collection_labels], 0.36, label="Test", color="#d97941")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(["Playlists", "Rows", "Unique TIDs"])
    axes[0, 0].set_title("Playlist Collection Scale", loc="left", fontweight="bold")
    axes[0, 0].set_ylabel("Count (log scale)")
    axes[0, 0].legend(loc="best")
    _set_axis_style(axes[0, 0])

    positive = _as_float(summary.get("positive_rows"))
    negative = _as_float(summary.get("negative_rows"))
    axes[0, 1].bar(["Positive", "Negative"], [positive, negative], color=["#2f6f9f", "#d97941"])
    axes[0, 1].set_title("Interaction Sample Balance", loc="left", fontweight="bold")
    axes[0, 1].set_ylabel("Rows")
    _set_axis_style(axes[0, 1])
    for idx, value in enumerate([positive, negative]):
        axes[0, 1].text(idx, value * 1.01, f"{int(value):,}", ha="center", va="bottom", fontsize=9)

    coverage = summary.get("query_target_train_coverage", {})
    coverage_labels = ["Query known", "Target known"]
    coverage_values = [
        _as_float(coverage.get("query_known_rate")),
        _as_float(coverage.get("target_known_rate")),
    ]
    axes[1, 0].bar(coverage_labels, coverage_values, color=["#5aa469", "#8f6bb8"])
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 0].set_title("Test Track Coverage in Training Catalog", loc="left", fontweight="bold")
    axes[1, 0].set_ylabel("Known rate")
    _set_axis_style(axes[1, 0])
    for idx, value in enumerate(coverage_values):
        axes[1, 0].text(idx, value + 0.02, f"{value:.1%}", ha="center", va="bottom", fontsize=9)

    embedding_labels = ["Requested", "Selected", "Present", "Extracted"]
    embedding_values = [
        _as_float(embedding.get("requested_track_ids")),
        _as_float(embedding.get("selected_embedding_files")),
        _as_float(embedding.get("selected_files_present")),
        _as_float(embedding.get("extracted_files")),
    ]
    axes[1, 1].bar(embedding_labels, embedding_values, color=["#707070", "#2f6f9f", "#5aa469", "#d97941"])
    axes[1, 1].set_title("Audio Embedding Availability", loc="left", fontweight="bold")
    axes[1, 1].set_ylabel("Files")
    axes[1, 1].tick_params(axis="x", rotation=12)
    _set_axis_style(axes[1, 1])
    for idx, value in enumerate(embedding_values):
        axes[1, 1].text(idx, value * 1.01 if value else 0.02, f"{int(value):,}", ha="center", va="bottom", fontsize=8.5)

    fig.suptitle("Automatic Playlist Continuation Dataset and Coverage", fontsize=14, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save_figure(fig, visual_dir / "dataset_embedding_coverage.png")


def _first_ranks(metrics: dict) -> list[float]:
    rows = metrics.get("per_playlist", [])
    ranks = []
    for row in rows:
        rank = row.get("first_relevant_rank")
        if rank is not None:
            ranks.append(float(rank))
    return ranks


def _save_first_relevant_rank_histogram(run_dir: Path, visual_dir: Path) -> Path | None:
    cf = _optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    cf_ranks = _first_ranks(cf)
    audio_ranks = _first_ranks(audio)
    if not cf_ranks and not audio_ranks:
        return None

    bins = np.asarray([1, 2, 5, 10, 20, 50, 100, 500, 1000, 5000, 20000], dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="#fbfaf7")
    if cf_ranks:
        ax.hist(cf_ranks, bins=bins, alpha=0.65, color="#2f6f9f", label="Collaborative filtering")
    if audio_ranks:
        ax.hist(audio_ranks, bins=bins, alpha=0.65, color="#d97941", label="Audio similarity")
    ax.set_xscale("log")
    ax.set_xlabel("First relevant rank (log scale)")
    ax.set_ylabel("Playlist count")
    ax.set_title("Where the First Correct Recommendation Appears", loc="left", fontweight="bold")
    ax.legend(loc="best")
    _set_axis_style(ax)
    fig.tight_layout()
    return _save_figure(fig, visual_dir / "first_relevant_rank_histogram.png")


def _parse_list_cell(value: str) -> list[str]:
    if not value:
        return []
    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        return []
    if not isinstance(parsed, list):
        return []
    return [str(item) for item in parsed]


def _preview_rows(path: Path, limit: int = 24) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "playlist_id": str(row.get("playlist_id", "")),
                    "top_10": _parse_list_cell(row.get("top_10", "")),
                    "targets": _parse_list_cell(row.get("targets", "")),
                }
            )
            if len(rows) >= limit:
                break
    return rows


def _hit_matrix(rows: list[dict]) -> np.ndarray:
    matrix = np.zeros((len(rows), 10), dtype=float)
    for row_idx, row in enumerate(rows):
        targets = set(row["targets"])
        for col_idx, tid in enumerate(row["top_10"][:10]):
            if tid in targets:
                matrix[row_idx, col_idx] = 1.0
    return matrix


def _save_ranking_hitmap(run_dir: Path, visual_dir: Path) -> Path | None:
    cf_rows = _preview_rows(run_dir / "rankings" / "collaborative_filtering_preview.csv")
    audio_rows = _preview_rows(run_dir / "rankings" / "audio_similarity_preview.csv")
    if not cf_rows and not audio_rows:
        return None

    panels = [
        ("Collaborative Filtering", cf_rows, "#2f6f9f"),
        ("Audio Similarity", audio_rows, "#d97941"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.2), facecolor="#fbfaf7", sharex=True)
    for ax, (title, rows, _color) in zip(axes, panels):
        matrix = _hit_matrix(rows)
        if matrix.size:
            ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
            hit_rows = int(np.sum(np.any(matrix > 0, axis=1)))
        else:
            matrix = np.zeros((1, 10), dtype=float)
            ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="Greens", vmin=0.0, vmax=1.0)
            hit_rows = 0
        ax.set_title(f"{title}\n{hit_rows}/{len(rows)} playlists hit in top 10", loc="left", fontweight="bold")
        ax.set_xlabel("Recommendation position")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["playlist_id"] for row in rows], fontsize=8)
        ax.set_ylabel("Playlist ID")
        for spine in ax.spines.values():
            spine.set_alpha(0.25)
    fig.suptitle("Top-10 Recommendation Hit Map", fontsize=14, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save_figure(fig, visual_dir / "top10_recommendation_hitmap.png")


def _save_recommender_storyboard(run_dir: Path, visual_dir: Path) -> Path | None:
    cf = _optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    cf_rows = _preview_rows(run_dir / "rankings" / "collaborative_filtering_preview.csv", limit=18)
    audio_rows = _preview_rows(run_dir / "rankings" / "audio_similarity_preview.csv", limit=18)
    if not cf and not audio and not cf_rows and not audio_rows:
        return None

    fig = plt.figure(figsize=(18.4, 10.2), facecolor=BG)
    gs = fig.add_gridspec(3, 12, height_ratios=[0.55, 2.35, 2.35], hspace=0.62, wspace=0.95)
    title_ax = fig.add_subplot(gs[0, :])
    _clean_axis(title_ax)
    title_ax.text(0.0, 0.58, "Recommendation Quality Storyboard", color=INK, fontsize=23, fontweight="bold", transform=title_ax.transAxes)
    title_ax.text(0.0, 0.16, "The collaborative model finds useful continuations much earlier than the audio-only similarity baseline.", color=MUTED, fontsize=11.5, transform=title_ax.transAxes)

    metric_names = [
        ("hit_rate_at_10", "Hit@10"),
        ("target_precision_at_10", "Target P@10"),
        ("mrr", "MRR"),
        ("ndcg_at_10", "NDCG@10"),
    ]
    for idx, (metrics, label, color) in enumerate([(cf, "Collaborative filtering", CF), (audio, "Audio similarity", AUDIO)]):
        ax = fig.add_subplot(gs[1, idx * 3 : (idx + 1) * 3])
        _clean_axis(ax)
        ax.add_patch(patches.FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.02,rounding_size=0.035", facecolor=PANEL, edgecolor="#ded6c9", transform=ax.transAxes))
        ax.text(0.08, 0.84, label, transform=ax.transAxes, fontsize=14, fontweight="bold", color=INK)
        for row_idx, (key, name) in enumerate(metric_names):
            value = _metric(metrics, key)
            y = 0.66 - row_idx * 0.15
            ax.text(0.08, y, name, transform=ax.transAxes, color=MUTED, fontsize=10)
            ax.text(0.86, y, f"{value:.3f}", transform=ax.transAxes, color=color, fontsize=14, fontweight="bold", ha="right")
            ax.add_patch(patches.Rectangle((0.08, y - 0.055), 0.78, 0.025, facecolor="#eee7dc", edgecolor="none", transform=ax.transAxes))
            ax.add_patch(patches.Rectangle((0.08, y - 0.055), min(max(value / 0.5, 0.0), 1.0) * 0.78, 0.025, facecolor=color, edgecolor="none", transform=ax.transAxes))

    ax_gain = fig.add_subplot(gs[1, 6:9])
    gains = []
    gain_labels = []
    compact_labels = {"hit_rate_at_10": "Hit", "target_precision_at_10": "TP", "mrr": "MRR", "ndcg_at_10": "NDCG"}
    for key, _label in metric_names:
        base = max(_metric(audio, key), 1e-9)
        gains.append(_metric(cf, key) / base)
        gain_labels.append(compact_labels[key])
    ax_gain.barh(np.arange(len(gain_labels)), gains, color=GOLD)
    ax_gain.axvline(1.0, color=INK, linewidth=1.1, alpha=0.7)
    ax_gain.set_yticks(np.arange(len(gain_labels)))
    ax_gain.set_yticklabels(gain_labels)
    ax_gain.set_xlabel("CF / audio score ratio")
    ax_gain.set_title("Lift Over Audio Baseline", loc="left", fontweight="bold")
    ax_gain.set_xlim(0.0, max(gains) * 1.20)
    for idx, value in enumerate(gains):
        ax_gain.text(value + max(gains) * 0.025, idx, f"{value:.1f}x", va="center", color=INK, fontsize=9.5)
    _set_axis_style(ax_gain)

    ax_depth = fig.add_subplot(gs[1, 9:])
    for label, metrics, color, marker in [("CF", cf, CF, "o"), ("Audio", audio, AUDIO, "s")]:
        ax_depth.plot(K_VALUES, [_metric(metrics, f"target_precision_at_{k}") for k in K_VALUES], color=color, marker=marker, linewidth=2.5, label=label)
    ax_depth.set_title("Target Recall by Depth", loc="left", fontweight="bold")
    ax_depth.set_xscale("log")
    ax_depth.set_xlim(4.5, 115)
    ax_depth.set_xlabel("k")
    ax_depth.set_ylabel("score")
    ax_depth.set_xticks(K_VALUES)
    ax_depth.set_xticklabels([str(k) for k in K_VALUES])
    ax_depth.set_ylim(0.0, 1.02)
    ax_depth.legend(loc="lower right")
    _set_axis_style(ax_depth)

    for idx, (rows, label, color) in enumerate([(cf_rows, "Collaborative filtering top-10 hits", CF), (audio_rows, "Audio similarity top-10 hits", AUDIO)]):
        ax = fig.add_subplot(gs[2, idx * 6 : (idx + 1) * 6])
        matrix = _hit_matrix(rows)
        if matrix.size:
            row_order = sorted(range(matrix.shape[0]), key=lambda row: (-int(np.sum(matrix[row])), np.argmax(matrix[row] > 0) if np.any(matrix[row] > 0) else 99, int(rows[row]["playlist_id"]) if rows[row]["playlist_id"].isdigit() else row))
            matrix = matrix[row_order]
            rows = [rows[row] for row in row_order]
            ax.imshow(matrix, aspect="auto", cmap=HITMAP_CMAP, interpolation="nearest", vmin=0.0, vmax=1.0)
        ax.set_title(label, loc="left", fontweight="bold")
        ax.set_xlabel("Recommendation position")
        ax.set_ylabel("Playlist")
        ax.set_xticks(range(10))
        ax.set_xticklabels([str(i) for i in range(1, 11)])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([row["playlist_id"] for row in rows], fontsize=7.8)
        for spine in ax.spines.values():
            spine.set_color("#c8c0b4")
        ax.text(0.985, 1.02, "green cells are correct held-out tracks", transform=ax.transAxes, ha="right", va="bottom", color=MUTED, fontsize=8.5)

    return _save_figure(fig, visual_dir / "recommender_storyboard.png")


def _save_recommendation_examples(run_dir: Path, visual_dir: Path) -> Path | None:
    cf_rows = _preview_rows(run_dir / "rankings" / "collaborative_filtering_preview.csv", limit=14)
    if not cf_rows:
        return None

    scored = []
    for row in cf_rows:
        hits = [tid in set(row["targets"]) for tid in row["top_10"][:10]]
        first_hit = hits.index(True) + 1 if any(hits) else 99
        scored.append((sum(hits), first_hit, row, hits))
    scored.sort(key=lambda item: (-item[0], item[1], int(item[2]["playlist_id"]) if item[2]["playlist_id"].isdigit() else 0))
    selected = scored[:10]

    fig, ax = plt.subplots(figsize=(16, 8.6), facecolor=BG)
    _clean_axis(ax)
    ax.text(0.02, 0.94, "Recommendation Examples: Top-10 CF Continuations", transform=ax.transAxes, fontsize=22, fontweight="bold", color=INK)
    ax.text(0.02, 0.895, "Each row is one validation playlist. Green ranks are held-out target tracks recovered by the recommender.", transform=ax.transAxes, fontsize=11.5, color=MUTED)

    left = 0.045
    top = 0.80
    row_h = 0.065
    box_w = 0.065
    gap = 0.007
    label_w = 0.13
    for pos in range(10):
        ax.text(left + label_w + pos * (box_w + gap) + box_w / 2.0, top + 0.055, str(pos + 1), ha="center", va="bottom", transform=ax.transAxes, color=MUTED, fontsize=9.5, fontweight="bold")

    for idx, (hit_count, first_hit, row, hits) in enumerate(selected):
        y = top - idx * row_h
        ax.text(left, y + 0.018, f"Playlist {row['playlist_id']}", transform=ax.transAxes, color=INK, fontsize=10.5, fontweight="bold")
        ax.text(left + 0.085, y + 0.018, f"{hit_count}/{max(len(row['targets']), 1)} target hits", transform=ax.transAxes, color=MUTED, fontsize=9)
        for pos, tid in enumerate(row["top_10"][:10]):
            x = left + label_w + pos * (box_w + gap)
            color = GREEN if hits[pos] else "#e7e2d8"
            edge = "#1f5b3a" if hits[pos] else "#d2c9bb"
            ax.add_patch(patches.FancyBboxPatch((x, y), box_w, 0.045, boxstyle="round,pad=0.004,rounding_size=0.008", facecolor=color, edgecolor=edge, linewidth=1.0, transform=ax.transAxes))
            text_color = "white" if hits[pos] else "#6d645a"
            ax.text(x + box_w / 2.0, y + 0.023, tid[-4:], ha="center", va="center", transform=ax.transAxes, fontsize=8.5, color=text_color, fontweight="bold")
        if first_hit < 99:
            ax.text(left + label_w + 10 * (box_w + gap) + 0.018, y + 0.019, f"first hit rank {first_hit}", transform=ax.transAxes, color=GREEN, fontsize=9.5, fontweight="bold")
        else:
            ax.text(left + label_w + 10 * (box_w + gap) + 0.018, y + 0.019, "no top-10 hit", transform=ax.transAxes, color=MUTED, fontsize=9.5)

    ax.add_patch(patches.FancyBboxPatch((0.02, 0.08), 0.96, 0.79, boxstyle="round,pad=0.018,rounding_size=0.025", facecolor="none", edgecolor="#ded6c9", linewidth=1.0, transform=ax.transAxes))
    return _save_figure(fig, visual_dir / "recommendation_examples_panel.png")


def _load_wav_mono(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        frames = handle.readframes(handle.getnframes())

    if width == 1:
        audio = np.frombuffer(frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif width == 2:
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    elif width == 4:
        audio = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483647.0
    else:
        raise ValueError(f"Unsupported WAV sample width {width} for {path}")

    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > 1.0:
        audio = audio / peak
    return sample_rate, audio.astype(np.float32, copy=False)


def _recommended_wavs(run_dir: Path) -> list[tuple[str, Path, str]]:
    summary = _optional_json(run_dir / "synthesis" / "synthesis_summary.json")
    recommended = summary.get("recommended_wavs", {})
    items = [
        ("better_adsr_warm_pad", "Better ADSR Warm Pad", recommended.get("better_adsr_warm_pad")),
        ("better_lfo_filter_sweep", "Better LFO Filter Sweep", recommended.get("better_lfo_filter_sweep")),
    ]
    result = []
    for stem, title, value in items:
        path = Path(value) if value else run_dir / "synthesis" / f"{stem}.wav"
        if path.exists():
            result.append((stem, path, title))
    if result:
        return result

    fallbacks = [
        ("adsr_sawtooth", run_dir / "synthesis" / "adsr_sawtooth.wav", "ADSR Sawtooth"),
        ("lfo_filtered_sawtooth", run_dir / "synthesis" / "lfo_filtered_sawtooth.wav", "LFO Filtered Sawtooth"),
    ]
    return [(stem, path, title) for stem, path, title in fallbacks if path.exists()]


def _audio_envelope(audio: np.ndarray, sample_rate: int, max_points: int = 1400) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window = max(1, int(np.ceil(audio.shape[0] / max_points)))
    centers = []
    lows = []
    highs = []
    for start in range(0, audio.shape[0], window):
        chunk = audio[start : start + window]
        centers.append((start + chunk.shape[0] / 2.0) / float(sample_rate))
        lows.append(float(np.min(chunk)))
        highs.append(float(np.max(chunk)))
    return np.asarray(centers), np.asarray(lows), np.asarray(highs)


def _save_synthesis_showcase(run_dir: Path, visual_dir: Path) -> Path | None:
    wavs = _recommended_wavs(run_dir)
    if not wavs:
        return None

    fig = plt.figure(figsize=(16, 9.5), facecolor=BG)
    gs = fig.add_gridspec(len(wavs) + 1, 6, height_ratios=[0.48] + [1.8] * len(wavs), hspace=0.44, wspace=0.42)
    title_ax = fig.add_subplot(gs[0, :])
    _clean_axis(title_ax)
    title_ax.text(0.0, 0.58, "Synthesis Output Showcase", transform=title_ax.transAxes, fontsize=23, fontweight="bold", color=INK)
    title_ax.text(0.0, 0.16, "Waveform envelopes and spectrograms for the improved WAV renders written by the APC workflow.", transform=title_ax.transAxes, fontsize=11.5, color=MUTED)

    for row_idx, (stem, path, title) in enumerate(wavs, start=1):
        sample_rate, audio = _load_wav_mono(path)
        t, low, high = _audio_envelope(audio, sample_rate)
        ax_wave = fig.add_subplot(gs[row_idx, :2])
        ax_wave.fill_between(t, low, high, color=CF if row_idx == 1 else AUDIO, alpha=0.35)
        ax_wave.plot(t, high, color=CF if row_idx == 1 else AUDIO, linewidth=1.0)
        ax_wave.plot(t, low, color=CF if row_idx == 1 else AUDIO, linewidth=1.0)
        ax_wave.axhline(0.0, color="#7d7468", linewidth=0.8, alpha=0.7)
        ax_wave.set_title(f"{title}: envelope", loc="left", fontweight="bold")
        ax_wave.set_xlabel("Time [s]")
        ax_wave.set_ylabel("Amplitude")
        ax_wave.set_ylim(-1.05, 1.05)
        _set_axis_style(ax_wave)

        ax_spec = fig.add_subplot(gs[row_idx, 2:5])
        ax_spec.specgram(audio, NFFT=2048, Fs=sample_rate, noverlap=1536, cmap="magma")
        ax_spec.set_title(f"{title}: spectrogram", loc="left", fontweight="bold")
        ax_spec.set_xlabel("Time [s]")
        ax_spec.set_ylabel("Frequency [Hz]")
        ax_spec.set_ylim(0, 6000)

        ax_stats = fig.add_subplot(gs[row_idx, 5])
        _clean_axis(ax_stats)
        duration = audio.shape[0] / float(sample_rate)
        rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        ax_stats.add_patch(patches.FancyBboxPatch((0.0, 0.02), 1.0, 0.96, boxstyle="round,pad=0.018,rounding_size=0.035", facecolor=PANEL, edgecolor="#ded6c9", transform=ax_stats.transAxes))
        ax_stats.text(0.08, 0.82, stem.replace("_", " ").title(), transform=ax_stats.transAxes, fontsize=10.5, color=INK, fontweight="bold")
        for idx, line in enumerate([f"{duration:.1f} sec", f"{sample_rate:,} Hz", f"peak {peak:.3f}", f"rms {rms:.3f}"]):
            ax_stats.text(0.08, 0.63 - idx * 0.15, line, transform=ax_stats.transAxes, fontsize=10.5, color=MUTED)

    return _save_figure(fig, visual_dir / "synthesis_showcase.png")


def _save_wav_visuals(run_dir: Path, visual_dir: Path, include_gifs: bool) -> list[Path]:
    assets: list[Path] = []
    for stem, wav_path, title in _recommended_wavs(run_dir):
        sample_rate, audio = _load_wav_mono(wav_path)
        waveform_path = visual_dir / f"{stem}_waveform.png"
        spectrogram_path = visual_dir / f"{stem}_spectrogram.png"
        _save_waveform_plot(audio, sample_rate, waveform_path, f"{title}: Waveform", color="#2f6f9f")
        _save_spectrogram_plot(audio, sample_rate, spectrogram_path, f"{title}: Spectrogram")
        assets.extend([waveform_path, spectrogram_path])
        if include_gifs:
            gif_path = visual_dir / f"{stem}_waveform_evolution.gif"
            _save_simple_waveform_gif(audio, sample_rate, gif_path, f"{title}: Waveform Evolution", color="#2f6f9f", frame_count=32)
            assets.append(gif_path)
    return assets


def render_gallery(run_dir: Path, visual_dir: Path, include_gifs: bool = False) -> list[Path]:
    ensure_dir(visual_dir)
    assets: list[Path] = []

    for renderer in [
        _save_executive_dashboard,
        _save_training_diagnostics,
        _save_recommender_storyboard,
        _save_recommendation_examples,
        _save_synthesis_showcase,
        _save_training_validation_curve,
        _save_metric_comparison,
        _save_retrieval_depth_curves,
        _save_dataset_coverage_panel,
        _save_first_relevant_rank_histogram,
        _save_ranking_hitmap,
    ]:
        asset = renderer(run_dir, visual_dir)
        if asset is not None:
            assets.append(asset)

    assets.extend(_save_wav_visuals(run_dir, visual_dir, include_gifs=include_gifs))

    manifest = {
        "run_dir": str(run_dir),
        "visual_dir": str(visual_dir),
        "include_gifs": include_gifs,
        "assets": [str(path) for path in assets],
    }
    save_json(visual_dir / "visual_manifest.json", manifest)
    assets.append(visual_dir / "visual_manifest.json")
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render automatic playlist continuation visual assets.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Run folder containing metrics, rankings, models, and synthesis outputs.")
    parser.add_argument("--visual-dir", type=Path, default=None, help="Destination for rendered visuals. Defaults to RUN_DIR/visuals.")
    parser.add_argument("--include-gifs", action="store_true", help="Also render waveform evolution GIFs for the recommended WAVs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = (args.run_dir or _latest_run_dir()).resolve()
    visual_dir = (args.visual_dir or run_dir / "visuals").resolve()
    assets = render_gallery(run_dir, visual_dir, include_gifs=args.include_gifs)
    print(f"[Automatic Playlist Visualiser] Wrote {len(assets)} assets under {visual_dir}")
    for asset in assets:
        print(asset)


if __name__ == "__main__":
    main()
