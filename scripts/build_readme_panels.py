#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import csv
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except ModuleNotFoundError:
    LogisticRegression = None
    accuracy_score = None
    train_test_split = None
    make_pipeline = None
    StandardScaler = None

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
    AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT,
    AUDIO_OUTPUT_DIR,
    CLASSIFIER_OUTPUT_DIR,
    OUTPUT_ROOT,
    VISUAL_AUDIO_DIR,
    VISUAL_CLASSIFIER_DIR,
    ensure_dir,
    load_json,
    read_csv_rows,
)


README_DIR = OUTPUT_ROOT / "readme"

BG = (247, 244, 238)
CARD = (255, 252, 247)
CARD_HEADER = (243, 238, 229)
TEXT = (46, 56, 68)
MUTED = (105, 116, 127)
LINE = (214, 205, 193)
ACCENT_AUDIO = (56, 140, 177)
ACCENT_CLASSIFIER = (195, 103, 66)
PIANO_COLOR = "#4C78A8"
DRUM_COLOR = "#E07A5F"
BASELINE_COLOR = "#7A8794"
ENHANCED_COLOR = "#3FA76E"
ACCENT_APC = (31, 111, 139)
ACCENT_APC_AUDIO = (208, 113, 63)
ACCENT_APC_GOOD = (47, 133, 90)
ACCENT_APC_GOLD = (214, 162, 61)

APC_PANEL_FILES = {
    "dashboard": "apc_full_run_dashboard.png",
    "training": "training_diagnostics_showcase.png",
    "recommender": "recommender_storyboard.png",
    "examples": "recommendation_examples_panel.png",
    "synthesis": "synthesis_showcase.png",
}

BASELINE_FIELDS = (
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
)
ENHANCED_FIELDS = (
    "lowest_pitch",
    "highest_pitch",
    "unique_pitch_num",
    "average_pitch_value",
    "pitch_span",
    "log_beats",
    "log_note_density",
    "average_velocity_norm",
    "drum_channel_ratio",
)
FEATURE_FIELDS = ENHANCED_FIELDS
FEATURE_LABELS = [
    "Min Pitch",
    "Max Pitch",
    "Unique Notes",
    "Mean Pitch",
    "Pitch Span",
    "Log Beats",
    "Log Density",
    "Velocity",
    "Drum Ratio",
]
SWEEP_SEEDS = (0, 1, 2, 3, 7, 11, 42, 99)


@dataclass(frozen=True)
class ImageCardSpec:
    label: str
    path: Path
    animated: bool = False


@dataclass(frozen=True)
class MetricCardSpec:
    title: str
    value: str
    subtitle: str


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    if bold:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
            ]
        )
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


FONT_TITLE = load_font(42, bold=True)
FONT_SUBTITLE = load_font(22, bold=False)
FONT_CARD = load_font(24, bold=True)
FONT_METRIC_TITLE = load_font(20, bold=False)
FONT_METRIC_VALUE = load_font(34, bold=True)
FONT_METRIC_SUB = load_font(16, bold=False)


def _mpl(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    return tuple(channel / 255.0 for channel in rgb)


def _trim_image(img: Image.Image, tolerance: int = 14) -> Image.Image:
    src = img.convert("RGB")
    arr = np.asarray(src)
    if arr.size == 0:
        return src
    corners = np.asarray(
        [
            arr[0, 0],
            arr[0, -1],
            arr[-1, 0],
            arr[-1, -1],
        ],
        dtype=np.int16,
    )
    bg = np.rint(corners.mean(axis=0)).astype(np.int16)
    delta = np.abs(arr.astype(np.int16) - bg).max(axis=2)
    ys, xs = np.where(delta > tolerance)
    if ys.size == 0 or xs.size == 0:
        return src
    pad = 8
    x0 = max(0, int(xs.min()) - pad)
    y0 = max(0, int(ys.min()) - pad)
    x1 = min(src.width, int(xs.max()) + pad + 1)
    y1 = min(src.height, int(ys.max()) + pad + 1)
    return src.crop((x0, y0, x1, y1))


def fit_image(img: Image.Image, width: int, height: int) -> Image.Image:
    src = _trim_image(img)
    scale = min(width / max(src.width, 1), height / max(src.height, 1))
    new_w = max(1, int(src.width * scale))
    new_h = max(1, int(src.height * scale))
    resized = src.resize((new_w, new_h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (width, height), CARD)
    canvas.paste(resized, ((width - new_w) // 2, (height - new_h) // 2))
    return canvas


def _figure_to_image(fig: plt.Figure, width: int, height: int) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as img:
        rendered = img.convert("RGB").copy()
    return fit_image(rendered, width, height)


def _sample_indices(num_src: int, num_dst: int) -> list[int]:
    if num_src <= 1:
        return [0] * max(1, num_dst)
    if num_dst <= 1:
        return [0]
    return [round(i * (num_src - 1) / (num_dst - 1)) for i in range(num_dst)]


def _load_card_frames(spec: ImageCardSpec, width: int, height: int, frame_count: int) -> list[Image.Image]:
    if spec.animated and spec.path.suffix.lower() == ".gif":
        with Image.open(spec.path) as gif:
            src_frames = getattr(gif, "n_frames", 1)
            indices = _sample_indices(src_frames, frame_count)
            frames = []
            for idx in indices:
                gif.seek(idx)
                frames.append(fit_image(gif.convert("RGB").copy(), width, height))
            return frames

    with Image.open(spec.path) as img:
        frame = fit_image(img, width, height)
    return [frame.copy() for _ in range(frame_count)]


def _build_shell(width: int, height: int, title: str, subtitle: str) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    canvas = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle((12, 12, width - 12, height - 12), radius=28, fill=BG, outline=LINE, width=2)
    title_box = draw.textbbox((0, 0), title, font=FONT_TITLE)
    title_w = title_box[2] - title_box[0]
    draw.text(((width - title_w) / 2, 18), title, fill=TEXT, font=FONT_TITLE)
    subtitle_box = draw.textbbox((0, 0), subtitle, font=FONT_SUBTITLE)
    subtitle_w = subtitle_box[2] - subtitle_box[0]
    draw.text(((width - subtitle_w) / 2, 70), subtitle, fill=MUTED, font=FONT_SUBTITLE)
    return canvas, draw


def _paste_card(
    canvas: Image.Image,
    draw: ImageDraw.ImageDraw,
    image: Image.Image,
    rect: tuple[int, int, int, int],
    label: str,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=18, fill=CARD, outline=LINE, width=2)
    header_h = 48
    draw.rounded_rectangle((x0, y0, x1, y0 + header_h), radius=18, fill=CARD_HEADER, outline=CARD_HEADER)
    draw.rectangle((x0, y0 + header_h - 2, x1, y0 + header_h), fill=accent)
    draw.text((x0 + 16, y0 + 11), label, fill=TEXT, font=FONT_CARD)
    content = ImageEnhance.Contrast(image).enhance(1.01)
    fitted = fit_image(content, x1 - x0 - 12, y1 - y0 - header_h - 12)
    canvas.paste(fitted, (x0 + 6, y0 + header_h + 6))


def _draw_metric_card(
    draw: ImageDraw.ImageDraw,
    rect: tuple[int, int, int, int],
    metric: MetricCardSpec,
    accent: tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle(rect, radius=16, fill=CARD, outline=LINE, width=2)
    draw.rounded_rectangle((x0, y0, x0 + 10, y1), radius=16, fill=accent, outline=accent)
    draw.text((x0 + 22, y0 + 14), metric.title, fill=MUTED, font=FONT_METRIC_TITLE)
    draw.text((x0 + 22, y0 + 40), metric.value, fill=TEXT, font=FONT_METRIC_VALUE)
    draw.text((x0 + 22, y1 - 26), metric.subtitle, fill=MUTED, font=FONT_METRIC_SUB)


def _render_grid_panel(
    title: str,
    subtitle: str,
    cards: list[ImageCardSpec],
    metrics: list[MetricCardSpec],
    output_gif: Path,
    cols: int,
    rows: int,
    accent: tuple[int, int, int],
    frame_count: int,
    frame_duration_ms: int,
    canvas_size: tuple[int, int],
) -> None:
    width, height = canvas_size
    pad = 24
    title_h = 112
    metric_h = 114
    cell_w = (width - pad * (cols + 1)) // cols
    cell_h = (height - title_h - metric_h - pad * (rows + 2)) // rows
    card_frames = [_load_card_frames(spec, cell_w - 12, cell_h - 52, frame_count) for spec in cards]

    rendered_frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        canvas, draw = _build_shell(width, height, title, subtitle)
        for idx, spec in enumerate(cards):
            row = idx // cols
            col = idx % cols
            x0 = pad + col * (cell_w + pad)
            y0 = title_h + pad + row * (cell_h + pad)
            rect = (x0, y0, x0 + cell_w, y0 + cell_h)
            _paste_card(canvas, draw, card_frames[idx][frame_idx], rect, spec.label, accent)

        metric_w = (width - pad * (len(metrics) + 1)) // len(metrics)
        metric_y = height - metric_h - pad
        for idx, metric in enumerate(metrics):
            x0 = pad + idx * (metric_w + pad)
            _draw_metric_card(draw, (x0, metric_y, x0 + metric_w, metric_y + metric_h), metric, accent)

        rendered_frames.append(canvas.quantize(colors=224, method=Image.MEDIANCUT))

    ensure_dir(output_gif.parent)
    rendered_frames[0].save(
        output_gif,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _style_plot_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(_mpl(CARD))
    ax.grid(True, alpha=0.25, color=_mpl(LINE), linewidth=0.8)
    ax.tick_params(colors=_mpl(MUTED), labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(_mpl(LINE))
        spine.set_linewidth(1.1)


def _row_matrix(rows: list[dict], fields: tuple[str, ...]) -> np.ndarray:
    return np.asarray([[float(row[field]) for field in fields] for row in rows], dtype=float)


def _profile_from_rows(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    matrix = _row_matrix(rows, FEATURE_FIELDS)
    labels = np.asarray([row["label"] for row in rows])
    mins = matrix.min(axis=0)
    maxs = matrix.max(axis=0)
    denom = np.where(maxs > mins, maxs - mins, 1.0)
    normalized = (matrix - mins) / denom
    piano_profile = normalized[labels == "piano"].mean(axis=0)
    drum_profile = normalized[labels == "drums"].mean(axis=0)
    return piano_profile, drum_profile


def _render_feature_profile_image(rows: list[dict], width: int, height: int, progress: float = 1.0) -> Image.Image:
    piano_profile, drum_profile = _profile_from_rows(rows)
    x = np.arange(len(FEATURE_LABELS))
    start_level = np.full_like(piano_profile, 0.5, dtype=float)
    eased = 1.0 - (1.0 - progress) ** 2
    piano_profile = start_level + eased * (piano_profile - start_level)
    drum_profile = start_level + eased * (drum_profile - start_level)

    fig, ax = plt.subplots(figsize=(7.4, 4.1), facecolor=_mpl(CARD))
    _style_plot_axis(ax)
    ax.plot(x, piano_profile, color=PIANO_COLOR, linewidth=2.4, marker="o", label="piano")
    ax.plot(x, drum_profile, color=DRUM_COLOR, linewidth=2.4, marker="o", label="drums")
    ax.fill_between(x, piano_profile, color=PIANO_COLOR, alpha=0.10)
    ax.fill_between(x, drum_profile, color=DRUM_COLOR, alpha=0.10)
    ax.set_title("All Feature Dimensions", color=_mpl(TEXT), fontsize=13, fontweight="bold")
    ax.set_ylabel("Class Mean (min-max normalized)", color=_mpl(MUTED), fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(x, FEATURE_LABELS, rotation=28, ha="right")
    legend = ax.legend(loc="upper left", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))
    return _figure_to_image(fig, width, height)


def _render_feature_space_image(rows: list[dict], width: int, height: int) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0), facecolor=_mpl(CARD))
    ax_ratio, ax_span = axes
    for ax in axes:
        _style_plot_axis(ax)

    labels = np.asarray([row["label"] for row in rows])
    avg_pitch = np.asarray([float(row["average_pitch_value"]) for row in rows], dtype=float)
    drum_ratio = np.asarray([float(row["drum_channel_ratio"]) for row in rows], dtype=float)
    unique_count = np.asarray([float(row["unique_pitch_num"]) for row in rows], dtype=float)
    pitch_span = np.asarray([float(row["pitch_span"]) for row in rows], dtype=float)

    for mask, color, label in (
        (labels == "piano", PIANO_COLOR, "piano"),
        (labels == "drums", DRUM_COLOR, "drums"),
    ):
        ax_ratio.scatter(avg_pitch[mask], drum_ratio[mask], s=28, alpha=0.72, color=color, edgecolors="none", label=label)
        ax_span.scatter(unique_count[mask], pitch_span[mask], s=28, alpha=0.72, color=color, edgecolors="none", label=label)

    ax_ratio.set_title("Average Pitch vs Drum Ratio", color=_mpl(TEXT), fontsize=12.5, fontweight="bold")
    ax_ratio.set_xlabel("Average Pitch Value", color=_mpl(MUTED), fontsize=9.5)
    ax_ratio.set_ylabel("Drum Channel Ratio", color=_mpl(MUTED), fontsize=9.5)
    ax_ratio.set_xlim(-2, 80)
    ax_ratio.set_ylim(-0.05, 1.05)
    legend = ax_ratio.legend(loc="lower left", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))

    ax_span.set_title("Unique Notes vs Pitch Span", color=_mpl(TEXT), fontsize=12.5, fontweight="bold")
    ax_span.set_xlabel("Unique Pitch Count", color=_mpl(MUTED), fontsize=9.5)
    ax_span.set_ylabel("Pitch Span", color=_mpl(MUTED), fontsize=9.5)
    ax_span.set_xlim(-2, max(80.0, float(np.max(unique_count)) + 5.0))
    ax_span.set_ylim(-5, max(100.0, float(np.max(pitch_span)) + 10.0))

    return _figure_to_image(fig, width, height)


def _feature_vector_from_row(row: dict, fields: tuple[str, ...]) -> list[float]:
    return [float(row[field]) for field in fields]


def _compute_seed_sweep(rows: list[dict]) -> dict:
    if any(item is None for item in (LogisticRegression, accuracy_score, train_test_split, make_pipeline, StandardScaler)):
        raise ModuleNotFoundError("Install scikit-learn to build classifier README panels.")

    y = [int(row["target"]) for row in rows]
    baseline_X = [_feature_vector_from_row(row, BASELINE_FIELDS) for row in rows]
    enhanced_X = [_feature_vector_from_row(row, ENHANCED_FIELDS) for row in rows]
    baseline_scores: list[float] = []
    enhanced_scores: list[float] = []

    for seed in SWEEP_SEEDS:
        Xb_train, Xb_test, y_train, y_test = train_test_split(
            baseline_X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )
        Xe_train, Xe_test, _, _ = train_test_split(
            enhanced_X,
            y,
            test_size=0.2,
            random_state=seed,
            stratify=y,
        )

        baseline_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )
        enhanced_model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, random_state=seed),
        )

        baseline_model.fit(Xb_train, y_train)
        enhanced_model.fit(Xe_train, y_train)
        baseline_scores.append(float(accuracy_score(y_test, baseline_model.predict(Xb_test))))
        enhanced_scores.append(float(accuracy_score(y_test, enhanced_model.predict(Xe_test))))

    return {
        "seeds": list(SWEEP_SEEDS),
        "baseline_scores": baseline_scores,
        "enhanced_scores": enhanced_scores,
        "baseline_mean": float(np.mean(baseline_scores)),
        "enhanced_mean": float(np.mean(enhanced_scores)),
        "baseline_min": float(np.min(baseline_scores)),
        "enhanced_min": float(np.min(enhanced_scores)),
        "baseline_max": float(np.max(baseline_scores)),
        "enhanced_max": float(np.max(enhanced_scores)),
    }


def _render_accuracy_summary_image(summary: dict, sweep: dict, width: int, height: int) -> Image.Image:
    labels = ["Fixed Split", "8-Seed Mean"]
    baseline_values = [summary["baseline_accuracy"], sweep["baseline_mean"]]
    enhanced_values = [summary["enhanced_accuracy"], sweep["enhanced_mean"]]
    x = np.arange(len(labels))
    bar_w = 0.34

    fig, ax = plt.subplots(figsize=(6.6, 4.0), facecolor=_mpl(CARD))
    _style_plot_axis(ax)
    bars_a = ax.bar(x - bar_w / 2, baseline_values, width=bar_w, color=BASELINE_COLOR, label="baseline")
    bars_b = ax.bar(x + bar_w / 2, enhanced_values, width=bar_w, color=ENHANCED_COLOR, label="enhanced")
    ax.set_ylim(0.88, 1.03)
    ax.set_ylabel("Accuracy", color=_mpl(MUTED), fontsize=9)
    ax.set_xticks(x, labels)
    ax.set_title("Accuracy Summary", color=_mpl(TEXT), fontsize=13, fontweight="bold")
    legend = ax.legend(loc="upper left", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))
    for bar_group in (bars_a, bars_b):
        for bar in bar_group:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.004,
                f"{bar.get_height():.3f}",
                ha="center",
                va="bottom",
                color=_mpl(TEXT),
                fontsize=10,
            )
    return _figure_to_image(fig, width, height)


def _render_seed_sweep_image(sweep: dict, width: int, height: int) -> Image.Image:
    seeds = np.asarray(sweep["seeds"], dtype=int)
    baseline_scores = np.asarray(sweep["baseline_scores"], dtype=float)
    enhanced_scores = np.asarray(sweep["enhanced_scores"], dtype=float)

    fig, ax = plt.subplots(figsize=(6.8, 4.0), facecolor=_mpl(CARD))
    _style_plot_axis(ax)
    ax.plot(seeds, baseline_scores, color=BASELINE_COLOR, linewidth=2.2, marker="o", label="baseline")
    ax.plot(seeds, enhanced_scores, color=ENHANCED_COLOR, linewidth=2.2, marker="o", label="enhanced")
    ax.axhline(sweep["baseline_mean"], color=BASELINE_COLOR, linewidth=1.2, linestyle="--", alpha=0.75)
    ax.axhline(sweep["enhanced_mean"], color=ENHANCED_COLOR, linewidth=1.2, linestyle="--", alpha=0.75)
    ax.set_ylim(0.88, 1.03)
    ax.set_title("Accuracy Across Random Splits", color=_mpl(TEXT), fontsize=13, fontweight="bold")
    ax.set_xlabel("Random Seed", color=_mpl(MUTED), fontsize=9)
    ax.set_ylabel("Accuracy", color=_mpl(MUTED), fontsize=9)
    legend = ax.legend(loc="lower right", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))
    return _figure_to_image(fig, width, height)


def _render_accuracy_dashboard_image(
    summary: dict,
    sweep: dict,
    width: int,
    height: int,
    progress: float = 1.0,
    visible_seed_count: int | None = None,
) -> Image.Image:
    seeds = np.asarray(sweep["seeds"], dtype=int)
    baseline_scores = np.asarray(sweep["baseline_scores"], dtype=float)
    enhanced_scores = np.asarray(sweep["enhanced_scores"], dtype=float)
    if visible_seed_count is None:
        visible_seed_count = len(seeds)
    eased = 1.0 - (1.0 - progress) ** 2

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), facecolor=_mpl(CARD))
    ax_bar, ax_line = axes
    for ax in axes:
        _style_plot_axis(ax)

    labels = ["Fixed Split", "8-Seed Mean"]
    x = np.arange(len(labels))
    bar_w = 0.34
    baseline_final = np.asarray([summary["baseline_accuracy"], sweep["baseline_mean"]], dtype=float)
    enhanced_final = np.asarray([summary["enhanced_accuracy"], sweep["enhanced_mean"]], dtype=float)
    floor = 0.88
    baseline_values = floor + eased * (baseline_final - floor)
    enhanced_values = floor + eased * (enhanced_final - floor)
    bars_a = ax_bar.bar(x - bar_w / 2, baseline_values, width=bar_w, color=BASELINE_COLOR, label="baseline")
    bars_b = ax_bar.bar(x + bar_w / 2, enhanced_values, width=bar_w, color=ENHANCED_COLOR, label="enhanced")
    ax_bar.set_ylim(0.88, 1.03)
    ax_bar.set_title("Fixed Split vs Mean Accuracy", color=_mpl(TEXT), fontsize=12.5, fontweight="bold")
    ax_bar.set_ylabel("Accuracy", color=_mpl(MUTED), fontsize=9.5)
    ax_bar.set_xticks(x, labels)
    legend = ax_bar.legend(loc="upper left", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))
    for bar, final_value in zip(bars_a, [summary["baseline_accuracy"], sweep["baseline_mean"]]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{final_value:.3f}", ha="center", color=_mpl(TEXT), fontsize=9.5)
    for bar, final_value in zip(bars_b, [summary["enhanced_accuracy"], sweep["enhanced_mean"]]):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{final_value:.3f}", ha="center", color=_mpl(TEXT), fontsize=9.5)

    visible_seeds = seeds[:visible_seed_count]
    visible_baseline = baseline_scores[:visible_seed_count]
    visible_enhanced = enhanced_scores[:visible_seed_count]
    ax_line.plot(visible_seeds, visible_baseline, color=BASELINE_COLOR, linewidth=2.2, marker="o", label="baseline")
    ax_line.plot(visible_seeds, visible_enhanced, color=ENHANCED_COLOR, linewidth=2.2, marker="o", label="enhanced")
    ax_line.axhline(sweep["baseline_mean"], color=BASELINE_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)
    ax_line.axhline(sweep["enhanced_mean"], color=ENHANCED_COLOR, linewidth=1.2, linestyle="--", alpha=0.7)
    ax_line.set_ylim(0.88, 1.03)
    ax_line.set_title("Random-Split Robustness", color=_mpl(TEXT), fontsize=12.5, fontweight="bold")
    ax_line.set_xlabel("Random Seed", color=_mpl(MUTED), fontsize=9.5)
    ax_line.set_ylabel("Accuracy", color=_mpl(MUTED), fontsize=9.5)
    legend = ax_line.legend(loc="lower right", frameon=False)
    if legend is not None:
        for text in legend.get_texts():
            text.set_color(_mpl(TEXT))

    return _figure_to_image(fig, width, height)


def _render_confusion_pair_image(
    baseline_cm: np.ndarray,
    enhanced_cm: np.ndarray,
    width: int,
    height: int,
    progress: float = 1.0,
) -> Image.Image:
    eased = 1.0 - (1.0 - progress) ** 2
    baseline_current = np.rint(baseline_cm * eased).astype(int)
    enhanced_current = np.rint(enhanced_cm * eased).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), facecolor=_mpl(CARD))
    for ax, matrix, title, cmap in (
        (axes[0], baseline_current, "Baseline Confusion", "Blues"),
        (axes[1], enhanced_current, "Enhanced Confusion", "Greens"),
    ):
        _style_plot_axis(ax)
        ax.imshow(matrix, cmap=cmap, vmin=0, vmax=int(max(baseline_cm.max(), enhanced_cm.max())))
        ax.set_title(title, color=_mpl(TEXT), fontsize=12.5, fontweight="bold")
        ax.set_xticks([0, 1], labels=["drums", "piano"])
        ax.set_yticks([0, 1], labels=["drums", "piano"])
        ax.set_xlabel("Predicted", color=_mpl(MUTED), fontsize=9.5)
        ax.set_ylabel("True", color=_mpl(MUTED), fontsize=9.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(matrix[i, j])), ha="center", va="center", color="#111111", fontsize=14, fontweight="bold")

    return _figure_to_image(fig, width, height)


def _interleave_rows(rows: list[dict]) -> list[dict]:
    piano = [row for row in rows if row["label"] == "piano"]
    drums = [row for row in rows if row["label"] == "drums"]
    ordered: list[dict] = []
    for idx in range(max(len(piano), len(drums))):
        if idx < len(piano):
            ordered.append(piano[idx])
        if idx < len(drums):
            ordered.append(drums[idx])
    return ordered


def build_audio_panels(output_dir: Path) -> None:
    audio_summary = load_json(AUDIO_OUTPUT_DIR / "audio_summary.json")
    metrics = [
        MetricCardSpec("Lead Notes", str(len(audio_summary["lead_notes"])), "melody sequence"),
        MetricCardSpec("Sample Rate", "44.1 kHz", "wav export"),
        MetricCardSpec("Delay Tail", f"{audio_summary['delay_tail_seconds']:.2f} s", "echo offset"),
        MetricCardSpec("Rendered Clips", str(len(audio_summary["audio_files"])), "waveform variants"),
    ]

    width, height = 1720, 860
    pad = 24
    title_h = 112
    top_h = 236
    mid_h = 272
    metric_h = 106
    canvas, draw = _build_shell(width, height, "Audio Synthesis", "Melody rendering, envelope change, and spectral structure")

    top_cards = [
        ImageCardSpec("Sine Waveform", VISUAL_AUDIO_DIR / "melody_sine_waveform.png"),
        ImageCardSpec("Sawtooth Waveform", VISUAL_AUDIO_DIR / "melody_sawtooth_waveform.png"),
        ImageCardSpec("Layered Waveform", VISUAL_AUDIO_DIR / "melody_stacked_waveform.png"),
        ImageCardSpec("Fade Comparison", VISUAL_AUDIO_DIR / "fade_comparison.png"),
    ]
    bottom_cards = [
        ImageCardSpec("Delay Comparison", VISUAL_AUDIO_DIR / "delay_comparison.png"),
        ImageCardSpec("Sine Spectrogram", VISUAL_AUDIO_DIR / "sine_spectrogram.png"),
        ImageCardSpec("Sawtooth Spectrogram", VISUAL_AUDIO_DIR / "sawtooth_spectrogram.png"),
    ]

    top_w = (width - pad * 5) // 4
    top_y = title_h + pad
    for idx, spec in enumerate(top_cards):
        x0 = pad + idx * (top_w + pad)
        rect = (x0, top_y, x0 + top_w, top_y + top_h)
        frame = _load_card_frames(spec, top_w - 12, top_h - 52, 1)[0]
        _paste_card(canvas, draw, frame, rect, spec.label, ACCENT_AUDIO)

    bottom_w = (width - pad * 4) // 3
    bottom_y = top_y + top_h + pad
    for idx, spec in enumerate(bottom_cards):
        x0 = pad + idx * (bottom_w + pad)
        rect = (x0, bottom_y, x0 + bottom_w, bottom_y + mid_h)
        frame = _load_card_frames(spec, bottom_w - 12, mid_h - 52, 1)[0]
        _paste_card(canvas, draw, frame, rect, spec.label, ACCENT_AUDIO)

    metric_w = (width - pad * 5) // 4
    metric_y = height - metric_h - pad
    for idx, metric in enumerate(metrics):
        x0 = pad + idx * (metric_w + pad)
        _draw_metric_card(draw, (x0, metric_y, x0 + metric_w, metric_y + metric_h), metric, ACCENT_AUDIO)

    static_out = output_dir / "readme_audio_static_panel.png"
    ensure_dir(static_out.parent)
    canvas.save(static_out, format="PNG", optimize=True)

    animated_cards = [
        ImageCardSpec("Sine Melody", VISUAL_AUDIO_DIR / "melody_sine_evolution.gif", animated=True),
        ImageCardSpec("Sawtooth Melody", VISUAL_AUDIO_DIR / "melody_sawtooth_evolution.gif", animated=True),
        ImageCardSpec("Layered Mix", VISUAL_AUDIO_DIR / "melody_stack_evolution.gif", animated=True),
        ImageCardSpec("Fade Evolution", VISUAL_AUDIO_DIR / "melody_fade_evolution.gif", animated=True),
        ImageCardSpec("Delay Echo", VISUAL_AUDIO_DIR / "melody_delay_evolution.gif", animated=True),
        ImageCardSpec("Spectral Structure", VISUAL_AUDIO_DIR / "sawtooth_spectrogram.png"),
    ]
    _render_grid_panel(
        title="Audio Synthesis",
        subtitle="Sine, sawtooth, fade, delay, layering, and harmonic content",
        cards=animated_cards,
        metrics=metrics,
        output_gif=output_dir / "readme_audio_animated_panel.gif",
        cols=3,
        rows=2,
        accent=ACCENT_AUDIO,
        frame_count=32,
        frame_duration_ms=120,
        canvas_size=(1560, 980),
    )


def build_classifier_panels(output_dir: Path) -> None:
    summary = load_json(CLASSIFIER_OUTPUT_DIR / "classifier_summary.json")
    baseline = load_json(CLASSIFIER_OUTPUT_DIR / "baseline_metrics.json")
    enhanced = load_json(CLASSIFIER_OUTPUT_DIR / "enhanced_metrics.json")
    rows = read_csv_rows(CLASSIFIER_OUTPUT_DIR / "feature_rows.csv")
    sweep = _compute_seed_sweep(rows)

    metrics = [
        MetricCardSpec("Rows", str(summary["row_count"]), "midi files"),
        MetricCardSpec("Fixed Split", f"{summary['baseline_accuracy']:.3f} / {summary['enhanced_accuracy']:.3f}", "baseline / enhanced"),
        MetricCardSpec("8-Seed Mean", f"{sweep['baseline_mean']:.3f} / {sweep['enhanced_mean']:.3f}", "baseline / enhanced"),
        MetricCardSpec("8-Seed Min", f"{sweep['baseline_min']:.3f} / {sweep['enhanced_min']:.3f}", "baseline / enhanced"),
    ]

    width, height = 1680, 900
    pad = 24
    title_h = 112
    cell_h = 270
    metric_h = 106
    canvas, draw = _build_shell(width, height, "Symbolic Classification", "Feature separation, stability across splits, and confusion structure")

    grid_w = (width - pad * 3) // 2
    top_y = title_h + pad
    bottom_y = top_y + cell_h + pad
    static_cards: list[tuple[tuple[int, int, int, int], str, Image.Image]] = [
        ((pad, top_y, pad + grid_w, top_y + cell_h), "Feature Space", _render_feature_space_image(rows, grid_w - 12, cell_h - 60)),
        ((pad * 2 + grid_w, top_y, pad * 2 + grid_w * 2, top_y + cell_h), "Feature Profile", _render_feature_profile_image(rows, grid_w - 12, cell_h - 60)),
        ((pad, bottom_y, pad + grid_w, bottom_y + cell_h), "Accuracy and Robustness", _render_accuracy_dashboard_image(summary, sweep, grid_w - 12, cell_h - 60)),
        ((pad * 2 + grid_w, bottom_y, pad * 2 + grid_w * 2, bottom_y + cell_h), "Confusion Structure", _render_confusion_pair_image(np.asarray(baseline["confusion_matrix"], dtype=float), np.asarray(enhanced["confusion_matrix"], dtype=float), grid_w - 12, cell_h - 60)),
    ]
    for rect, label, frame in static_cards:
        _paste_card(canvas, draw, frame, rect, label, ACCENT_CLASSIFIER)

    metric_w = (width - pad * 5) // 4
    metric_y = height - metric_h - pad
    for idx, metric in enumerate(metrics):
        x0 = pad + idx * (metric_w + pad)
        _draw_metric_card(draw, (x0, metric_y, x0 + metric_w, metric_y + metric_h), metric, ACCENT_CLASSIFIER)

    static_out = output_dir / "readme_classifier_static_panel.png"
    ensure_dir(static_out.parent)
    canvas.save(static_out, format="PNG", optimize=True)

    ordered_rows = _interleave_rows(rows)
    baseline_cm = np.asarray(baseline["confusion_matrix"], dtype=float)
    enhanced_cm = np.asarray(enhanced["confusion_matrix"], dtype=float)
    seeds = np.asarray(sweep["seeds"], dtype=int)

    frame_count = 18
    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        progress = (frame_idx + 1) / frame_count
        visible_count = max(10, int(round(progress * len(ordered_rows))))
        visible_seed_count = max(2, int(round(progress * len(seeds))))
        visible_rows = ordered_rows[:visible_count]

        frame_canvas, frame_draw = _build_shell(width, height, "Symbolic Classification", "Feature separation, stability across splits, and confusion structure")
        animated_cards: list[tuple[tuple[int, int, int, int], str, Image.Image]] = [
            ((pad, top_y, pad + grid_w, top_y + cell_h), "Feature Space", _render_feature_space_image(visible_rows, grid_w - 12, cell_h - 60)),
            ((pad * 2 + grid_w, top_y, pad * 2 + grid_w * 2, top_y + cell_h), "Feature Profile", _render_feature_profile_image(visible_rows, grid_w - 12, cell_h - 60, progress=progress)),
            ((pad, bottom_y, pad + grid_w, bottom_y + cell_h), "Accuracy and Robustness", _render_accuracy_dashboard_image(summary, sweep, grid_w - 12, cell_h - 60, progress=progress, visible_seed_count=visible_seed_count)),
            ((pad * 2 + grid_w, bottom_y, pad * 2 + grid_w * 2, bottom_y + cell_h), "Confusion Structure", _render_confusion_pair_image(baseline_cm, enhanced_cm, grid_w - 12, cell_h - 60, progress=progress)),
        ]
        for rect, label, image in animated_cards:
            _paste_card(frame_canvas, frame_draw, image, rect, label, ACCENT_CLASSIFIER)

        animated_metrics = [
            MetricCardSpec("Rows Shown", f"{visible_count}/{len(ordered_rows)}", "progressive reveal"),
            MetricCardSpec("Fixed Split", f"{summary['baseline_accuracy']:.3f} / {summary['enhanced_accuracy']:.3f}", "baseline / enhanced"),
            MetricCardSpec("8-Seed Mean", f"{sweep['baseline_mean']:.3f} / {sweep['enhanced_mean']:.3f}", "baseline / enhanced"),
            MetricCardSpec("8-Seed Min", f"{sweep['baseline_min']:.3f} / {sweep['enhanced_min']:.3f}", "baseline / enhanced"),
        ]
        metric_w = (width - pad * 5) // 4
        metric_y = height - metric_h - pad
        for idx, metric in enumerate(animated_metrics):
            x0 = pad + idx * (metric_w + pad)
            _draw_metric_card(frame_draw, (x0, metric_y, x0 + metric_w, metric_y + metric_h), metric, ACCENT_CLASSIFIER)

        frames.append(frame_canvas.quantize(colors=224, method=Image.MEDIANCUT))

    gif_out = output_dir / "readme_classifier_animated_panel.gif"
    ensure_dir(gif_out.parent)
    frames[0].save(
        gif_out,
        save_all=True,
        append_images=frames[1:],
        duration=150,
        loop=0,
        optimize=False,
        disposal=2,
    )


def _latest_apc_run_dir() -> Path:
    candidates = [
        path
        for path in AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT.glob("full_run_*")
        if path.is_dir()
    ]
    if not candidates:
        return AUTOMATIC_PLAYLIST_CONTINUATION_OUTPUT_ROOT
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _load_optional_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return load_json(path)


def _ensure_apc_visuals(run_dir: Path) -> Path:
    visual_dir = run_dir / "visuals"
    required = [visual_dir / filename for filename in APC_PANEL_FILES.values()]
    if all(path.exists() for path in required):
        return visual_dir

    from scripts.visualiser.render_automatic_playlist_continuation_gallery import render_gallery

    render_gallery(run_dir, visual_dir, include_gifs=False)
    return visual_dir


def _apc_metric_cards(run_dir: Path) -> list[MetricCardSpec]:
    summary = _load_optional_json(run_dir / "metrics" / "playlist_continuation_summary.json")
    training = _load_optional_json(run_dir / "metrics" / "training_validation_summary.json")
    cf = _load_optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _load_optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    embedding = _load_optional_json(run_dir / "metrics" / "embedding_summary.json")

    train = summary.get("train", {})
    best = training.get("best_epoch_by_validation_accuracy_at_10", {})
    requested = float(embedding.get("requested_track_ids", 0) or 0)
    present = float(embedding.get("selected_files_present", 0) or 0)
    coverage = present / requested if requested else 0.0

    return [
        MetricCardSpec("Train Playlists", f"{int(train.get('playlists', 0)):,}", f"{int(train.get('track_rows', 0)):,} track rows"),
        MetricCardSpec("Best Epoch", str(int(best.get("epoch", 0))) if best else "n/a", f"Hit@10 {float(best.get('validation_accuracy_at_10', 0.0)):.2f}"),
        MetricCardSpec("CF Quality", f"{float(cf.get('hit_rate_at_10', 0.0)):.2f}", f"MRR {float(cf.get('mrr', 0.0)):.3f}"),
        MetricCardSpec("Audio Baseline", f"{float(audio.get('hit_rate_at_10', 0.0)):.2f}", f"embeddings {coverage:.0%}"),
    ]


def _apc_specs(visual_dir: Path) -> list[ImageCardSpec]:
    return [
        ImageCardSpec("Full Run Dashboard", visual_dir / APC_PANEL_FILES["dashboard"]),
        ImageCardSpec("Training Diagnostics", visual_dir / APC_PANEL_FILES["training"]),
        ImageCardSpec("Recommendation Storyboard", visual_dir / APC_PANEL_FILES["recommender"]),
        ImageCardSpec("Top-10 Examples", visual_dir / APC_PANEL_FILES["examples"]),
        ImageCardSpec("Synthesis Showcase", visual_dir / APC_PANEL_FILES["synthesis"]),
    ]


def _save_apc_static_panel(run_dir: Path, visual_dir: Path, output_dir: Path, metrics: list[MetricCardSpec]) -> Path:
    width, height = 1800, 1040
    pad = 24
    title_h = 116
    top_h = 430
    bottom_h = 265
    metric_h = 110
    canvas, draw = _build_shell(
        width,
        height,
        "Automatic Playlist Continuation",
        "Dataset coverage, WRMF training, recommendation quality, and synthesis outputs",
    )

    specs = _apc_specs(visual_dir)
    top_y = title_h + pad
    left_w = 1030
    right_w = width - pad * 3 - left_w
    _paste_card(
        canvas,
        draw,
        _load_card_frames(specs[0], left_w - 12, top_h - 52, 1)[0],
        (pad, top_y, pad + left_w, top_y + top_h),
        specs[0].label,
        ACCENT_APC,
    )
    _paste_card(
        canvas,
        draw,
        _load_card_frames(specs[2], right_w - 12, top_h - 52, 1)[0],
        (pad * 2 + left_w, top_y, width - pad, top_y + top_h),
        specs[2].label,
        ACCENT_APC_GOOD,
    )

    bottom_y = top_y + top_h + pad
    bottom_w = (width - pad * 4) // 3
    for idx, spec in enumerate([specs[1], specs[3], specs[4]]):
        x0 = pad + idx * (bottom_w + pad)
        _paste_card(
            canvas,
            draw,
            _load_card_frames(spec, bottom_w - 12, bottom_h - 52, 1)[0],
            (x0, bottom_y, x0 + bottom_w, bottom_y + bottom_h),
            spec.label,
            [ACCENT_APC_GOLD, ACCENT_APC_GOOD, ACCENT_APC_AUDIO][idx],
        )

    metric_w = (width - pad * (len(metrics) + 1)) // len(metrics)
    metric_y = height - metric_h - pad
    for idx, metric in enumerate(metrics):
        x0 = pad + idx * (metric_w + pad)
        _draw_metric_card(draw, (x0, metric_y, x0 + metric_w, metric_y + metric_h), metric, ACCENT_APC)

    static_out = output_dir / "readme_automatic_playlist_continuation_static_panel.png"
    ensure_dir(static_out.parent)
    canvas.save(static_out, format="PNG", optimize=True)
    return static_out


def _figure_to_exact_image(fig: plt.Figure, width: int, height: int, dpi: int = 100) -> Image.Image:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as img:
        rendered = img.convert("RGB").copy()
    if rendered.size != (width, height):
        rendered = rendered.resize((width, height), Image.Resampling.LANCZOS)
    return rendered


def _apc_training_records(run_dir: Path) -> list[dict]:
    curve = _load_optional_json(run_dir / "metrics" / "training_validation_curve.json")
    records = curve.get("records", []) if isinstance(curve, dict) else []
    if not records:
        history = _load_optional_json(run_dir / "models" / "wrmf" / "history.json").get("history", [])
        records = [
            {
                "epoch": idx + 1,
                "train_loss": item.get("loss", 0.0),
                "validation_accuracy_at_10": 0.0,
                "validation_precision_at_10": 0.0,
                "validation_mrr": 0.0,
                "validation_ndcg_at_10": 0.0,
            }
            for idx, item in enumerate(history)
        ]

    clean_records: list[dict] = []
    for idx, record in enumerate(records):
        clean_records.append(
            {
                "epoch": int(record.get("epoch", idx + 1)),
                "train_loss": float(record.get("train_loss", record.get("loss", 0.0)) or 0.0),
                "validation_accuracy_at_10": float(record.get("validation_accuracy_at_10", 0.0) or 0.0),
                "validation_precision_at_10": float(record.get("validation_precision_at_10", 0.0) or 0.0),
                "validation_mrr": float(record.get("validation_mrr", 0.0) or 0.0),
                "validation_ndcg_at_10": float(record.get("validation_ndcg_at_10", 0.0) or 0.0),
            }
        )
    if not clean_records:
        clean_records = [
            {
                "epoch": 1,
                "train_loss": 0.0,
                "validation_accuracy_at_10": 0.0,
                "validation_precision_at_10": 0.0,
                "validation_mrr": 0.0,
                "validation_ndcg_at_10": 0.0,
            }
        ]
    return sorted(clean_records, key=lambda item: item["epoch"])


def _apc_preview_hit_matrix(run_dir: Path, max_rows: int = 10) -> tuple[list[str], np.ndarray]:
    preview_path = run_dir / "rankings" / "collaborative_filtering_preview.csv"
    rows: list[tuple[str, list[int]]] = []
    if preview_path.exists():
        with preview_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    top_10 = ast.literal_eval(row.get("top_10", "[]"))
                    targets = set(ast.literal_eval(row.get("targets", "[]")))
                except (SyntaxError, ValueError):
                    continue
                hits = [1 if track_id in targets else 0 for track_id in top_10[:10]]
                hits.extend([0] * (10 - len(hits)))
                rows.append((str(row.get("playlist_id", len(rows))), hits[:10]))

    if not rows:
        return [f"P{idx + 1}" for idx in range(max_rows)], np.zeros((max_rows, 10), dtype=int)

    rows.sort(key=lambda item: (-sum(item[1]), item[0]))
    selected = rows[:max_rows]
    labels = [f"P{playlist_id}" for playlist_id, _ in selected]
    matrix = np.asarray([hits for _, hits in selected], dtype=int)
    return labels, matrix


def _visible_curve_points(epochs: np.ndarray, values: np.ndarray, epoch_cursor: float) -> tuple[np.ndarray, np.ndarray]:
    if len(epochs) == 1:
        return epochs.copy(), values.copy()
    visible = epochs <= epoch_cursor
    x = list(epochs[visible])
    y = list(values[visible])
    if not x:
        x = [float(epochs[0])]
        y = [float(values[0])]
    elif epoch_cursor > x[-1] and epoch_cursor < epochs[-1]:
        x.append(float(epoch_cursor))
        y.append(float(np.interp(epoch_cursor, epochs, values)))
    return np.asarray(x, dtype=float), np.asarray(y, dtype=float)


def _metric_at_depth(metrics: dict, name: str, depths: list[int]) -> np.ndarray:
    values = [float(metrics.get(f"{name}_at_{depth}", 0.0) or 0.0) for depth in depths]
    return np.asarray(values, dtype=float)


def _save_apc_animated_panel(run_dir: Path, visual_dir: Path, output_dir: Path, metrics: list[MetricCardSpec]) -> Path:
    del visual_dir
    width, height, dpi = 1280, 760, 100
    records = _apc_training_records(run_dir)
    cf = _load_optional_json(run_dir / "metrics" / "collaborative_filtering_metrics.json")
    audio = _load_optional_json(run_dir / "metrics" / "audio_similarity_metrics.json")
    embedding = _load_optional_json(run_dir / "metrics" / "embedding_summary.json")
    playlists, hit_matrix = _apc_preview_hit_matrix(run_dir, max_rows=10)

    epochs = np.asarray([record["epoch"] for record in records], dtype=float)
    train_loss = np.asarray([record["train_loss"] for record in records], dtype=float)
    hit_at_10 = np.asarray([record["validation_accuracy_at_10"] for record in records], dtype=float)
    target_p_at_10 = np.asarray([record["validation_precision_at_10"] for record in records], dtype=float)
    mrr = np.asarray([record["validation_mrr"] for record in records], dtype=float)
    ndcg = np.asarray([record["validation_ndcg_at_10"] for record in records], dtype=float)
    best_hit = np.maximum.accumulate(hit_at_10)

    depths = [5, 10, 20, 50, 100]
    cf_depth = _metric_at_depth(cf, "hit_rate", depths)
    audio_depth = _metric_at_depth(audio, "hit_rate", depths)
    quality_labels = ["Hit@10", "Target P@10", "MRR", "NDCG@10"]
    cf_quality = np.asarray(
        [
            float(cf.get("hit_rate_at_10", 0.0) or 0.0),
            float(cf.get("target_precision_at_10", 0.0) or 0.0),
            float(cf.get("mrr", 0.0) or 0.0),
            float(cf.get("ndcg_at_10", 0.0) or 0.0),
        ],
        dtype=float,
    )
    audio_quality = np.asarray(
        [
            float(audio.get("hit_rate_at_10", 0.0) or 0.0),
            float(audio.get("target_precision_at_10", 0.0) or 0.0),
            float(audio.get("mrr", 0.0) or 0.0),
            float(audio.get("ndcg_at_10", 0.0) or 0.0),
        ],
        dtype=float,
    )
    requested = float(embedding.get("requested_track_ids", 0) or 0)
    present = float(embedding.get("selected_files_present", 0) or 0)
    embedding_coverage = present / requested if requested else 0.0
    lift = cf_quality[0] / max(audio_quality[0], 1e-9)
    final_best_idx = int(np.argmax(hit_at_10))

    frame_count = 36
    frames: list[Image.Image] = []
    for frame_idx in range(frame_count):
        progress = frame_idx / max(frame_count - 1, 1)
        eased = 1.0 - (1.0 - progress) ** 2
        epoch_cursor = float(np.interp(eased, [0.0, 1.0], [epochs[0], epochs[-1]]))
        epoch_label = int(round(epoch_cursor))
        current_loss = float(np.interp(epoch_cursor, epochs, train_loss))
        current_hit = float(np.interp(epoch_cursor, epochs, hit_at_10))
        current_precision = float(np.interp(epoch_cursor, epochs, target_p_at_10))
        current_mrr = float(np.interp(epoch_cursor, epochs, mrr))
        current_best_hit = float(np.interp(epoch_cursor, epochs, best_hit))
        depth_visible = max(2, min(len(depths), int(np.ceil(eased * len(depths)))))
        topk_visible = max(1, min(10, int(np.ceil(eased * 10))))
        bar_progress = min(1.0, max(0.0, (progress - 0.12) / 0.68))

        fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi, facecolor=_mpl(BG))
        fig.text(0.5, 0.965, "Automatic Playlist Continuation", ha="center", va="center", color=_mpl(TEXT), fontsize=24, fontweight="bold")
        fig.text(
            0.5,
            0.928,
            "Training progress, validation quality, recommendation depth, and top-10 hits rendered from the run metrics",
            ha="center",
            va="center",
            color=_mpl(MUTED),
            fontsize=11,
        )

        grid = fig.add_gridspec(
            3,
            4,
            left=0.045,
            right=0.965,
            top=0.875,
            bottom=0.205,
            hspace=0.48,
            wspace=0.35,
            height_ratios=[1.04, 0.96, 0.88],
        )
        ax_loss = fig.add_subplot(grid[0, 0:2])
        ax_validation = fig.add_subplot(grid[1, 0:2])
        ax_depth = fig.add_subplot(grid[2, 0:2])
        ax_quality = fig.add_subplot(grid[0, 2:4])
        ax_hits = fig.add_subplot(grid[1:3, 2:4])

        for ax in (ax_loss, ax_validation, ax_depth, ax_quality, ax_hits):
            _style_plot_axis(ax)

        x_loss, y_loss = _visible_curve_points(epochs, train_loss, epoch_cursor)
        ax_loss.plot(x_loss, y_loss, color=_mpl(ACCENT_APC), linewidth=3.0, marker="o", markersize=4.8)
        ax_loss.fill_between(x_loss, y_loss, np.nanmin(train_loss) * 0.93, color=_mpl(ACCENT_APC), alpha=0.12)
        ax_loss.scatter([epoch_cursor], [current_loss], s=90, color=_mpl(ACCENT_APC_GOLD), edgecolor="white", linewidth=1.0, zorder=5)
        ax_loss.set_title("WRMF loss updates each epoch", color=_mpl(TEXT), fontsize=13, fontweight="bold", loc="left")
        ax_loss.set_xlim(epochs[0], epochs[-1])
        ax_loss.set_ylim(max(0.0, float(np.nanmin(train_loss)) * 0.86), float(np.nanmax(train_loss)) * 1.08)
        ax_loss.set_xlabel("Epoch", color=_mpl(MUTED), fontsize=9)
        ax_loss.set_ylabel("Training loss", color=_mpl(MUTED), fontsize=9)
        ax_loss.text(
            0.98,
            0.88,
            f"epoch {epoch_label}/{int(epochs[-1])}\nloss {current_loss:.3f}",
            transform=ax_loss.transAxes,
            ha="right",
            va="top",
            color=_mpl(TEXT),
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.35", facecolor=_mpl(CARD), edgecolor=_mpl(LINE), linewidth=0.9),
        )

        x_hit, y_hit = _visible_curve_points(epochs, hit_at_10, epoch_cursor)
        _, y_best = _visible_curve_points(epochs, best_hit, epoch_cursor)
        _, y_precision = _visible_curve_points(epochs, target_p_at_10, epoch_cursor)
        _, y_mrr = _visible_curve_points(epochs, mrr, epoch_cursor)
        _, y_ndcg = _visible_curve_points(epochs, ndcg, epoch_cursor)
        ax_validation.plot(x_hit, y_hit, color=_mpl(ACCENT_APC_GOOD), linewidth=2.5, marker="o", markersize=4.3, label="Hit@10")
        ax_validation.plot(x_hit, y_best, color=_mpl(ACCENT_APC_GOLD), linewidth=2.5, linestyle="--", label="best Hit@10")
        ax_validation.plot(x_hit, y_precision, color=_mpl(ACCENT_APC_AUDIO), linewidth=2.0, label="target P@10")
        ax_validation.plot(x_hit, y_ndcg, color=(0.48, 0.35, 0.68), linewidth=2.0, label="NDCG@10")
        ax_validation.plot(x_hit, y_mrr, color=_mpl(MUTED), linewidth=1.8, label="MRR")
        if epochs[final_best_idx] <= epoch_cursor:
            ax_validation.scatter([epochs[final_best_idx]], [hit_at_10[final_best_idx]], s=78, color=_mpl(ACCENT_APC_GOLD), edgecolor="white", linewidth=1.0, zorder=5)
            ax_validation.text(epochs[final_best_idx] + 0.1, hit_at_10[final_best_idx] + 0.025, "best", color=_mpl(TEXT), fontsize=9)
        ax_validation.set_title("Validation metrics change during training", color=_mpl(TEXT), fontsize=13, fontweight="bold", loc="left")
        ax_validation.set_xlim(epochs[0], epochs[-1])
        ax_validation.set_ylim(0.0, max(0.54, float(np.nanmax(hit_at_10)) + 0.06))
        ax_validation.set_xlabel("Epoch", color=_mpl(MUTED), fontsize=9)
        ax_validation.set_ylabel("Score", color=_mpl(MUTED), fontsize=9)
        legend = ax_validation.legend(loc="upper left", ncol=3, frameon=False, fontsize=8.3)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(_mpl(TEXT))

        ax_depth.plot(depths[:depth_visible], cf_depth[:depth_visible], color=_mpl(ACCENT_APC), linewidth=2.7, marker="o", label="Collaborative filtering")
        ax_depth.plot(depths[:depth_visible], audio_depth[:depth_visible], color=_mpl(ACCENT_APC_AUDIO), linewidth=2.3, marker="o", label="Audio similarity")
        ax_depth.set_title("Recall opportunity grows with recommendation depth", color=_mpl(TEXT), fontsize=13, fontweight="bold", loc="left")
        ax_depth.set_xlim(4, 105)
        ax_depth.set_ylim(0.0, max(0.72, float(np.nanmax(cf_depth)) + 0.06))
        ax_depth.set_xticks(depths)
        ax_depth.set_xlabel("Recommendation depth", color=_mpl(MUTED), fontsize=9)
        ax_depth.set_ylabel("Hit rate", color=_mpl(MUTED), fontsize=9)
        legend = ax_depth.legend(loc="lower right", frameon=False, fontsize=8.5)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(_mpl(TEXT))

        y = np.arange(len(quality_labels))
        ax_quality.barh(y - 0.18, audio_quality * bar_progress, height=0.32, color=_mpl(ACCENT_APC_AUDIO), label="Audio")
        ax_quality.barh(y + 0.18, cf_quality * bar_progress, height=0.32, color=_mpl(ACCENT_APC), label="CF")
        for idx, value in enumerate(cf_quality):
            if bar_progress > 0.86:
                ax_quality.text(value + 0.012, idx + 0.18, f"{value:.3f}" if value < 0.1 else f"{value:.2f}", color=_mpl(TEXT), va="center", fontsize=9)
        ax_quality.set_title("Model quality separates from the audio baseline", color=_mpl(TEXT), fontsize=13, fontweight="bold", loc="left")
        ax_quality.set_yticks(y, quality_labels)
        ax_quality.set_xlim(0.0, max(0.52, float(np.nanmax(cf_quality)) + 0.08))
        ax_quality.invert_yaxis()
        legend = ax_quality.legend(loc="lower right", frameon=False, fontsize=8.5)
        if legend is not None:
            for text in legend.get_texts():
                text.set_color(_mpl(TEXT))

        revealed = np.full(hit_matrix.shape, np.nan, dtype=float)
        revealed[:, :topk_visible] = np.where(hit_matrix[:, :topk_visible] > 0, 2.0, 1.0)
        cmap = plt.matplotlib.colors.ListedColormap([_mpl((232, 226, 216)), _mpl(ACCENT_APC_GOOD)])
        ax_hits.imshow(revealed - 1.0, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        ax_hits.set_title(f"Top-10 target hits reveal over ranks 1-{topk_visible}", color=_mpl(TEXT), fontsize=13, fontweight="bold", loc="left")
        ax_hits.set_xticks(np.arange(10), [str(idx) for idx in range(1, 11)])
        ax_hits.set_yticks(np.arange(len(playlists)), playlists)
        ax_hits.tick_params(axis="y", labelsize=7.5)
        ax_hits.set_xlabel("Recommendation rank", color=_mpl(MUTED), fontsize=9)
        ax_hits.set_ylabel("Playlist", color=_mpl(MUTED), fontsize=9)
        ax_hits.set_xticks(np.arange(-0.5, 10, 1), minor=True)
        ax_hits.set_yticks(np.arange(-0.5, len(playlists), 1), minor=True)
        ax_hits.grid(which="minor", color=_mpl(CARD), linewidth=1.1)
        ax_hits.tick_params(which="minor", bottom=False, left=False)
        for row_idx, col_idx in zip(*np.where((hit_matrix > 0) & (np.arange(10)[None, :] < topk_visible))):
            ax_hits.text(col_idx, row_idx, "hit", ha="center", va="center", color="white", fontsize=7.5, fontweight="bold")

        cards = [
            ("Epoch", f"{epoch_label}/{int(epochs[-1])}", "line extends over time"),
            ("Loss", f"{current_loss:.3f}", f"start {train_loss[0]:.3f}"),
            ("Best Hit@10", f"{current_best_hit:.2f}", f"current {current_hit:.2f}"),
            ("Target P@10", f"{current_precision:.2f}", f"MRR {current_mrr:.3f}"),
            ("CF lift", f"{lift * bar_progress:.1f}x", "vs audio Hit@10"),
            ("Embeddings", f"{embedding_coverage * min(1.0, progress * 1.4):.0%}", "selected files present"),
        ]
        ax_cards = fig.add_axes([0.035, 0.035, 0.93, 0.125])
        ax_cards.set_axis_off()
        card_gap = 0.014
        card_w = (1.0 - card_gap * (len(cards) - 1)) / len(cards)
        for idx, (title, value, subtitle) in enumerate(cards):
            x0 = idx * (card_w + card_gap)
            ax_cards.add_patch(
                mpatches.FancyBboxPatch(
                    (x0, 0.04),
                    card_w,
                    0.88,
                    boxstyle="round,pad=0.012,rounding_size=0.032",
                    facecolor=_mpl(CARD),
                    edgecolor=_mpl(LINE),
                    linewidth=1.2,
                    transform=ax_cards.transAxes,
                )
            )
            ax_cards.add_patch(
                mpatches.FancyBboxPatch(
                    (x0, 0.04),
                    0.012,
                    0.88,
                    boxstyle="round,pad=0.012,rounding_size=0.032",
                    facecolor=_mpl(ACCENT_APC if idx != 2 else ACCENT_APC_GOLD),
                    edgecolor=_mpl(ACCENT_APC if idx != 2 else ACCENT_APC_GOLD),
                    linewidth=0,
                    transform=ax_cards.transAxes,
                )
            )
            ax_cards.text(x0 + 0.025, 0.73, title, transform=ax_cards.transAxes, color=_mpl(MUTED), fontsize=9.2, va="center")
            ax_cards.text(x0 + 0.025, 0.43, value, transform=ax_cards.transAxes, color=_mpl(TEXT), fontsize=16, fontweight="bold", va="center")
            ax_cards.text(x0 + 0.025, 0.16, subtitle, transform=ax_cards.transAxes, color=_mpl(MUTED), fontsize=8.4, va="center")

        frames.append(_figure_to_exact_image(fig, width, height, dpi).quantize(colors=192, method=Image.MEDIANCUT))

    gif_out = output_dir / "readme_automatic_playlist_continuation_animated_panel.gif"
    ensure_dir(gif_out.parent)
    frames[0].save(
        gif_out,
        save_all=True,
        append_images=frames[1:],
        duration=130,
        loop=0,
        optimize=False,
        disposal=2,
    )
    return gif_out


def _write_apc_readme_snippet(output_dir: Path) -> Path:
    snippet = output_dir / "README_automatic_playlist_continuation_visuals.md"
    lines = [
        "### Automatic Playlist Continuation",
        "",
        "![Automatic Playlist Continuation Animated Panel](readme_automatic_playlist_continuation_animated_panel.gif)",
        "",
        "![Automatic Playlist Continuation Static Panel](readme_automatic_playlist_continuation_static_panel.png)",
        "",
    ]
    snippet.write_text("\n".join(lines), encoding="utf-8")
    return snippet


def build_automatic_playlist_continuation_panels(run_dir: Path, output_dir: Path | None = None) -> None:
    run_dir = run_dir.resolve()
    visual_dir = _ensure_apc_visuals(run_dir)
    target_dir = ensure_dir((output_dir or run_dir / "readme").resolve())
    metrics = _apc_metric_cards(run_dir)
    static_out = _save_apc_static_panel(run_dir, visual_dir, target_dir, metrics)
    gif_out = _save_apc_animated_panel(run_dir, visual_dir, target_dir, metrics)
    snippet = _write_apc_readme_snippet(target_dir)
    print(f"[README Panels] APC static panel: {static_out}")
    print(f"[README Panels] APC animated panel: {gif_out}")
    print(f"[README Panels] APC README snippet: {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build README-ready presentation panels from generated outputs.")
    parser.add_argument("--output-dir", type=Path, default=README_DIR)
    parser.add_argument(
        "--suite",
        choices=("legacy", "audio", "classifier", "automatic_playlist_continuation", "all"),
        default="legacy",
        help="Which README panel set to build. Default preserves the original audio+classifier behavior.",
    )
    parser.add_argument("--apc-run-dir", type=Path, default=None, help="Automatic playlist continuation run folder.")
    parser.add_argument("--apc-output-dir", type=Path, default=None, help="Output folder for APC README panels. Defaults to APC_RUN_DIR/readme.")
    args = parser.parse_args()

    if args.suite in ("legacy", "audio", "all"):
        output_dir = ensure_dir(args.output_dir)
        build_audio_panels(output_dir)
        print(f"[README Panels] Wrote audio assets under {output_dir}")

    if args.suite in ("legacy", "classifier", "all"):
        output_dir = ensure_dir(args.output_dir)
        build_classifier_panels(output_dir)
        print(f"[README Panels] Wrote classifier assets under {output_dir}")

    if args.suite in ("automatic_playlist_continuation", "all"):
        build_automatic_playlist_continuation_panels(args.apc_run_dir or _latest_apc_run_dir(), args.apc_output_dir)


if __name__ == "__main__":
    main()
