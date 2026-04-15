#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFont
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import (
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Build README-ready presentation panels from generated outputs.")
    parser.add_argument("--output-dir", type=Path, default=README_DIR)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    build_audio_panels(output_dir)
    build_classifier_panels(output_dir)
    print(f"[README Panels] Wrote assets under {output_dir}")


if __name__ == "__main__":
    main()
