#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.constants import (  # noqa: E402
    EVENT_TYPE_LABELS,
    FIELD_SPECS,
    INSTRUMENT_COLORS,
    INSTRUMENT_LABELS,
    TIME_STEPS_PER_BEAT,
)
from scripts.multitrack_generation.data import collect_split_files, resolve_processed_dir, summarize_packed_split, summarize_split  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402
from utils.project_paths import (  # noqa: E402
    MULTITRACK_GENERATION_GENERATED_DIR,
    MULTITRACK_GENERATION_LOG_ROOT,
    MULTITRACK_GENERATION_OUTPUT_ROOT,
    MULTITRACK_GENERATION_README_DIR,
    MULTITRACK_GENERATION_VISUAL_DIR,
)


BG = "#f4f6f4"
PANEL = "#ffffff"
INK = "#14202b"
MUTED = "#5e6b76"
GRID = "#d7dee3"
GRID_LIGHT = "#eef2f4"
ACCENT = "#2563ad"
ACCENT_2 = "#0f766e"
WARN = "#b45309"
BAD = "#a23b68"
GOOD = "#2c7a4b"
FIELD_COLORS = ["#2563ad", "#0f766e", "#b45309", "#a23b68", "#6d5bd0", "#2c7a4b"]
INSTRUMENT_COLOR_LIST = [INSTRUMENT_COLORS[label] for label in INSTRUMENT_LABELS]


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates: list[str] = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/root/.vscode-server/bin/41dd792b5e652393e7787322889ed5fdc58bd75b/node_modules/katex/dist/fonts/KaTeX_SansSerif-Bold.ttf"
        if bold
        else "/root/.vscode-server/bin/41dd792b5e652393e7787322889ed5fdc58bd75b/node_modules/katex/dist/fonts/KaTeX_SansSerif-Regular.ttf",
        "/root/.vscode-server/bin/41dd792b5e652393e7787322889ed5fdc58bd75b/node_modules/katex/dist/fonts/KaTeX_Main-Bold.ttf"
        if bold
        else "/root/.vscode-server/bin/41dd792b5e652393e7787322889ed5fdc58bd75b/node_modules/katex/dist/fonts/KaTeX_Main-Regular.ttf",
    ]
    vscode_roots = sorted(Path("/root/.vscode-server/bin").glob("*/node_modules/katex/dist/fonts"))
    for root in vscode_roots:
        candidates.append(str(root / ("KaTeX_SansSerif-Bold.ttf" if bold else "KaTeX_SansSerif-Regular.ttf")))
        candidates.append(str(root / ("KaTeX_Main-Bold.ttf" if bold else "KaTeX_Main-Regular.ttf")))
    for candidate in candidates:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


@lru_cache(maxsize=512)
def _text_scale(size: int, bold: bool = False) -> float:
    font = _font(10, bold)
    bbox = font.getbbox("Ag")
    base_height = max(1, bbox[3] - bbox[1])
    return max(1.0, float(size) / float(base_height))


def _measure_text(text: str, size: int, bold: bool = False) -> tuple[int, int]:
    font = _font(size, bold)
    bbox = font.getbbox(text or " ")
    return max(1, int(math.ceil(bbox[2] - bbox[0]))), max(1, int(math.ceil(bbox[3] - bbox[1])))


def _hex(color: str) -> tuple[int, int, int]:
    color = color.lstrip("#")
    return tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))


def _blend(a: str, b: str, t: float) -> tuple[int, int, int]:
    ar, ag, ab = _hex(a)
    br, bg, bb = _hex(b)
    t = max(0.0, min(float(t), 1.0))
    return (int(ar + (br - ar) * t), int(ag + (bg - ag) * t), int(ab + (bb - ab) * t))


def _text(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int, color: str = INK, bold: bool = False, anchor: str | None = None) -> None:
    font = _font(size, bold)
    text = str(text or " ")
    bbox = draw.textbbox((0, 0), text, font=font)
    width = max(1, int(math.ceil(bbox[2] - bbox[0])))
    height = max(1, int(math.ceil(bbox[3] - bbox[1])))
    x, y = int(xy[0]), int(xy[1])
    if anchor:
        if anchor[0] == "r":
            x -= width
        elif anchor[0] == "m":
            x -= width // 2
        if len(anchor) > 1 and anchor[1] == "m":
            y -= height // 2
    draw.text((x - bbox[0], y - bbox[1]), text, fill=_hex(color), font=font)


def _fit_text(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], text: str, size: int, color: str = INK, bold: bool = False) -> None:
    words = text.split()
    lines: list[str] = []
    current = ""
    width = box[2] - box[0]
    for word in words:
        candidate = f"{current} {word}".strip()
        if _measure_text(candidate, size, bold)[0] <= width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    y = box[1]
    for line in lines:
        line_height = _measure_text(line, size, bold)[1]
        if y + line_height > box[3]:
            break
        _text(draw, (box[0], y), line, size, color, bold)
        y += int(line_height * 1.35)


def _fmt_int(value: int | float) -> str:
    value = int(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 10_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,}"


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _canvas(width: int, height: int, title: str, subtitle: str | None = None) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), _hex(BG))
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 112), fill=_hex("#eaf0ee"))
    _text(draw, (42, 22), title, 42, INK, bold=True)
    if subtitle:
        _fit_text(draw, (44, 72, width - 44, 108), subtitle, 20, MUTED)
    return image, draw


def _panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str | None = None) -> None:
    draw.rounded_rectangle(box, radius=8, fill=_hex(PANEL), outline=_hex("#cbd5dc"), width=1)
    if title:
        _text(draw, (box[0] + 22, box[1] + 16), title, 24, INK, bold=True)


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max <= src_min:
        return (dst_min + dst_max) / 2.0
    return dst_min + (float(value) - src_min) / (src_max - src_min) * (dst_max - dst_min)


def _metric_tile(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], label: str, value: str, color: str) -> None:
    _panel(draw, box)
    draw.rounded_rectangle((box[0], box[1], box[0] + 8, box[3]), radius=4, fill=_hex(color))
    _text(draw, (box[0] + 26, box[1] + 18), label, 18, MUTED, bold=True)
    _text(draw, (box[0] + 26, box[1] + 54), value, 38, INK, bold=True)


def _draw_line_chart(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    x_values: list[float],
    y_values: list[float],
    title: str,
    color: str,
    y_label: str = "",
    y_min: float | None = None,
    y_max: float | None = None,
) -> None:
    _panel(draw, box, title)
    left, top, right, bottom = box[0] + 72, box[1] + 62, box[2] - 30, box[3] - 52
    for index in range(5):
        y = int(top + index * (bottom - top) / 4)
        draw.line((left, y, right, y), fill=_hex(GRID_LIGHT), width=1)
    draw.line((left, bottom, right, bottom), fill=_hex(GRID), width=2)
    draw.line((left, top, left, bottom), fill=_hex(GRID), width=2)
    if not y_values:
        _text(draw, (left + 20, top + 60), "No values yet", 17, MUTED)
        return
    if not x_values:
        x_values = list(range(len(y_values)))
    lo = min(y_values) if y_min is None else y_min
    hi = max(y_values) if y_max is None else y_max
    if abs(hi - lo) < 1e-9:
        hi = lo + 1.0
    x_lo, x_hi = min(x_values), max(x_values)
    if abs(x_hi - x_lo) < 1e-9:
        x_hi = x_lo + 1.0
    for tick in range(5):
        value = lo + (hi - lo) * (4 - tick) / 4
        y = int(top + tick * (bottom - top) / 4)
        _text(draw, (box[0] + 18, y - 10), f"{value:.2f}", 15, MUTED)
    points = [
        (
            int(_scale(x, x_lo, x_hi, left, right)),
            int(_scale(y, lo, hi, bottom, top)),
        )
        for x, y in zip(x_values, y_values)
    ]
    if len(points) > 1:
        draw.line(points, fill=_hex(color), width=3)
    for point in points[:: max(1, len(points) // 20)]:
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=_hex(color))
    _text(draw, (left, bottom + 18), f"step {_fmt_int(x_lo)}", 15, MUTED)
    _text(draw, (right - 116, bottom + 18), f"step {_fmt_int(x_hi)}", 15, MUTED)
    if y_label:
        _text(draw, (left, top - 28), y_label, 15, MUTED)


def _draw_horizontal_bars(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    values: list[float],
    labels: list[str],
    colors: list[str],
    title: str,
    value_suffix: str = "",
) -> None:
    _panel(draw, box, title)
    left, top, right, bottom = box[0] + 170, box[1] + 58, box[2] - 34, box[3] - 28
    max_value = max(max(values) if values else 0.0, 1e-9)
    row_count = max(len(values), 1)
    row_h = min(42, max(18, int((bottom - top) / row_count)))
    if top + row_h * row_count > bottom:
        top = box[1] + 48
        bottom = box[3] - 16
        row_h = max(12, int((bottom - top) / row_count))
    label_size = 16 if row_h >= 24 else 12
    value_size = 15 if row_h >= 24 else 12
    for index, (label, value, color) in enumerate(zip(labels, values, colors)):
        y = top + index * row_h
        if y + row_h > box[3] - 4:
            break
        text_y = y + max(1, (row_h - label_size) // 2)
        bar_top_offset = max(4, row_h // 4)
        bar_bottom_offset = max(bar_top_offset + 4, row_h - max(3, row_h // 5))
        bar_top = y + bar_top_offset
        bar_bottom = y + bar_bottom_offset
        _text(draw, (box[0] + 22, text_y), label, label_size, MUTED, bold=True)
        bar_right = int(_scale(value, 0.0, max_value, left, right))
        draw.rounded_rectangle((left, bar_top, right, bar_bottom), radius=5, fill=_hex("#edf2f4"))
        draw.rounded_rectangle((left, bar_top, max(left + 2, bar_right), bar_bottom), radius=5, fill=_hex(color))
        _text(draw, (right - 2, text_y), f"{value:.3f}{value_suffix}" if max_value <= 1.5 else f"{_fmt_int(value)}{value_suffix}", value_size, INK, anchor="ra")


def _draw_stacked_split_bars(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], split_counts: dict[str, list[int]]) -> None:
    _panel(draw, box, "Instrument mix by split")
    left, top, right = box[0] + 150, box[1] + 72, box[2] - 42
    row_h = 58
    for row, split in enumerate(("train", "valid", "test")):
        y0 = top + row * row_h
        counts = np.asarray(split_counts.get(split, [0] * len(INSTRUMENT_LABELS)), dtype=float)
        total = max(float(counts.sum()), 1.0)
        _text(draw, (box[0] + 28, y0 + 8), split, 16, MUTED, bold=True)
        cursor = left
        for index, count in enumerate(counts):
            width = int((right - left) * count / total)
            if width <= 0:
                continue
            draw.rectangle((cursor, y0, cursor + width, y0 + 30), fill=_hex(INSTRUMENT_COLOR_LIST[index]))
            cursor += width
        draw.rectangle((left, y0, right, y0 + 30), outline=_hex(GRID), width=1)
        _text(draw, (right, y0 + 36), f"{_fmt_int(total)} notes", 13, MUTED, anchor="ra")
    legend_y = box[3] - 52
    for index, label in enumerate(INSTRUMENT_LABELS):
        x = box[0] + 34 + index * 156
        draw.rounded_rectangle((x, legend_y, x + 24, legend_y + 14), radius=3, fill=_hex(INSTRUMENT_COLOR_LIST[index]))
        _text(draw, (x + 32, legend_y - 3), label, 13, MUTED)


def _draw_pitch_histogram(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], pitch_counts: list[int], title: str) -> None:
    _panel(draw, box, title)
    counts = np.asarray(pitch_counts, dtype=float)
    left, top, right, bottom = box[0] + 58, box[1] + 62, box[2] - 28, box[3] - 44
    transformed = np.sqrt(counts)
    max_value = max(float(transformed.max()) if transformed.size else 0.0, 1.0)
    for pitch in range(0, 128, 12):
        x = int(_scale(pitch, 0, 127, left, right))
        draw.line((x, top, x, bottom), fill=_hex(GRID_LIGHT), width=1)
        _text(draw, (x - 8, bottom + 14), str(pitch), 11, MUTED)
    bar_w = max(1, int((right - left) / 128))
    for pitch, value in enumerate(transformed):
        x0 = int(_scale(pitch, 0, 128, left, right))
        x1 = max(x0 + 1, x0 + bar_w)
        y = int(_scale(value, 0, max_value, bottom, top))
        color = _blend("#cfe0f7", ACCENT, value / max_value)
        draw.rectangle((x0, y, x1, bottom), fill=color)
    _text(draw, (left, bottom + 30), "MIDI pitch", 12, MUTED)


def _latest_value(metrics: dict | None, key: str, fallback: str = "n/a") -> str:
    if not metrics or key not in metrics:
        return fallback
    value = metrics[key]
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_dataset_overview(output_root: Path = MULTITRACK_GENERATION_OUTPUT_ROOT, max_files: int | None = None) -> Path:
    summary_path = output_root / "dataset" / "dataset_summary.json"
    summary = _read_json(summary_path)
    if summary is None:
        processed_dir = resolve_processed_dir(None)
        splits = collect_split_files(processed_dir)
        def summarize(name: str, files: list[Path]) -> dict:
            if max_files is None:
                packed = summarize_packed_split(processed_dir, name)
                if packed is not None:
                    return packed
            return summarize_split(files, max_files=max_files)
        summary = {
            "processed_dir": str(processed_dir),
            "splits": {
                "train": summarize("train", splits.train),
                "valid": summarize("valid", splits.valid),
                "test": summarize("test", splits.test),
            },
        }
        save_json(summary_path, summary)

    image, draw = _canvas(1800, 1100, "Multitrack Generation Dataset", f"Processed arrays: {summary['processed_dir']}")
    splits = summary["splits"]
    total_files = sum(int(splits[name]["files"]) for name in ("train", "valid", "test"))
    total_notes = sum(int(splits[name]["notes"]) for name in ("train", "valid", "test"))
    _metric_tile(draw, (42, 126, 398, 236), "files", _fmt_int(total_files), ACCENT)
    _metric_tile(draw, (422, 126, 778, 236), "notes", _fmt_int(total_notes), GOOD)
    _metric_tile(draw, (802, 126, 1158, 236), "train split", f"{_fmt_int(splits['train']['files'])} files", WARN)
    _metric_tile(draw, (1182, 126, 1538, 236), "test split", f"{_fmt_int(splits['test']['files'])} files", BAD)

    split_counts = {name: splits[name]["instrument_counts"] for name in ("train", "valid", "test")}
    _draw_stacked_split_bars(draw, (42, 270, 870, 560), split_counts)

    train_counts = np.asarray(splits["train"]["instrument_counts"], dtype=float)
    total_train = max(float(train_counts.sum()), 1.0)
    shares = (train_counts / total_train).tolist()
    _draw_horizontal_bars(
        draw,
        (910, 270, 1758, 560),
        shares,
        list(INSTRUMENT_LABELS),
        INSTRUMENT_COLOR_LIST,
        "Train instrument share",
        value_suffix="",
    )
    _draw_pitch_histogram(draw, (42, 600, 870, 1030), splits["train"]["pitch_counts"], "Train pitch distribution")
    _draw_pitch_histogram(draw, (910, 600, 1758, 1030), splits["test"]["pitch_counts"], "Test pitch distribution")

    output_path = ensure_dir(MULTITRACK_GENERATION_VISUAL_DIR / "dataset") / "dataset_overview.png"
    image.save(output_path)
    return output_path


def render_training_overview(run_name: str, output_root: Path = MULTITRACK_GENERATION_OUTPUT_ROOT) -> Path | None:
    run_dir = output_root / "runs" / run_name
    log_dir = MULTITRACK_GENERATION_LOG_ROOT / run_name
    history = _read_json(run_dir / "history.json")
    metrics = _read_json(run_dir / "final_metrics.json")
    train_rows = _read_csv(log_dir / "train_steps.csv")
    val_rows = _read_csv(log_dir / "validation_steps.csv")
    if history is None and metrics is None and not train_rows and not val_rows:
        return None

    if train_rows:
        train_steps = [float(row["step"]) for row in train_rows if row.get("step")]
        train_losses = [float(row["train_loss"]) for row in train_rows if row.get("train_loss")]
    else:
        train_losses = [float(item) for item in (history or {}).get("train_losses", [])]
        train_steps = list(range(1, len(train_losses) + 1))
    if val_rows:
        val_steps = [float(row["step"]) for row in val_rows if row.get("step")]
        val_losses = [float(row["val_loss"]) for row in val_rows if row.get("val_loss")]
        val_accs = [float(row["val_accuracy"]) for row in val_rows if row.get("val_accuracy")]
        latest_val = val_rows[-1]
    else:
        rows = (history or {}).get("history", [])
        val_steps = [float(row["step"]) for row in rows if "step" in row]
        val_losses = [float(row["val_loss"]) for row in rows if "val_loss" in row]
        val_accs = [float(row["val_accuracy"]) for row in rows if "val_accuracy" in row]
        latest_val = rows[-1] if rows else {}

    image, draw = _canvas(1800, 1100, f"Multitrack Training: {run_name}", str(run_dir))
    _metric_tile(draw, (42, 126, 382, 236), "latest step", _fmt_int(train_steps[-1] if train_steps else metrics.get("steps", 0) if metrics else 0), ACCENT)
    _metric_tile(draw, (404, 126, 744, 236), "latest train loss", f"{train_losses[-1]:.3f}" if train_losses else "n/a", WARN)
    _metric_tile(draw, (766, 126, 1106, 236), "latest val loss", f"{val_losses[-1]:.3f}" if val_losses else _latest_value(metrics, "final_val_loss"), BAD)
    _metric_tile(draw, (1128, 126, 1468, 236), "latest val acc", f"{val_accs[-1]:.3f}" if val_accs else _latest_value(metrics, "final_val_accuracy"), GOOD)

    _draw_line_chart(draw, (42, 270, 870, 630), train_steps, train_losses, "Training loss", ACCENT, "loss")
    _draw_line_chart(draw, (910, 270, 1758, 630), val_steps, val_losses, "Validation loss", BAD, "loss")
    _draw_line_chart(draw, (42, 670, 870, 1030), val_steps, val_accs, "Validation accuracy", GOOD, "accuracy", y_min=0.0, y_max=1.0)

    field_acc: list[float] = []
    field_loss: list[float] = []
    for field in FIELD_SPECS:
        if metrics and "per_field_accuracy" in metrics:
            field_acc.append(float(metrics["per_field_accuracy"].get(field.name, 0.0)))
            field_loss.append(float(metrics.get("per_field_loss", {}).get(field.name, 0.0)))
        elif latest_val:
            field_acc.append(float(latest_val.get(f"val_acc_{field.name}", 0.0)))
            field_loss.append(float(latest_val.get(f"val_loss_{field.name}", 0.0)))
        else:
            field_acc.append(0.0)
            field_loss.append(0.0)
    _draw_horizontal_bars(draw, (910, 670, 1328, 1030), field_acc, [f.name for f in FIELD_SPECS], FIELD_COLORS, "Per-field accuracy")
    _draw_horizontal_bars(draw, (1352, 670, 1758, 1030), field_loss, [f.name for f in FIELD_SPECS], FIELD_COLORS, "Per-field loss")

    output_path = ensure_dir(MULTITRACK_GENERATION_VISUAL_DIR / "training") / f"{run_name}_training_overview.png"
    image.save(output_path)
    return output_path


def _matrix_labels(field_name: str, size: int) -> list[str]:
    if field_name == "type":
        return list(EVENT_TYPE_LABELS)
    if field_name == "instrument":
        return list(INSTRUMENT_LABELS)
    if size <= 16:
        return [str(index) for index in range(size)]
    step = 8 if size <= 64 else 16
    return [str(index) if index % step == 0 else "" for index in range(size)]


def render_confusion_matrix(matrix_path: Path, output_path: Path, title: str, field_name: str) -> Path:
    matrix = np.load(matrix_path)
    labels = _matrix_labels(field_name, matrix.shape[0])
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, np.maximum(row_sums, 1), out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)
    total = int(matrix.sum())
    accuracy = float(np.trace(matrix) / total) if total else 0.0

    image, draw = _canvas(1400, 1040, title, "Rows are ground truth. Columns are predictions. Color is row-normalized probability.")
    _metric_tile(draw, (42, 126, 352, 226), "accuracy", f"{accuracy:.3f}", GOOD)
    _metric_tile(draw, (374, 126, 684, 226), "tokens", _fmt_int(total), ACCENT)
    _metric_tile(draw, (706, 126, 1016, 226), "classes", str(matrix.shape[0]), WARN)

    left, top, size = 86, 286, 704
    cell = size / max(matrix.shape[0], matrix.shape[1], 1)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = float(normalized[row, col])
            color = _blend("#f7fbff", "#1d4f91", math.sqrt(value))
            x0 = int(left + col * cell)
            y0 = int(top + row * cell)
            x1 = max(x0 + 1, int(left + (col + 1) * cell))
            y1 = max(y0 + 1, int(top + (row + 1) * cell))
            draw.rectangle((x0, y0, x1, y1), fill=color)
    draw.rectangle((left, top, left + size, top + size), outline=_hex("#7f8c99"), width=2)

    for index, label in enumerate(labels):
        if not label:
            continue
        center = int(left + (index + 0.5) * cell)
        _text(draw, (center, top + size + 14), label, 10, MUTED, anchor="ma")
        _text(draw, (left - 12, int(top + (index + 0.5) * cell)), label, 10, MUTED, anchor="ra")
    _text(draw, (left + size // 2, top + size + 46), "prediction", 15, MUTED, anchor="ma")
    _text(draw, (left - 42, top - 30), "truth", 15, MUTED)

    side = (850, 286, 1348, 990)
    _panel(draw, side, "Top off-diagonal errors")
    off_diag = matrix.astype(np.int64).copy()
    diag_len = min(off_diag.shape)
    off_diag[np.arange(diag_len), np.arange(diag_len)] = 0
    flat_indices = np.argsort(off_diag.ravel())[::-1][:12]
    y = side[1] + 62
    for rank, flat in enumerate(flat_indices, start=1):
        count = int(off_diag.ravel()[flat])
        if count <= 0:
            continue
        truth, pred = np.unravel_index(flat, off_diag.shape)
        truth_label = EVENT_TYPE_LABELS[truth] if field_name == "type" and truth < len(EVENT_TYPE_LABELS) else INSTRUMENT_LABELS[truth] if field_name == "instrument" and truth < len(INSTRUMENT_LABELS) else str(truth)
        pred_label = EVENT_TYPE_LABELS[pred] if field_name == "type" and pred < len(EVENT_TYPE_LABELS) else INSTRUMENT_LABELS[pred] if field_name == "instrument" and pred < len(INSTRUMENT_LABELS) else str(pred)
        _text(draw, (side[0] + 24, y), f"{rank:02d}. {truth_label} -> {pred_label}", 15, INK, bold=True)
        _text(draw, (side[2] - 24, y), _fmt_int(count), 15, MUTED, anchor="ra")
        y += 38
    if y == side[1] + 62:
        _text(draw, (side[0] + 24, y), "No off-diagonal errors in this matrix.", 15, MUTED)

    ensure_dir(output_path.parent)
    image.save(output_path)
    return output_path


def render_generated_piano_roll(notes_path: Path, output_path: Path, title: str, progress: float = 1.0) -> Path:
    notes = np.load(notes_path)
    summary = _read_json(notes_path.parent / "summary.json") or {}
    image, draw = _canvas(1800, 1000, title, str(notes_path))
    plot = (82, 178, 1336, 790)
    _panel(draw, (52, 132, 1370, 850), "Piano roll")
    draw.rectangle(plot, fill=_hex("#fbfdff"), outline=_hex(GRID), width=1)
    side = (1402, 132, 1756, 850)
    _panel(draw, side, "Generation summary")

    if notes.size == 0:
        _text(draw, (plot[0] + 34, plot[1] + 160), "No note events in this generation.", 22, MUTED)
    else:
        starts = notes[:, 0].astype(float) / TIME_STEPS_PER_BEAT
        durations = np.maximum(notes[:, 2].astype(float) / TIME_STEPS_PER_BEAT, 0.06)
        pitches = notes[:, 1].astype(float)
        instruments = notes[:, 3].astype(int)
        x_max = max(float(np.max(starts + durations)), 1.0)
        reveal_x = x_max * max(0.0, min(progress, 1.0))
        y_min = max(0.0, math.floor((float(np.min(pitches)) - 4) / 12) * 12)
        y_max = min(127.0, math.ceil((float(np.max(pitches)) + 4) / 12) * 12)
        if y_max <= y_min:
            y_max = y_min + 12
        beat_step = 4 if x_max > 32 else 1
        for beat in range(0, int(math.ceil(x_max)) + 1, beat_step):
            x = int(_scale(beat, 0, x_max, plot[0], plot[2]))
            draw.line((x, plot[1], x, plot[3]), fill=_hex(GRID_LIGHT if beat % 4 else GRID), width=1)
            _text(draw, (x, plot[3] + 12), str(beat), 11, MUTED, anchor="ma")
        for pitch in range(int(y_min), int(y_max) + 1, 12):
            y = int(_scale(pitch, y_min, y_max, plot[3], plot[1]))
            draw.line((plot[0], y, plot[2], y), fill=_hex(GRID_LIGHT), width=1)
            _text(draw, (plot[0] - 12, y), str(pitch), 11, MUTED, anchor="ra")
        visible = starts <= reveal_x if progress < 0.999 else np.ones_like(starts, dtype=bool)
        order = np.argsort(starts)
        for index in order:
            if not visible[index]:
                continue
            x0 = int(_scale(starts[index], 0.0, x_max, plot[0], plot[2]))
            x1 = int(_scale(starts[index] + durations[index], 0.0, x_max, plot[0], plot[2]))
            y = int(_scale(pitches[index], y_min, y_max, plot[3], plot[1]))
            instrument = max(0, min(int(instruments[index]), len(INSTRUMENT_LABELS) - 1))
            color = INSTRUMENT_COLOR_LIST[instrument]
            draw.rectangle((x0, y - 5, max(x1, x0 + 8), y + 5), fill=_hex(color))
        if progress < 0.999:
            x = int(_scale(reveal_x, 0.0, x_max, plot[0], plot[2]))
            draw.line((x, plot[1], x, plot[3]), fill=_hex(INK), width=3)
        _text(draw, (plot[0], plot[3] + 42), "Beat", 13, MUTED)
        _text(draw, (plot[0], plot[1] - 24), f"Pitch range {int(y_min)}-{int(y_max)}", 13, MUTED)

    y = side[1] + 64
    for label, value, color in [
        ("events", _fmt_int(summary.get("sequence_len", 0)), ACCENT),
        ("notes", _fmt_int(summary.get("note_count", int(notes.shape[0]) if notes.size else 0)), GOOD),
        ("directory", notes_path.parent.name, WARN),
    ]:
        _text(draw, (side[0] + 24, y), label, 13, MUTED, bold=True)
        _fit_text(draw, (side[0] + 24, y + 22, side[2] - 24, y + 58), str(value), 20, color, bold=True)
        y += 82
    counts = summary.get("instrument_counts", {})
    _text(draw, (side[0] + 24, y + 8), "instrument notes", 16, INK, bold=True)
    y += 44
    max_count = max([int(counts.get(label, 0)) for label in INSTRUMENT_LABELS] + [1])
    for index, label in enumerate(INSTRUMENT_LABELS):
        count = int(counts.get(label, 0))
        draw.rounded_rectangle((side[0] + 24, y + 4, side[2] - 72, y + 22), radius=4, fill=_hex("#edf2f4"))
        bar_w = int((side[2] - side[0] - 96) * count / max_count)
        draw.rounded_rectangle((side[0] + 24, y + 4, side[0] + 24 + bar_w, y + 22), radius=4, fill=_hex(INSTRUMENT_COLOR_LIST[index]))
        _text(draw, (side[0] + 24, y + 28), label, 12, MUTED)
        _text(draw, (side[2] - 24, y + 8), _fmt_int(count), 12, INK, anchor="ra")
        y += 54

    legend_y = 890
    for index, label in enumerate(INSTRUMENT_LABELS):
        x = 82 + index * 220
        draw.rounded_rectangle((x, legend_y, x + 42, legend_y + 18), radius=4, fill=_hex(INSTRUMENT_COLOR_LIST[index]))
        _text(draw, (x + 54, legend_y - 2), label, 16, MUTED)

    ensure_dir(output_path.parent)
    image.save(output_path)
    return output_path


def render_generation_visuals(run_name: str) -> list[Path]:
    generated_dir = MULTITRACK_GENERATION_GENERATED_DIR / run_name
    notes_path = generated_dir / "notes.npy"
    if not notes_path.exists():
        return []
    output_dir = ensure_dir(MULTITRACK_GENERATION_VISUAL_DIR / "generated")
    static_path = render_generated_piano_roll(notes_path, output_dir / f"{run_name}_piano_roll.png", f"Generated Multitrack Music: {run_name}")
    frames: list[Image.Image] = []
    for index in range(12):
        progress = (index + 1) / 12.0
        temp_path = output_dir / f".{run_name}_frame_{index:02d}.png"
        render_generated_piano_roll(notes_path, temp_path, f"Generated Multitrack Music: {run_name}", progress=progress)
        with Image.open(temp_path) as frame:
            frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=224))
        temp_path.unlink(missing_ok=True)
    gif_path = output_dir / f"{run_name}_piano_roll.gif"
    if frames:
        frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=130, loop=0, optimize=False)
    return [static_path, gif_path]


def render_available_confusions(run_name: str, output_root: Path = MULTITRACK_GENERATION_OUTPUT_ROOT) -> list[Path]:
    candidates = [
        output_root / "runs" / run_name / "metrics",
        output_root / "evaluation" / run_name / "test",
        output_root / "evaluation" / run_name / "valid",
    ]
    rendered: list[Path] = []
    for candidate in candidates:
        if not candidate.exists():
            continue
        for field in FIELD_SPECS:
            matrix_path = candidate / f"{field.name}_confusion.npy"
            if not matrix_path.exists():
                continue
            rendered.append(
                render_confusion_matrix(
                    matrix_path,
                    MULTITRACK_GENERATION_VISUAL_DIR / "confusion" / f"{run_name}_{field.name}_confusion.png",
                    f"{run_name}: {field.name} confusion",
                    field.name,
                )
            )
    return rendered


def _training_series(run_name: str, output_root: Path) -> tuple[list[float], list[float], list[float], list[float], list[float], dict | None]:
    run_dir = output_root / "runs" / run_name
    log_dir = MULTITRACK_GENERATION_LOG_ROOT / run_name
    history = _read_json(run_dir / "history.json")
    metrics = _read_json(run_dir / "final_metrics.json")
    train_rows = _read_csv(log_dir / "train_steps.csv")
    val_rows = _read_csv(log_dir / "validation_steps.csv")
    if train_rows:
        train_steps = [float(row["step"]) for row in train_rows if row.get("step")]
        train_losses = [float(row["train_loss"]) for row in train_rows if row.get("train_loss")]
    else:
        train_losses = [float(item) for item in (history or {}).get("train_losses", [])]
        train_steps = list(range(1, len(train_losses) + 1))
    if val_rows:
        val_steps = [float(row["step"]) for row in val_rows if row.get("step")]
        val_losses = [float(row["val_loss"]) for row in val_rows if row.get("val_loss")]
        val_accs = [float(row["val_accuracy"]) for row in val_rows if row.get("val_accuracy")]
    else:
        rows = (history or {}).get("history", [])
        val_steps = [float(row["step"]) for row in rows if "step" in row]
        val_losses = [float(row["val_loss"]) for row in rows if "val_loss" in row]
        val_accs = [float(row["val_accuracy"]) for row in rows if "val_accuracy" in row]
    return train_steps, train_losses, val_steps, val_losses, val_accs, metrics


def _draw_readme_piano_roll(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], notes: np.ndarray, progress: float) -> None:
    _panel(draw, box, "Generated piano roll")
    plot = (box[0] + 72, box[1] + 72, box[2] - 36, box[3] - 96)
    draw.rectangle(plot, fill=_hex("#fbfdff"), outline=_hex(GRID), width=1)
    if notes.size == 0:
        _text(draw, (plot[0] + 30, plot[1] + 120), "No notes were generated.", 24, MUTED)
        return
    starts = notes[:, 0].astype(float) / TIME_STEPS_PER_BEAT
    durations = np.maximum(notes[:, 2].astype(float) / TIME_STEPS_PER_BEAT, 0.08)
    pitches = notes[:, 1].astype(float)
    instruments = notes[:, 3].astype(int)
    x_max = max(float(np.max(starts + durations)), 1.0)
    y_min = max(0.0, math.floor((float(np.min(pitches)) - 6) / 12) * 12)
    y_max = min(127.0, math.ceil((float(np.max(pitches)) + 6) / 12) * 12)
    if y_max <= y_min:
        y_max = y_min + 12
    reveal = x_max * max(0.0, min(progress, 1.0))
    for beat in range(0, int(math.ceil(x_max)) + 1, 4):
        x = int(_scale(beat, 0.0, x_max, plot[0], plot[2]))
        draw.line((x, plot[1], x, plot[3]), fill=_hex(GRID), width=1)
        _text(draw, (x, plot[3] + 16), str(beat), 13, MUTED, anchor="ma")
    for pitch in range(int(y_min), int(y_max) + 1, 12):
        y = int(_scale(pitch, y_min, y_max, plot[3], plot[1]))
        draw.line((plot[0], y, plot[2], y), fill=_hex(GRID_LIGHT), width=1)
        _text(draw, (plot[0] - 16, y), str(pitch), 13, MUTED, anchor="ra")
    for index in np.argsort(starts):
        if starts[index] > reveal:
            continue
        x0 = int(_scale(starts[index], 0.0, x_max, plot[0], plot[2]))
        x1 = int(_scale(starts[index] + durations[index], 0.0, x_max, plot[0], plot[2]))
        y = int(_scale(pitches[index], y_min, y_max, plot[3], plot[1]))
        instrument = max(0, min(int(instruments[index]), len(INSTRUMENT_LABELS) - 1))
        draw.rectangle((x0, y - 6, max(x1, x0 + 8), y + 6), fill=_hex(INSTRUMENT_COLOR_LIST[instrument]))
    x = int(_scale(reveal, 0.0, x_max, plot[0], plot[2]))
    draw.line((x, plot[1], x, plot[3]), fill=_hex(INK), width=3)
    for index, label in enumerate(INSTRUMENT_LABELS):
        spacing = max(120, int((plot[2] - plot[0]) / len(INSTRUMENT_LABELS)))
        lx = plot[0] + index * spacing
        ly = box[3] - 54
        draw.rounded_rectangle((lx, ly, lx + 34, ly + 16), radius=4, fill=_hex(INSTRUMENT_COLOR_LIST[index]))
        _text(draw, (lx + 42, ly - 2), label, 14, MUTED)
    _text(draw, (plot[0], plot[3] + 44), "beat", 16, MUTED)


def _readme_confusion_path(output_root: Path, run_name: str, field_name: str) -> Path | None:
    for root in (
        output_root / "evaluation" / run_name / "test",
        output_root / "runs" / run_name / "metrics",
        output_root / "evaluation" / run_name / "valid",
    ):
        path = root / f"{field_name}_confusion.npy"
        if path.exists():
            return path
    return None


def _draw_readme_confusions(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    output_root: Path,
    run_name: str,
    progress: float,
) -> None:
    _panel(draw, box, "Final confusion matrices")
    fields = ["type", "position", "pitch", "duration", "instrument"]
    gap = 18
    left = box[0] + 24
    top = box[1] + 66
    slot_w = int((box[2] - box[0] - 48 - gap * (len(fields) - 1)) / len(fields))
    side = min(slot_w, box[3] - top - 44)
    for index, field_name in enumerate(fields):
        slot_x = left + index * (slot_w + gap)
        local_progress = max(0.0, min(1.0, progress * len(fields) - index))
        draw.rounded_rectangle((slot_x, top, slot_x + slot_w, box[3] - 24), radius=6, fill=_hex("#f7fafb"), outline=_hex("#d4dde3"))
        matrix_path = _readme_confusion_path(output_root, run_name, field_name)
        if matrix_path is None:
            _text(draw, (slot_x + 16, top + 50), "missing", 18, MUTED)
            continue
        matrix = np.load(matrix_path)
        row_sums = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(matrix, np.maximum(row_sums, 1), out=np.zeros_like(matrix, dtype=float), where=row_sums > 0)
        values = np.sqrt(np.clip(normalized * local_progress, 0.0, 1.0))
        low = np.asarray(_hex("#f7fbff"), dtype=np.float32)
        high = np.asarray(_hex("#1d4f91"), dtype=np.float32)
        rgb = np.clip(low + (high - low) * values[:, :, None], 0, 255).astype(np.uint8)
        heatmap = Image.fromarray(rgb, mode="RGB").resize((side, side), Image.Resampling.NEAREST)
        heat_x = slot_x + (slot_w - side) // 2
        heat_y = top + 18
        draw._image.paste(heatmap, (heat_x, heat_y))
        draw.rectangle((heat_x, heat_y, heat_x + side, heat_y + side), outline=_hex("#80909c"), width=1)
        total = int(matrix.sum())
        accuracy = float(np.trace(matrix) / total) if total else 0.0
        _text(draw, (slot_x + 16, box[3] - 54), field_name, 18, INK, bold=True)
        _text(draw, (slot_x + slot_w - 16, box[3] - 54), f"{accuracy:.2f}", 18, GOOD, bold=True, anchor="ra")


def _readme_dashboard_frame(
    training_run_name: str,
    generated_name: str,
    output_root: Path,
    progress: float,
) -> Image.Image:
    train_steps, train_losses, val_steps, val_losses, val_accs, metrics = _training_series(training_run_name, output_root)
    generated_dir = MULTITRACK_GENERATION_GENERATED_DIR / generated_name
    notes_path = generated_dir / "notes.npy"
    summary = _read_json(generated_dir / "summary.json") or {}
    notes = np.load(notes_path) if notes_path.exists() else np.empty((0, 4), dtype=np.int64)

    image, draw = _canvas(
        1600,
        1100,
        "Multitrack Transformer",
        "Full training run and rich generated sample",
    )
    step_limit = max(1, int(len(train_steps) * progress))
    val_limit = max(1, int(len(val_steps) * progress))
    visible_train_steps = train_steps[:step_limit]
    visible_train_losses = train_losses[:step_limit]
    visible_val_steps = val_steps[:val_limit]
    visible_val_losses = val_losses[:val_limit]
    visible_val_accs = val_accs[:val_limit]

    _draw_readme_confusions(draw, (42, 126, 1558, 350), output_root, training_run_name, progress)
    _draw_line_chart(draw, (42, 380, 770, 612), visible_train_steps, visible_train_losses, "Training loss", ACCENT, "loss")
    _draw_line_chart(draw, (42, 650, 770, 882), visible_val_steps, visible_val_accs, "Validation accuracy", GOOD, "accuracy", y_min=0.0, y_max=1.0)
    _draw_readme_piano_roll(draw, (810, 380, 1558, 1050), notes, progress)

    inst_box = (42, 896, 770, 1050)
    counts = summary.get("instrument_counts", {})
    values = [float(counts.get(label, 0)) for label in INSTRUMENT_LABELS]
    _draw_horizontal_bars(draw, inst_box, values, list(INSTRUMENT_LABELS), INSTRUMENT_COLOR_LIST, "Generated instrument balance")
    return image


def render_readme_panel(training_run_name: str, generated_name: str, output_root: Path = MULTITRACK_GENERATION_OUTPUT_ROOT) -> Path:
    output_path = ensure_dir(MULTITRACK_GENERATION_README_DIR) / "readme_multitrack_generation_static_panel.png"
    static = _readme_dashboard_frame(training_run_name, generated_name, output_root, progress=1.0)
    static.save(output_path)

    gif_path = MULTITRACK_GENERATION_README_DIR / "readme_multitrack_generation_animated_panel.gif"
    frames: list[Image.Image] = []
    for frame_index in range(18):
        progress = (frame_index + 1) / 18.0
        frame = _readme_dashboard_frame(training_run_name, generated_name, output_root, progress=progress)
        frames.append(frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=224))
    frames[0].save(gif_path, save_all=True, append_images=frames[1:], duration=135, loop=0, optimize=False)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render visuals for multitrack generation outputs.")
    parser.add_argument("--run-name", default="full_transformer")
    parser.add_argument("--training-run-name", default=None)
    parser.add_argument("--generated-name", default=None)
    parser.add_argument("--output-root", type=Path, default=MULTITRACK_GENERATION_OUTPUT_ROOT)
    parser.add_argument("--dataset-max-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_run_name = args.training_run_name or args.run_name
    generated_name = args.generated_name or args.run_name
    rendered = [render_dataset_overview(args.output_root, max_files=args.dataset_max_files)]
    training = render_training_overview(training_run_name, args.output_root)
    if training is not None:
        rendered.append(training)
    rendered.extend(render_generation_visuals(generated_name))
    rendered.extend(render_available_confusions(training_run_name, args.output_root))
    rendered.append(render_readme_panel(training_run_name, generated_name, args.output_root))
    for path in rendered:
        print(path)


if __name__ == "__main__":
    main()
