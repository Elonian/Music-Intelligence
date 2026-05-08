#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

try:
    from scipy.ndimage import gaussian_filter
except ModuleNotFoundError:  # pragma: no cover
    gaussian_filter = None


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.diffusion_based_music_generation.audio_io import tensor_to_audio_array, write_wav  # noqa: E402
from scripts.diffusion_based_music_generation.dataset import SR, spec_to_audio  # noqa: E402
from scripts.diffusion_based_music_generation.model import load_flow_model  # noqa: E402
from scripts.diffusion_based_music_generation.paths import OUTPUT_ROOT  # noqa: E402
from scripts.diffusion_based_music_generation.samplers import guided_velocity  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402


DEFAULT_CHECKPOINT = OUTPUT_ROOT / "runs" / "q4_full_guitar" / "checkpoints" / "model_ft.pt"
DEFAULT_SAMPLES_NPZ = OUTPUT_ROOT / "beat_baseline_pitch_guided" / "pitch_guided_beat_baseline_samples.npz"
DEFAULT_OUTPUT_DIR = OUTPUT_ROOT / "visuals" / "music_forming_flow"
DEFAULT_TRAIN_HISTORY = OUTPUT_ROOT / "runs" / "q4_full_guitar" / "history.json"
DEFAULT_TRAIN_SUMMARY = OUTPUT_ROOT / "runs" / "q4_full_guitar" / "train_summary.json"
DEFAULT_METHOD_SUMMARY = OUTPUT_ROOT / "beat_baseline_pitch_guided" / "beat_baseline_method_summary.json"
DEFAULT_COMPARISON_JSON = OUTPUT_ROOT / "beat_baseline_pitch_guided" / "pitch_score_baseline_comparison.json"
DEFAULT_EVALUATION_JSON = OUTPUT_ROOT / "beat_baseline_pitch_guided" / "evaluation" / "diffusion_generation_evaluation.json"
DEFAULT_SELECTED_CANDIDATES = OUTPUT_ROOT / "beat_baseline_pitch_guided" / "selected_candidates.csv"

BACKGROUND = "#f5efe5"
PANEL = "#fffaf1"
PANEL_ALT = "#fffdf7"
TEXT = "#1f2933"
MUTED = "#5b6573"
GRID = "#d7cdbc"
TEAL = "#0f766e"
TEAL_LIGHT = "#14b8a6"
GOLD = "#b7791f"
ROSE = "#d94f70"
BLUE = "#2563eb"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _npz_value(data: np.lib.npyio.NpzFile, key: str, index: int | None = None, default=None):
    if key not in data.files:
        return default
    values = data[key]
    if values.shape == ():
        return values.item()
    if index is None:
        return values
    value = values[index]
    return value.item() if hasattr(value, "item") else value


def _metadata_arrays(data: np.lib.npyio.NpzFile) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = int(data["samples"].shape[0])
    sampler_default = str(_npz_value(data, "sampler", default="heun"))
    steps_default = int(_npz_value(data, "n_steps", default=50))
    guidance_default = float(_npz_value(data, "guidance_scale", default=1.0))
    samplers = np.asarray(data["selected_sampler"] if "selected_sampler" in data.files else [sampler_default] * count)
    steps = np.asarray(data["selected_n_steps"] if "selected_n_steps" in data.files else [steps_default] * count, dtype=np.int32)
    guidance = np.asarray(
        data["selected_guidance_scale"] if "selected_guidance_scale" in data.files else [guidance_default] * count,
        dtype=np.float32,
    )
    return samplers.astype(str), steps, guidance


def _midi_to_hz(midi_pitch: int) -> float:
    return float(440.0 * 2 ** ((int(midi_pitch) - 69) / 12))


def _note_name(midi_pitch: int) -> str:
    names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    return f"{names[int(midi_pitch) % 12]}{int(midi_pitch) // 12 - 1}"


def _raw_magnitude(spec: np.ndarray) -> np.ndarray:
    values = np.asarray(spec, dtype=np.float32)
    return np.sqrt(np.square(values[0]) + np.square(values[1]))


def _log_magnitude(spec: np.ndarray, smooth: bool = False) -> np.ndarray:
    values = np.log1p(np.nan_to_num(_raw_magnitude(spec), nan=0.0, posinf=0.0, neginf=0.0))
    if smooth and gaussian_filter is not None:
        return gaussian_filter(values, sigma=(0.55, 0.25))
    return values


def _harmonic_bins(midi_pitch: int, freq_bins: int = 129) -> list[int]:
    f0 = _midi_to_hz(midi_pitch)
    bins = []
    harmonic = 1
    while f0 * harmonic <= SR / 2:
        bins.append(int(round((f0 * harmonic) / (SR / 2) * (freq_bins - 1))))
        harmonic += 1
    return [min(max(bin_index, 0), freq_bins - 1) for bin_index in bins]


def _visual_scores(data: np.lib.npyio.NpzFile) -> np.ndarray:
    samples = np.asarray(data["samples"], dtype=np.float32)
    noises = np.asarray(data["noises"], dtype=np.float32)
    pitches = np.asarray(data["pitches"], dtype=np.int64)
    selected_score = np.asarray(data["selected_score"], dtype=np.float32) if "selected_score" in data.files else np.zeros(samples.shape[0])
    scores = np.zeros(samples.shape[0], dtype=np.float32)

    for index in range(samples.shape[0]):
        mag = _raw_magnitude(samples[index])
        log_mag = np.log1p(mag)
        harmonic_mask = np.zeros(mag.shape[0], dtype=bool)
        for bin_index in _harmonic_bins(int(pitches[index]), mag.shape[0]):
            harmonic_mask[max(0, bin_index - 1) : min(mag.shape[0], bin_index + 2)] = True

        if np.any(~harmonic_mask):
            harmonic_ratio = float(mag[harmonic_mask].mean() / (mag[~harmonic_mask].mean() + 1e-6))
        else:
            harmonic_ratio = 0.0
        contrast = float(np.percentile(log_mag, 99.0) - np.percentile(log_mag, 50.0))
        movement = float(np.mean(np.abs(mag - _raw_magnitude(noises[index]))))
        mid_pitch_preference = float(np.exp(-((int(pitches[index]) - 64) / 13) ** 2))
        scores[index] = (
            0.35 * np.log1p(max(harmonic_ratio, 0.0))
            + 0.25 * contrast
            + 0.20 * np.log1p(max(movement, 0.0))
            + 0.20 * mid_pitch_preference
            + 0.08 * np.log1p(float(max(selected_score[index], 0.0)))
        )
    return scores


def _select_sample(data: np.lib.npyio.NpzFile, requested_index: int | None) -> tuple[int, str, np.ndarray]:
    sample_count = int(data["samples"].shape[0])
    scores = _visual_scores(data)
    if requested_index is not None:
        if requested_index < 0 or requested_index >= sample_count:
            raise ValueError(f"--sample-index must be between 0 and {sample_count - 1}")
        return int(requested_index), "requested sample", scores

    samplers, steps, guidance = _metadata_arrays(data)
    pitches = np.asarray(data["pitches"], dtype=np.int64)
    common_heun = (samplers == "heun") & (steps == 50) & np.isclose(guidance, 5.0) & (pitches >= 56) & (pitches <= 76)
    if np.any(common_heun):
        candidates = np.flatnonzero(common_heun)
        return int(candidates[np.argmax(scores[candidates])]), "highest visual-flow score among common HEUN samples", scores

    mid_pitch = (pitches >= 56) & (pitches <= 76)
    if np.any(mid_pitch):
        candidates = np.flatnonzero(mid_pitch)
        return int(candidates[np.argmax(scores[candidates])]), "highest visual-flow score among midrange pitches", scores
    return int(np.argmax(scores)), "highest visual-flow score", scores


def _select_flow_indices(data: np.lib.npyio.NpzFile, selected_index: int, visual_scores: np.ndarray, flow_samples: int) -> tuple[np.ndarray, str]:
    samplers, steps, guidance = _metadata_arrays(data)
    same_setting = (
        (samplers == samplers[selected_index])
        & (steps == steps[selected_index])
        & np.isclose(guidance, guidance[selected_index])
    )
    if int(same_setting.sum()) < min(8, flow_samples):
        same_setting = samplers == samplers[selected_index]
        reason = "same sampler"
    else:
        reason = "same sampler, step count, and guidance"

    candidates = np.flatnonzero(same_setting)
    order = candidates[np.argsort(visual_scores[candidates])[::-1]]
    ordered = [int(selected_index)]
    for index in order:
        if int(index) not in ordered:
            ordered.append(int(index))
        if len(ordered) >= int(flow_samples):
            break
    return np.asarray(ordered, dtype=np.int64), reason


def _device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def _time_batch(batch_size: int, value: float, device: torch.device) -> torch.Tensor:
    return torch.full((batch_size,), float(value), device=device)


def _record_batch_trajectory(
    model,
    noise: torch.Tensor,
    pitches: torch.Tensor,
    sampler: str,
    n_steps: int,
    guidance_scale: float,
    selected_position: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dt = 1.0 / int(n_steps)
    x = noise.clone()
    batch_size = int(x.shape[0])
    states = [x.detach().cpu().numpy().astype(np.float32)]
    times = [1.0]
    velocity_rms = []
    selected_velocity_maps = []

    with torch.no_grad():
        for index in range(int(n_steps)):
            t_value = 1.0 - index * dt
            t = _time_batch(batch_size, t_value, x.device)
            t_next = _time_batch(batch_size, max(t_value - dt, 0.0), x.device)

            if sampler == "euler":
                v = model(x, t, pitches)
                x = x - v * dt
            elif sampler == "cfg":
                v = guided_velocity(model, x, t, pitches, guidance_scale)
                x = x - v * dt
            elif sampler == "naive":
                v = model(x, t, pitches) * float(guidance_scale)
                x = x - v * dt
            elif sampler == "heun":
                k1 = guided_velocity(model, x, t, pitches, guidance_scale)
                x_predict = x - k1 * dt
                k2 = guided_velocity(model, x_predict, t_next, pitches, guidance_scale)
                v = 0.5 * (k1 + k2)
                x = x - v * dt
            elif sampler == "rk4":
                t_mid = _time_batch(batch_size, max(t_value - 0.5 * dt, 0.0), x.device)
                k1 = guided_velocity(model, x, t, pitches, guidance_scale)
                k2 = guided_velocity(model, x - 0.5 * dt * k1, t_mid, pitches, guidance_scale)
                k3 = guided_velocity(model, x - 0.5 * dt * k2, t_mid, pitches, guidance_scale)
                k4 = guided_velocity(model, x - dt * k3, t_next, pitches, guidance_scale)
                v = (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
                x = x - dt * v
            else:
                raise ValueError("sampler must be one of euler, cfg, naive, heun, rk4")

            selected_v = v[selected_position].detach().to(torch.float32)
            selected_velocity_maps.append(torch.sqrt(selected_v[0].square() + selected_v[1].square()).cpu().numpy().astype(np.float32))
            velocity_rms.append(float(v.detach().to(torch.float32).square().mean().sqrt().cpu()))
            states.append(x.detach().cpu().numpy().astype(np.float32))
            times.append(max(t_value - dt, 0.0))

    return (
        np.stack(states, axis=0).astype(np.float32),
        np.asarray(times, dtype=np.float32),
        np.asarray(velocity_rms, dtype=np.float32),
        np.stack(selected_velocity_maps, axis=0).astype(np.float32),
    )


def _display_limits(selected_states: np.ndarray, velocity_maps: np.ndarray) -> tuple[float, float, float]:
    spectra = np.stack([_log_magnitude(state, smooth=True) for state in selected_states], axis=0)
    vmin = float(np.percentile(spectra, 4.0))
    vmax = float(np.percentile(spectra, 99.6))
    energy = np.log1p(np.asarray(velocity_maps, dtype=np.float32))
    energy_vmax = float(np.percentile(energy, 99.0))
    return vmin, max(vmax, vmin + 1e-4), max(energy_vmax, 1e-4)


def _audio_from_spec(spec: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.asarray(spec, dtype=np.float32))
    with torch.no_grad():
        audio = spec_to_audio(tensor)
    return tensor_to_audio_array(audio, normalize=True)


def _phase_label(progress: float) -> str:
    if progress < 0.20:
        return "noise cloud"
    if progress < 0.48:
        return "field organizing spectra"
    if progress < 0.78:
        return "harmonics locking in"
    return "audio resolved"


def _frame_indices(n_states: int, frame_count: int) -> np.ndarray:
    count = max(2, min(int(frame_count), int(n_states)))
    return np.unique(np.linspace(0, n_states - 1, count).round().astype(int))


def _project_flow(states: np.ndarray, frame_indices: np.ndarray) -> np.ndarray:
    spectral = np.log1p(np.sqrt(np.square(states[:, :, 0]) + np.square(states[:, :, 1]))).astype(np.float32)
    anchor_indices = np.unique(np.concatenate([frame_indices, np.array([0, states.shape[0] - 1], dtype=np.int64)]))
    anchors = spectral[anchor_indices].reshape(-1, spectral.shape[-2] * spectral.shape[-1])
    mean = anchors.mean(axis=0, keepdims=True)
    centered = anchors - mean
    _, _singular, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2].astype(np.float32)
    positions = (spectral.reshape(-1, spectral.shape[-2] * spectral.shape[-1]) - mean) @ components.T
    positions = positions.reshape(states.shape[0], states.shape[1], 2)
    if float(positions[-1, :, 0].mean()) < float(positions[0, :, 0].mean()):
        positions[:, :, 0] *= -1.0
    if float(positions[-1, 0, 1]) < float(positions[0, 0, 1]):
        positions[:, :, 1] *= -1.0
    return positions.astype(np.float32)


def _axis_limits(positions: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x0, x1 = np.percentile(positions[:, :, 0], [1.0, 99.0])
    y0, y1 = np.percentile(positions[:, :, 1], [1.0, 99.0])
    xpad = max((x1 - x0) * 0.16, 0.25)
    ypad = max((y1 - y0) * 0.18, 0.25)
    return (float(x0 - xpad), float(x1 + xpad)), (float(y0 - ypad), float(y1 + ypad))


def _figure_to_image(fig: plt.Figure, dpi: int) -> np.ndarray:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)
    buffer.seek(0)
    return imageio.imread(buffer)


def _style_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_color("#d8ccb9")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _style_info_axis(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_ALT)
    for spine in ax.spines.values():
        spine.set_color("#d8ccb9")
        spine.set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def _format_number(value, digits: int = 3, suffix: str = "") -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(numeric) >= 1000:
        return f"{numeric:,.0f}{suffix}"
    if abs(numeric) >= 100:
        return f"{numeric:.1f}{suffix}"
    return f"{numeric:.{digits}f}{suffix}"


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if values.size == 0:
        return values
    window = max(1, min(int(window), int(values.size)))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _read_selected_candidate(path: Path, sample_index: int) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row.get("sample_index", -1)) == int(sample_index):
                return row
    return {}


def _read_selected_candidates(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _float_from_row(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _build_static_context(
    train_history_path: Path,
    train_summary_path: Path,
    method_summary_path: Path,
    comparison_json_path: Path,
    evaluation_json_path: Path,
    selected_candidates_path: Path,
    sample_index: int,
) -> dict:
    method_summary = _load_json(method_summary_path)
    context = {
        "train_history": _load_json(train_history_path),
        "train_summary": _load_json(train_summary_path),
        "method_summary": method_summary,
        "comparison": _load_json(comparison_json_path),
        "evaluation": _load_json(evaluation_json_path),
        "selected_candidate": _read_selected_candidate(selected_candidates_path, sample_index),
        "selected_candidates": _read_selected_candidates(selected_candidates_path),
    }
    if not context["comparison"] and method_summary:
        context["comparison"] = method_summary.get("pitch_score_comparison", {})
    return context


def _draw_training_curve(ax: plt.Axes, context: dict) -> None:
    _style_info_axis(ax)
    history = context.get("train_history", {}).get("history", [])
    train_summary = context.get("train_summary", {})
    analysis = context.get("method_summary", {}).get("training_analysis", {})
    if history:
        epochs = np.asarray([float(item.get("epoch", index + 1)) for index, item in enumerate(history)], dtype=np.float32)
        losses = np.asarray([float(item.get("loss", np.nan)) for item in history], dtype=np.float32)
        finite = np.isfinite(losses)
        epochs = epochs[finite]
        losses = losses[finite]
        smooth = _moving_average(losses, window=11)
        ax.plot(epochs, losses, color="#475569", linewidth=0.8, alpha=0.55)
        ax.plot(epochs, smooth, color=TEAL_LIGHT, linewidth=2.1)
        best_epoch = int(analysis.get("best_epoch", epochs[int(np.argmin(losses))] if losses.size else 0))
        best_loss = float(analysis.get("best_epoch_loss", np.nanmin(losses) if losses.size else np.nan))
        if np.isfinite(best_loss):
            ax.scatter([best_epoch], [best_loss], s=44, color=GOLD, edgecolor="#fff4c2", linewidth=0.8, zorder=4)
        ax.set_xlim(float(epochs.min()), float(epochs.max()))
        y0, y1 = np.percentile(losses, [1.0, 99.0])
        padding = max(float(y1 - y0) * 0.18, 0.002)
        ax.set_ylim(float(y0 - padding), float(y1 + padding))
    else:
        ax.text(0.5, 0.5, "training history not found", transform=ax.transAxes, ha="center", va="center", color=MUTED)

    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.35)
    ax.set_title("training curve", color=TEXT, fontsize=11.5, loc="left", pad=7)
    ax.set_xlabel("epoch", color=MUTED, fontsize=9)
    ax.set_ylabel("flow loss", color=MUTED, fontsize=9)
    final_loss = train_summary.get("final_loss")
    best_loss = analysis.get("best_epoch_loss")
    improvement = analysis.get("relative_improvement_from_epoch_1")
    ax.text(
        0.98,
        0.94,
        f"final {_format_number(final_loss)}\nbest {_format_number(best_loss)}\nimprovement {_format_number((improvement or 0) * 100, 1, '%')}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TEXT,
        fontsize=9,
        bbox={"facecolor": "#fff7e8", "edgecolor": "#d8ccb9", "boxstyle": "round,pad=0.35", "linewidth": 0.8},
    )


def _draw_baseline_comparison(ax: plt.Axes, context: dict) -> None:
    _style_info_axis(ax)
    comparison = context.get("comparison", {})
    method = comparison.get("selected_pitch_guided", {})
    baseline = comparison.get("notebook_baseline_pretrained_keyboard_heun25_gs6", {})
    metrics = [
        ("target ratio", baseline.get("mean_target_ratio"), method.get("mean_target_ratio")),
        ("margin rate", baseline.get("positive_margin_rate"), method.get("positive_margin_rate")),
        ("total score", baseline.get("mean_total_score"), method.get("mean_total_score")),
    ]
    x = np.arange(len(metrics), dtype=np.float32)
    width = 0.35
    base_values = np.asarray([float(item[1] or 0.0) for item in metrics], dtype=np.float32)
    method_values = np.asarray([float(item[2] or 0.0) for item in metrics], dtype=np.float32)
    ax.bar(x - width / 2, base_values, width=width, color="#9ca3af", alpha=0.85, label="baseline")
    ax.bar(x + width / 2, method_values, width=width, color=TEAL, alpha=0.92, label="pitch-guided")
    for index, value in enumerate(method_values):
        ax.text(index + width / 2, value + max(method_values.max(), 1.0) * 0.035, _format_number(value, 2), color=TEXT, fontsize=8, ha="center")
    for index, value in enumerate(base_values):
        ax.text(index - width / 2, value + max(method_values.max(), 1.0) * 0.035, _format_number(value, 2), color=MUTED, fontsize=8, ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels([item[0] for item in metrics], color=MUTED, fontsize=8)
    ax.set_ylim(0.0, max(float(max(base_values.max(), method_values.max())) * 1.25, 1.0))
    ax.grid(True, axis="y", color=GRID, linewidth=0.65, alpha=0.30)
    ax.set_title("pitch-guided selection beats baseline", color=TEXT, fontsize=11.5, loc="left", pad=7)
    ax.legend(loc="upper left", fontsize=8, facecolor="#fff7e8", edgecolor="#d8ccb9", labelcolor=TEXT)


def _draw_metric_cards(
    ax: plt.Axes,
    context: dict,
    sample_index: int,
    note: str,
    sampler_label: str,
    flow_count: int,
    selected_score,
    visual_score: float,
    final_mse: float,
    final_mae: float,
) -> None:
    _style_info_axis(ax)
    ax.set_xticks([])
    ax.set_yticks([])
    train_summary = context.get("train_summary", {})
    method_summary = context.get("method_summary", {})
    evaluation = context.get("evaluation", {})
    selected_candidate = context.get("selected_candidate", {})
    comparison = context.get("comparison", {})
    gain = comparison.get("relative_mean_target_ratio_gain")
    margin_gain = comparison.get("positive_margin_rate_gain")

    rows = [
        ("target sample", f"#{sample_index} | {note}"),
        ("solver", sampler_label.replace(", pitch guidance", "")),
        ("flow examples", str(flow_count)),
        ("training", f"{train_summary.get('epochs_completed', 'n/a')} epochs | {train_summary.get('steps', 'n/a')} steps"),
        ("dataset", f"{train_summary.get('dataset_files', 'n/a')} guitar files"),
        ("params", _format_number(train_summary.get("n_params"), 0)),
        ("candidates", str(method_summary.get("candidate_count", "n/a"))),
        ("selected score", _format_number(selected_score)),
        ("target ratio", _format_number(_float_from_row(selected_candidate, "target_ratio"))),
        ("margin ratio", _format_number(_float_from_row(selected_candidate, "margin_ratio"))),
        ("method gain", f"{_format_number((gain or 0) * 100, 1, '%')} target"),
        ("margin gain", f"{_format_number((margin_gain or 0) * 100, 1, '%')} rate"),
        ("shape", "100 x 2 x 129 x 63" if evaluation.get("shape_ok") else "unchecked"),
        ("replay mse", f"{final_mse:.2e}"),
        ("replay mae", f"{final_mae:.2e}"),
        ("visual score", _format_number(visual_score)),
    ]

    ax.text(0.04, 0.97, "numeric summary", transform=ax.transAxes, color=TEXT, fontsize=11.5, weight="bold", va="top")
    y = 0.88
    for label, value in rows:
        ax.text(0.05, y, label, transform=ax.transAxes, color=MUTED, fontsize=8.7, ha="left", va="center")
        ax.text(0.95, y, value, transform=ax.transAxes, color=TEXT, fontsize=8.7, ha="right", va="center")
        y -= 0.052
    ax.text(
        0.05,
        0.035,
        "numeric checks from training, generation, and replay",
        transform=ax.transAxes,
        color="#6b7280",
        fontsize=7.8,
        ha="left",
    )


def _draw_q5_score_curve(ax: plt.Axes, context: dict, sample_index: int) -> None:
    _style_info_axis(ax)
    rows = context.get("selected_candidates", [])
    parsed = []
    for row in rows:
        try:
            parsed.append(
                (
                    int(row["sample_index"]),
                    int(row["pitch"]),
                    float(row["total_score"]),
                    float(row["target_ratio"]),
                    float(row["margin_ratio"]),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    if not parsed:
        ax.text(0.5, 0.5, "selected-candidate table not found", transform=ax.transAxes, ha="center", va="center", color=MUTED)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    parsed.sort(key=lambda item: item[0])
    indices = np.asarray([item[0] for item in parsed], dtype=np.int32)
    pitches = np.asarray([item[1] for item in parsed], dtype=np.int32)
    scores = np.asarray([item[2] for item in parsed], dtype=np.float32)
    target_ratios = np.asarray([item[3] for item in parsed], dtype=np.float32)
    margins = np.asarray([item[4] for item in parsed], dtype=np.float32)
    selected_mask = indices == int(sample_index)
    x = pitches.astype(np.float32)

    ax.plot(x, scores, color=TEAL, linewidth=1.9, label="total score")
    ax.plot(x, target_ratios, color=GOLD, linewidth=1.55, alpha=0.92, label="target ratio")
    ax.axhline(float(np.mean(scores)), color="#94a3b8", linewidth=0.9, alpha=0.60)
    if np.any(selected_mask):
        ax.scatter(x[selected_mask], scores[selected_mask], s=58, color=ROSE, edgecolor="#ffe4e6", linewidth=0.9, zorder=5)
        ax.scatter(x[selected_mask], target_ratios[selected_mask], s=42, color=ROSE, edgecolor="#ffe4e6", linewidth=0.75, zorder=5)
    positive = margins > 0.0
    ax.fill_between(x, 0.0, np.maximum(margins, 0.0) * 20.0, where=positive, color=BLUE, alpha=0.15, linewidth=0)

    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.35)
    ax.set_title("candidate-selection curves", color=TEXT, fontsize=11.5, loc="left", pad=7)
    ax.set_xlabel("MIDI pitch", color=MUTED, fontsize=9)
    ax.set_ylabel("score / ratio", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, facecolor="#fff7e8", edgecolor="#d8ccb9", labelcolor=TEXT)
    ax.text(
        0.98,
        0.94,
        f"mean score {_format_number(np.mean(scores))}\npositive margin {100.0 * np.mean(positive):.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=TEXT,
        fontsize=8.6,
        bbox={"facecolor": "#fff7e8", "edgecolor": "#d8ccb9", "boxstyle": "round,pad=0.30", "linewidth": 0.75},
    )


def _curve_to_unit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return values
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros_like(values)
    low, high = np.percentile(finite, [2.0, 98.0])
    if high <= low + 1e-8:
        return np.zeros_like(values)
    return np.clip((values - low) / (high - low), 0.0, 1.0)


def _harmonic_ratio_for_spec(spec: np.ndarray, pitch_value: int) -> float:
    mag = _raw_magnitude(spec)
    harmonic_mask = np.zeros(mag.shape[0], dtype=bool)
    for bin_index in _harmonic_bins(int(pitch_value), mag.shape[0]):
        harmonic_mask[max(0, bin_index - 1) : min(mag.shape[0], bin_index + 2)] = True
    if not np.any(~harmonic_mask):
        return 0.0
    return float(mag[harmonic_mask].mean() / (mag[~harmonic_mask].mean() + 1e-6))


def _draw_generation_dynamics(
    ax: plt.Axes,
    selected_states: np.ndarray,
    velocity_maps: np.ndarray,
    pitch_value: int,
) -> None:
    _style_info_axis(ax)
    step_x = np.linspace(0.0, 1.0, selected_states.shape[0])
    harmonic = np.asarray([_harmonic_ratio_for_spec(state, pitch_value) for state in selected_states], dtype=np.float32)
    contrast = np.asarray(
        [float(np.percentile(_log_magnitude(state), 99.0) - np.percentile(_log_magnitude(state), 50.0)) for state in selected_states],
        dtype=np.float32,
    )
    field_energy = np.log1p(np.asarray([float(np.mean(item)) for item in velocity_maps], dtype=np.float32))
    field_energy = np.pad(field_energy, (0, max(0, selected_states.shape[0] - field_energy.shape[0])), mode="edge")

    ax.plot(step_x, _curve_to_unit(harmonic), color=GOLD, linewidth=2.0, label="harmonic strength")
    ax.plot(step_x, _curve_to_unit(contrast), color=TEAL, linewidth=2.0, label="spectral contrast")
    ax.plot(step_x, _curve_to_unit(field_energy), color=BLUE, linewidth=1.7, alpha=0.86, label="field energy")
    ax.fill_between(step_x, 0.0, _curve_to_unit(field_energy), color=BLUE, alpha=0.10, linewidth=0)
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.35)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("generation dynamics over flow steps", color=TEXT, fontsize=11.5, loc="left", pad=7)
    ax.set_xlabel("")
    ax.set_ylabel("normalized value", color=MUTED, fontsize=9)
    ax.legend(loc="upper left", fontsize=8, facecolor="#fff7e8", edgecolor="#d8ccb9", labelcolor=TEXT)
    ax.text(
        0.98,
        0.07,
        f"final harmonic {_format_number(harmonic[-1])}\nfinal contrast {_format_number(contrast[-1])}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color=TEXT,
        fontsize=8.6,
        bbox={"facecolor": "#fff7e8", "edgecolor": "#d8ccb9", "boxstyle": "round,pad=0.30", "linewidth": 0.75},
    )


def _overlay_harmonic_guides(ax: plt.Axes, pitch_value: int, alpha: float = 0.34) -> None:
    for order, bin_index in enumerate(_harmonic_bins(pitch_value)):
        linewidth = 1.0 if order == 0 else 0.65
        ax.axhline(bin_index, color=GOLD, linewidth=linewidth, alpha=max(alpha - order * 0.025, 0.12))


def _draw_flow_plane(
    ax: plt.Axes,
    positions: np.ndarray,
    state_index: int,
    selected_position: int,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    _style_axis(ax)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(True, color=GRID, linewidth=0.65, alpha=0.35)

    if state_index > 0:
        snapshots = np.unique(np.linspace(0, state_index, min(6, state_index + 1)).round().astype(int))
        for order, snapshot in enumerate(snapshots):
            alpha = 0.12 + 0.08 * order
            size = 16 + 2.5 * order
            color = "#b7c2cf" if snapshot < state_index else TEAL_LIGHT
            ax.scatter(
                positions[snapshot, :, 0],
                positions[snapshot, :, 1],
                s=size,
                color=color,
                alpha=min(alpha, 0.68),
                linewidths=0,
                zorder=1 + order,
            )

    segments_future = [positions[:, item, :] for item in range(positions.shape[1])]
    ax.add_collection(LineCollection(segments_future, colors="#b7c2cf", linewidths=0.70, alpha=0.35))

    segments_now = [positions[: state_index + 1, item, :] for item in range(positions.shape[1])]
    ax.add_collection(LineCollection(segments_now, colors=TEAL, linewidths=0.95, alpha=0.34))

    ax.scatter(positions[0, :, 0], positions[0, :, 1], s=18, color="#94a3b8", alpha=0.58, linewidths=0)
    ax.scatter(positions[-1, :, 0], positions[-1, :, 1], s=28, marker="x", color=GOLD, alpha=0.55, linewidths=1.0)
    ax.scatter(positions[state_index, :, 0], positions[state_index, :, 1], s=38, color=TEAL_LIGHT, alpha=0.82, edgecolor=BACKGROUND, linewidth=0.45)

    if state_index > 0:
        previous = max(0, state_index - 2)
        current_xy = positions[state_index, :, :]
        delta_xy = current_xy - positions[previous, :, :]
        stride = max(1, positions.shape[1] // 12)
        ax.quiver(
            current_xy[::stride, 0],
            current_xy[::stride, 1],
            delta_xy[::stride, 0],
            delta_xy[::stride, 1],
            angles="xy",
            scale_units="xy",
            scale=1.0,
            width=0.0042,
            color=BLUE,
            alpha=0.45,
        )

    selected_path = positions[: state_index + 1, selected_position, :]
    ax.plot(selected_path[:, 0], selected_path[:, 1], color=ROSE, linewidth=2.8, alpha=0.95)
    ax.scatter(
        [positions[state_index, selected_position, 0]],
        [positions[state_index, selected_position, 1]],
        s=118,
        color=ROSE,
        edgecolor="#ffe4e6",
        linewidth=1.2,
        zorder=6,
    )
    ax.text(0.03, 0.96, "multi-sample flow field", transform=ax.transAxes, color=TEXT, fontsize=11.5, va="top")
    ax.text(0.04, 0.06, "noise cloud", transform=ax.transAxes, color=MUTED, fontsize=9.5, ha="left")
    ax.text(0.96, 0.06, "generated music", transform=ax.transAxes, color=GOLD, fontsize=9.5, ha="right")


def _draw_progress(ax: plt.Axes, progress: float) -> None:
    ax.set_facecolor(BACKGROUND)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.plot([0.0, 1.0], [0.55, 0.55], color="#d7cdbc", linewidth=8.5, solid_capstyle="round")
    ax.plot([0.0, progress], [0.55, 0.55], color=TEAL, linewidth=8.5, solid_capstyle="round")
    ax.scatter([progress], [0.55], s=96, color=GOLD, edgecolor="#fff4c2", linewidth=1.0, zorder=3)
    for value, label in [(0.0, "noise"), (0.32, "organize"), (0.67, "harmonics"), (1.0, "audio")]:
        ax.text(value, 0.08, label, color=MUTED, fontsize=9.5, ha="center" if 0 < value < 1 else ("left" if value == 0 else "right"))


def _render_flow_frame(
    positions: np.ndarray,
    state_index: int,
    selected_position: int,
    selected_state: np.ndarray,
    final_state: np.ndarray,
    velocity_map: np.ndarray,
    current_audio: np.ndarray,
    final_audio: np.ndarray,
    pitch_value: int,
    note: str,
    sampler_label: str,
    progress: float,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    spec_vmin: float,
    spec_vmax: float,
    energy_vmax: float,
    dpi: int,
) -> np.ndarray:
    current_log = _log_magnitude(selected_state, smooth=True)
    final_log = _log_magnitude(final_state, smooth=True)
    energy = np.log1p(velocity_map)
    if gaussian_filter is not None:
        energy = gaussian_filter(energy, sigma=(0.75, 0.35))
    time = np.arange(current_audio.shape[0], dtype=float) / float(SR)

    fig = plt.figure(figsize=(15.2, 8.55), facecolor=BACKGROUND)
    gs = fig.add_gridspec(
        4,
        2,
        width_ratios=[1.05, 1.65],
        height_ratios=[0.30, 2.95, 0.95, 1.05],
        hspace=0.20,
        wspace=0.08,
        left=0.045,
        right=0.985,
        top=0.895,
        bottom=0.065,
    )
    fig.text(0.045, 0.955, "Music generation as a learned flow", color=TEXT, fontsize=22.5, weight="bold")
    fig.text(
        0.045,
        0.922,
        f"{note} guitar sample | {sampler_label} | {_phase_label(progress)}",
        color=MUTED,
        fontsize=11.5,
    )

    ax_bar = fig.add_subplot(gs[0, :])
    _draw_progress(ax_bar, progress)

    ax_flow = fig.add_subplot(gs[1:, 0])
    _draw_flow_plane(ax_flow, positions, state_index, selected_position, xlim, ylim)

    ax_spec = fig.add_subplot(gs[1, 1])
    _style_axis(ax_spec)
    ax_spec.imshow(current_log, origin="lower", aspect="auto", cmap="inferno", vmin=spec_vmin, vmax=spec_vmax)
    contour_levels = np.percentile(final_log, [82.0, 90.0, 96.0])
    ax_spec.contour(final_log, levels=np.unique(contour_levels), colors=GOLD, linewidths=[0.45, 0.65, 0.85], alpha=0.48)
    _overlay_harmonic_guides(ax_spec, pitch_value, alpha=0.30)
    ax_spec.set_xticks([])
    ax_spec.set_yticks([])
    ax_spec.set_ylabel("frequency", color=MUTED, fontsize=9.5)
    ax_spec.set_title("selected spectrogram forming, with final structure ghosted in gold", color=TEXT, fontsize=11.5, loc="left", pad=7)

    ax_energy = fig.add_subplot(gs[2, 1])
    _style_axis(ax_energy)
    ax_energy.imshow(energy, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=energy_vmax)
    _overlay_harmonic_guides(ax_energy, pitch_value, alpha=0.18)
    ax_energy.set_xticks([])
    ax_energy.set_yticks([])
    ax_energy.set_title("where the trained vector field is pushing now", color=TEXT, fontsize=10.5, loc="left", pad=5)

    ax_wave = fig.add_subplot(gs[3, 1])
    _style_axis(ax_wave)
    ax_wave.plot(time, final_audio, color="#94a3b8", linewidth=1.0, alpha=0.62)
    ax_wave.plot(time, current_audio, color=TEAL_LIGHT, linewidth=1.35, alpha=0.96)
    ax_wave.fill_between(time, current_audio, 0.0, color=TEAL, alpha=0.17, linewidth=0.0)
    ax_wave.axhline(0.0, color="#94a3b8", linewidth=0.65, alpha=0.55)
    ax_wave.set_xlim(0.0, max(float(time[-1]), 1e-6))
    ax_wave.set_ylim(-1.08, 1.08)
    ax_wave.set_yticks([])
    ax_wave.set_xlabel("time", color=MUTED, fontsize=9.5)
    ax_wave.set_title("waveform becoming the final generated audio", color=TEXT, fontsize=10.5, loc="left", pad=5)

    return _figure_to_image(fig, dpi=dpi)


def render_flow_gif(
    selected_states: np.ndarray,
    positions: np.ndarray,
    selected_position: int,
    frame_indices: np.ndarray,
    velocity_maps: np.ndarray,
    output_path: Path,
    pitch_value: int,
    note: str,
    sampler_label: str,
    duration: float,
    dpi: int,
) -> None:
    spec_vmin, spec_vmax, energy_vmax = _display_limits(selected_states[frame_indices], velocity_maps)
    xlim, ylim = _axis_limits(positions)
    final_state = selected_states[-1]
    final_audio = _audio_from_spec(final_state)
    frame_audios = {int(index): _audio_from_spec(selected_states[int(index)]) for index in frame_indices}
    ensure_dir(output_path.parent)
    duration_ms = max(20, int(round(float(duration) * 1000.0)))
    writer = imageio.get_writer(output_path, mode="I", duration=duration_ms, loop=0)
    for frame_number, state_index in enumerate(frame_indices):
        state_index = int(state_index)
        progress = frame_number / max(len(frame_indices) - 1, 1)
        velocity_index = min(state_index, velocity_maps.shape[0] - 1)
        frame = (
            _render_flow_frame(
                positions=positions,
                state_index=state_index,
                selected_position=selected_position,
                selected_state=selected_states[state_index],
                final_state=final_state,
                velocity_map=velocity_maps[velocity_index],
                current_audio=frame_audios[state_index],
                final_audio=final_audio,
                pitch_value=pitch_value,
                note=note,
                sampler_label=sampler_label,
                progress=progress,
                xlim=xlim,
                ylim=ylim,
                spec_vmin=spec_vmin,
                spec_vmax=spec_vmax,
                energy_vmax=energy_vmax,
                dpi=dpi,
            )
        )
        writer.append_data(frame)
    writer.close()


def render_storyboard(
    selected_states: np.ndarray,
    positions: np.ndarray,
    selected_position: int,
    velocity_maps: np.ndarray,
    output_path: Path,
    pitch_value: int,
    note: str,
    sampler_label: str,
    static_context: dict,
    sample_index: int,
    flow_count: int,
    selected_score,
    visual_score: float,
    final_mse: float,
    final_mae: float,
    dpi: int,
) -> None:
    positions_to_show = [0.0, 0.28, 0.62, 1.0]
    labels = ["noise cloud", "field organizes", "harmonics lock", "generated audio"]
    indices = [min(int(round(pos * (selected_states.shape[0] - 1))), selected_states.shape[0] - 1) for pos in positions_to_show]
    spec_vmin, spec_vmax, energy_vmax = _display_limits(selected_states[indices], velocity_maps)
    xlim, ylim = _axis_limits(positions)
    final_log = _log_magnitude(selected_states[-1], smooth=True)

    fig = plt.figure(figsize=(18.0, 11.0), facecolor=BACKGROUND)
    gs = fig.add_gridspec(
        4,
        6,
        width_ratios=[1.35, 1.0, 1.0, 1.0, 1.0, 1.20],
        height_ratios=[2.55, 0.92, 1.15, 0.06],
        hspace=0.18,
        wspace=0.10,
        left=0.04,
        right=0.985,
        top=0.875,
        bottom=0.075,
    )
    fig.text(0.04, 0.956, "Diffusion music generation: flow, training, and pitch-guided result", color=TEXT, fontsize=24, weight="bold")
    fig.text(
        0.04,
        0.922,
        f"Target {note} | {sampler_label} | many examples move through the same trained field",
        color=MUTED,
        fontsize=12,
    )

    ax_flow = fig.add_subplot(gs[0:2, 0])
    _draw_flow_plane(ax_flow, positions, selected_states.shape[0] - 1, selected_position, xlim, ylim)
    ax_flow.set_title("complete learned flow", color=TEXT, fontsize=12, loc="left", pad=8)

    for column, (state_index, label) in enumerate(zip(indices, labels), start=1):
        spec = selected_states[state_index]
        spec_log = _log_magnitude(spec, smooth=True)
        audio = _audio_from_spec(spec)
        time = np.arange(audio.shape[0], dtype=float) / float(SR)
        energy = np.log1p(velocity_maps[min(state_index, velocity_maps.shape[0] - 1)])
        if gaussian_filter is not None:
            energy = gaussian_filter(energy, sigma=(0.75, 0.35))

        ax_spec = fig.add_subplot(gs[0, column])
        _style_axis(ax_spec)
        ax_spec.imshow(spec_log, origin="lower", aspect="auto", cmap="inferno", vmin=spec_vmin, vmax=spec_vmax)
        ax_spec.contour(final_log, levels=np.unique(np.percentile(final_log, [86.0, 94.0])), colors=GOLD, linewidths=0.65, alpha=0.42)
        _overlay_harmonic_guides(ax_spec, pitch_value, alpha=0.28)
        ax_spec.set_xticks([])
        ax_spec.set_yticks([])
        ax_spec.set_title(label, color=TEXT, fontsize=11.5, pad=7)
        if column < 4:
            fig.text(0.341 + (column - 1) * 0.130, 0.565, "->", color=GOLD, fontsize=22, ha="center", va="center")

        ax_wave = fig.add_subplot(gs[1, column])
        _style_axis(ax_wave)
        ax_wave.imshow(energy, origin="lower", aspect="auto", cmap="viridis", vmin=0.0, vmax=energy_vmax, alpha=0.48)
        ax_wave.plot(np.linspace(0, energy.shape[1] - 1, audio.shape[0]), (audio * 0.42 + 0.50) * (energy.shape[0] - 1), color=TEAL_LIGHT, linewidth=0.75, alpha=0.90)
        ax_wave.set_xticks([])
        ax_wave.set_yticks([])

    ax_dynamics = fig.add_subplot(gs[0:2, 5])
    _draw_generation_dynamics(ax_dynamics, selected_states, velocity_maps, pitch_value)

    ax_train = fig.add_subplot(gs[2, 0:3])
    _draw_training_curve(ax_train, static_context)

    ax_compare = fig.add_subplot(gs[2, 3:5])
    _draw_baseline_comparison(ax_compare, static_context)

    ax_q5_curve = fig.add_subplot(gs[2, 5])
    _draw_q5_score_curve(ax_q5_curve, static_context, sample_index)

    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=dpi, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)


def _write_audio_snapshots(selected_states: np.ndarray, output_dir: Path) -> dict[str, str]:
    ensure_dir(output_dir)
    positions = {
        "noise_state": 0.0,
        "early_flow": 0.28,
        "late_flow": 0.62,
        "final_audio": 1.0,
    }
    paths: dict[str, str] = {}
    for name, position in positions.items():
        index = min(int(round(position * (selected_states.shape[0] - 1))), selected_states.shape[0] - 1)
        audio = _audio_from_spec(selected_states[index])
        path = output_dir / f"{name}.wav"
        write_wav(path, audio, sample_rate=SR)
        paths[name] = str(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Render diffusion music generation flow visuals.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--samples-npz", type=Path, default=DEFAULT_SAMPLES_NPZ)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample-index", type=int, default=None)
    parser.add_argument("--flow-samples", type=int, default=24)
    parser.add_argument("--frame-count", type=int, default=42)
    parser.add_argument("--frame-duration", type=float, default=0.115)
    parser.add_argument("--dpi", type=int, default=110)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--skip-storyboard", action="store_true")
    parser.add_argument("--train-history", type=Path, default=DEFAULT_TRAIN_HISTORY)
    parser.add_argument("--train-summary", type=Path, default=DEFAULT_TRAIN_SUMMARY)
    parser.add_argument("--method-summary", type=Path, default=DEFAULT_METHOD_SUMMARY)
    parser.add_argument("--comparison-json", type=Path, default=DEFAULT_COMPARISON_JSON)
    parser.add_argument("--evaluation-json", type=Path, default=DEFAULT_EVALUATION_JSON)
    parser.add_argument("--selected-candidates", type=Path, default=DEFAULT_SELECTED_CANDIDATES)
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    data = np.load(args.samples_npz, allow_pickle=False)
    sample_index, selection_reason, visual_scores = _select_sample(data, args.sample_index)
    flow_indices, flow_selection_reason = _select_flow_indices(data, sample_index, visual_scores, args.flow_samples)
    selected_position = int(np.where(flow_indices == sample_index)[0][0])
    pitch_value = int(_npz_value(data, "pitches", sample_index))
    note = _note_name(pitch_value)
    samplers, steps, guidance = _metadata_arrays(data)
    sampler = str(samplers[sample_index])
    n_steps = int(steps[sample_index])
    guidance_scale = float(guidance[sample_index])
    selected_score = _npz_value(data, "selected_score", sample_index, None)

    device = _device_from_arg(args.device)
    model, _checkpoint_payload = load_flow_model(str(args.checkpoint), device=str(device))
    model.eval()

    noise = torch.from_numpy(data["noises"][flow_indices]).to(device=device, dtype=torch.float32)
    pitches = torch.from_numpy(np.asarray(data["pitches"][flow_indices], dtype=np.int64)).to(device=device, dtype=torch.long)
    states, times, velocity_rms, selected_velocity_maps = _record_batch_trajectory(
        model=model,
        noise=noise,
        pitches=pitches,
        sampler=sampler,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        selected_position=selected_position,
    )
    frame_indices = _frame_indices(states.shape[0], args.frame_count)
    positions = _project_flow(states, frame_indices)
    selected_states = states[:, selected_position]
    sampler_label = f"{sampler.upper()} flow, pitch guidance"

    gif_path = output_dir / "music_forming_over_diffusion_flow.gif"
    storyboard_path = output_dir / "music_forming_storyboard.png"
    trajectory_path = output_dir / "replayed_flow_trajectory.npz"
    audio_paths = _write_audio_snapshots(selected_states, output_dir / "audio_snapshots")
    replay_final = selected_states[-1]
    saved_final = np.asarray(data["samples"][sample_index], dtype=np.float32)
    final_mse = float(np.mean(np.square(replay_final - saved_final)))
    final_mae = float(np.mean(np.abs(replay_final - saved_final)))
    static_context = _build_static_context(
        train_history_path=args.train_history,
        train_summary_path=args.train_summary,
        method_summary_path=args.method_summary,
        comparison_json_path=args.comparison_json,
        evaluation_json_path=args.evaluation_json,
        selected_candidates_path=args.selected_candidates,
        sample_index=sample_index,
    )

    if not args.skip_gif:
        render_flow_gif(
            selected_states=selected_states,
            positions=positions,
            selected_position=selected_position,
            frame_indices=frame_indices,
            velocity_maps=selected_velocity_maps,
            output_path=gif_path,
            pitch_value=pitch_value,
            note=note,
            sampler_label=sampler_label,
            duration=args.frame_duration,
            dpi=args.dpi,
        )
    if not args.skip_storyboard:
        render_storyboard(
            selected_states=selected_states,
            positions=positions,
            selected_position=selected_position,
            velocity_maps=selected_velocity_maps,
            output_path=storyboard_path,
            pitch_value=pitch_value,
            note=note,
            sampler_label=sampler_label,
            static_context=static_context,
            sample_index=sample_index,
            flow_count=int(flow_indices.shape[0]),
            selected_score=selected_score,
            visual_score=float(visual_scores[sample_index]),
            final_mse=final_mse,
            final_mae=final_mae,
            dpi=max(args.dpi, 120),
        )

    np.savez_compressed(
        trajectory_path,
        selected_states=selected_states.astype(np.float32),
        selected_velocity_maps=selected_velocity_maps.astype(np.float32),
        times=times,
        frame_indices=frame_indices.astype(np.int32),
        flow_positions=positions.astype(np.float32),
        flow_indices=flow_indices.astype(np.int32),
        pitch=np.array(pitch_value, dtype=np.int32),
        sampler=np.array(sampler),
        n_steps=np.array(n_steps, dtype=np.int32),
        guidance_scale=np.array(guidance_scale, dtype=np.float32),
        velocity_rms=velocity_rms,
    )

    manifest = {
        "sample_index": sample_index,
        "selection_reason": selection_reason,
        "flow_indices": flow_indices.astype(int).tolist(),
        "flow_selection_reason": flow_selection_reason,
        "target_pitch": pitch_value,
        "target_note": note,
        "sampler": sampler,
        "n_steps": n_steps,
        "guidance_scale": guidance_scale,
        "selected_score": None if selected_score is None else float(selected_score),
        "visual_score": float(visual_scores[sample_index]),
        "checkpoint": str(args.checkpoint),
        "samples_npz": str(args.samples_npz),
        "device": str(device),
        "replay_final_mse_vs_saved_sample": final_mse,
        "replay_final_mae_vs_saved_sample": final_mae,
        "outputs": {
            "gif": str(gif_path),
            "storyboard": str(storyboard_path),
            "trajectory": str(trajectory_path),
            "audio_snapshots": audio_paths,
        },
    }
    save_json(output_dir / "visual_manifest.json", manifest)
    print(f"[Diffusion Visualiser] Wrote GIF: {gif_path}")
    print(f"[Diffusion Visualiser] Wrote storyboard: {storyboard_path}")
    print(f"[Diffusion Visualiser] Manifest: {output_dir / 'visual_manifest.json'}")


if __name__ == "__main__":
    main()
