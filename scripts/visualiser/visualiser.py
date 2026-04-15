from __future__ import annotations

import io
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from utils import ensure_dir


def _time_axis(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.arange(audio.shape[0], dtype=float) / float(sample_rate)


def _downsample(audio: np.ndarray, sample_rate: int, max_points: int = 2400) -> tuple[np.ndarray, np.ndarray]:
    if audio.shape[0] <= max_points:
        return _time_axis(audio, sample_rate), audio
    stride = max(1, audio.shape[0] // max_points)
    indices = np.arange(0, audio.shape[0], stride)
    return indices / float(sample_rate), audio[indices]


def _envelope_summary(
    audio: np.ndarray,
    sample_rate: int,
    max_points: int = 2200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        empty = np.array([], dtype=float)
        return empty, empty, empty, empty

    window = max(1, int(np.ceil(audio.shape[0] / max_points)))
    centers = []
    lows = []
    highs = []
    rms = []
    for start in range(0, audio.shape[0], window):
        chunk = audio[start : start + window]
        centers.append((start + chunk.shape[0] / 2.0) / float(sample_rate))
        lows.append(float(np.min(chunk)))
        highs.append(float(np.max(chunk)))
        rms.append(float(np.sqrt(np.mean(np.square(chunk)))))
    return (
        np.asarray(centers, dtype=float),
        np.asarray(lows, dtype=float),
        np.asarray(highs, dtype=float),
        np.asarray(rms, dtype=float),
    )


def _metric_window(audio: np.ndarray, center_index: int, window_samples: int) -> np.ndarray:
    if audio.size == 0:
        return np.array([], dtype=float)
    start = max(0, center_index - window_samples // 2)
    end = min(audio.size, start + window_samples)
    start = max(0, end - window_samples)
    return np.asarray(audio[start:end], dtype=float)


def _estimate_window_metrics(audio: np.ndarray, sample_rate: int) -> tuple[float, float]:
    window = np.asarray(audio, dtype=float)
    if window.size == 0:
        return 0.0, 0.0

    rms = float(np.sqrt(np.mean(np.square(window))))
    if window.size < 32 or np.max(np.abs(window)) <= 1e-8:
        return rms, 0.0

    tapered = window * np.hanning(window.size)
    spectrum = np.abs(np.fft.rfft(tapered))
    if spectrum.size <= 1:
        return rms, 0.0

    freqs = np.fft.rfftfreq(window.size, d=1.0 / float(sample_rate))
    peak_index = int(np.argmax(spectrum[1:]) + 1)
    return rms, float(freqs[peak_index])


def _zoom_limits(current_time: float, duration: float, zoom_seconds: float) -> tuple[float, float]:
    span = min(max(zoom_seconds, 1e-3), max(duration, 1e-3))
    half = span / 2.0
    left = max(0.0, current_time - half)
    right = min(duration, current_time + half)
    if right - left < span:
        if left <= 1e-9:
            right = min(duration, span)
        else:
            left = max(0.0, duration - span)
    return left, max(right, left + 1e-3)


def _style_wave_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("#fffdf8")
    ax.grid(True, alpha=0.22, color="#b7ab98", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_color("#cfc3b1")
        spine.set_linewidth(0.9)


def save_waveform_plot(audio: np.ndarray, sample_rate: int, output_path: Path, title: str, color: str = "tab:blue") -> None:
    audio = np.asarray(audio, dtype=float)
    t, y = _downsample(audio, sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t, y, color=color, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0.0, max(audio.shape[0] / float(sample_rate), 1e-6))
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_waveform_comparison(
    audio_a: np.ndarray,
    audio_b: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
    label_a: str,
    label_b: str,
) -> None:
    audio_a = np.asarray(audio_a, dtype=float)
    audio_b = np.asarray(audio_b, dtype=float)
    t_a, y_a = _downsample(audio_a, sample_rate)
    t_b, y_b = _downsample(audio_b, sample_rate)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(t_a, y_a, linewidth=1.2, alpha=0.75, label=label_a)
    ax.plot(t_b, y_b, linewidth=1.2, alpha=0.75, label=label_b)
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0.0, max(max(audio_a.shape[0], audio_b.shape[0]) / float(sample_rate), 1e-6))
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_spectrogram_plot(audio: np.ndarray, sample_rate: int, output_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.specgram(np.asarray(audio, dtype=float), NFFT=1024, Fs=sample_rate, noverlap=768, cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Frequency [Hz]")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_progressive_waveform_gif(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
    color: str = "tab:blue",
    frame_count: int = 32,
) -> None:
    audio = np.asarray(audio, dtype=float)
    frames = []
    for frame_idx in range(1, frame_count + 1):
        end = max(1, int(frame_idx / frame_count * audio.shape[0]))
        t, y = _downsample(audio[:end], sample_rate)
        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.plot(t, y, color=color, linewidth=1.5)
        ax.fill_between(t, y, 0.0, color=color, alpha=0.15)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, max(audio.shape[0] / sample_rate, 1e-6))
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.2)
        fig.tight_layout()

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=110)
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    ensure_dir(output_path.parent)
    imageio.mimsave(output_path, frames, duration=0.08)


def save_waveform_story_gif(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
    color: str = "#2455a4",
    frame_count: int = 40,
    zoom_seconds: float = 0.03,
) -> None:
    audio = np.asarray(audio, dtype=float)
    total_duration = max(audio.shape[0] / float(sample_rate), 1e-6)
    full_t, full_low, full_high, full_rms = _envelope_summary(audio, sample_rate, max_points=2600)
    window_samples = max(256, int(sample_rate * zoom_seconds))
    frames = []

    for frame_idx in range(frame_count):
        progress = frame_idx / max(frame_count - 1, 1)
        sample_index = min(max(int(progress * max(audio.shape[0] - 1, 0)), 0), max(audio.shape[0] - 1, 0))
        current_time = sample_index / float(sample_rate)
        reveal_mask = full_t <= current_time
        current_value = float(audio[sample_index]) if audio.size else 0.0
        metric_window = _metric_window(audio, sample_index, window_samples)
        rms, dominant_frequency = _estimate_window_metrics(metric_window, sample_rate)
        zoom_left, zoom_right = _zoom_limits(current_time, total_duration, zoom_seconds)

        fig = plt.figure(figsize=(10.5, 6.6), facecolor="#f4efe6", constrained_layout=True)
        gs = fig.add_gridspec(2, 1, height_ratios=[1.25, 0.95], hspace=0.28)
        ax_full = fig.add_subplot(gs[0])
        ax_zoom = fig.add_subplot(gs[1])
        _style_wave_axes(ax_full)
        _style_wave_axes(ax_zoom)

        ax_full.fill_between(full_t, full_low, full_high, color="#dfd7ca", alpha=0.8, linewidth=0.0)
        ax_full.plot(full_t, full_high, color="#b9afa2", linewidth=0.85, alpha=0.95)
        ax_full.plot(full_t, full_low, color="#b9afa2", linewidth=0.85, alpha=0.95)
        ax_full.plot(full_t, full_rms, color="#9f9587", linewidth=0.9, alpha=0.95)
        ax_full.plot(full_t, -full_rms, color="#9f9587", linewidth=0.9, alpha=0.95)
        if np.any(reveal_mask):
            ax_full.fill_between(full_t[reveal_mask], full_low[reveal_mask], full_high[reveal_mask], color=color, alpha=0.34, linewidth=0.0)
            ax_full.plot(full_t[reveal_mask], full_high[reveal_mask], color=color, linewidth=1.1, alpha=0.95)
            ax_full.plot(full_t[reveal_mask], full_low[reveal_mask], color=color, linewidth=1.1, alpha=0.95)
            ax_full.plot(full_t[reveal_mask], full_rms[reveal_mask], color="#1f1f1f", linewidth=1.05, alpha=0.85)
            ax_full.plot(full_t[reveal_mask], -full_rms[reveal_mask], color="#1f1f1f", linewidth=1.05, alpha=0.85)
        ax_full.axvspan(0.0, current_time, color=color, alpha=0.05)
        ax_full.axvline(current_time, color="#161616", linewidth=1.7, alpha=0.9)
        ax_full.scatter([current_time], [current_value], s=32, color="#161616", zorder=5)
        ax_full.set_xlim(0.0, total_duration)
        ax_full.set_ylim(-1.1, 1.1)
        ax_full.set_title(title, fontsize=14, fontweight="bold", loc="left", color="#2a2118")
        ax_full.set_xlabel("Timeline [s]", color="#4b4034")
        ax_full.set_ylabel("Amplitude", color="#4b4034")
        ax_full.tick_params(colors="#5b4f43")
        ax_full.text(
            0.985,
            0.94,
            f"time {current_time:0.2f}s / {total_duration:0.2f}s\nrms {rms:0.3f}\npeak freq {dominant_frequency:0.1f} Hz",
            transform=ax_full.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#2a2118",
            bbox={"boxstyle": "round,pad=0.45", "facecolor": "#fffaf2", "edgecolor": "#d8cbb9", "linewidth": 1.0},
        )

        zoom_t = _time_axis(audio, sample_rate)
        zoom_mask = (zoom_t >= zoom_left) & (zoom_t <= zoom_right)
        ax_zoom.plot(zoom_t[zoom_mask], audio[zoom_mask], color=color, linewidth=1.7)
        ax_zoom.fill_between(zoom_t[zoom_mask], audio[zoom_mask], 0.0, color=color, alpha=0.18)
        ax_zoom.axhline(0.0, color="#8d8377", linewidth=0.9, alpha=0.8)
        ax_zoom.axvline(current_time, color="#161616", linewidth=1.7, alpha=0.9)
        ax_zoom.scatter([current_time], [current_value], s=30, color="#161616", zorder=5)
        ax_zoom.set_xlim(zoom_left, zoom_right)
        ax_zoom.set_ylim(-1.1, 1.1)
        ax_zoom.set_title("Live Window", fontsize=12, loc="left", color="#2a2118")
        ax_zoom.set_xlabel("Local Time [s]", color="#4b4034")
        ax_zoom.set_ylabel("Amplitude", color="#4b4034")
        ax_zoom.tick_params(colors="#5b4f43")

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=125, facecolor=fig.get_facecolor())
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    ensure_dir(output_path.parent)
    imageio.mimsave(output_path, frames, duration=0.08, loop=0)


def save_simple_waveform_gif(
    audio: np.ndarray,
    sample_rate: int,
    output_path: Path,
    title: str,
    color: str = "tab:blue",
    frame_count: int = 36,
) -> None:
    audio = np.asarray(audio, dtype=float)
    if audio.size == 0:
        ensure_dir(output_path.parent)
        imageio.mimsave(output_path, [], duration=0.08, loop=0)
        return

    total_duration = audio.shape[0] / float(sample_rate)
    full_t, full_y = _downsample(audio, sample_rate, max_points=3200)
    frames = []

    for frame_idx in range(1, frame_count + 1):
        end = max(1, int(frame_idx / frame_count * audio.shape[0]))
        t, y = _downsample(audio[:end], sample_rate, max_points=3200)
        current_time = end / float(sample_rate)

        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#faf7f1")
        ax.plot(full_t, full_y, color="#d8d0c5", linewidth=1.0, alpha=0.9)
        ax.plot(t, y, color=color, linewidth=1.5)
        ax.fill_between(t, y, 0.0, color=color, alpha=0.14)
        ax.axvline(current_time, color="#161616", linewidth=1.2, alpha=0.85)
        ax.set_title(title)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")
        ax.set_xlim(0.0, total_duration)
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.22)
        ax.text(
            0.985,
            0.94,
            f"time {current_time:.2f}s / {total_duration:.2f}s",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            color="#1f1f1f",
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fffaf2", "edgecolor": "#d8cbb9", "linewidth": 1.0},
        )
        fig.tight_layout()
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor=fig.get_facecolor())
        plt.close(fig)
        buffer.seek(0)
        frames.append(imageio.imread(buffer))

    ensure_dir(output_path.parent)
    imageio.mimsave(output_path, frames, duration=0.08, loop=0)


def save_confusion_matrix(matrix: np.ndarray, labels: list[str], output_path: Path, title: str) -> None:
    matrix = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_feature_scatter(rows: list[dict], output_path: Path, title: str, x_key: str, y_key: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for label, color in [("piano", "tab:blue"), ("drums", "tab:orange")]:
        points = [row for row in rows if row["label"] == label]
        x = [float(row[x_key]) for row in points]
        y = [float(row[y_key]) for row in points]
        ax.scatter(x, y, s=18, alpha=0.6, label=label, color=color)
    ax.set_title(title)
    ax.set_xlabel(x_key.replace("_", " ").title())
    ax.set_ylabel(y_key.replace("_", " ").title())
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_accuracy_bar(values: dict[str, float], output_path: Path, title: str) -> None:
    labels = list(values.keys())
    scores = [float(values[label]) for label in labels]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(labels, scores, color=["tab:gray", "tab:green"])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    for idx, score in enumerate(scores):
        ax.text(idx, score + 0.02, f"{score:.3f}", ha="center")
    fig.tight_layout()
    ensure_dir(output_path.parent)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
