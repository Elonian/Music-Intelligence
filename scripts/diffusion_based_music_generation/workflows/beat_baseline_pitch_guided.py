#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.evaluate_diffusion_generation import evaluate_generation  # noqa: E402
from scripts.diffusion_based_music_generation.audio_io import spectrogram_stats, write_sample_wavs  # noqa: E402
from scripts.diffusion_based_music_generation.dataset import FREQ_BINS, N_FFT, SR, TIME_FRAMES  # noqa: E402
from scripts.diffusion_based_music_generation.generation import make_pitch_grid, sample_batch  # noqa: E402
from scripts.diffusion_based_music_generation.paths import OUTPUT_ROOT, PRETRAINED_KEYBOARD_CKPT, RUNS_DIR  # noqa: E402
from scripts.diffusion_based_music_generation.training import select_device, set_random_seed  # noqa: E402
from utils.io_helpers import ensure_dir, save_json  # noqa: E402


RUN_NAME = "beat_baseline_pitch_guided"
DEFAULT_CHECKPOINT = RUNS_DIR / "q4_full_guitar" / "checkpoints" / "model_ft.pt"
DEFAULT_HISTORY = RUNS_DIR / "q4_full_guitar" / "history.json"
METHOD = (
    "Pitch-guided candidate selection: generate multiple candidates for every requested MIDI pitch "
    "with stronger ODE settings, score harmonic energy around the target pitch against nearby "
    "off-target pitches, reject unstable spectra, and keep the best candidate per pitch."
)


@dataclass(frozen=True)
class CandidateSetting:
    label: str
    sampler: str
    n_steps: int
    guidance_scale: float


DEFAULT_CANDIDATE_SETTINGS = (
    CandidateSetting("heun_50_gs5", "heun", 50, 5.0),
    CandidateSetting("heun_64_gs6", "heun", 64, 6.0),
    CandidateSetting("rk4_32_gs6", "rk4", 32, 6.0),
    CandidateSetting("rk4_50_gs65", "rk4", 50, 6.5),
)


@dataclass
class BeatBaselineConfig:
    checkpoint: Path = DEFAULT_CHECKPOINT
    baseline_checkpoint: Path = PRETRAINED_KEYBOARD_CKPT
    history: Path = DEFAULT_HISTORY
    output_dir: Path = OUTPUT_ROOT / RUN_NAME
    n_samples: int = 100
    pitch_start: int = 48
    pitch_span: int = 36
    noise_variants: int = 3
    batch_size: int = 24
    seed: int = 0
    max_wavs: int = 12
    require_cuda: bool = False


def _serializable_config(config: BeatBaselineConfig) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _midi_to_hz(pitch: int) -> float:
    return 440.0 * (2.0 ** ((int(pitch) - 69) / 12.0))


def _training_analysis(history_path: Path) -> dict:
    if not history_path.exists():
        return {"history_path": str(history_path), "available": False}
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    rows = payload.get("history", [])
    losses = [float(row["loss"]) for row in rows if row.get("loss") is not None]
    if not losses:
        return {"history_path": str(history_path), "available": False}

    tail = losses[-30:] if len(losses) >= 30 else losses
    best_index = int(np.argmin(losses))
    first = float(losses[0])
    final = float(losses[-1])
    best = float(losses[best_index])
    return {
        "history_path": str(history_path),
        "available": True,
        "epochs": len(rows),
        "optimizer_steps": int(rows[-1].get("step", 0)),
        "first_epoch_loss": first,
        "final_epoch_loss": final,
        "best_epoch": int(rows[best_index].get("epoch", best_index + 1)),
        "best_epoch_loss": best,
        "tail_30_mean_loss": float(np.mean(tail)),
        "tail_30_std_loss": float(np.std(tail)),
        "relative_improvement_from_epoch_1": float((first - final) / max(first, 1e-8)),
        "final_within_3_percent_of_best": bool(final <= best * 1.03),
    }


def _harmonic_energy(mag_freq: np.ndarray, pitch: int, radius: int = 1) -> float:
    bin_hz = SR / N_FFT
    total = 0.0
    weight_total = 0.0
    harmonic = 1
    while True:
        frequency = _midi_to_hz(pitch) * harmonic
        if frequency >= SR / 2:
            break
        center = int(round(frequency / bin_hz))
        if center <= 0 or center >= len(mag_freq):
            harmonic += 1
            continue
        low = max(0, center - radius)
        high = min(len(mag_freq), center + radius + 1)
        weight = 1.0 / np.sqrt(float(harmonic))
        total += float(mag_freq[low:high].mean()) * weight
        weight_total += weight
        harmonic += 1
    return total / max(weight_total, 1e-8)


def _score_one(spec: torch.Tensor, pitch: int) -> dict:
    values = spec.detach().cpu().to(torch.float32)
    mag_freq = torch.sqrt(values[0].square() + values[1].square()).mean(dim=-1).numpy()
    target = _harmonic_energy(mag_freq, pitch)
    off_scores = [_harmonic_energy(mag_freq, pitch + offset) for offset in (-2, -1, 1, 2)]
    off_max = max(off_scores) if off_scores else 0.0
    global_energy = float(np.mean(mag_freq)) + 1e-8
    target_ratio = target / global_energy
    margin_ratio = (target - off_max) / (abs(off_max) + 1e-8)
    sample_std = float(values.std())
    sample_absmax = float(values.abs().max())
    finite = bool(torch.isfinite(values).all())
    artifact_penalty = 0.0
    artifact_penalty += max(0.0, 0.12 - sample_std) * 8.0
    artifact_penalty += max(0.0, sample_std - 1.55) * 1.5
    artifact_penalty += max(0.0, sample_absmax - 18.0) * 0.08
    if not finite:
        artifact_penalty += 1_000.0
    total_score = 2.5 * target_ratio + 1.5 * margin_ratio - artifact_penalty
    return {
        "target_harmonic": float(target),
        "off_target_harmonic": float(off_max),
        "target_ratio": float(target_ratio),
        "margin_ratio": float(margin_ratio),
        "sample_std": sample_std,
        "sample_absmax": sample_absmax,
        "finite": finite,
        "artifact_penalty": float(artifact_penalty),
        "total_score": float(total_score),
    }


def _score_collection(samples: torch.Tensor, pitches: torch.Tensor) -> dict:
    rows = [_score_one(samples[index], int(pitches[index])) for index in range(samples.shape[0])]
    target_ratios = np.array([row["target_ratio"] for row in rows], dtype=np.float32)
    margins = np.array([row["margin_ratio"] for row in rows], dtype=np.float32)
    totals = np.array([row["total_score"] for row in rows], dtype=np.float32)
    finite = all(row["finite"] for row in rows)
    return {
        "sample_count": int(samples.shape[0]),
        "finite": bool(finite),
        "mean_target_ratio": float(target_ratios.mean()),
        "mean_margin_ratio": float(margins.mean()),
        "positive_margin_rate": float((margins > 0).mean()),
        "mean_total_score": float(totals.mean()),
        "min_total_score": float(totals.min()),
        "max_total_score": float(totals.max()),
    }


def _load_model(checkpoint: Path, device: torch.device):
    from scripts.diffusion_based_music_generation.model import load_flow_model

    model, checkpoint_payload = load_flow_model(str(checkpoint), device=str(device))
    model.eval()
    return model, checkpoint_payload


def _generate_reference(
    checkpoint: Path,
    pitches: torch.Tensor,
    seed: int,
    batch_size: int,
    device: torch.device,
    sampler: str = "heun",
    n_steps: int = 25,
    guidance_scale: float = 6.0,
) -> torch.Tensor:
    set_random_seed(seed)
    model, _checkpoint_payload = _load_model(checkpoint, device)
    noises = torch.randn(pitches.shape[0], 2, FREQ_BINS, TIME_FRAMES, device=device)
    outputs: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, pitches.shape[0], batch_size):
            stop = min(start + batch_size, pitches.shape[0])
            generated = sample_batch(
                model,
                noises[start:stop].clone(),
                pitches[start:stop],
                sampler=sampler,
                n_steps=n_steps,
                guidance_scale=guidance_scale,
            )
            outputs.append(generated.detach().cpu())
    return torch.cat(outputs, dim=0)


def run_pitch_guided_generation(config: BeatBaselineConfig) -> dict:
    if not config.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {config.checkpoint}")

    set_random_seed(config.seed)
    device = select_device(config.require_cuda)
    output_dir = ensure_dir(config.output_dir)
    model, checkpoint_payload = _load_model(config.checkpoint, device)

    target_pitches = make_pitch_grid(config.n_samples, config.pitch_start, config.pitch_span, device=device)
    candidate_pitches = target_pitches.repeat_interleave(config.noise_variants)
    candidate_count = int(candidate_pitches.shape[0])
    candidate_noises = torch.randn(candidate_count, 2, FREQ_BINS, TIME_FRAMES, device=device)

    best_samples: list[torch.Tensor | None] = [None] * config.n_samples
    best_noises: list[torch.Tensor | None] = [None] * config.n_samples
    best_scores: list[float] = [-float("inf")] * config.n_samples
    best_rows: list[dict | None] = [None] * config.n_samples
    score_rows: list[dict] = []

    with torch.no_grad():
        for setting in DEFAULT_CANDIDATE_SETTINGS:
            generated_batches: list[torch.Tensor] = []
            for start in range(0, candidate_count, config.batch_size):
                stop = min(start + config.batch_size, candidate_count)
                generated = sample_batch(
                    model,
                    candidate_noises[start:stop].clone(),
                    candidate_pitches[start:stop],
                    sampler=setting.sampler,
                    n_steps=setting.n_steps,
                    guidance_scale=setting.guidance_scale,
                )
                generated_batches.append(generated.detach().cpu())

            generated_all = torch.cat(generated_batches, dim=0)
            for candidate_index in range(candidate_count):
                sample_index = candidate_index // config.noise_variants
                variant_index = candidate_index % config.noise_variants
                pitch = int(candidate_pitches[candidate_index].detach().cpu())
                score = _score_one(generated_all[candidate_index], pitch)
                row = {
                    "sample_index": sample_index,
                    "variant_index": variant_index,
                    "pitch": pitch,
                    "setting": setting.label,
                    "sampler": setting.sampler,
                    "n_steps": setting.n_steps,
                    "guidance_scale": setting.guidance_scale,
                    **score,
                    "selected": False,
                }
                if score["total_score"] > best_scores[sample_index]:
                    best_scores[sample_index] = score["total_score"]
                    best_samples[sample_index] = generated_all[candidate_index].clone()
                    best_noises[sample_index] = candidate_noises[candidate_index].detach().cpu().clone()
                    best_rows[sample_index] = row
                score_rows.append(row)

    selected_rows: list[dict] = []
    for sample_index, row in enumerate(best_rows):
        if row is None:
            raise RuntimeError(f"No candidate selected for sample {sample_index}")
        row = dict(row)
        row["selected"] = True
        selected_rows.append(row)

    samples = torch.stack([sample for sample in best_samples if sample is not None], dim=0)
    noises = torch.stack([noise for noise in best_noises if noise is not None], dim=0)
    pitches_cpu = target_pitches.detach().cpu()

    selected_setting = np.array([row["setting"] for row in selected_rows])
    selected_sampler = np.array([row["sampler"] for row in selected_rows])
    selected_steps = np.array([row["n_steps"] for row in selected_rows], dtype=np.int32)
    selected_guidance = np.array([row["guidance_scale"] for row in selected_rows], dtype=np.float32)
    selected_score = np.array([row["total_score"] for row in selected_rows], dtype=np.float32)

    samples_path = output_dir / "pitch_guided_beat_baseline_samples.npz"
    np.savez_compressed(
        samples_path,
        samples=samples.numpy().astype(np.float32),
        noises=noises.numpy().astype(np.float32),
        pitches=pitches_cpu.numpy().astype(np.int64),
        sampler=np.array("pitch_guided_candidate_selection"),
        n_steps=np.array(max(setting.n_steps for setting in DEFAULT_CANDIDATE_SETTINGS), dtype=np.int32),
        guidance_scale=np.array(max(setting.guidance_scale for setting in DEFAULT_CANDIDATE_SETTINGS), dtype=np.float32),
        selected_setting=selected_setting,
        selected_sampler=selected_sampler,
        selected_n_steps=selected_steps,
        selected_guidance_scale=selected_guidance,
        selected_score=selected_score,
        method=np.array(METHOD),
        source_checkpoint=np.array(str(config.checkpoint)),
    )

    checkpoint_output = output_dir / "pitch_guided_beat_baseline_model.pt"
    checkpoint_payload = dict(checkpoint_payload)
    checkpoint_payload["beat_baseline_method"] = METHOD
    checkpoint_payload["beat_baseline_source_checkpoint"] = str(config.checkpoint)
    checkpoint_payload["beat_baseline_candidate_settings"] = [asdict(setting) for setting in DEFAULT_CANDIDATE_SETTINGS]
    torch.save(checkpoint_payload, checkpoint_output)

    _write_csv(output_dir / "candidate_scores.csv", score_rows)
    _write_csv(output_dir / "selected_candidates.csv", selected_rows)
    wav_files = write_sample_wavs(samples, output_dir / "audio", pitches_cpu, max_wavs=config.max_wavs)
    evaluation = evaluate_generation(samples_path, output_dir=output_dir / "evaluation", max_audio=config.max_wavs)

    pitch_score = _score_collection(samples, pitches_cpu)
    baseline_score = None
    if config.baseline_checkpoint.exists():
        baseline_samples = _generate_reference(
            config.baseline_checkpoint,
            target_pitches,
            seed=config.seed,
            batch_size=config.batch_size,
            device=device,
            sampler="heun",
            n_steps=25,
            guidance_scale=6.0,
        )
        baseline_score = _score_collection(baseline_samples, pitches_cpu)

    score_comparison = {
        "metric_note": "Higher is better for these local harmonic pitch-proxy scores.",
        "selected_pitch_guided": pitch_score,
        "notebook_baseline_pretrained_keyboard_heun25_gs6": baseline_score,
    }
    if baseline_score is not None:
        score_comparison["relative_mean_target_ratio_gain"] = float(
            (pitch_score["mean_target_ratio"] - baseline_score["mean_target_ratio"])
            / max(abs(baseline_score["mean_target_ratio"]), 1e-8)
        )
        score_comparison["positive_margin_rate_gain"] = float(
            pitch_score["positive_margin_rate"] - baseline_score["positive_margin_rate"]
        )
    save_json(output_dir / "pitch_score_baseline_comparison.json", score_comparison)

    selected_counts: dict[str, int] = {}
    for row in selected_rows:
        selected_counts[row["setting"]] = selected_counts.get(row["setting"], 0) + 1

    summary = {
        "run_name": RUN_NAME,
        "method": METHOD,
        "output_dir": str(output_dir),
        "samples_path": str(samples_path),
        "checkpoint_path": str(checkpoint_output),
        "source_checkpoint": str(config.checkpoint),
        "candidate_settings": [asdict(setting) for setting in DEFAULT_CANDIDATE_SETTINGS],
        "candidate_count": len(score_rows),
        "candidates_per_sample": len(DEFAULT_CANDIDATE_SETTINGS) * config.noise_variants,
        "selected_setting_counts": selected_counts,
        "selected_score_mean": float(selected_score.mean()),
        "selected_score_min": float(selected_score.min()),
        "selected_score_max": float(selected_score.max()),
        "training_analysis": _training_analysis(config.history),
        "pitch_score_comparison": score_comparison,
        "stats": spectrogram_stats(samples),
        "wav_files": wav_files,
        "evaluation": evaluation,
        "config": _serializable_config(config),
    }
    save_json(output_dir / "beat_baseline_method_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pitch-guided candidate selection for Part 5 beat-baseline generation.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-checkpoint", type=Path, default=PRETRAINED_KEYBOARD_CKPT)
    parser.add_argument("--history", type=Path, default=DEFAULT_HISTORY)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT / RUN_NAME)
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--pitch-start", type=int, default=48)
    parser.add_argument("--pitch-span", type=int, default=36)
    parser.add_argument("--noise-variants", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-wavs", type=int, default=12)
    parser.add_argument("--require-cuda", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_pitch_guided_generation(
        BeatBaselineConfig(
            checkpoint=args.checkpoint,
            baseline_checkpoint=args.baseline_checkpoint,
            history=args.history,
            output_dir=args.output_dir,
            n_samples=args.n_samples,
            pitch_start=args.pitch_start,
            pitch_span=args.pitch_span,
            noise_variants=args.noise_variants,
            batch_size=args.batch_size,
            seed=args.seed,
            max_wavs=args.max_wavs,
            require_cuda=args.require_cuda,
        )
    )
    print(
        json.dumps(
            {
                "samples_path": summary["samples_path"],
                "checkpoint_path": summary["checkpoint_path"],
                "candidate_count": summary["candidate_count"],
                "candidates_per_sample": summary["candidates_per_sample"],
                "sample_count": summary["evaluation"]["sample_count"],
                "shape_ok": summary["evaluation"]["shape_ok"],
                "selected_setting_counts": summary["selected_setting_counts"],
                "training_final_within_3_percent_of_best": summary["training_analysis"].get(
                    "final_within_3_percent_of_best"
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
