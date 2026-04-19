#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.multitrack_generation.generation import (  # noqa: E402
    GenerationConfig,
    generate_sequence,
    prompt_by_name,
    save_generation_bundle,
)
from scripts.multitrack_generation.models import MODEL_SPECS, load_model_checkpoint  # noqa: E402
from scripts.multitrack_generation.quality import (  # noqa: E402
    PAPER_OBJECTIVE_METRICS,
    find_generated_note_files,
    iter_reference_notes,
    load_generated_notes,
    note_quality_metrics,
    paper_metric_distance,
    summarize_metric_rows,
)
from scripts.multitrack_generation.events import sequence_to_note_array  # noqa: E402
from utils.io_helpers import ensure_dir, save_json, write_csv_rows  # noqa: E402
from utils.project_paths import (  # noqa: E402
    MULTITRACK_GENERATION_EVALUATION_DIR,
    MULTITRACK_GENERATION_GENERATED_DIR,
    MULTITRACK_GENERATION_OUTPUT_ROOT,
)


SHAPE_METRICS = ("notes_per_beat", "active_instrument_count", "average_polyphony", "instrument_entropy")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate multitrack generation with notebook metrics plus paper-style objective music metrics."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=MULTITRACK_GENERATION_OUTPUT_ROOT / "runs" / "full_transformer" / "checkpoints" / "best_model.pt",
    )
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default=None)
    parser.add_argument("--positional-mode", choices=["sequence", "notebook"], default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--run-name", default="full_transformer")
    parser.add_argument("--generated-path", type=Path, default=MULTITRACK_GENERATION_GENERATED_DIR / "full_transformer")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-reference-files", type=int, default=0, help="0 means all files in the split.")
    parser.add_argument("--reference-max-beats", type=int, default=32)
    parser.add_argument("--tempo-bpm", type=int, default=120)
    parser.add_argument("--search", action="store_true", help="Generate candidates and keep the best paper-metric match.")
    parser.add_argument("--search-name", default=None)
    parser.add_argument("--num-search-samples", type=int, default=18)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--prompts", default="empty,piano_guitar_bass,twinkle")
    parser.add_argument("--seed", type=int, default=420)
    parser.add_argument("--min-notes", type=int, default=64)
    return parser.parse_args()


def _scalar_row(prefix: dict, metrics: dict, score: dict | None = None) -> dict:
    row = dict(prefix)
    keep = [
        "note_count",
        "length_beats",
        "length_seconds",
        "notes_per_beat",
        "unique_pitch_count",
        "unique_pitch_class_count",
        "active_instrument_count",
        "instrument_entropy",
        "average_polyphony",
        "max_polyphony",
        "mean_duration_steps",
        "pitch_class_entropy",
        "scale_consistency_percent",
        "groove_consistency_percent",
        "sequence_len",
        "raw_note_events",
        "unique_note_events",
        "duplicate_note_rate",
        "note_instrument_violation_rate",
    ]
    for key in keep:
        value = metrics.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            row[key] = f"{float(value):.6f}" if isinstance(value, (float, np.floating)) else int(value)
        elif value is None:
            row[key] = ""
    row["best_scale"] = metrics.get("best_scale") or ""
    row["instrument_counts"] = json.dumps(metrics.get("instrument_counts", {}), sort_keys=True)
    if score:
        for key, value in score.items():
            if key.endswith("_components"):
                row[key] = json.dumps(value, sort_keys=True)
            elif isinstance(value, (int, float)):
                row[key] = f"{float(value):.6f}"
    return row


def _stat_mean(summary: dict, key: str) -> float | None:
    item = summary.get(key)
    if not isinstance(item, dict) or item.get("mean") is None:
        return None
    return float(item["mean"])


def _normalized_metric_distance(metrics: dict, reference_summary: dict, keys: tuple[str, ...], weight: float) -> tuple[float, dict]:
    total = 0.0
    components: dict[str, float] = {}
    for key in keys:
        value = metrics.get(key)
        reference = reference_summary.get(key)
        if value is None or not isinstance(reference, dict):
            continue
        ref_mean = float(reference.get("mean") or 0.0)
        ref_std = max(float(reference.get("std") or 0.0), abs(ref_mean) * 0.05, 1e-6)
        distance = abs(float(value) - ref_mean) / ref_std
        components[key] = float(distance)
        total += weight * distance
    return float(total), components


def _quality_score(metrics: dict, reference_summary: dict, min_notes: int) -> dict:
    paper = paper_metric_distance(metrics, reference_summary)
    shape_distance, shape_components = _normalized_metric_distance(metrics, reference_summary, SHAPE_METRICS, weight=0.35)
    note_count = int(metrics.get("note_count") or 0)
    penalty = 0.0
    if note_count <= 0:
        penalty += 100.0
    elif note_count < min_notes:
        penalty += 4.0 * (min_notes - note_count) / max(min_notes, 1)
    if float(metrics.get("length_beats") or 0.0) < 8.0:
        penalty += 2.0
    penalty += 3.0 * float(metrics.get("duplicate_note_rate") or 0.0)
    penalty += 5.0 * float(metrics.get("note_instrument_violation_rate") or 0.0)
    if int(metrics.get("active_instrument_count") or 0) == 0:
        penalty += 3.0
    score = float(paper["paper_metric_distance"]) + shape_distance + penalty
    return {
        "quality_score": score,
        "paper_metric_distance": float(paper["paper_metric_distance"]),
        "paper_metric_distance_components": paper["paper_metric_distance_components"],
        "shape_metric_distance": shape_distance,
        "shape_metric_distance_components": shape_components,
        "validity_penalty": float(penalty),
    }


def _candidate_configs(prompts: list[str], count: int, seed: int, max_seq_len: int) -> list[tuple[str, GenerationConfig]]:
    decoding_grid = [
        {"decoding": "topk", "temperature": 0.85, "top_k": 5, "top_p": 0.90},
        {"decoding": "topk", "temperature": 0.90, "top_k": 8, "top_p": 0.90},
        {"decoding": "topk", "temperature": 0.95, "top_k": 12, "top_p": 0.90},
        {"decoding": "topp", "temperature": 0.90, "top_k": 8, "top_p": 0.88},
        {"decoding": "topp", "temperature": 0.95, "top_k": 8, "top_p": 0.92},
        {"decoding": "random", "temperature": 0.85, "top_k": 8, "top_p": 0.90},
        {"decoding": "random", "temperature": 1.00, "top_k": 8, "top_p": 0.90},
    ]
    configs: list[tuple[str, GenerationConfig]] = []
    pairs = list(itertools.product(prompts, decoding_grid))
    for index in range(count):
        prompt, spec = pairs[index % len(pairs)]
        configs.append(
            (
                prompt,
                GenerationConfig(
                    decoding=spec["decoding"],
                    temperature=spec["temperature"],
                    top_k=int(spec["top_k"]),
                    top_p=spec["top_p"],
                    max_seq_len=max_seq_len,
                    seed=seed + index,
                ),
            )
        )
    return configs


def _prompt_richness_penalty(prompt_name: str, metrics: dict) -> float:
    active = int(metrics.get("active_instrument_count") or 0)
    entropy = float(metrics.get("instrument_entropy") or 0.0)
    if prompt_name == "all_instruments":
        return 1.15 * max(0, 5 - active) + 0.75 * max(0.0, 1.5 - entropy)
    if prompt_name == "piano_guitar_bass":
        return 0.95 * max(0, 3 - active) + 0.55 * max(0.0, 1.0 - entropy)
    return 0.0


def _evaluate_existing_generated(path: Path, reference_summary: dict, tempo_bpm: int, min_notes: int) -> tuple[list[dict], list[dict]]:
    if not path.exists():
        return [], []
    metrics_rows: list[dict] = []
    csv_rows: list[dict] = []
    for notes_file in find_generated_note_files(path):
        notes, sequence = load_generated_notes(notes_file)
        metrics = note_quality_metrics(notes, sequence=sequence, tempo_bpm=tempo_bpm)
        score = _quality_score(metrics, reference_summary, min_notes=min_notes)
        name = notes_file.parent.name if notes_file.name == "notes.npy" else notes_file.stem
        metrics["name"] = name
        metrics["path"] = str(notes_file)
        metrics.update(score)
        metrics_rows.append(metrics)
        csv_rows.append(_scalar_row({"kind": "existing", "name": name, "path": str(notes_file)}, metrics, score))
    return metrics_rows, csv_rows


def _run_candidate_search(
    args: argparse.Namespace,
    reference_summary: dict,
    output_dir: Path,
) -> tuple[list[dict], dict | None]:
    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_checkpoint(args.checkpoint, model_name=args.model, device=device, positional_mode=args.positional_mode)
    prompts = [item.strip() for item in args.prompts.split(",") if item.strip()]
    candidates = _candidate_configs(prompts, args.num_search_samples, args.seed, args.max_seq_len)
    rows: list[dict] = []
    best: dict | None = None
    for index, (prompt_name, config) in enumerate(candidates, start=1):
        start = time.perf_counter()
        sequence = generate_sequence(model, prompt=prompt_by_name(prompt_name), config=config, device=device)
        elapsed = max(time.perf_counter() - start, 1e-9)
        notes = sequence_to_note_array(sequence)
        metrics = note_quality_metrics(notes, sequence=sequence, tempo_bpm=args.tempo_bpm)
        score = _quality_score(metrics, reference_summary, min_notes=args.min_notes)
        richness_penalty = _prompt_richness_penalty(prompt_name, metrics)
        if richness_penalty:
            score["prompt_richness_penalty"] = float(richness_penalty)
            score["quality_score"] = float(score["quality_score"]) + float(richness_penalty)
        note_count = max(int(metrics.get("note_count") or 0), 1)
        candidate = {
            "kind": "candidate",
            "name": f"candidate_{index:03d}",
            "prompt": prompt_name,
            "config": asdict(config),
            "generation_seconds": float(elapsed),
            "inference_notes_per_second": float(note_count / elapsed),
            "metrics": metrics,
            "score": score,
            "sequence": sequence,
        }
        flat_metrics = dict(metrics)
        flat_metrics["inference_notes_per_second"] = candidate["inference_notes_per_second"]
        rows.append(
            _scalar_row(
                {
                    "kind": "candidate",
                    "name": candidate["name"],
                    "prompt": prompt_name,
                    "decoding": config.decoding,
                    "temperature": f"{config.temperature:.3f}",
                    "top_k": config.top_k,
                    "top_p": f"{config.top_p:.3f}",
                    "seed": config.seed,
                    "generation_seconds": f"{elapsed:.6f}",
                    "inference_notes_per_second": f"{candidate['inference_notes_per_second']:.6f}",
                },
                flat_metrics,
                score,
            )
        )
        if best is None or float(score["quality_score"]) < float(best["score"]["quality_score"]):
            best = candidate
        print(
            f"[{index}/{len(candidates)}] prompt={prompt_name} decoding={config.decoding} "
            f"notes={metrics['note_count']} score={score['quality_score']:.3f}",
            flush=True,
        )
    write_csv_rows(output_dir / "candidate_scores.csv", rows)
    if best is not None:
        name = args.search_name or f"{args.run_name}_quality_selected"
        summary = save_generation_bundle(
            best["sequence"],
            output_dir=MULTITRACK_GENERATION_GENERATED_DIR,
            name=name,
            config=GenerationConfig(**best["config"]),
            tempo_bpm=args.tempo_bpm,
        )
        best_payload = {
            key: value
            for key, value in best.items()
            if key != "sequence"
        }
        best_payload["saved_bundle"] = summary
        save_json(output_dir / "best_candidate.json", best_payload)
    return rows, best


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir or (MULTITRACK_GENERATION_EVALUATION_DIR / args.run_name / "generation_quality"))
    max_reference_files = None if args.max_reference_files <= 0 else args.max_reference_files
    reference_rows = [
        note_quality_metrics(notes, tempo_bpm=args.tempo_bpm)
        for notes in iter_reference_notes(
            args.data_dir,
            split=args.split,
            max_files=max_reference_files,
            max_beats=args.reference_max_beats,
        )
    ]
    reference_summary = summarize_metric_rows(reference_rows)
    save_json(output_dir / "reference_quality_summary.json", reference_summary)
    write_csv_rows(
        output_dir / "reference_quality_samples.csv",
        [_scalar_row({"kind": "reference", "name": f"{args.split}_{index:05d}"}, row) for index, row in enumerate(reference_rows)],
    )

    existing_metrics, existing_rows = _evaluate_existing_generated(
        args.generated_path,
        reference_summary,
        tempo_bpm=args.tempo_bpm,
        min_notes=args.min_notes,
    )
    if existing_rows:
        write_csv_rows(output_dir / "generated_quality.csv", existing_rows)

    candidate_rows: list[dict] = []
    best_candidate: dict | None = None
    if args.search:
        candidate_rows, best_candidate = _run_candidate_search(args, reference_summary, output_dir)

    existing_best = min(existing_metrics, key=lambda item: float(item["quality_score"])) if existing_metrics else None
    best_candidate_score = float(best_candidate["score"]["quality_score"]) if best_candidate is not None else None
    existing_best_score = float(existing_best["quality_score"]) if existing_best is not None else None
    summary = {
        "run_name": args.run_name,
        "split": args.split,
        "reference_files": len(reference_rows),
        "paper_objective_metrics": list(PAPER_OBJECTIVE_METRICS),
        "paper_metric_rule": "For the paper objective metrics, closer to the reference split mean is better.",
        "reference_summary_path": str(output_dir / "reference_quality_summary.json"),
        "generated_quality_path": str(output_dir / "generated_quality.csv") if existing_rows else None,
        "candidate_scores_path": str(output_dir / "candidate_scores.csv") if candidate_rows else None,
        "best_candidate_path": str(output_dir / "best_candidate.json") if best_candidate is not None else None,
        "existing_best": None
        if existing_best is None
        else {
            "name": existing_best["name"],
            "quality_score": existing_best_score,
            "note_count": existing_best.get("note_count"),
            "paper_metric_distance": existing_best.get("paper_metric_distance"),
        },
        "best_candidate": None
        if best_candidate is None
        else {
            "name": best_candidate["name"],
            "prompt": best_candidate["prompt"],
            "config": best_candidate["config"],
            "quality_score": best_candidate_score,
            "note_count": best_candidate["metrics"].get("note_count"),
            "paper_metric_distance": best_candidate["score"].get("paper_metric_distance"),
        },
        "candidate_improved_over_existing": (
            None
            if best_candidate_score is None or existing_best_score is None
            else bool(best_candidate_score < existing_best_score)
        ),
        "reference_means": {key: _stat_mean(reference_summary, key) for key in PAPER_OBJECTIVE_METRICS},
    }
    save_json(output_dir / "generation_quality_summary.json", summary)
    print(output_dir / "generation_quality_summary.json")


if __name__ == "__main__":
    main()
