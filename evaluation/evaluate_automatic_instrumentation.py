#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.automatic_music_instrumentation.core.data import (  # noqa: E402
    INSTRUMENT_LABELS,
    N_CLASSES,
    PAD_LABEL,
    build_dataloader,
    collect_split_files,
)
from scripts.automatic_music_instrumentation.core.metrics import (  # noqa: E402
    normalize_confusion_matrix,
    save_confusion_matrix_plot,
)
from scripts.automatic_music_instrumentation.core.models import MODEL_SPECS  # noqa: E402
from scripts.automatic_music_instrumentation.core.pitch_zones import (  # noqa: E402
    pitch_zone_predict_events,
)
from scripts.automatic_music_instrumentation.core.training import (  # noqa: E402
    load_checkpoint_model,
    select_device,
)
from utils.io_helpers import ensure_dir, save_json, write_csv_rows  # noqa: E402
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT  # noqa: E402


DEFAULT_OUTPUT_DIR = AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT / "evaluation"


def _safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator, dtype=float), where=denominator != 0)


def confusion_matrix_from_labels(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[truth, pred] += 1
    return matrix


def classification_report_from_matrix(matrix: np.ndarray) -> dict:
    matrix = np.asarray(matrix, dtype=np.int64)
    true_positive = np.diag(matrix).astype(float)
    support = matrix.sum(axis=1).astype(float)
    predicted = matrix.sum(axis=0).astype(float)
    precision = _safe_divide(true_positive, predicted)
    recall = _safe_divide(true_positive, support)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    total = float(matrix.sum())
    accuracy = float(true_positive.sum() / total) if total else 0.0
    weighted_f1 = float(np.sum(f1 * support) / total) if total else 0.0
    weighted_precision = float(np.sum(precision * support) / total) if total else 0.0
    weighted_recall = float(np.sum(recall * support) / total) if total else 0.0

    per_class = []
    for index, label in enumerate(INSTRUMENT_LABELS):
        per_class.append(
            {
                "label": label,
                "support": int(support[index]),
                "predicted": int(predicted[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
        )

    return {
        "accuracy": accuracy,
        "balanced_accuracy": float(np.mean(recall)) if len(recall) else 0.0,
        "macro_precision": float(np.mean(precision)) if len(precision) else 0.0,
        "macro_recall": float(np.mean(recall)) if len(recall) else 0.0,
        "macro_f1": float(np.mean(f1)) if len(f1) else 0.0,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
        "per_class": per_class,
    }


def _checkpoint_run_name(checkpoint_path: Path) -> str:
    if checkpoint_path.parent.name == "checkpoints":
        return checkpoint_path.parent.parent.name
    return checkpoint_path.stem


def _checkpoint_model_name(checkpoint_path: Path, override: str | None = None) -> str | None:
    if override:
        return override
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    config = payload.get("config", {}) if isinstance(payload, dict) else {}
    model_name = config.get("model_name")
    return str(model_name) if model_name else None


def _save_matrix_outputs(output_dir: Path, prefix: str, matrix: np.ndarray, write_plots: bool) -> None:
    np.save(output_dir / f"{prefix}_confusion_matrix.npy", matrix)
    np.save(output_dir / f"{prefix}_confusion_matrix_normalized.npy", normalize_confusion_matrix(matrix))
    if write_plots:
        save_confusion_matrix_plot(matrix, output_dir / f"{prefix}_confusion_matrix.png")
        save_confusion_matrix_plot(matrix, output_dir / f"{prefix}_confusion_matrix_normalized.png", normalized=True)


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint_path: Path,
    files: list[Path],
    split: str,
    output_dir: Path,
    model_name_override: str | None,
    batch_size: int,
    num_workers: int,
    require_cuda: bool,
    max_seq_len: int,
    write_plots: bool,
) -> dict:
    device = select_device(require_cuda)
    model_name = _checkpoint_model_name(checkpoint_path, model_name_override)
    model = load_checkpoint_model(checkpoint_path, model_name=model_name, device=device)
    resolved_model_name = model_name or "checkpoint_model"
    run_name = _checkpoint_run_name(checkpoint_path)
    prefix = f"{run_name}_{split}"

    loader = build_dataloader(
        files,
        batch_size=batch_size,
        max_seq_len=max_seq_len,
        augmentation=False,
        shuffle=False,
        num_workers=num_workers,
        preload=False,
    )
    mean_loss = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    summed_loss = nn.CrossEntropyLoss(ignore_index=PAD_LABEL, reduction="sum")
    sequence_loss_total = 0.0
    note_loss_total = 0.0
    sequence_count = 0
    note_count = 0
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []

    if device.type == "cuda":
        torch.cuda.synchronize()
    started_at = time.perf_counter()
    for samples, labels in loader:
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        padding_mask = labels == PAD_LABEL
        logits = model(samples, src_key_padding_mask=padding_mask)
        sequence_loss_total += float(mean_loss(logits.transpose(1, 2), labels).detach().cpu()) * len(samples)
        note_loss_total += float(summed_loss(logits.transpose(1, 2), labels).detach().cpu())
        sequence_count += int(len(samples))
        note_count += int(torch.count_nonzero(~padding_mask).detach().cpu())
        predictions = torch.argmax(logits, dim=-1)
        y_true_parts.append(labels[~padding_mask].detach().cpu().numpy())
        y_pred_parts.append(predictions[~padding_mask].detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started_at

    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.int64)
    matrix = confusion_matrix_from_labels(y_true, y_pred)
    report = classification_report_from_matrix(matrix)
    _save_matrix_outputs(output_dir, prefix, matrix, write_plots)

    metrics = {
        "name": run_name,
        "model": resolved_model_name,
        "split": split,
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "files": len(files),
        "sequences": sequence_count,
        "notes": note_count,
        "loss_sequence_weighted": sequence_loss_total / sequence_count if sequence_count else 0.0,
        "loss_note_weighted": note_loss_total / note_count if note_count else 0.0,
        "elapsed_seconds": elapsed,
        "sequences_per_second": sequence_count / elapsed if elapsed > 0 else 0.0,
        "notes_per_second": note_count / elapsed if elapsed > 0 else 0.0,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalize_confusion_matrix(matrix).tolist(),
        **report,
    }
    save_json(output_dir / f"{prefix}_metrics.json", metrics)
    return metrics


def evaluate_pitch_zones(
    files: list[Path],
    split: str,
    output_dir: Path,
    write_plots: bool,
) -> dict:
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    started_at = time.perf_counter()
    for file_path in files:
        array = np.load(file_path)
        if array.ndim != 2 or array.shape[1] != 4 or array.size == 0:
            continue
        y_true_parts.append(array[:, 3].astype(np.int64))
        y_pred_parts.append(pitch_zone_predict_events(array))
    elapsed = time.perf_counter() - started_at
    y_true = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=np.int64)
    y_pred = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=np.int64)
    matrix = confusion_matrix_from_labels(y_true, y_pred)
    report = classification_report_from_matrix(matrix)
    prefix = f"pitch_zones_{split}"
    _save_matrix_outputs(output_dir, prefix, matrix, write_plots)
    metrics = {
        "name": "pitch_zones",
        "model": "pitch_zones",
        "split": split,
        "checkpoint": "",
        "device": "cpu",
        "files": len(files),
        "sequences": len(files),
        "notes": int(len(y_true)),
        "loss_sequence_weighted": None,
        "loss_note_weighted": None,
        "elapsed_seconds": elapsed,
        "sequences_per_second": len(files) / elapsed if elapsed > 0 else 0.0,
        "notes_per_second": len(y_true) / elapsed if elapsed > 0 else 0.0,
        "confusion_matrix": matrix.tolist(),
        "normalized_confusion_matrix": normalize_confusion_matrix(matrix).tolist(),
        **report,
    }
    save_json(output_dir / f"{prefix}_metrics.json", metrics)
    return metrics


def resolve_checkpoints(checkpoints: list[Path] | None, run_dirs: list[Path] | None) -> list[Path]:
    resolved: list[Path] = []
    for checkpoint in checkpoints or []:
        resolved.append(checkpoint.expanduser().resolve())
    for run_dir in run_dirs or []:
        candidate = run_dir.expanduser().resolve() / "checkpoints" / "best_model.pt"
        if candidate.exists():
            resolved.append(candidate)
    return resolved


def write_summary(output_dir: Path, metrics: list[dict]) -> None:
    save_json(output_dir / "automatic_instrumentation_evaluation_summary.json", {"evaluations": metrics})
    rows = []
    for item in metrics:
        rows.append(
            {
                "name": item["name"],
                "model": item["model"],
                "split": item["split"],
                "files": item["files"],
                "notes": item["notes"],
                "loss_sequence_weighted": "" if item["loss_sequence_weighted"] is None else f"{item['loss_sequence_weighted']:.6f}",
                "loss_note_weighted": "" if item["loss_note_weighted"] is None else f"{item['loss_note_weighted']:.6f}",
                "accuracy": f"{item['accuracy']:.6f}",
                "balanced_accuracy": f"{item['balanced_accuracy']:.6f}",
                "macro_f1": f"{item['macro_f1']:.6f}",
                "weighted_f1": f"{item['weighted_f1']:.6f}",
                "notes_per_second": f"{item['notes_per_second']:.2f}",
            }
        )
    write_csv_rows(output_dir / "automatic_instrumentation_evaluation_summary.csv", rows)
    lines = [
        "| Name | Model | Split | Notes | Loss | Accuracy | Balanced Accuracy | Macro F1 | Weighted F1 | Notes/s |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {name} | {model} | {split} | {notes} | {loss_sequence_weighted} | {accuracy} | "
            "{balanced_accuracy} | {macro_f1} | {weighted_f1} | {notes_per_second} |".format(**row)
        )
    (output_dir / "automatic_instrumentation_evaluation_summary.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate automatic instrumentation models and rule baselines.")
    parser.add_argument("--checkpoint", type=Path, nargs="*", default=None, help="One or more model checkpoints.")
    parser.add_argument("--run-dir", type=Path, nargs="*", default=None, help="Run dirs containing checkpoints/best_model.pt.")
    parser.add_argument("--model", choices=sorted(MODEL_SPECS), default=None, help="Override model name for checkpoints without config.")
    parser.add_argument("--include-pitch-zones", action="store_true", help="Include the fixed pitch-zone rule baseline.")
    parser.add_argument("--data-dir", type=Path, default=None, help="Processed data dir or automatic_music_instrumentation dir.")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(args.output_dir)
    split_files = collect_split_files(args.data_dir)
    files = getattr(split_files, args.split)
    if args.max_files is not None:
        files = files[: args.max_files]

    metrics: list[dict] = []
    if args.include_pitch_zones:
        metrics.append(evaluate_pitch_zones(files, args.split, output_dir, write_plots=not args.no_plots))

    checkpoints = resolve_checkpoints(args.checkpoint, args.run_dir)
    for checkpoint_path in checkpoints:
        metrics.append(
            evaluate_checkpoint(
                checkpoint_path=checkpoint_path,
                files=files,
                split=args.split,
                output_dir=output_dir,
                model_name_override=args.model,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                require_cuda=args.require_cuda,
                max_seq_len=args.max_seq_len,
                write_plots=not args.no_plots,
            )
        )

    if not metrics:
        raise SystemExit("No evaluations requested. Pass --include-pitch-zones and/or --checkpoint/--run-dir.")
    write_summary(output_dir, metrics)
    print(f"[Automatic Instrumentation Eval] Wrote {output_dir / 'automatic_instrumentation_evaluation_summary.json'}")
    print(f"[Automatic Instrumentation Eval] Wrote {output_dir / 'automatic_instrumentation_evaluation_summary.csv'}")
    print(json.dumps([{key: item[key] for key in ('name', 'model', 'accuracy', 'macro_f1')} for item in metrics], indent=2))


if __name__ == "__main__":
    main()
