from __future__ import annotations

import csv
import json
import logging
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import LinearLR

from scripts.multitrack_generation.data import build_dataloader, collect_split_files, resolve_processed_dir
from scripts.multitrack_generation.metrics import evaluate_model, save_confusion_matrices, sequence_loss, target_mask_from_lengths
from scripts.multitrack_generation.models import build_model, model_parameter_count
from utils.io_helpers import ensure_dir, save_json
from utils.project_paths import MULTITRACK_GENERATION_LOG_ROOT, MULTITRACK_GENERATION_OUTPUT_ROOT


@dataclass
class TrainConfig:
    model_name: str = "full"
    run_name: str | None = None
    data_dir: Path | None = None
    output_root: Path = MULTITRACK_GENERATION_OUTPUT_ROOT
    log_root: Path = MULTITRACK_GENERATION_LOG_ROOT
    positional_mode: str = "sequence"
    max_beats: int = 32
    max_seq_len: int = 1024
    batch_size: int = 16
    num_epochs: int = 50
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    val_steps: int = 1000
    num_workers: int = 0
    random_seed: int = 42
    max_train_files: int | None = None
    max_valid_files: int | None = None
    preload: bool = False
    use_packed_cache: bool = True
    require_cuda: bool = False
    amp: bool = False
    log_steps: int = 25
    max_val_batches: int | None = None


def set_random_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(require_cuda: bool = False) -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if require_cuda:
        raise RuntimeError("CUDA was required, but torch.cuda.is_available() is false.")
    return torch.device("cpu")


def run_dir_for(config: TrainConfig) -> Path:
    return config.output_root / "runs" / (config.run_name or config.model_name)


def log_dir_for(config: TrainConfig) -> Path:
    return config.log_root / (config.run_name or config.model_name)


def _serializable_config(config: TrainConfig) -> dict:
    return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}


def _build_logger(log_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"multitrack_generation.{log_dir.name}")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _append_csv(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _checkpoint_payload(model: torch.nn.Module, config: TrainConfig, step: int, val_loss: float | None = None) -> dict:
    return {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": _serializable_config(config),
        "step": int(step),
        "val_loss": None if val_loss is None else float(val_loss),
    }


def train_model(config: TrainConfig) -> dict:
    set_random_seed(config.random_seed)
    device = select_device(config.require_cuda)
    processed_dir = resolve_processed_dir(config.data_dir)
    run_dir = ensure_dir(run_dir_for(config))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    metrics_dir = ensure_dir(run_dir / "metrics")
    log_dir = ensure_dir(log_dir_for(config))
    logger = _build_logger(log_dir)
    save_json(run_dir / "config.json", _serializable_config(config))
    save_json(log_dir / "config.json", _serializable_config(config))
    (log_dir / "run_dir.txt").write_text(f"{run_dir}\n", encoding="utf-8")

    splits = collect_split_files(processed_dir)
    train_files = splits.train[: config.max_train_files] if config.max_train_files else splits.train
    valid_files = splits.valid[: config.max_valid_files] if config.max_valid_files else splits.valid
    if not train_files:
        raise FileNotFoundError(f"no train .npy files found under {processed_dir / 'train'}")
    if not valid_files:
        raise FileNotFoundError(f"no valid .npy files found under {processed_dir / 'valid'}")

    logger.info("run_dir=%s", run_dir)
    logger.info("processed_dir=%s", processed_dir)
    logger.info("train_files=%d valid_files=%d", len(train_files), len(valid_files))
    logger.info("device=%s model=%s positional_mode=%s", device, config.model_name, config.positional_mode)

    train_loader = build_dataloader(
        train_files,
        batch_size=config.batch_size,
        max_beats=config.max_beats,
        max_seq_len=config.max_seq_len,
        augmentation=True,
        shuffle=True,
        num_workers=config.num_workers,
        preload=config.preload,
        use_packed=config.use_packed_cache and config.max_train_files is None,
        processed_dir=processed_dir,
        split_name="train",
    )
    valid_loader = build_dataloader(
        valid_files,
        batch_size=config.batch_size,
        max_beats=config.max_beats,
        max_seq_len=config.max_seq_len,
        augmentation=False,
        shuffle=False,
        num_workers=config.num_workers,
        preload=config.preload,
        use_packed=config.use_packed_cache and config.max_valid_files is None,
        processed_dir=processed_dir,
        split_name="valid",
    )

    model = build_model(config.model_name, positional_mode=config.positional_mode).to(device)
    logger.info("parameters=%d", model_parameter_count(model))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = (
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps)
        if config.warmup_steps > 0
        else None
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    history: list[dict] = []
    train_losses: list[float] = []
    best_val_loss = float("inf")
    step = 0

    for epoch in range(config.num_epochs):
        logger.info("epoch=%d/%d", epoch + 1, config.num_epochs)
        model.train()
        for batch_index, (batch, lengths) in enumerate(train_loader, start=1):
            batch = batch.to(device, non_blocking=True)
            lengths = lengths.to(device, non_blocking=True)
            if batch.shape[1] < 2:
                continue
            sources = batch[:, :-1, :]
            targets = batch[:, 1:, :]
            mask = target_mask_from_lengths(lengths, targets.shape[1], device)
            if not bool(mask.any()):
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                outputs = model(sources)
                loss, per_field_train = sequence_loss(outputs, targets, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            if scheduler is not None:
                scheduler.step()

            step += 1
            train_loss = float(loss.detach().cpu())
            train_losses.append(train_loss)
            if config.log_steps > 0 and (step == 1 or step % config.log_steps == 0):
                row = {
                    "epoch": epoch + 1,
                    "batch": batch_index,
                    "step": step,
                    "train_loss": train_loss,
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                    **{f"train_loss_{key}": value for key, value in per_field_train.items()},
                }
                _append_csv(log_dir / "train_steps.csv", row)
                logger.info("step=%d train_loss=%.6f lr=%.8f", step, train_loss, float(optimizer.param_groups[0]["lr"]))

            if config.val_steps > 0 and step % config.val_steps == 0:
                val_metrics = evaluate_model(model, valid_loader, device, max_batches=config.max_val_batches, collect_confusion=False)
                val_loss = float(val_metrics["loss"])
                row = {
                    "epoch": epoch + 1,
                    "step": step,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "val_accuracy": float(val_metrics["accuracy"]),
                    **{f"val_loss_{key}": value for key, value in val_metrics["per_field_loss"].items()},
                    **{f"val_acc_{key}": value for key, value in val_metrics["per_field_accuracy"].items()},
                }
                history.append(row)
                _append_csv(log_dir / "validation_steps.csv", row)
                torch.save(_checkpoint_payload(model, config, step, val_loss), checkpoints_dir / "latest_model.pt")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(_checkpoint_payload(model, config, step, val_loss), checkpoints_dir / "best_model.pt")
                    logger.info("new_best step=%d val_loss=%.6f val_accuracy=%.6f", step, val_loss, float(val_metrics["accuracy"]))
                else:
                    logger.info("validation step=%d val_loss=%.6f val_accuracy=%.6f", step, val_loss, float(val_metrics["accuracy"]))
                model.train()

    final_metrics = evaluate_model(model, valid_loader, device, max_batches=config.max_val_batches, collect_confusion=True)
    final_val_loss = float(final_metrics["loss"])
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        torch.save(_checkpoint_payload(model, config, step, final_val_loss), checkpoints_dir / "best_model.pt")
    torch.save(_checkpoint_payload(model, config, step, final_val_loss), checkpoints_dir / "final_model.pt")
    torch.save(_checkpoint_payload(model, config, step, final_val_loss), checkpoints_dir / "latest_model.pt")

    save_confusion_matrices(final_metrics["confusion_matrices"], metrics_dir)
    summary = {
        "run_dir": str(run_dir),
        "log_dir": str(log_dir),
        "processed_dir": str(processed_dir),
        "model_name": config.model_name,
        "positional_mode": config.positional_mode,
        "device": str(device),
        "train_files": len(train_files),
        "valid_files": len(valid_files),
        "steps": int(step),
        "best_val_loss": float(best_val_loss),
        "final_val_loss": final_val_loss,
        "final_val_accuracy": float(final_metrics["accuracy"]),
        "per_field_loss": final_metrics["per_field_loss"],
        "per_field_accuracy": final_metrics["per_field_accuracy"],
        "active_tokens": int(final_metrics["active_tokens"]),
    }
    save_json(run_dir / "history.json", {"history": history, "train_losses": train_losses})
    save_json(run_dir / "final_metrics.json", summary)
    logger.info("complete steps=%d final_val_loss=%.6f final_val_accuracy=%.6f", step, final_val_loss, float(final_metrics["accuracy"]))
    return summary


def load_history(run_dir: Path) -> dict:
    return json.loads((run_dir / "history.json").read_text(encoding="utf-8"))
