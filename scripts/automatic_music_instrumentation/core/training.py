from __future__ import annotations

import json
import logging
import random
from csv import DictWriter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from scripts.automatic_music_instrumentation.core.data import (
    N_CLASSES,
    PAD_LABEL,
    build_dataloader,
    build_packed_split,
    collect_split_files,
    has_packed_split,
    resolve_processed_dir,
)
from scripts.automatic_music_instrumentation.core.metrics import evaluate_model, save_confusion_matrix_plot
from scripts.automatic_music_instrumentation.core.models import build_model
from utils.io_helpers import ensure_dir, save_json
from utils.project_paths import AUTOMATIC_INSTRUMENTATION_LOG_ROOT, AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT


@dataclass
class TrainConfig:
    model_name: str = "full_transformer"
    run_name: str | None = None
    data_dir: Path | None = None
    output_root: Path = AUTOMATIC_INSTRUMENTATION_OUTPUT_ROOT
    log_root: Path = AUTOMATIC_INSTRUMENTATION_LOG_ROOT
    max_beats: int = 32
    max_seq_len: int = 1024
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    val_steps: int = 500
    num_workers: int = 4
    random_seed: int = 42
    max_train_files: int | None = None
    max_valid_files: int | None = None
    preload: bool = True
    use_packed_cache: bool = True
    build_packed_cache: bool = False
    require_cuda: bool = False
    amp: bool = False
    log_steps: int = 50


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
    run_name = config.run_name or config.model_name
    return config.output_root / "runs" / run_name


def log_dir_for(config: TrainConfig) -> Path:
    run_name = config.run_name or config.model_name
    return config.log_root / run_name


def _build_logger(logs_dir: Path) -> logging.Logger:
    logger = logging.getLogger(f"automatic_instrumentation.{logs_dir.name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(logs_dir / "training.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def _append_csv_row(path: Path, row: dict) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def save_training_curves(
    run_dir: Path,
    train_losses: list[float],
    val_losses: list[float],
    val_accs: list[float],
    val_steps: int,
) -> None:
    figures_dir = ensure_dir(run_dir / "figures")
    np.save(run_dir / "train_losses.npy", np.asarray(train_losses, dtype=float))
    np.save(run_dir / "val_losses.npy", np.asarray(val_losses, dtype=float))
    np.save(run_dir / "val_accs.npy", np.asarray(val_accs, dtype=float))
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    plt.figure(figsize=(8, 5))
    if train_losses:
        plt.plot(np.arange(len(train_losses)), train_losses, color="C0", alpha=0.25, label="Train loss")
        if len(train_losses) >= 100:
            window = 100
            moving = np.convolve(train_losses, np.ones(window), "valid") / window
            plt.plot(np.arange(len(moving)) + window / 2, moving, color="C0", label="Train loss MA")
    if val_losses:
        plt.plot(np.arange(len(val_losses)) * val_steps, val_losses, color="C1", marker="o", label="Validation loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "losses.png", dpi=160)
    plt.close()

    if val_accs:
        plt.figure(figsize=(8, 5))
        plt.plot(np.arange(len(val_accs)) * val_steps, val_accs, color="C2", marker="o", label="Validation accuracy")
        plt.xlabel("Step")
        plt.ylabel("Accuracy")
        plt.ylim(0.0, 1.0)
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figures_dir / "validation_accuracy.png", dpi=160)
        plt.close()


def _checkpoint_payload(model: nn.Module, config: TrainConfig, step: int, val_loss: float | None = None) -> dict:
    return {
        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()},
        "step": step,
        "val_loss": val_loss,
    }


def train_one_model(config: TrainConfig) -> dict:
    set_random_seed(config.random_seed)
    device = select_device(config.require_cuda)
    processed_dir = resolve_processed_dir(config.data_dir)
    run_dir = ensure_dir(run_dir_for(config))
    checkpoints_dir = ensure_dir(run_dir / "checkpoints")
    logs_dir = ensure_dir(log_dir_for(config))
    logger = _build_logger(logs_dir)
    serialized_config = {key: str(value) if isinstance(value, Path) else value for key, value in asdict(config).items()}
    save_json(run_dir / "config.json", serialized_config)
    save_json(logs_dir / "config.json", serialized_config)
    (logs_dir / "run_dir.txt").write_text(f"{run_dir}\n", encoding="utf-8")
    logger.info("Run directory: %s", run_dir)
    logger.info("Log directory: %s", logs_dir)
    logger.info("Model: %s | device: %s", config.model_name, device)
    logger.info("Collecting train/valid split files from %s", processed_dir)
    splits = collect_split_files(config.data_dir)
    train_files = splits.train[: config.max_train_files] if config.max_train_files else splits.train
    valid_files = splits.valid[: config.max_valid_files] if config.max_valid_files else splits.valid
    if not train_files or not valid_files:
        raise FileNotFoundError("Could not find train/valid .npy files. Check --data-dir.")
    logger.info("Train files: %d | valid files: %d", len(train_files), len(valid_files))

    train_uses_packed = config.use_packed_cache and config.max_train_files is None
    valid_uses_packed = config.use_packed_cache and config.max_valid_files is None
    if train_uses_packed and not has_packed_split(processed_dir, "train"):
        if config.build_packed_cache:
            logger.info("Building packed train cache. This is a one-time dataset preparation step.")
            build_packed_split(train_files, processed_dir, "train", progress=logger.info)
        else:
            train_uses_packed = False
            logger.warning("Packed train cache is missing, so training will use per-file loading.")
    if valid_uses_packed and not has_packed_split(processed_dir, "valid"):
        if config.build_packed_cache:
            logger.info("Building packed valid cache. This is a one-time dataset preparation step.")
            build_packed_split(valid_files, processed_dir, "valid", progress=logger.info)
        else:
            valid_uses_packed = False
            logger.warning("Packed valid cache is missing, so validation will use per-file loading.")

    logger.info(
        "Building dataloaders | packed_train=%s packed_valid=%s preload=%s workers=%d batch_size=%d",
        train_uses_packed,
        valid_uses_packed,
        config.preload,
        config.num_workers,
        config.batch_size,
    )
    train_loader = build_dataloader(
        train_files,
        batch_size=config.batch_size,
        max_beats=config.max_beats,
        max_seq_len=config.max_seq_len,
        augmentation=True,
        shuffle=True,
        num_workers=config.num_workers,
        preload=config.preload,
        use_packed=train_uses_packed,
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
        use_packed=valid_uses_packed,
        processed_dir=processed_dir,
        split_name="valid",
    )
    logger.info("Dataloaders ready")

    model = build_model(config.model_name, n_classes=N_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_LABEL)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=config.warmup_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")

    train_losses: list[float] = []
    val_losses: list[float] = []
    val_accs: list[float] = []
    history: list[dict] = []
    best_val_loss = float("inf")
    step = 0

    for epoch in range(config.num_epochs):
        model.train()
        logger.info("Starting epoch %d/%d", epoch + 1, config.num_epochs)
        for batch_index, (samples, labels) in enumerate(train_loader, start=1):
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            padding_mask = labels == PAD_LABEL
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=config.amp and device.type == "cuda"):
                logits = model(samples, src_key_padding_mask=padding_mask)
                loss = criterion(logits.transpose(1, 2), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_losses.append(float(loss.detach().cpu()))
            step += 1
            current_lr = float(scheduler.get_last_lr()[0])
            if config.log_steps > 0 and (step == 1 or step % config.log_steps == 0):
                train_row = {
                    "epoch": epoch,
                    "batch": batch_index,
                    "step": step,
                    "train_loss": float(train_losses[-1]),
                    "learning_rate": current_lr,
                }
                _append_csv_row(logs_dir / "train_steps.csv", train_row)
                logger.info(
                    "step=%d epoch=%d batch=%d train_loss=%.6f lr=%.8f",
                    step,
                    epoch,
                    batch_index,
                    train_row["train_loss"],
                    current_lr,
                )

            if step % config.val_steps == 0:
                val_metrics = evaluate_model(model, valid_loader, criterion, device)
                val_loss = float(val_metrics["loss"])
                val_acc = float(val_metrics["accuracy"])
                val_losses.append(val_loss)
                val_accs.append(val_acc)
                history_row = {"epoch": epoch, "step": step, "train_loss": train_losses[-1], "val_loss": val_loss, "val_acc": val_acc}
                history.append(history_row)
                _append_csv_row(logs_dir / "validation_steps.csv", history_row)
                logger.info("validation step=%d val_loss=%.6f val_acc=%.6f", step, val_loss, val_acc)
                checkpoint = _checkpoint_payload(model, config, step, val_loss)
                torch.save(checkpoint, checkpoints_dir / f"model_{step}.pt")
                torch.save(checkpoint, checkpoints_dir / "latest_model.pt")
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(checkpoint, checkpoints_dir / "best_model.pt")
                    logger.info("new best checkpoint at step=%d val_loss=%.6f", step, val_loss)
                model.train()

        save_training_curves(run_dir, train_losses, val_losses, val_accs, config.val_steps)
        save_json(run_dir / "history.json", {"history": history, "val_accs": val_accs, "best_val_loss": best_val_loss})

    final_metrics = evaluate_model(model, valid_loader, criterion, device)
    final_val_loss = float(final_metrics["loss"])
    final_checkpoint = _checkpoint_payload(model, config, step, final_val_loss)
    torch.save(final_checkpoint, checkpoints_dir / "final_model.pt")
    torch.save(final_checkpoint, checkpoints_dir / "latest_model.pt")
    if best_val_loss == float("inf"):
        best_val_loss = final_val_loss
        torch.save(final_checkpoint, checkpoints_dir / "best_model.pt")
    np.save(run_dir / "val_confusion_matrix.npy", final_metrics["confusion_matrix"])
    save_confusion_matrix_plot(final_metrics["confusion_matrix"], run_dir / "figures" / "val_confusion_matrix.png")
    save_json(
        run_dir / "final_metrics.json",
        {
            "model_name": config.model_name,
            "device": str(device),
            "log_dir": str(logs_dir),
            "train_files": len(train_files),
            "valid_files": len(valid_files),
            "best_val_loss": best_val_loss,
            "final_val_loss": final_val_loss,
            "final_val_accuracy": float(final_metrics["accuracy"]),
            "num_predictions": int(final_metrics["num_predictions"]),
        },
    )
    logger.info(
        "completed final_val_loss=%.6f final_val_accuracy=%.6f best_val_loss=%.6f",
        final_val_loss,
        float(final_metrics["accuracy"]),
        best_val_loss,
    )
    return {
        "run_dir": str(run_dir),
        "log_dir": str(logs_dir),
        "model_name": config.model_name,
        "device": str(device),
        "best_val_loss": best_val_loss,
        "final_val_loss": final_val_loss,
        "final_val_accuracy": float(final_metrics["accuracy"]),
    }


def load_checkpoint_model(checkpoint_path: Path, model_name: str | None = None, device: torch.device | None = None) -> nn.Module:
    device = device or select_device(False)
    payload = torch.load(checkpoint_path, map_location=device)
    config = payload.get("config", {})
    resolved_model_name = model_name or config.get("model_name")
    if resolved_model_name is None:
        raise ValueError("model_name is required when the checkpoint has no config.")
    model = build_model(str(resolved_model_name), n_classes=N_CLASSES).to(device)
    state = payload["model_state"] if "model_state" in payload else payload
    model.load_state_dict(state)
    model.eval()
    return model
