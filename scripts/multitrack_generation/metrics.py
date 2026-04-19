from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from scripts.multitrack_generation.constants import FIELD_NAMES, FIELD_SPECS


def target_mask_from_lengths(lengths: torch.Tensor, target_seq_len: int, device: torch.device) -> torch.Tensor:
    """Mask valid next-token positions after shifting ``x[:, :-1] -> x[:, 1:]``."""
    positions = torch.arange(target_seq_len, device=device).unsqueeze(0)
    valid_lengths = torch.clamp(lengths.to(device) - 1, min=0).unsqueeze(1)
    return positions < valid_lengths


def sequence_loss(
    outputs: list[torch.Tensor],
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    total = targets.new_tensor(0.0, dtype=torch.float32)
    per_field: dict[str, float] = {}
    active_count = int(mask.sum().item())
    if active_count == 0:
        return total, {name: 0.0 for name in FIELD_NAMES}
    for field_index, field in enumerate(FIELD_SPECS):
        logits = outputs[field_index][mask]
        field_targets = targets[:, :, field_index][mask].clamp(0, field.vocab_size - 1)
        field_loss = F.cross_entropy(logits, field_targets, reduction="sum")
        total = total + field_loss
        per_field[field.name] = float((field_loss / active_count).detach().cpu())
    return total / active_count, per_field


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
    collect_confusion: bool = True,
) -> dict:
    model.eval()
    loss_sums = np.zeros(len(FIELD_SPECS), dtype=np.float64)
    correct = np.zeros(len(FIELD_SPECS), dtype=np.int64)
    active_total = 0
    confusion = [
        np.zeros((field.vocab_size, field.vocab_size), dtype=np.int64)
        for field in FIELD_SPECS
    ]

    for batch_index, (batch, lengths) in enumerate(dataloader, start=1):
        batch = batch.to(device, non_blocking=True)
        lengths = lengths.to(device, non_blocking=True)
        if batch.shape[1] < 2:
            continue
        sources = batch[:, :-1, :]
        targets = batch[:, 1:, :]
        mask = target_mask_from_lengths(lengths, targets.shape[1], device)
        active_count = int(mask.sum().item())
        if active_count == 0:
            continue

        outputs = model(sources)
        active_total += active_count
        for field_index, field in enumerate(FIELD_SPECS):
            logits = outputs[field_index]
            field_targets = targets[:, :, field_index].clamp(0, field.vocab_size - 1)
            loss = F.cross_entropy(logits[mask], field_targets[mask], reduction="sum")
            predictions = logits.argmax(dim=-1)
            loss_sums[field_index] += float(loss.detach().cpu())
            correct[field_index] += int((predictions[mask] == field_targets[mask]).sum().detach().cpu())
            if collect_confusion:
                truth_np = field_targets[mask].detach().cpu().numpy().astype(np.int64)
                pred_np = predictions[mask].detach().cpu().numpy().astype(np.int64)
                np.add.at(confusion[field_index], (truth_np, pred_np), 1)

        if max_batches is not None and batch_index >= max_batches:
            break

    denom = max(active_total, 1)
    per_field_loss = {
        field.name: float(loss_sums[index] / denom)
        for index, field in enumerate(FIELD_SPECS)
    }
    per_field_accuracy = {
        field.name: float(correct[index] / denom)
        for index, field in enumerate(FIELD_SPECS)
    }
    return {
        "loss": float(loss_sums.sum() / denom),
        "per_field_loss": per_field_loss,
        "accuracy": float(correct.sum() / max(denom * len(FIELD_SPECS), 1)),
        "per_field_accuracy": per_field_accuracy,
        "active_tokens": int(active_total),
        "confusion_matrices": confusion,
    }


def save_confusion_matrices(confusion_matrices: list[np.ndarray], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for field, matrix in zip(FIELD_SPECS, confusion_matrices):
        np.save(output_dir / f"{field.name}_confusion.npy", matrix)


def load_confusion_matrices(input_dir: Path) -> dict[str, np.ndarray]:
    matrices: dict[str, np.ndarray] = {}
    for field in FIELD_SPECS:
        filename = input_dir / f"{field.name}_confusion.npy"
        if filename.exists():
            matrices[field.name] = np.load(filename)
    return matrices
