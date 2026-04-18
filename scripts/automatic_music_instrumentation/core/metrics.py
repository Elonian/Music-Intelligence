from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from scripts.automatic_music_instrumentation.core.data import INSTRUMENT_LABELS, N_CLASSES, PAD_LABEL


def flatten_masked_predictions(
    logits: torch.Tensor,
    labels: torch.Tensor,
    padding_label: int = PAD_LABEL,
) -> tuple[np.ndarray, np.ndarray]:
    mask = labels != padding_label
    predictions = torch.argmax(logits, dim=-1)
    return labels[mask].detach().cpu().numpy(), predictions[mask].detach().cpu().numpy()


def confusion_matrix_numpy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for truth, pred in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            matrix[truth, pred] += 1
    return matrix


def normalize_confusion_matrix(matrix: np.ndarray) -> np.ndarray:
    row_sums = matrix.sum(axis=1, keepdims=True)
    return np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    padding_label: int = PAD_LABEL,
) -> dict:
    model.eval()
    losses: list[float] = []
    y_true_parts: list[np.ndarray] = []
    y_pred_parts: list[np.ndarray] = []
    for samples, labels in loader:
        samples = samples.to(device)
        labels = labels.to(device)
        padding_mask = labels == padding_label
        logits = model(samples, src_key_padding_mask=padding_mask)
        loss = criterion(logits.transpose(1, 2), labels)
        losses.append(float(loss.detach().cpu()))
        y_true, y_pred = flatten_masked_predictions(logits, labels, padding_label=padding_label)
        y_true_parts.append(y_true)
        y_pred_parts.append(y_pred)
    y_true_all = np.concatenate(y_true_parts) if y_true_parts else np.array([], dtype=int)
    y_pred_all = np.concatenate(y_pred_parts) if y_pred_parts else np.array([], dtype=int)
    accuracy = float(np.mean(y_true_all == y_pred_all)) if len(y_true_all) else 0.0
    matrix = confusion_matrix_numpy(y_true_all, y_pred_all)
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": accuracy,
        "num_predictions": int(len(y_true_all)),
        "confusion_matrix": matrix,
        "normalized_confusion_matrix": normalize_confusion_matrix(matrix),
    }


def save_confusion_matrix_plot(matrix: np.ndarray, path: Path, normalized: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return

    display = normalize_confusion_matrix(matrix) if normalized else matrix
    plt.figure(figsize=(7, 6))
    plt.imshow(display, cmap="Blues")
    plt.colorbar()
    plt.xlabel("Prediction")
    plt.ylabel("Ground truth")
    plt.xticks(range(N_CLASSES), INSTRUMENT_LABELS, rotation=30, ha="right")
    plt.yticks(range(N_CLASSES), INSTRUMENT_LABELS)
    for row in range(N_CLASSES):
        for col in range(N_CLASSES):
            value = display[row, col]
            text = f"{value:.2f}" if normalized else str(int(value))
            plt.text(col, row, text, ha="center", va="center", color="black", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
