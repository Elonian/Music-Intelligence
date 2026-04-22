from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader

from scripts.automatic_playlist_continuation.data import split_query_targets
from scripts.automatic_playlist_continuation.models import WRMF, WRMFDataset
from utils.playlist_continuation import PlaylistDict


@dataclass(frozen=True)
class TrainResult:
    model: WRMF
    history: list[dict]
    config: dict

    def summary(self) -> dict:
        return {
            "history": self.history,
            "config": self.config,
            "final_loss": self.history[-1]["loss"] if self.history else None,
        }


def _as_numpy_matrix(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _cosine_scores(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float32).reshape(-1)
    matrix = np.asarray(matrix, dtype=np.float32)
    query_norm = max(float(np.linalg.norm(query)), 1e-12)
    matrix_norm = np.maximum(np.linalg.norm(matrix, axis=1), 1e-12)
    return matrix @ query / (matrix_norm * query_norm)


def get_playlist_embedding(model: WRMF, playlist: list[str], tid_to_idx: dict[str, int]) -> torch.Tensor:
    item_ids = [tid_to_idx[tid] for tid in playlist if tid in tid_to_idx]
    if not item_ids:
        raise ValueError("Playlist has no known tracks in tid_to_idx.")
    indices = torch.as_tensor(item_ids, dtype=torch.long, device=model.item_factors.weight.device)
    return model.item_factors(indices).mean(dim=0)


def generate_recommendations(
    model: WRMF,
    playlist: list[str],
    all_item_embeddings: torch.Tensor,
    idx_to_tid: dict[int, str],
    tid_to_idx: dict[str, int],
    N: int = 10,
    *,
    n: int | None = None,
) -> tuple[list[float], list[str]]:
    if n is not None:
        N = n
    playlist_embedding = get_playlist_embedding(model, playlist, tid_to_idx)
    playlist_vector = _as_numpy_matrix(playlist_embedding)
    item_matrix = _as_numpy_matrix(all_item_embeddings)
    similarities = _cosine_scores(playlist_vector, item_matrix)
    playlist_item_ids = {tid_to_idx[tid] for tid in playlist if tid in tid_to_idx}
    sorted_indices = np.argsort(-similarities)
    recommendation_indices = [int(index) for index in sorted_indices if int(index) not in playlist_item_ids][:N]
    return [float(similarities[index]) for index in recommendation_indices], [idx_to_tid[index] for index in recommendation_indices]


def make_cf_rankings(
    model: WRMF,
    test_playlists: PlaylistDict,
    all_item_embeddings: torch.Tensor,
    idx_to_tid: dict[int, str],
    tid_to_idx: dict[str, int],
    query_size: int = 2,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    rankings: dict[str, list[str]] = {}
    targets: dict[str, list[str]] = {}
    known_tids = set(tid_to_idx)
    max_items = len(idx_to_tid)
    for playlist_id, playlist in test_playlists.items():
        query, target = split_query_targets(playlist, query_size=query_size, known_tids=known_tids)
        if not query:
            rankings[playlist_id] = []
            targets[playlist_id] = target
            continue
        _scores, recommendations = generate_recommendations(
            model,
            query,
            all_item_embeddings,
            idx_to_tid,
            tid_to_idx,
            N=max_items,
        )
        rankings[playlist_id] = recommendations
        targets[playlist_id] = target
    return rankings, targets


def train_wrmf(
    data: list[tuple[int, int, int]],
    num_users: int,
    num_items: int,
    num_factors: int = 16,
    alpha: float = 40.0,
    lambda_reg: float = 0.1,
    learning_rate: float = 0.01,
    batch_size: int = 1024,
    epochs: int = 10,
    seed: int = 42,
    device: torch.device | str | None = None,
    num_workers: int = 0,
) -> TrainResult:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    resolved_device = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WRMF(
        num_users=num_users,
        num_items=num_items,
        num_factors=num_factors,
        alpha=alpha,
        lambda_reg=lambda_reg,
    ).to(resolved_device)
    loader = DataLoader(
        WRMFDataset(data),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history: list[dict] = []
    for epoch in range(epochs):
        model.train()
        losses: list[float] = []
        for user, item, label in loader:
            user = user.to(resolved_device)
            item = item.to(resolved_device)
            label = label.to(resolved_device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.compute_loss(user, item, label)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)) if losses else 0.0})
    config = {
        "num_users": num_users,
        "num_items": num_items,
        "num_factors": num_factors,
        "alpha": alpha,
        "lambda_reg": lambda_reg,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "epochs": epochs,
        "seed": seed,
        "device": str(resolved_device),
    }
    return TrainResult(model=model, history=history, config=config)


def train_result_payload(result: TrainResult) -> dict:
    payload = result.summary()
    payload["config"] = dict(payload["config"])
    return payload
