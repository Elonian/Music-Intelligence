from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import Dataset


class WRMFDataset(Dataset):
    def __init__(self, data: list[tuple[int, int, int]]) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        playlist_idx, track_idx, label = self.data[index]
        return (
            torch.tensor(playlist_idx, dtype=torch.long),
            torch.tensor(track_idx, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32),
        )


class WRMF(nn.Module):
    """Weighted regularized matrix factorization with sigmoid preference scores."""

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_factors: int = 16,
        alpha: float = 40.0,
        lambda_reg: float = 0.1,
    ) -> None:
        super().__init__()
        self.user_factors = nn.Embedding(num_users, num_factors)
        self.item_factors = nn.Embedding(num_items, num_factors)
        self.alpha = float(alpha)
        self.lambda_reg = float(lambda_reg)
        nn.init.normal_(self.user_factors.weight, std=0.01)
        nn.init.normal_(self.item_factors.weight, std=0.01)

    def forward(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return (self.user_factors(user) * self.item_factors(item)).sum(dim=1)

    def preference_scores(self, user: torch.Tensor, item: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(user, item))

    def compute_loss(self, user: torch.Tensor, item: torch.Tensor, feedback: torch.Tensor) -> torch.Tensor:
        feedback = feedback.float()
        prediction = self.preference_scores(user, item)
        confidence = 1.0 + self.alpha * feedback
        reconstruction = (confidence * (feedback - prediction).pow(2)).mean()
        regularization = self.lambda_reg * (
            self.user_factors(user).pow(2).sum(dim=1).mean()
            + self.item_factors(item).pow(2).sum(dim=1).mean()
        )
        return reconstruction + regularization
