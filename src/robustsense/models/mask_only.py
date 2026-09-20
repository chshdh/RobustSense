"""Availability-only diagnostic baseline for Phase V2-2."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn

from robustsense.constants import MODALITIES
from robustsense.training.torch_losses import masked_weighted_bce_with_logits


class MaskOnlyModel(nn.Module):
    """A diagnostic model whose API cannot receive sensor feature values."""

    model_id = "mask-only"

    def __init__(self, n_labels: int, hidden_dim: int = 16):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(len(MODALITIES), hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, n_labels),
        )

    def forward(self, availability: torch.Tensor) -> torch.Tensor:
        if availability.ndim != 2 or availability.shape[1] != len(MODALITIES):
            raise ValueError("Mask-only input must have exactly six availability bits")
        return self.network(availability.to(torch.float32))


def permute_availability_within_users(
    availability: np.ndarray, user_ids: np.ndarray, *, seed: int
) -> np.ndarray:
    values = np.asarray(availability)
    users = np.asarray(user_ids).astype(str)
    if values.ndim != 2 or values.shape[1] != len(MODALITIES) or len(values) != len(users):
        raise ValueError("Availability and user IDs must align with six modalities")
    result = values.copy()
    rng = np.random.default_rng(seed)
    for user in sorted(set(users.tolist())):
        indices = np.flatnonzero(users == user)
        result[indices] = values[rng.permutation(indices)]
    return result


def fit_mask_only_model(
    train_availability: np.ndarray,
    train_targets: np.ndarray,
    train_target_mask: np.ndarray,
    *,
    n_labels: int,
    epochs: int = 50,
    learning_rate: float = 1.0e-2,
    seed: int = 13,
) -> tuple[MaskOnlyModel, list[dict[str, Any]]]:
    """Fit the diagnostic on an already user-separated training fold."""
    if epochs <= 0:
        raise ValueError("epochs must be positive")
    torch.manual_seed(seed)
    model = MaskOnlyModel(n_labels=n_labels)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    availability = torch.as_tensor(train_availability, dtype=torch.float32)
    targets = torch.as_tensor(train_targets, dtype=torch.float32)
    target_mask = torch.as_tensor(train_target_mask, dtype=torch.bool)
    history: list[dict[str, Any]] = []
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(availability)
        loss = masked_weighted_bce_with_logits(logits, targets, target_mask)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch + 1, "train_loss": float(loss.detach())})
    return model, history
