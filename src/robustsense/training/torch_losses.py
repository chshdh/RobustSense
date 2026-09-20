"""Torch losses that preserve unknown-label masks."""

from __future__ import annotations

import torch
from torch.nn import functional as F


def masked_weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor,
    pos_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    if logits.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("logits, targets, and target_mask must have identical shapes")
    known = target_mask.bool()
    if not torch.any(known):
        raise ValueError("Cannot compute loss when every target is unknown")
    clean_targets = torch.nan_to_num(targets, nan=0.0)
    per_entry = F.binary_cross_entropy_with_logits(
        logits, clean_targets, pos_weight=pos_weight, reduction="none"
    )
    return per_entry[known].mean()


def training_pos_weight(
    targets: torch.Tensor, target_mask: torch.Tensor, cap: float = 20.0
) -> torch.Tensor:
    known = target_mask.bool()
    positives = ((targets == 1) & known).sum(dim=0).to(torch.float32)
    negatives = ((targets == 0) & known).sum(dim=0).to(torch.float32)
    if torch.any(positives == 0):
        missing = torch.nonzero(positives == 0).flatten().tolist()
        raise ValueError(f"Training split has labels without positives: {missing}")
    return (negatives / positives).clamp(min=0.01, max=cap)
