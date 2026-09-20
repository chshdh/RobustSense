"""Auxiliary losses for quality-aware robust fusion."""

from __future__ import annotations

import torch


def reliability_mse(
    reliability: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if reliability.shape != target.shape:
        raise ValueError("Reliability prediction and target shapes differ")
    return torch.mean((reliability - target.to(reliability.dtype)) ** 2)


def consistency_mse(
    clean_logits: torch.Tensor,
    corrupted_logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    if clean_logits.shape != corrupted_logits.shape or clean_logits.shape != target_mask.shape:
        raise ValueError("Consistency tensors must have matching shapes")
    known = target_mask.to(torch.bool)
    if not known.any():
        return corrupted_logits.sum() * 0.0
    clean_probability = torch.sigmoid(clean_logits.detach())
    corrupted_probability = torch.sigmoid(corrupted_logits)
    return torch.mean((clean_probability[known] - corrupted_probability[known]) ** 2)


def reliability_ranking_loss(
    cleaner_reliability: torch.Tensor,
    more_severe_reliability: torch.Tensor,
    comparison_mask: torch.Tensor,
    margin: float = 0.1,
) -> torch.Tensor:
    """Encourage the same sample/modality to rank cleaner above more severe."""
    if cleaner_reliability.shape != more_severe_reliability.shape:
        raise ValueError("Cleaner and severe reliability tensors must have matching shapes")
    if comparison_mask.shape != cleaner_reliability.shape:
        raise ValueError("Ranking comparison mask must match reliability shape")
    if margin < 0:
        raise ValueError("Ranking margin cannot be negative")
    selected = comparison_mask.to(torch.bool)
    losses = torch.relu(
        float(margin) - cleaner_reliability + more_severe_reliability
    )
    if not selected.any():
        return losses.sum() * 0.0
    return losses[selected].mean()
