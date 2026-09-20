"""Masked multi-label losses."""

from __future__ import annotations

import numpy as np


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))


def masked_binary_cross_entropy(
    logits: np.ndarray, targets: np.ndarray, target_mask: np.ndarray
) -> float:
    if logits.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("logits, targets, and target_mask must have identical shapes")
    known = target_mask.astype(bool)
    if not known.any():
        raise ValueError("Cannot compute loss when every target is unknown")
    clean_targets = np.nan_to_num(targets, nan=0.0)
    per_entry = np.maximum(logits, 0.0) - logits * clean_targets + np.log1p(np.exp(-np.abs(logits)))
    return float(per_entry[known].mean())
