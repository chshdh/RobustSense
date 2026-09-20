"""Validation-only deterministic per-label threshold tuning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def tune_per_label_thresholds(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    grid: Sequence[float],
    *,
    source_split: str,
) -> np.ndarray:
    if source_split != "val":
        raise ValueError("Thresholds may only be tuned from the validation split")
    if probabilities.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("probabilities, targets, and target_mask must align")
    thresholds = np.full(targets.shape[1], 0.5, dtype=np.float32)
    clean_targets = np.nan_to_num(targets, nan=0.0).astype(bool)
    candidates = np.asarray(tuple(grid), dtype=np.float64)
    if candidates.ndim != 1 or not len(candidates):
        raise ValueError("Threshold grid cannot be empty")
    for label_index in range(targets.shape[1]):
        known = target_mask[:, label_index].astype(bool)
        if not known.any() or not clean_targets[known, label_index].any():
            continue
        truth = clean_targets[known, label_index]
        scores = probabilities[known, label_index]
        best: tuple[float, float, float] | None = None
        best_threshold = 0.5
        for threshold in candidates:
            prediction = scores >= threshold
            true_positive = np.logical_and(prediction, truth).sum()
            false_positive = np.logical_and(prediction, ~truth).sum()
            false_negative = np.logical_and(~prediction, truth).sum()
            denominator = 2 * true_positive + false_positive + false_negative
            f1 = float(2 * true_positive / denominator) if denominator else 0.0
            key = (f1, -abs(float(threshold) - 0.5), -float(threshold))
            if best is None or key > best:
                best = key
                best_threshold = float(threshold)
        thresholds[label_index] = best_threshold
    return thresholds
