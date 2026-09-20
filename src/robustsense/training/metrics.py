"""Masked multi-label metrics for the smoke pipeline."""

from __future__ import annotations

import numpy as np


def _safe_f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return float(2 * tp / denominator) if denominator else float("nan")


def masked_f1_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    thresholds: np.ndarray | float = 0.5,
) -> dict[str, object]:
    if probabilities.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("probabilities, targets, and target_mask must have identical shapes")
    known = target_mask.astype(bool)
    predictions = probabilities >= thresholds
    truth = np.nan_to_num(targets, nan=0.0).astype(bool)

    label_f1: list[float | None] = []
    for label_index in range(targets.shape[1]):
        mask = known[:, label_index]
        pred = predictions[mask, label_index]
        actual = truth[mask, label_index]
        tp = int(np.logical_and(pred, actual).sum())
        fp = int(np.logical_and(pred, ~actual).sum())
        fn = int(np.logical_and(~pred, actual).sum())
        value = _safe_f1(tp, fp, fn)
        label_f1.append(None if np.isnan(value) else value)

    valid_label_f1 = [value for value in label_f1 if value is not None]
    pred_known = predictions[known]
    truth_known = truth[known]
    tp = int(np.logical_and(pred_known, truth_known).sum())
    fp = int(np.logical_and(pred_known, ~truth_known).sum())
    fn = int(np.logical_and(~pred_known, truth_known).sum())
    micro = _safe_f1(tp, fp, fn)
    return {
        "macro_f1": float(np.mean(valid_label_f1)) if valid_label_f1 else None,
        "micro_f1": None if np.isnan(micro) else micro,
        "per_label_f1": label_f1,
        "known_target_count": int(known.sum()),
    }
