"""Masked multi-label metrics for Phase-2 model runs."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def phase2_multilabel_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray | float,
) -> dict[str, object]:
    if probabilities.shape != targets.shape or targets.shape != target_mask.shape:
        raise ValueError("probabilities, targets, and target_mask must align")
    if probabilities.shape[1] != len(labels):
        raise ValueError("Label names do not match prediction width")
    known = target_mask.astype(bool)
    truth = np.nan_to_num(targets, nan=0.0).astype(bool)
    predictions = probabilities >= thresholds
    per_label: list[dict[str, object]] = []
    f1_values: list[float] = []
    average_precisions: list[float] = []

    for index, label in enumerate(labels):
        mask = known[:, index]
        label_truth = truth[mask, index]
        label_predictions = predictions[mask, index]
        tp = int(np.logical_and(label_predictions, label_truth).sum())
        fp = int(np.logical_and(label_predictions, ~label_truth).sum())
        fn = int(np.logical_and(~label_predictions, label_truth).sum())
        precision = _ratio(tp, tp + fp)
        recall = _ratio(tp, tp + fn)
        f1 = _ratio(2 * tp, 2 * tp + fp + fn)
        ap: float | None = None
        if label_truth.any() and (~label_truth).any():
            ap = float(average_precision_score(label_truth, probabilities[mask, index]))
            average_precisions.append(ap)
        if f1 is not None:
            f1_values.append(f1)
        per_label.append(
            {
                "label": label,
                "known_support": int(mask.sum()),
                "positive_support": int(label_truth.sum()),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "average_precision": ap,
            }
        )

    known_truth = truth[known]
    known_predictions = predictions[known]
    tp = int(np.logical_and(known_predictions, known_truth).sum())
    fp = int(np.logical_and(known_predictions, ~known_truth).sum())
    fn = int(np.logical_and(~known_predictions, known_truth).sum())
    micro_f1 = _ratio(2 * tp, 2 * tp + fp + fn)
    brier = float(np.mean((probabilities[known] - truth[known].astype(float)) ** 2))
    return {
        "macro_f1": float(np.mean(f1_values)) if f1_values else None,
        "micro_f1": micro_f1,
        "mean_average_precision": (
            float(np.mean(average_precisions)) if average_precisions else None
        ),
        "brier_score": brier,
        "known_target_count": int(known.sum()),
        "sample_count": int(len(targets)),
        "per_label": per_label,
    }
