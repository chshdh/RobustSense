"""Validation-only abstention threshold selection for P2."""

from __future__ import annotations

import numpy as np


def apply_abstention_threshold(
    system_reliability: np.ndarray, threshold: float
) -> np.ndarray:
    values = np.asarray(system_reliability, dtype=np.float64)
    if values.ndim != 1 or not len(values):
        raise ValueError("System reliability must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("System reliability must contain finite values in [0, 1]")
    if not 0 <= float(threshold) <= 1:
        raise ValueError("Abstention threshold must be in [0, 1]")
    return values < float(threshold)


def tune_abstention_threshold(
    system_reliability: np.ndarray,
    target_coverage: float = 0.9,
    *,
    source_split: str,
) -> dict[str, float | int | str]:
    """Choose the highest observed threshold retaining at least target coverage."""
    if source_split != "val":
        raise ValueError("Abstention thresholds may only be tuned from validation")
    if not 0 < float(target_coverage) <= 1:
        raise ValueError("target_coverage must be in (0, 1]")
    values = np.asarray(system_reliability, dtype=np.float64)
    apply_abstention_threshold(values, 0.0)
    max_rejected = int(np.floor((1.0 - float(target_coverage)) * len(values) + 1.0e-12))
    index = min(max_rejected, len(values) - 1)
    threshold = float(np.sort(values, kind="stable")[index])
    abstain = apply_abstention_threshold(values, threshold)
    return {
        "source_split": source_split,
        "method": "highest_observed_threshold_with_at_least_target_coverage",
        "target_coverage": float(target_coverage),
        "threshold": threshold,
        "validation_sample_count": int(len(values)),
        "validation_coverage": float((~abstain).mean()),
    }

