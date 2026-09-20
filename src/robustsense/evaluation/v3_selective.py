"""V3-1 公平选择性预测评估指标。"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from robustsense.evaluation.v2_metrics import risk_coverage_curve


def _probability_matrix(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or not values.size:
        raise ValueError("Probabilities must be a non-empty two-dimensional array")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    return values


def mean_binary_certainty(probabilities: np.ndarray) -> np.ndarray:
    """Return one minus mean normalized Bernoulli entropy for each sample."""
    values = _probability_matrix(probabilities)
    clipped = np.clip(values, 1.0e-12, 1.0 - 1.0e-12)
    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / np.log(2.0)
    return np.clip(1.0 - entropy.mean(axis=1), 0.0, 1.0)


def mean_normalized_threshold_margin(
    probabilities: np.ndarray, thresholds: np.ndarray | float
) -> np.ndarray:
    """Return mean distance from validation-derived decision thresholds in [0, 1]."""
    values = _probability_matrix(probabilities)
    cutoffs = np.asarray(thresholds, dtype=np.float64)
    if cutoffs.ndim == 0:
        cutoffs = np.full(values.shape[1], float(cutoffs), dtype=np.float64)
    if cutoffs.shape != (values.shape[1],):
        raise ValueError("Thresholds must be scalar or have one value per label")
    if not np.isfinite(cutoffs).all() or ((cutoffs <= 0) | (cutoffs >= 1)).any():
        raise ValueError("Thresholds must be finite and strictly inside (0, 1)")
    positive_side = (values - cutoffs) / (1.0 - cutoffs)
    negative_side = (cutoffs - values) / cutoffs
    margins = np.where(values >= cutoffs, positive_side, negative_side)
    return np.clip(margins.mean(axis=1), 0.0, 1.0)


def sample_error_indicator(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    thresholds: np.ndarray | float,
) -> np.ndarray:
    """Mark samples with at least one wrong known label; no-known samples become NaN."""
    values = _probability_matrix(probabilities)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    if values.shape != expected.shape or expected.shape != known.shape:
        raise ValueError("Probabilities, targets, and target masks must align")
    cutoffs = np.asarray(thresholds, dtype=np.float64)
    if cutoffs.ndim == 0:
        cutoffs = np.full(values.shape[1], float(cutoffs), dtype=np.float64)
    if cutoffs.shape != (values.shape[1],):
        raise ValueError("Thresholds must be scalar or have one value per label")
    truth = np.nan_to_num(expected, nan=0.0).astype(bool)
    wrong = ((values >= cutoffs) != truth) & known
    usable = known.any(axis=1)
    result = np.full(len(values), np.nan, dtype=np.float64)
    result[usable] = wrong[usable].any(axis=1).astype(np.float64)
    return result


def error_detection_metrics(
    confidence: np.ndarray, errors: np.ndarray, *, run_id: str, score_method: str
) -> dict[str, Any]:
    """Evaluate low confidence as a detector of sample-level classification error."""
    scores = np.asarray(confidence, dtype=np.float64)
    labels = np.asarray(errors, dtype=np.float64)
    if scores.shape != labels.shape or scores.ndim != 1:
        raise ValueError("Confidence and error indicators must be aligned vectors")
    if not run_id or not score_method:
        raise ValueError("Run ID and score method are required")
    selected = np.isfinite(scores) & np.isfinite(labels)
    binary = labels[selected].astype(np.int64)
    error_scores = 1.0 - scores[selected]
    has_two_classes = len(np.unique(binary)) == 2
    return {
        "run_id": run_id,
        "score_method": score_method,
        "valid_count": int(selected.sum()),
        "error_count": int(binary.sum()),
        "error_prevalence": float(binary.mean()) if len(binary) else np.nan,
        "error_detection_auroc": (
            float(roc_auc_score(binary, error_scores)) if has_two_classes else np.nan
        ),
        "error_detection_auprc": (
            float(average_precision_score(binary, error_scores))
            if has_two_classes
            else np.nan
        ),
    }


def selective_curve(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    confidence: np.ndarray,
    *,
    labels: list[str],
    thresholds: np.ndarray | float,
    coverage_grid: np.ndarray,
    run_id: str,
    score_method: str,
) -> list[dict[str, Any]]:
    """Build a matched-coverage curve from a predeclared confidence score."""
    scores = np.asarray(confidence, dtype=np.float64)
    if scores.shape != (len(probabilities),) or not np.isfinite(scores).all():
        raise ValueError("Confidence must contain one finite value per sample")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Confidence must be in [0, 1]")
    rows = risk_coverage_curve(
        probabilities,
        targets,
        target_mask,
        scores,
        labels=labels,
        thresholds=thresholds,
        coverage_grid=coverage_grid,
        run_id=run_id,
    )
    ordered = np.argsort(-scores, kind="stable")
    for row in rows:
        accepted_count = int(row["accepted_count"])
        row["score_method"] = score_method
        row["confidence_cutoff"] = (
            float(scores[ordered[accepted_count - 1]]) if accepted_count else np.nan
        )
    return rows


def normalized_aurc(curve: list[dict[str, Any]]) -> float:
    """Integrate risk over the actually achieved coverage interval."""
    pairs = sorted(
        (
            (float(row["coverage"]), float(row["risk_masked_bce"]))
            for row in curve
            if np.isfinite(row["coverage"]) and np.isfinite(row["risk_masked_bce"])
        ),
        key=lambda pair: pair[0],
    )
    if len(pairs) < 2:
        return np.nan
    coverage = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    risk = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    if np.any(np.diff(coverage) <= 0):
        raise ValueError("Curve coverage must be strictly increasing")
    width = coverage[-1] - coverage[0]
    return float(np.trapezoid(risk, coverage) / width) if width > 0 else np.nan


def summarize_selector(
    curve: list[dict[str, Any]],
    confidence: np.ndarray,
    errors: np.ndarray,
    *,
    run_id: str,
    score_method: str,
    fixed_coverages: tuple[float, ...],
) -> dict[str, Any]:
    """Create one compact, machine-readable summary for a selector."""
    detection = error_detection_metrics(
        confidence, errors, run_id=run_id, score_method=score_method
    )
    values = np.asarray(confidence, dtype=np.float64)
    result: dict[str, Any] = {
        **detection,
        "normalized_aurc": normalized_aurc(curve),
        "confidence_mean": float(values.mean()),
        "confidence_std": float(values.std()),
        "confidence_min": float(values.min()),
        "confidence_max": float(values.max()),
    }
    for requested in fixed_coverages:
        matches = [
            row
            for row in curve
            if np.isclose(float(row["requested_coverage"]), requested, atol=1.0e-12)
        ]
        if len(matches) != 1:
            raise ValueError(f"Coverage {requested} is absent or duplicated")
        row = matches[0]
        suffix = f"{requested:.2f}".replace(".", "_")
        result[f"actual_coverage_{suffix}"] = float(row["coverage"])
        result[f"risk_masked_bce_{suffix}"] = float(row["risk_masked_bce"])
        result[f"macro_f1_{suffix}"] = float(row["macro_f1"])
        result[f"micro_f1_{suffix}"] = float(row["micro_f1"])
    return result
