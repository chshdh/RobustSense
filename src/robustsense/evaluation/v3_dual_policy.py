"""Validation-thresholded dual-objective decision policies for V3-4."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from robustsense.evaluation.v2_metrics import masked_binary_cross_entropy_per_sample
from robustsense.training.phase2_metrics import phase2_multilabel_metrics

POLICY_MODES = ("risk_control", "error_alert")


def tune_coverage_threshold(
    confidence: np.ndarray, *, target_coverage: float, source_split: str
) -> dict[str, Any]:
    """Derive a deployable scalar threshold from validation confidence only."""
    scores = np.asarray(confidence, dtype=np.float64)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Confidence must be a non-empty finite vector")
    if ((scores < 0) | (scores > 1)).any():
        raise ValueError("Confidence must be in [0, 1]")
    if not 0 < target_coverage <= 1:
        raise ValueError("Target coverage must be inside (0, 1]")
    if source_split != "val":
        raise ValueError("Coverage thresholds must be derived from validation")
    count = int(np.ceil(target_coverage * len(scores)))
    ordered = np.sort(scores)[::-1]
    if count == len(scores):
        threshold = 0.0
    else:
        accepted_boundary = float(ordered[count - 1])
        rejected_boundary = float(ordered[count])
        threshold = (
            (accepted_boundary + rejected_boundary) / 2.0
            if accepted_boundary > rejected_boundary
            else accepted_boundary
        )
    accepted = scores >= threshold
    return {
        "source_split": source_split,
        "target_coverage": float(target_coverage),
        "threshold": float(threshold),
        "validation_sample_count": int(len(scores)),
        "validation_accepted_count": int(accepted.sum()),
        "validation_achieved_coverage": float(accepted.mean()),
        "validation_coverage_absolute_error": float(
            abs(accepted.mean() - target_coverage)
        ),
    }


@dataclass(frozen=True)
class DualPolicyDecisionLayer:
    """Fail-closed mode and target-coverage threshold dispatcher."""

    thresholds: dict[str, dict[str, float]]

    def decide(
        self, mode: str, confidence: np.ndarray, *, target_coverage: float
    ) -> np.ndarray:
        if mode not in POLICY_MODES or mode not in self.thresholds:
            raise ValueError(f"Unknown or unavailable policy mode: {mode}")
        key = f"{target_coverage:.2f}"
        if key not in self.thresholds[mode]:
            raise ValueError(f"Target coverage is not frozen for {mode}: {target_coverage}")
        scores = np.asarray(confidence, dtype=np.float64)
        if scores.ndim != 1 or not np.isfinite(scores).all():
            raise ValueError("Confidence must be a finite vector")
        return scores >= float(self.thresholds[mode][key])

    def to_artifact(self) -> dict[str, Any]:
        return {
            "version": 1,
            "policy": "explicit_dual_objective",
            "modes": list(POLICY_MODES),
            "threshold_source_split": "val",
            "thresholds": self.thresholds,
            "automatic_mode_selection": False,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> DualPolicyDecisionLayer:
        if artifact.get("threshold_source_split") != "val":
            raise ValueError("Policy thresholds are not validation-derived")
        if artifact.get("automatic_mode_selection") is not False:
            raise ValueError("Automatic objective selection is prohibited")
        return cls(
            thresholds={
                str(mode): {str(key): float(value) for key, value in values.items()}
                for mode, values in artifact["thresholds"].items()
            }
        )


def fixed_threshold_policy_metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    confidence: np.ndarray,
    errors: np.ndarray,
    *,
    labels: list[str],
    classification_thresholds: np.ndarray,
    acceptance_threshold: float,
    target_coverage: float,
) -> dict[str, Any]:
    """Evaluate a validation-derived threshold without test-time rank selection."""
    predicted = np.asarray(probabilities, dtype=np.float64)
    expected = np.asarray(targets, dtype=np.float64)
    known = np.asarray(target_mask, dtype=bool)
    scores = np.asarray(confidence, dtype=np.float64)
    error_events = np.asarray(errors, dtype=np.float64)
    if predicted.shape != expected.shape or expected.shape != known.shape:
        raise ValueError("Probabilities, targets, and masks must align")
    if scores.shape != (len(predicted),) or error_events.shape != scores.shape:
        raise ValueError("Confidence and errors must have one value per sample")
    if not np.isfinite(scores).all() or not np.isfinite(error_events).all():
        raise ValueError("Confidence and errors must be finite")
    accepted = scores >= acceptance_threshold
    rejected = ~accepted
    accepted_metrics: dict[str, Any] = {}
    if accepted.any():
        accepted_metrics = phase2_multilabel_metrics(
            predicted[accepted],
            expected[accepted],
            known[accepted],
            labels,
            classification_thresholds,
        )
    losses = masked_binary_cross_entropy_per_sample(predicted, expected, known)
    error_count = float(error_events.sum())
    rejected_error_count = float(error_events[rejected].sum())
    return {
        "target_coverage": float(target_coverage),
        "acceptance_threshold": float(acceptance_threshold),
        "test_sample_count": int(len(predicted)),
        "accepted_count": int(accepted.sum()),
        "rejected_count": int(rejected.sum()),
        "test_coverage": float(accepted.mean()),
        "coverage_absolute_error": float(abs(accepted.mean() - target_coverage)),
        "risk_masked_bce": (
            float(np.nanmean(losses[accepted])) if accepted.any() else np.nan
        ),
        "macro_f1": float(accepted_metrics.get("macro_f1", np.nan)),
        "micro_f1": float(accepted_metrics.get("micro_f1", np.nan)),
        "accepted_error_prevalence": (
            float(error_events[accepted].mean()) if accepted.any() else np.nan
        ),
        "rejected_error_precision": (
            float(error_events[rejected].mean()) if rejected.any() else np.nan
        ),
        "rejected_error_recall": (
            rejected_error_count / error_count if error_count > 0 else np.nan
        ),
    }
