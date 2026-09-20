"""Validation-fitted task-aligned confidence selectors for V3-2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


def _probabilities(values: np.ndarray) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if probabilities.ndim != 2 or not probabilities.size:
        raise ValueError("Probabilities must be a non-empty matrix")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities must be finite")
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    return probabilities


def per_label_binary_certainty(probabilities: np.ndarray) -> np.ndarray:
    """Return normalized Bernoulli certainty without averaging across labels."""
    values = _probabilities(probabilities)
    clipped = np.clip(values, 1.0e-12, 1.0 - 1.0e-12)
    entropy = -(
        clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped)
    ) / np.log(2.0)
    return np.clip(1.0 - entropy, 0.0, 1.0)


def per_label_normalized_threshold_margin(
    probabilities: np.ndarray, thresholds: np.ndarray
) -> np.ndarray:
    """Return validation-threshold distance for every label."""
    values = _probabilities(probabilities)
    cutoffs = np.asarray(thresholds, dtype=np.float64)
    if cutoffs.shape != (values.shape[1],):
        raise ValueError("Thresholds must have one value per label")
    if not np.isfinite(cutoffs).all() or ((cutoffs <= 0) | (cutoffs >= 1)).any():
        raise ValueError("Thresholds must be finite and inside (0, 1)")
    positive_side = (values - cutoffs) / (1.0 - cutoffs)
    negative_side = (cutoffs - values) / cutoffs
    return np.clip(np.where(values >= cutoffs, positive_side, negative_side), 0.0, 1.0)


def build_task_confidence_features(
    collected: dict[str, Any],
    *,
    labels: list[str],
    modality_names: list[str],
    thresholds: np.ndarray,
    include_sensor_features: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build output-only or output-plus-sensor features without using targets."""
    probabilities = _probabilities(collected["probabilities"])
    if probabilities.shape[1] != len(labels):
        raise ValueError("Label names do not match probability width")
    certainty = per_label_binary_certainty(probabilities)
    margins = per_label_normalized_threshold_margin(probabilities, thresholds)
    parts = [certainty, margins]
    names = [f"label_certainty::{label}" for label in labels]
    names.extend(f"label_threshold_margin::{label}" for label in labels)

    if include_sensor_features:
        expected_shape = (len(probabilities), len(modality_names))
        availability = np.asarray(collected.get("availability"), dtype=np.float64)
        weights = np.asarray(collected.get("fusion_weights"), dtype=np.float64)
        reliability = np.asarray(collected.get("reliability"), dtype=np.float64)
        system = np.asarray(collected.get("system_reliability"), dtype=np.float64).reshape(-1)
        for feature_name, values in (
            ("availability", availability),
            ("fusion weights", weights),
            ("modality reliability", reliability),
        ):
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f"{feature_name} must be a finite sample-modality matrix")
        if system.shape != (len(probabilities),) or not np.isfinite(system).all():
            raise ValueError("System reliability must be one finite value per sample")
        masked_reliability = reliability * availability
        parts.extend([availability, weights, masked_reliability, system[:, None]])
        names.extend(f"availability::{name}" for name in modality_names)
        names.extend(f"fusion_weight::{name}" for name in modality_names)
        names.extend(f"masked_reliability::{name}" for name in modality_names)
        names.append("system_reliability")

    features = np.column_stack(parts)
    if not np.isfinite(features).all() or features.shape[1] != len(names):
        raise ValueError("Task-confidence features are invalid")
    return features, names


@dataclass(frozen=True)
class TaskConfidenceSelector:
    """A serializable standardized logistic error selector."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    regularization_c: float
    solver: str
    max_iter: int
    iterations: int

    def predict_confidence(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        width = len(self.feature_names)
        if values.ndim != 2 or values.shape[1] != width:
            raise ValueError("Feature matrix width does not match selector")
        if not np.isfinite(values).all():
            raise ValueError("Feature matrix must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        logits = standardized @ self.coefficients + self.intercept
        error_probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        return np.clip(1.0 - error_probability, 0.0, 1.0)

    def to_artifact(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model": "standardized_logistic_error_selector",
            "target": "at_least_one_wrong_known_label",
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "regularization_c": self.regularization_c,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "iterations": self.iterations,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> TaskConfidenceSelector:
        return cls(
            feature_names=tuple(artifact["feature_names"]),
            feature_mean=np.asarray(artifact["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(artifact["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(artifact["coefficients"], dtype=np.float64),
            intercept=float(artifact["intercept"]),
            regularization_c=float(artifact["regularization_c"]),
            solver=str(artifact["solver"]),
            max_iter=int(artifact["max_iter"]),
            iterations=int(artifact["iterations"]),
        )


def fit_task_confidence_selector(
    features: np.ndarray,
    feature_names: list[str],
    errors: np.ndarray,
    *,
    regularization_c: float,
    solver: str,
    max_iter: int,
) -> TaskConfidenceSelector:
    """Fit a fixed-policy logistic error predictor on validation samples only."""
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(errors, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match feature width")
    if targets.shape != (len(values),):
        raise ValueError("Errors must have one value per sample")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("Training features and errors must be finite")
    binary = targets.astype(np.int64)
    if not np.array_equal(binary.astype(np.float64), targets):
        raise ValueError("Error targets must be binary")
    if len(np.unique(binary)) != 2:
        raise ValueError("Both correct and error samples are required")
    if regularization_c <= 0 or max_iter <= 0:
        raise ValueError("Regularization and max_iter must be positive")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (values - mean) / scale
    model = LogisticRegression(
        C=regularization_c,
        solver=solver,
        max_iter=max_iter,
        class_weight=None,
        fit_intercept=True,
    )
    model.fit(standardized, binary)
    return TaskConfidenceSelector(
        feature_names=tuple(feature_names),
        feature_mean=mean,
        feature_scale=scale,
        coefficients=model.coef_[0].astype(np.float64),
        intercept=float(model.intercept_[0]),
        regularization_c=float(regularization_c),
        solver=solver,
        max_iter=int(max_iter),
        iterations=int(model.n_iter_[0]),
    )
