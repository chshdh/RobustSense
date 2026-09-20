"""User-grouped continuous-risk selectors for V3-3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from robustsense.evaluation.v3_task_confidence import (
    per_label_binary_certainty,
    per_label_normalized_threshold_margin,
)


def build_compact_risk_features(
    collected: dict[str, Any],
    *,
    modality_names: list[str],
    thresholds: np.ndarray,
    include_sensor_features: bool,
) -> tuple[np.ndarray, list[str]]:
    """Build a compact feature set without targets or user identity."""
    probabilities = np.asarray(collected["probabilities"], dtype=np.float64)
    certainty = per_label_binary_certainty(probabilities)
    margins = per_label_normalized_threshold_margin(probabilities, thresholds)
    predicted_positive = probabilities >= np.asarray(thresholds, dtype=np.float64)
    parts = [
        certainty.mean(axis=1),
        certainty.min(axis=1),
        margins.mean(axis=1),
        margins.min(axis=1),
        predicted_positive.mean(axis=1),
        probabilities.std(axis=1),
    ]
    names = [
        "mean_label_certainty",
        "min_label_certainty",
        "mean_label_threshold_margin",
        "min_label_threshold_margin",
        "predicted_positive_fraction",
        "probability_std",
    ]
    if include_sensor_features:
        expected_shape = (len(probabilities), len(modality_names))
        availability = np.asarray(collected.get("availability"), dtype=np.float64)
        weights = np.asarray(collected.get("fusion_weights"), dtype=np.float64)
        reliability = np.asarray(collected.get("reliability"), dtype=np.float64)
        system = np.asarray(collected.get("system_reliability"), dtype=np.float64).reshape(-1)
        for feature_name, values in (
            ("availability", availability),
            ("fusion weights", weights),
            ("reliability", reliability),
        ):
            if values.shape != expected_shape or not np.isfinite(values).all():
                raise ValueError(f"{feature_name} must be a finite sample-modality matrix")
        if system.shape != (len(probabilities),) or not np.isfinite(system).all():
            raise ValueError("System reliability must be one finite value per sample")
        counts = availability.sum(axis=1)
        if (counts <= 0).any():
            raise ValueError("Every sample needs at least one available modality")
        mean_available_reliability = (reliability * availability).sum(axis=1) / counts
        min_available_reliability = np.where(
            availability.astype(bool), reliability, np.inf
        ).min(axis=1)
        parts.extend(
            [
                counts / len(modality_names),
                system,
                mean_available_reliability,
                min_available_reliability,
                np.square(weights).sum(axis=1),
            ]
        )
        names.extend(
            [
                "available_modality_fraction",
                "system_reliability",
                "mean_available_modality_reliability",
                "min_available_modality_reliability",
                "fusion_weight_concentration",
            ]
        )
    features = np.column_stack(parts)
    if features.shape[1] != len(names) or not np.isfinite(features).all():
        raise ValueError("Compact risk features are invalid")
    return features, names


@dataclass(frozen=True)
class RidgeRiskSelector:
    """Serializable standardized Ridge predictor of per-sample BCE risk."""

    feature_names: tuple[str, ...]
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    coefficients: np.ndarray
    intercept: float
    alpha: float
    solver: str

    def predict_risk(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(self.feature_names):
            raise ValueError("Feature matrix width does not match selector")
        if not np.isfinite(values).all():
            raise ValueError("Features must be finite")
        standardized = (values - self.feature_mean) / self.feature_scale
        return standardized @ self.coefficients + self.intercept

    def predict_confidence(self, features: np.ndarray) -> np.ndarray:
        risk = self.predict_risk(features)
        return 1.0 / (1.0 + np.exp(np.clip(risk, -40.0, 40.0)))

    def to_artifact(self) -> dict[str, Any]:
        return {
            "version": 1,
            "model": "standardized_ridge_risk_selector",
            "target": "masked_bce_per_sample",
            "feature_names": list(self.feature_names),
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "coefficients": self.coefficients.tolist(),
            "intercept": self.intercept,
            "alpha": self.alpha,
            "solver": self.solver,
        }

    @classmethod
    def from_artifact(cls, artifact: dict[str, Any]) -> RidgeRiskSelector:
        return cls(
            feature_names=tuple(artifact["feature_names"]),
            feature_mean=np.asarray(artifact["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(artifact["feature_scale"], dtype=np.float64),
            coefficients=np.asarray(artifact["coefficients"], dtype=np.float64),
            intercept=float(artifact["intercept"]),
            alpha=float(artifact["alpha"]),
            solver=str(artifact["solver"]),
        )


def fit_ridge_risk_selector(
    features: np.ndarray,
    feature_names: list[str],
    losses: np.ndarray,
    *,
    alpha: float,
    solver: str,
) -> RidgeRiskSelector:
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(losses, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(feature_names):
        raise ValueError("Feature names do not match feature width")
    if targets.shape != (len(values),):
        raise ValueError("Losses must have one value per sample")
    if not np.isfinite(values).all() or not np.isfinite(targets).all():
        raise ValueError("Features and losses must be finite")
    if alpha <= 0:
        raise ValueError("Ridge alpha must be positive")
    mean = values.mean(axis=0)
    scale = values.std(axis=0)
    scale = np.where(scale > 1.0e-12, scale, 1.0)
    standardized = (values - mean) / scale
    model = Ridge(alpha=alpha, solver=solver, fit_intercept=True)
    model.fit(standardized, targets)
    return RidgeRiskSelector(
        feature_names=tuple(feature_names),
        feature_mean=mean,
        feature_scale=scale,
        coefficients=np.asarray(model.coef_, dtype=np.float64),
        intercept=float(model.intercept_),
        alpha=float(alpha),
        solver=solver,
    )


def normalized_risk_aurc_from_losses(
    losses: np.ndarray, confidence: np.ndarray, coverage_grid: np.ndarray
) -> float:
    risk = np.asarray(losses, dtype=np.float64)
    scores = np.asarray(confidence, dtype=np.float64)
    coverage = np.asarray(coverage_grid, dtype=np.float64)
    if risk.shape != scores.shape or risk.ndim != 1:
        raise ValueError("Losses and confidence must be aligned vectors")
    if not np.isfinite(risk).all() or not np.isfinite(scores).all():
        raise ValueError("Losses and confidence must be finite")
    if ((coverage <= 0) | (coverage > 1)).any():
        raise ValueError("Coverage must be inside (0, 1]")
    order = np.argsort(-scores, kind="stable")
    actual = []
    mean_risk = []
    for requested in coverage:
        count = int(np.ceil(float(requested) * len(risk)))
        actual.append(count / len(risk))
        mean_risk.append(float(risk[order[:count]].mean()))
    actual_values = np.asarray(actual, dtype=np.float64)
    risk_values = np.asarray(mean_risk, dtype=np.float64)
    width = actual_values[-1] - actual_values[0]
    if width <= 0 or np.any(np.diff(actual_values) <= 0):
        raise ValueError("Coverage grid does not create a valid integration interval")
    return float(np.trapezoid(risk_values, actual_values) / width)


def select_grouped_ridge_risk_selector(
    features: np.ndarray,
    feature_names: list[str],
    losses: np.ndarray,
    groups: np.ndarray,
    *,
    alphas: list[float],
    n_splits: int,
    solver: str,
    coverage_grid: np.ndarray,
) -> tuple[RidgeRiskSelector, dict[str, Any]]:
    """Select alpha by group-held-out OOF AURC, then refit all validation samples."""
    values = np.asarray(features, dtype=np.float64)
    targets = np.asarray(losses, dtype=np.float64)
    group_values = np.asarray(groups).astype(str)
    if len(group_values) != len(values):
        raise ValueError("Groups must have one value per sample")
    if len(np.unique(group_values)) < n_splits or n_splits < 2:
        raise ValueError("Not enough unique groups for grouped cross-validation")
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("Alpha candidates must be positive")
    splitter = GroupKFold(n_splits=n_splits)
    folds = list(splitter.split(values, targets, groups=group_values))
    fold_contract = []
    for fold_index, (train_indices, val_indices) in enumerate(folds):
        train_groups = set(group_values[train_indices])
        val_groups = set(group_values[val_indices])
        fold_contract.append(
            {
                "fold_index": fold_index,
                "train_sample_count": int(len(train_indices)),
                "validation_sample_count": int(len(val_indices)),
                "train_group_count": len(train_groups),
                "validation_group_count": len(val_groups),
                "group_overlap_count": len(train_groups & val_groups),
            }
        )
    candidates = []
    for alpha in alphas:
        oof_risk = np.full(len(values), np.nan, dtype=np.float64)
        for train_indices, val_indices in folds:
            selector = fit_ridge_risk_selector(
                values[train_indices],
                feature_names,
                targets[train_indices],
                alpha=float(alpha),
                solver=solver,
            )
            oof_risk[val_indices] = selector.predict_risk(values[val_indices])
        if not np.isfinite(oof_risk).all():
            raise RuntimeError("Grouped OOF prediction did not cover every validation sample")
        oof_confidence = 1.0 / (1.0 + np.exp(np.clip(oof_risk, -40.0, 40.0)))
        candidates.append(
            {
                "alpha": float(alpha),
                "oof_normalized_aurc": normalized_risk_aurc_from_losses(
                    targets, oof_confidence, coverage_grid
                ),
                "oof_mse": float(np.mean(np.square(oof_risk - targets))),
            }
        )
    selected = min(
        candidates,
        key=lambda row: (row["oof_normalized_aurc"], -row["alpha"]),
    )
    final = fit_ridge_risk_selector(
        values,
        feature_names,
        targets,
        alpha=selected["alpha"],
        solver=solver,
    )
    selection = {
        "selection_split": "val_grouped_oof",
        "selection_metric": "normalized_aurc",
        "tie_break": "larger_alpha",
        "n_splits": n_splits,
        "unique_group_count": int(len(np.unique(group_values))),
        "selected_alpha": selected["alpha"],
        "selected_oof_normalized_aurc": selected["oof_normalized_aurc"],
        "candidates": candidates,
        "fold_contract": fold_contract,
    }
    return final, selection
