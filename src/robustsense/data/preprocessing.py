"""Fold-specific robust preprocessing fitted on training users only."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robustsense.constants import MODALITIES
from robustsense.utils.io import read_json, write_json


class PreprocessingError(ValueError):
    """Raised when a fold cannot be preprocessed without leakage or ambiguity."""


def user_list_sha256(users: list[str]) -> str:
    return hashlib.sha256(("\n".join(sorted(users)) + "\n").encode()).hexdigest()


@dataclass(frozen=True)
class RobustPreprocessor:
    feature_names: tuple[str, ...]
    medians: np.ndarray
    iqrs: np.ndarray
    fit_user_sha256: str
    outlier_threshold: float = 4.0
    clip_value: float = 20.0
    minimum_observed_fraction: float = 0.01

    @classmethod
    def fit(
        cls,
        features: np.ndarray,
        feature_names: list[str] | tuple[str, ...],
        fit_users: list[str],
        *,
        outlier_threshold: float = 4.0,
        clip_value: float = 20.0,
        minimum_observed_fraction: float = 0.01,
    ) -> RobustPreprocessor:
        if features.ndim != 2 or features.shape[1] != len(feature_names):
            raise PreprocessingError("Feature matrix and feature names do not align")
        if not fit_users:
            raise PreprocessingError("Preprocessor requires explicit training users")
        if not 0.0 < minimum_observed_fraction <= 1.0:
            raise PreprocessingError("minimum_observed_fraction must be in (0, 1]")
        finite_features = np.where(np.isfinite(features), features, np.nan)
        finite_counts = np.isfinite(finite_features).sum(axis=0)
        if np.any(finite_counts == 0):
            missing = [feature_names[index] for index in np.flatnonzero(finite_counts == 0)]
            raise PreprocessingError(f"Training data has all-missing features: {missing}")
        quartiles = np.nanpercentile(finite_features, [25.0, 50.0, 75.0], axis=0)
        medians = quartiles[1].astype(np.float32)
        iqrs = (quartiles[2] - quartiles[0]).astype(np.float32)
        iqrs[~np.isfinite(iqrs) | (iqrs == 0)] = 1.0
        return cls(
            feature_names=tuple(feature_names),
            medians=medians,
            iqrs=iqrs,
            fit_user_sha256=user_list_sha256(fit_users),
            outlier_threshold=float(outlier_threshold),
            clip_value=float(clip_value),
            minimum_observed_fraction=float(minimum_observed_fraction),
        )

    def transform(
        self, features: np.ndarray, modality_slices: dict[str, tuple[int, int]]
    ) -> dict[str, np.ndarray]:
        if features.ndim != 2 or features.shape[1] != len(self.feature_names):
            raise PreprocessingError("Transform feature shape differs from fitted schema")
        if tuple(modality_slices) != MODALITIES:
            raise PreprocessingError("Modality slices must use the frozen modality order")
        feature_masks = np.isfinite(features)
        clean = np.where(feature_masks, features, self.medians)
        robust_z = (clean - self.medians) / self.iqrs
        quality = np.zeros((features.shape[0], len(MODALITIES), 5), dtype=np.float32)
        availability = np.zeros((features.shape[0], len(MODALITIES)), dtype=bool)

        for modality_index, modality in enumerate(MODALITIES):
            start, end = modality_slices[modality]
            modality_mask = feature_masks[:, start:end]
            observed_count = modality_mask.sum(axis=1)
            observed_fraction = observed_count / (end - start)
            is_available = observed_fraction >= self.minimum_observed_fraction
            modality_z = robust_z[:, start:end]
            abs_z = np.abs(modality_z)
            observed_denominator = np.maximum(observed_count, 1)
            outlier_fraction = (
                ((abs_z > self.outlier_threshold) & modality_mask).sum(axis=1)
                / observed_denominator
            )
            mean_abs = (np.where(modality_mask, abs_z, 0.0).sum(axis=1) / observed_denominator)
            max_abs = np.where(modality_mask, abs_z, 0.0).max(axis=1)
            availability[:, modality_index] = is_available
            quality[:, modality_index] = np.column_stack(
                [
                    is_available,
                    observed_fraction,
                    outlier_fraction,
                    np.minimum(mean_abs, self.clip_value),
                    np.minimum(max_abs, self.clip_value),
                ]
            )
            robust_z[~is_available, start:end] = 0.0
            feature_masks[~is_available, start:end] = False

        scaled = np.clip(robust_z, -self.clip_value, self.clip_value).astype(np.float32)
        scaled[~feature_masks] = 0.0
        return {
            "features": scaled,
            "feature_masks": feature_masks,
            "availability": availability,
            "quality_features": quality,
        }

    def save(self, directory: str | Path, fit_users: list[str]) -> tuple[Path, Path]:
        output = Path(directory)
        output.mkdir(parents=True, exist_ok=True)
        arrays_path = output / "preprocessor.npz"
        metadata_path = output / "preprocessor.json"
        np.savez(
            arrays_path,
            feature_names=np.asarray(self.feature_names),
            medians=self.medians,
            iqrs=self.iqrs,
        )
        write_json(
            metadata_path,
            {
                "version": 1,
                "fit_split": "train",
                "fit_users": sorted(fit_users),
                "fit_user_sha256": self.fit_user_sha256,
                "feature_count": len(self.feature_names),
                "outlier_threshold": self.outlier_threshold,
                "clip_value": self.clip_value,
                "minimum_observed_fraction": self.minimum_observed_fraction,
                "arrays_file": arrays_path.name,
            },
        )
        return arrays_path, metadata_path

    @classmethod
    def load(cls, directory: str | Path) -> RobustPreprocessor:
        root = Path(directory)
        metadata = read_json(root / "preprocessor.json")
        with np.load(root / metadata["arrays_file"]) as arrays:
            return cls(
                feature_names=tuple(arrays["feature_names"].astype(str)),
                medians=arrays["medians"].astype(np.float32),
                iqrs=arrays["iqrs"].astype(np.float32),
                fit_user_sha256=metadata["fit_user_sha256"],
                outlier_threshold=float(metadata["outlier_threshold"]),
                clip_value=float(metadata["clip_value"]),
                minimum_observed_fraction=float(metadata["minimum_observed_fraction"]),
            )
