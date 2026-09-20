"""Deterministic synthetic fixture used only for Phase-0 smoke tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robustsense.constants import LABELS, MODALITIES, MODALITY_DIMS
from robustsense.data.splits import user_level_split
from robustsense.utils.io import write_json
from robustsense.utils.reproducibility import set_global_seed


@dataclass(frozen=True)
class PreparedSyntheticData:
    dataset_path: Path
    manifest_path: Path
    split_manifest_path: Path


def modality_slices() -> dict[str, tuple[int, int]]:
    result: dict[str, tuple[int, int]] = {}
    start = 0
    for modality in MODALITIES:
        end = start + MODALITY_DIMS[modality]
        result[modality] = (start, end)
        start = end
    return result


def prepare_synthetic(config: dict, project_root: Path, fold: int = 0) -> PreparedSyntheticData:
    if config.get("dataset") != "synthetic":
        raise ValueError("Phase 0 only prepares the synthetic dataset; real data begins in Phase 1")

    seed = int(config.get("seed", 13))
    rng = set_global_seed(seed)
    n_users = int(config.get("n_users", 10))
    samples_per_user = int(config.get("samples_per_user", 12))
    missing_probability = float(config.get("missing_modality_probability", 0.16))
    unknown_probability = float(config.get("unknown_label_probability", 0.18))
    if n_users < 5:
        raise ValueError("Synthetic fixture needs at least five users")

    slices = modality_slices()
    feature_dim = sum(MODALITY_DIMS.values())
    sample_count = n_users * samples_per_user
    users = np.repeat([f"synthetic_user_{i:02d}" for i in range(n_users)], samples_per_user)
    timestamps = np.tile(np.arange(samples_per_user, dtype=np.int64) * 60, n_users)
    timestamps += 1_700_000_000

    features = rng.normal(size=(sample_count, feature_dim)).astype(np.float32)
    availability = rng.random((sample_count, len(MODALITIES))) >= missing_probability
    for row in range(sample_count):
        if not availability[row].any():
            availability[row, row % len(MODALITIES)] = True

    feature_mask = np.ones_like(features, dtype=bool)
    for modality_index, modality in enumerate(MODALITIES):
        start, end = slices[modality]
        unavailable = ~availability[:, modality_index]
        feature_mask[unavailable, start:end] = False
        features[unavailable, start:end] = np.nan

    latent_features = np.nan_to_num(features, nan=0.0)
    hidden_weights = rng.normal(scale=0.45, size=(feature_dim, len(LABELS)))
    user_bias = rng.normal(scale=0.25, size=(n_users, len(LABELS)))
    logits = latent_features @ hidden_weights + np.repeat(user_bias, samples_per_user, axis=0)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -20.0, 20.0)))
    targets = (rng.random(probabilities.shape) < probabilities).astype(np.float32)
    target_mask = rng.random(targets.shape) >= unknown_probability
    for row in range(sample_count):
        if not target_mask[row].any():
            target_mask[row, row % len(LABELS)] = True
    targets[~target_mask] = np.nan

    split = user_level_split(users.tolist(), test_fold=fold)
    split_codes = np.empty(sample_count, dtype="U5")
    for name, split_users in split.items():
        split_codes[np.isin(users, split_users)] = name

    dataset_path = project_root / config.get("output_path", "data/processed/synthetic_dev.npz")
    manifest_dir = project_root / config.get("manifest_dir", "data/manifests")
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dataset_path,
        features=features,
        feature_mask=feature_mask,
        availability=availability,
        targets=targets,
        target_mask=target_mask,
        user_id=users,
        timestamp=timestamps,
        split=split_codes,
    )

    manifest_path = manifest_dir / "synthetic_data_manifest.json"
    split_manifest_path = manifest_dir / f"synthetic_split_fold{fold}.json"
    write_json(
        manifest_path,
        {
            "synthetic_only": True,
            "seed": seed,
            "sample_count": sample_count,
            "user_count": n_users,
            "labels": list(LABELS),
            "modalities": {
                name: {"dim": MODALITY_DIMS[name], "slice": list(slices[name])}
                for name in MODALITIES
            },
            "unknown_label_count": int((~target_mask).sum()),
            "naturally_missing_modality_count": int((~availability).sum()),
        },
    )
    write_json(split_manifest_path, {"test_fold": fold, "validation_fold": (fold + 1) % 5, **split})
    return PreparedSyntheticData(dataset_path, manifest_path, split_manifest_path)
