"""Prepare audited ExtraSensory files into leakage-safe fold caches."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.constants import MODALITIES
from robustsense.data.extrasensory import (
    discover_user_files,
    parse_official_folds,
    sha256_file,
    split_from_official_folds,
)
from robustsense.data.preprocessing import RobustPreprocessor, user_list_sha256
from robustsense.utils.io import read_json, write_json


def _materialize_audited_split_manifest(
    config: dict[str, Any], root: Path, fold: int
) -> Path:
    """Derive another outer split from already-audited official fold files."""
    manifest_dir = root / config.get("manifest_dir", "data/manifests")
    schema_path = manifest_dir / "extrasensory_schema_manifest.json"
    data_manifest_path = manifest_dir / "extrasensory_data_manifest.json"
    for required in (schema_path, data_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing Phase-1 artifact {required}; run the ExtraSensory audit first"
            )
    folds = parse_official_folds(
        root / config.get("extracted_dir", "data/extracted") / "folds"
    )
    user_files = discover_user_files(
        root / config.get("extracted_dir", "data/extracted") / "features"
    )
    official_users = {user for values in folds.values() for user in values["test"]}
    if set(user_files) != official_users:
        raise ValueError("Audited feature users no longer match official fold users")
    split = split_from_official_folds(folds, fold)
    payload = dict(split)
    for name in ("train", "val", "test"):
        payload[f"{name}_user_sha256"] = user_list_sha256(payload[name])
    path = manifest_dir / f"extrasensory_split_fold{fold}.json"
    write_json(path, payload)
    return path


def _load_raw_users(
    users: list[str],
    user_files: dict[str, Path],
    feature_columns: list[str],
    label_columns: list[str],
    timestamp_column: str,
) -> dict[str, np.ndarray]:
    frames: list[pd.DataFrame] = []
    use_columns = [timestamp_column, *feature_columns, *label_columns]
    dtypes = {column: "float32" for column in [*feature_columns, *label_columns]}
    dtypes[timestamp_column] = "int64"
    user_ids: list[np.ndarray] = []
    for user in users:
        frame = pd.read_csv(user_files[user], usecols=use_columns, dtype=dtypes)
        frames.append(frame)
        user_ids.append(np.full(len(frame), user, dtype=f"U{len(user)}"))
    combined = pd.concat(frames, ignore_index=True)
    return {
        "features": combined.loc[:, feature_columns].to_numpy(dtype=np.float32),
        "targets": combined.loc[:, label_columns].to_numpy(dtype=np.float32),
        "user_id": np.concatenate(user_ids),
        "timestamp": combined[timestamp_column].to_numpy(dtype=np.int64),
    }


def _save_processed_split(
    path: Path,
    raw: dict[str, np.ndarray],
    preprocessor: RobustPreprocessor,
    modality_slices: dict[str, tuple[int, int]],
) -> dict[str, int]:
    transformed = preprocessor.transform(raw["features"], modality_slices)
    target_mask = np.isfinite(raw["targets"])
    valid_modalities = transformed["availability"].any(axis=1)
    valid_targets = target_mask.any(axis=1)
    keep = valid_modalities & valid_targets
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=transformed["features"][keep],
        feature_masks=transformed["feature_masks"][keep],
        availability=transformed["availability"][keep],
        quality_features=transformed["quality_features"][keep],
        targets=raw["targets"][keep],
        target_mask=target_mask[keep],
        user_id=raw["user_id"][keep],
        timestamp=raw["timestamp"][keep],
    )
    return {
        "source_rows": int(len(keep)),
        "kept_rows": int(keep.sum()),
        "excluded_rows": int((~keep).sum()),
        "excluded_no_modality": int((~valid_modalities).sum()),
        "excluded_no_targets": int((~valid_targets).sum()),
    }


def prepare_extrasensory(
    config: dict[str, Any], project_root: str | Path, fold: int = 0
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    manifest_dir = root / config.get("manifest_dir", "data/manifests")
    schema_path = manifest_dir / "extrasensory_schema_manifest.json"
    split_path = manifest_dir / f"extrasensory_split_fold{fold}.json"
    data_manifest_path = manifest_dir / "extrasensory_data_manifest.json"
    if not split_path.is_file():
        split_path = _materialize_audited_split_manifest(config, root, fold)
    for required in (schema_path, split_path, data_manifest_path):
        if not required.is_file():
            raise FileNotFoundError(
                f"Missing Phase-1 artifact {required}; run the ExtraSensory audit first"
            )

    schema = read_json(schema_path)
    split = read_json(split_path)
    modality_columns = {name: schema["modalities"][name] for name in MODALITIES}
    feature_columns = [column for name in MODALITIES for column in modality_columns[name]]
    modality_slices: dict[str, tuple[int, int]] = {}
    start = 0
    for name in MODALITIES:
        end = start + len(modality_columns[name])
        modality_slices[name] = (start, end)
        start = end
    label_prefix = config.get("label_prefix", "label:")
    label_columns = [f"{label_prefix}{name}" for name in schema["selected_labels"]]
    user_files = discover_user_files(
        root / config.get("extracted_dir", "data/extracted") / "features"
    )

    expected_split_users = set(split["train"]) | set(split["val"]) | set(split["test"])
    if expected_split_users != set(user_files):
        raise ValueError("Processed split users do not match the audited feature files")
    if split["train_user_sha256"] != user_list_sha256(split["train"]):
        raise ValueError("Training user-list hash does not match the Phase-1 split manifest")

    output_dir = (
        root / config.get("processed_dir", "data/processed") / "extrasensory" / f"fold{fold}"
    )
    train_raw = _load_raw_users(
        split["train"],
        user_files,
        feature_columns,
        label_columns,
        config.get("timestamp_column", "timestamp"),
    )
    preprocessing = config.get("preprocessing", {})
    preprocessor = RobustPreprocessor.fit(
        train_raw["features"],
        feature_columns,
        split["train"],
        outlier_threshold=float(preprocessing.get("outlier_threshold", 4.0)),
        clip_value=float(preprocessing.get("clip_value", 20.0)),
        minimum_observed_fraction=float(
            preprocessing.get("minimum_observed_fraction", 0.01)
        ),
    )
    arrays_path, metadata_path = preprocessor.save(output_dir, split["train"])

    split_counts: dict[str, dict[str, int]] = {}
    split_files: dict[str, dict[str, Any]] = {}
    for split_name in ("train", "val", "test"):
        raw = (
            train_raw
            if split_name == "train"
            else _load_raw_users(
                split[split_name],
                user_files,
                feature_columns,
                label_columns,
                config.get("timestamp_column", "timestamp"),
            )
        )
        output_path = output_dir / f"{split_name}.npz"
        split_counts[split_name] = _save_processed_split(
            output_path, raw, preprocessor, modality_slices
        )
        split_files[split_name] = {
            "path": output_path.name,
            "bytes": output_path.stat().st_size,
            "sha256": sha256_file(output_path),
            "user_count": len(split[split_name]),
            **split_counts[split_name],
        }

    processed_manifest = {
        "version": 1,
        "dataset": "ExtraSensory",
        "fold": fold,
        "source_schema_sha256": sha256_file(schema_path),
        "source_data_manifest_sha256": sha256_file(data_manifest_path),
        "source_split_manifest_sha256": sha256_file(split_path),
        "preprocessor_arrays_sha256": sha256_file(arrays_path),
        "preprocessor_metadata_sha256": sha256_file(metadata_path),
        "fit_split": "train",
        "fit_user_sha256": preprocessor.fit_user_sha256,
        "labels": schema["selected_labels"],
        "quality_feature_names": [
            "availability",
            "observed_fraction",
            "outlier_fraction",
            "mean_abs_robust_z",
            "max_abs_robust_z",
        ],
        "modality_slices": {name: list(values) for name, values in modality_slices.items()},
        "feature_count": len(feature_columns),
        "splits": split_files,
    }
    processed_manifest_path = output_dir / "processed_manifest.json"
    write_json(processed_manifest_path, processed_manifest)
    return {
        "status": "passed",
        "fold": fold,
        "output_dir": str(output_dir),
        "processed_manifest": str(processed_manifest_path),
        "preprocessor": str(metadata_path),
        "split_counts": split_counts,
    }
