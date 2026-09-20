"""Strict extraction, schema discovery, and auditing for ExtraSensory."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd

from robustsense.constants import LABELS, MODALITIES
from robustsense.data.splits import assert_disjoint_split
from robustsense.utils.io import read_json, write_json


class DataAuditError(ValueError):
    """Raised when source data violates a Phase-1 contract."""


@dataclass(frozen=True)
class FeatureSchema:
    header: tuple[str, ...]
    timestamp_column: str
    feature_columns: tuple[str, ...]
    label_columns: tuple[str, ...]
    selected_label_columns: tuple[str, ...]
    metadata_columns: tuple[str, ...]
    modality_columns: dict[str, tuple[str, ...]]
    ignored_features: tuple[str, ...]


@dataclass(frozen=True)
class ExtractionResult:
    destination: Path
    archive_sha256: str
    archive_bytes: int
    member_count: int
    skipped: bool


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_member_path(name: str) -> PurePosixPath:
    normalized = name.replace("\\", "/")
    member = PurePosixPath(normalized)
    if (
        not normalized
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
        or any(":" in part for part in member.parts)
        or ".." in member.parts
    ):
        raise DataAuditError(f"Unsafe ZIP member path: {name!r}")
    return member


def safe_extract_zip(archive: str | Path, destination: str | Path) -> list[Path]:
    """Extract a ZIP only after validating every member against Zip Slip and links."""
    archive_path = Path(archive)
    destination_path = Path(destination).resolve()
    destination_path.mkdir(parents=True, exist_ok=True)
    extracted: list[Path] = []
    seen: set[str] = set()

    try:
        with zipfile.ZipFile(archive_path) as bundle:
            validated: list[tuple[zipfile.ZipInfo, Path]] = []
            for info in bundle.infolist():
                member = _safe_member_path(info.filename)
                unix_mode = info.external_attr >> 16
                if stat.S_ISLNK(unix_mode):
                    raise DataAuditError(f"ZIP symbolic links are not allowed: {info.filename}")
                target = destination_path.joinpath(*member.parts).resolve()
                if os.path.commonpath([destination_path, target]) != str(destination_path):
                    raise DataAuditError(f"ZIP member escapes destination: {info.filename}")
                key = os.path.normcase(str(target))
                if key in seen:
                    raise DataAuditError(f"Duplicate ZIP destination: {info.filename}")
                seen.add(key)
                validated.append((info, target))

            for info, target in validated:
                if info.is_dir() or info.filename.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("wb") as output:
                    shutil.copyfileobj(source, output)
                extracted.append(target)
    except zipfile.BadZipFile as exc:
        raise DataAuditError(f"Invalid or incomplete ZIP archive: {archive_path}") from exc
    return extracted


def extract_archive_verified(archive: str | Path, destination: str | Path) -> ExtractionResult:
    """Extract atomically and reuse output only when its local archive hash matches."""
    archive_path = Path(archive).resolve()
    destination_path = Path(destination).resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Required archive is missing: {archive_path}")
    archive_hash = sha256_file(archive_path)
    marker_name = ".robustsense-extraction.json"
    marker_path = destination_path / marker_name
    if destination_path.exists():
        if marker_path.is_file():
            marker = read_json(marker_path)
            if marker.get("archive_sha256") == archive_hash:
                return ExtractionResult(
                    destination_path,
                    archive_hash,
                    archive_path.stat().st_size,
                    int(marker["member_count"]),
                    True,
                )
        if any(destination_path.iterdir()):
            raise DataAuditError(
                f"Extraction directory is non-empty but does not match the archive: "
                f"{destination_path}"
            )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination_path.name}-partial-", dir=destination_path.parent
    ) as temporary:
        staging = Path(temporary) / destination_path.name
        members = safe_extract_zip(archive_path, staging)
        write_json(
            staging / marker_name,
            {
                "archive": archive_path.name,
                "archive_bytes": archive_path.stat().st_size,
                "archive_sha256": archive_hash,
                "checksum_scope": "local_integrity_only_not_official",
                "member_count": len(members),
            },
        )
        if destination_path.exists():
            destination_path.rmdir()
        staging.replace(destination_path)
    return ExtractionResult(
        destination_path,
        archive_hash,
        archive_path.stat().st_size,
        len(members),
        False,
    )


def canonical_uuid(value: str) -> str:
    try:
        return str(UUID(value.strip())).upper()
    except ValueError as exc:
        raise DataAuditError(f"Invalid user UUID: {value!r}") from exc


def uuid_from_feature_path(path: str | Path) -> str:
    suffix = ".features_labels.csv.gz"
    name = Path(path).name
    if not name.endswith(suffix):
        raise DataAuditError(f"Unexpected ExtraSensory filename: {name}")
    return canonical_uuid(name[: -len(suffix)])


def discover_user_files(feature_root: str | Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(Path(feature_root).rglob("*.features_labels.csv.gz")):
        user = uuid_from_feature_path(path)
        if user in files:
            raise DataAuditError(f"Duplicate feature file for user {user}")
        files[user] = path
    if not files:
        raise DataAuditError(f"No per-user csv.gz files found under {feature_root}")
    return files


def read_csv_header(path: str | Path) -> tuple[str, ...]:
    try:
        with gzip.open(path, "rt", encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
    except (OSError, StopIteration) as exc:
        raise DataAuditError(f"Cannot read CSV header from {path}") from exc
    if not header:
        raise DataAuditError(f"Empty CSV header: {path}")
    duplicates = sorted({name for name in header if header.count(name) > 1})
    if duplicates:
        raise DataAuditError(f"Duplicate header columns in {path}: {duplicates}")
    return tuple(header)


def inspect_shared_header(user_files: dict[str, Path]) -> tuple[str, ...]:
    reference_user, reference_path = next(iter(user_files.items()))
    reference = read_csv_header(reference_path)
    for user, path in user_files.items():
        header = read_csv_header(path)
        if header != reference:
            raise DataAuditError(
                f"Schema conflict between users {reference_user} and {user}: "
                f"{reference_path.name} != {path.name}"
            )
    return reference


def build_feature_schema(header: tuple[str, ...], config: dict[str, Any]) -> FeatureSchema:
    timestamp = str(config.get("timestamp_column", "timestamp"))
    label_prefix = str(config.get("label_prefix", "label:"))
    metadata = tuple(str(value) for value in config.get("metadata_columns", ["label_source"]))
    if header.count(timestamp) != 1:
        raise DataAuditError(f"Expected exactly one timestamp column {timestamp!r}")
    missing_metadata = sorted(set(metadata) - set(header))
    if missing_metadata:
        raise DataAuditError(f"Missing metadata columns: {missing_metadata}")

    label_columns = tuple(column for column in header if column.startswith(label_prefix))
    if not label_columns:
        raise DataAuditError(f"No label columns use prefix {label_prefix!r}")
    selected_labels = tuple(str(value) for value in config.get("labels", LABELS))
    selected_columns = tuple(f"{label_prefix}{label}" for label in selected_labels)
    missing_labels = sorted(set(selected_columns) - set(label_columns))
    if missing_labels:
        raise DataAuditError(f"Configured labels missing from header: {missing_labels}")

    excluded = {timestamp, *metadata, *label_columns}
    feature_columns = tuple(column for column in header if column not in excluded)
    if not feature_columns:
        raise DataAuditError("No feature columns found")

    configured_modalities = config.get("modalities", {})
    if set(configured_modalities) != set(MODALITIES):
        raise DataAuditError(
            f"Modality config must define exactly {list(MODALITIES)}; "
            f"got {sorted(configured_modalities)}"
        )
    modality_patterns = {
        modality: tuple(re.compile(pattern) for pattern in configured_modalities[modality])
        for modality in MODALITIES
    }
    ignored_patterns = tuple(
        re.compile(pattern) for pattern in config.get("ignored_feature_patterns", [])
    )
    assignments: dict[str, list[str]] = {modality: [] for modality in MODALITIES}
    ignored: list[str] = []
    unmapped: list[str] = []
    conflicts: dict[str, list[str]] = {}

    for column in feature_columns:
        matches = [
            modality
            for modality, patterns in modality_patterns.items()
            if any(pattern.search(column) for pattern in patterns)
        ]
        ignored_match = any(pattern.search(column) for pattern in ignored_patterns)
        if len(matches) == 1 and not ignored_match:
            assignments[matches[0]].append(column)
        elif not matches and ignored_match:
            ignored.append(column)
        elif not matches:
            unmapped.append(column)
        else:
            conflicts[column] = matches + (["ignored"] if ignored_match else [])

    if unmapped or conflicts:
        details: list[str] = []
        if unmapped:
            details.append(f"unmapped={unmapped}")
        if conflicts:
            details.append(f"multiply_mapped={conflicts}")
        raise DataAuditError(
            "Feature schema is not exhaustive and exclusive: " + "; ".join(details)
        )
    empty_modalities = [name for name, columns in assignments.items() if not columns]
    if empty_modalities:
        raise DataAuditError(f"Modalities with zero feature columns: {empty_modalities}")

    return FeatureSchema(
        header=header,
        timestamp_column=timestamp,
        feature_columns=feature_columns,
        label_columns=label_columns,
        selected_label_columns=selected_columns,
        metadata_columns=metadata,
        modality_columns={name: tuple(values) for name, values in assignments.items()},
        ignored_features=tuple(ignored),
    )


_FOLD_FILE = re.compile(r"fold_(\d+)_(train|test)_(android|iphone)_uuids\.txt$")


def parse_official_folds(
    fold_root: str | Path, n_folds: int = 5
) -> dict[int, dict[str, list[str]]]:
    components: dict[tuple[int, str, str], list[str]] = {}
    for path in Path(fold_root).rglob("*.txt"):
        match = _FOLD_FILE.fullmatch(path.name)
        if not match:
            continue
        fold, split, platform = int(match.group(1)), match.group(2), match.group(3)
        key = (fold, split, platform)
        if key in components:
            raise DataAuditError(f"Duplicate official fold component: {path.name}")
        users = [
            canonical_uuid(line)
            for line in path.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        if not users:
            raise DataAuditError(f"Empty official fold component: {path.name}")
        if len(users) != len(set(users)):
            raise DataAuditError(f"Duplicate user in official fold component: {path.name}")
        components[key] = users

    expected_keys = {
        (fold, split, platform)
        for fold in range(n_folds)
        for split in ("train", "test")
        for platform in ("android", "iphone")
    }
    missing = sorted(expected_keys - set(components))
    unexpected = sorted(set(components) - expected_keys)
    if missing or unexpected:
        raise DataAuditError(
            f"Invalid official fold files: missing={missing}, unexpected={unexpected}"
        )

    folds: dict[int, dict[str, list[str]]] = {}
    test_sets: list[set[str]] = []
    for fold in range(n_folds):
        test = set(components[(fold, "test", "android")]) | set(
            components[(fold, "test", "iphone")]
        )
        train = set(components[(fold, "train", "android")]) | set(
            components[(fold, "train", "iphone")]
        )
        if not test or not train:
            raise DataAuditError(f"Official fold {fold} has an empty train or test set")
        if test & train:
            raise DataAuditError(f"Official fold {fold} train/test users overlap")
        folds[fold] = {"train": sorted(train), "test": sorted(test)}
        test_sets.append(test)

    for left in range(n_folds):
        for right in range(left + 1, n_folds):
            overlap = test_sets[left] & test_sets[right]
            if overlap:
                raise DataAuditError(
                    f"Official test folds {left}/{right} overlap: {sorted(overlap)}"
                )
    all_users = set().union(*test_sets)
    for fold in range(n_folds):
        if set(folds[fold]["train"]) != all_users - set(folds[fold]["test"]):
            raise DataAuditError(f"Official fold {fold} training list is not the test complement")
    return folds


def split_from_official_folds(
    folds: dict[int, dict[str, list[str]]], test_fold: int
) -> dict[str, list[str] | int]:
    if test_fold not in folds:
        raise DataAuditError(f"Unknown outer fold {test_fold}")
    validation_fold = (test_fold + 1) % len(folds)
    test = set(folds[test_fold]["test"])
    val = set(folds[validation_fold]["test"])
    train = set(folds[test_fold]["train"]) - val
    split = {"train": sorted(train), "val": sorted(val), "test": sorted(test)}
    assert_disjoint_split(split)
    return {"test_fold": test_fold, "validation_fold": validation_fold, **split}


def _user_list_sha256(users: list[str]) -> str:
    return hashlib.sha256(("\n".join(users) + "\n").encode()).hexdigest()


def _schema_payload(schema: FeatureSchema, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "header": list(schema.header),
        "header_sha256": hashlib.sha256(
            (",".join(schema.header) + "\n").encode("utf-8")
        ).hexdigest(),
        "timestamp_column": schema.timestamp_column,
        "metadata_columns": list(schema.metadata_columns),
        "feature_count": len(schema.feature_columns),
        "label_count": len(schema.label_columns),
        "selected_labels": [
            column.removeprefix(config.get("label_prefix", "label:"))
            for column in schema.selected_label_columns
        ],
        "modalities": {name: list(schema.modality_columns[name]) for name in MODALITIES},
        "ignored_features": list(schema.ignored_features),
    }


def _save_heatmap(matrix: np.ndarray, labels: list[str], path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    size = max(6.0, min(12.0, len(labels) * 0.62))
    figure, axis = plt.subplots(figsize=(size, size))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(len(labels)), labels=labels, rotation=90)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.set_title(title)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _validate_numeric_frame(raw: pd.DataFrame, numeric: pd.DataFrame, kind: str, user: str) -> None:
    invalid = raw.notna() & numeric.isna()
    if invalid.to_numpy().any():
        row, column = np.argwhere(invalid.to_numpy())[0]
        value = raw.iloc[int(row), int(column)]
        raise DataAuditError(
            f"Non-numeric {kind} value for user {user}, column {raw.columns[int(column)]}: "
            f"{value!r}"
        )


def audit_user_files(
    user_files: dict[str, Path],
    schema: FeatureSchema,
    folds: dict[int, dict[str, list[str]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    label_names = [
        column.removeprefix(config.get("label_prefix", "label:"))
        for column in schema.selected_label_columns
    ]
    user_to_fold = {
        user: fold for fold, values in folds.items() for user in values["test"]
    }
    data_users = set(user_files)
    fold_users = set(user_to_fold)
    if data_users != fold_users:
        raise DataAuditError(
            f"Feature/fold user mismatch: missing_files={sorted(fold_users - data_users)}, "
            f"unassigned_files={sorted(data_users - fold_users)}"
        )

    feature_count = len(schema.feature_columns)
    label_count = len(schema.selected_label_columns)
    feature_finite = np.zeros(feature_count, dtype=np.int64)
    label_known = np.zeros(label_count, dtype=np.int64)
    label_positive = np.zeros(label_count, dtype=np.int64)
    label_positive_users: list[set[str]] = [set() for _ in range(label_count)]
    fold_label_positive_users: list[list[set[str]]] = [
        [set() for _ in range(label_count)] for _ in folds
    ]
    fold_known = np.zeros((len(folds), label_count), dtype=np.int64)
    fold_positive = np.zeros((len(folds), label_count), dtype=np.int64)
    modality_available = np.zeros(len(MODALITIES), dtype=np.int64)
    modality_coavailability = np.zeros((len(MODALITIES), len(MODALITIES)), dtype=np.int64)
    label_cooccurrence = np.zeros((label_count, label_count), dtype=np.int64)
    modality_indices = {
        modality: [
            schema.feature_columns.index(column)
            for column in schema.modality_columns[modality]
        ]
        for modality in MODALITIES
    }
    sample_count = 0
    excluded_no_modality = 0
    excluded_no_targets = 0
    excluded_total = 0
    user_rows: list[dict[str, Any]] = []
    file_manifest: list[dict[str, Any]] = []
    chunk_size = int(config.get("audit_chunk_size", 25000))

    for user, path in sorted(user_files.items()):
        user_sample_count = 0
        user_no_modality = 0
        user_no_targets = 0
        user_excluded = 0
        timestamps_seen: set[int] = set()
        timestamp_min: int | None = None
        timestamp_max: int | None = None
        previous_timestamp: int | None = None
        timestamps_sorted = True
        fold = user_to_fold[user]

        for chunk in pd.read_csv(path, chunksize=chunk_size, low_memory=False):
            if tuple(chunk.columns) != schema.header:
                raise DataAuditError(f"Schema changed while reading {path}")
            timestamps_numeric = pd.to_numeric(chunk[schema.timestamp_column], errors="coerce")
            if timestamps_numeric.isna().any() or not np.isfinite(timestamps_numeric).all():
                raise DataAuditError(f"Invalid timestamp for user {user}")
            timestamp_values = timestamps_numeric.to_numpy(dtype=np.float64)
            if not np.equal(timestamp_values, np.floor(timestamp_values)).all():
                raise DataAuditError(f"Non-integer timestamp for user {user}")
            integer_timestamps = timestamp_values.astype(np.int64)
            duplicated = set(integer_timestamps) & timestamps_seen
            if len(integer_timestamps) != len(set(integer_timestamps)) or duplicated:
                raise DataAuditError(f"Duplicate (user, timestamp) row for user {user}")
            timestamps_seen.update(int(value) for value in integer_timestamps)
            if previous_timestamp is not None and integer_timestamps[0] < previous_timestamp:
                timestamps_sorted = False
            if len(integer_timestamps) > 1 and np.any(np.diff(integer_timestamps) < 0):
                timestamps_sorted = False
            previous_timestamp = int(integer_timestamps[-1])
            current_min = int(integer_timestamps.min())
            current_max = int(integer_timestamps.max())
            timestamp_min = (
                current_min if timestamp_min is None else min(timestamp_min, current_min)
            )
            timestamp_max = (
                current_max if timestamp_max is None else max(timestamp_max, current_max)
            )

            raw_features = chunk.loc[:, schema.feature_columns]
            numeric_features = raw_features.apply(pd.to_numeric, errors="coerce")
            _validate_numeric_frame(raw_features, numeric_features, "feature", user)
            finite = np.isfinite(numeric_features.to_numpy(dtype=np.float64))
            feature_finite += finite.sum(axis=0)

            raw_labels = chunk.loc[:, schema.label_columns]
            numeric_labels = raw_labels.apply(pd.to_numeric, errors="coerce")
            _validate_numeric_frame(raw_labels, numeric_labels, "label", user)
            all_label_values = numeric_labels.to_numpy(dtype=np.float64)
            invalid_values = np.isfinite(all_label_values) & ~np.isin(all_label_values, [0.0, 1.0])
            if invalid_values.any():
                row, column = np.argwhere(invalid_values)[0]
                raise DataAuditError(
                    f"Illegal label value for user {user}, column "
                    f"{schema.label_columns[int(column)]}: "
                    f"{all_label_values[int(row), int(column)]}"
                )

            selected_values = numeric_labels.loc[:, schema.selected_label_columns].to_numpy(
                dtype=np.float64
            )
            known = np.isfinite(selected_values)
            positive = known & (selected_values == 1.0)
            label_known += known.sum(axis=0)
            label_positive += positive.sum(axis=0)
            fold_known[fold] += known.sum(axis=0)
            fold_positive[fold] += positive.sum(axis=0)
            label_cooccurrence += positive.astype(np.int64).T @ positive.astype(np.int64)
            for index in np.flatnonzero(positive.any(axis=0)):
                label_positive_users[int(index)].add(user)
                fold_label_positive_users[fold][int(index)].add(user)

            availability = np.column_stack(
                [finite[:, modality_indices[modality]].any(axis=1) for modality in MODALITIES]
            )
            modality_available += availability.sum(axis=0)
            modality_coavailability += availability.astype(np.int64).T @ availability.astype(
                np.int64
            )
            no_modality = ~availability.any(axis=1)
            no_targets = ~known.any(axis=1)
            excluded = no_modality | no_targets
            user_no_modality += int(no_modality.sum())
            user_no_targets += int(no_targets.sum())
            user_excluded += int(excluded.sum())
            user_sample_count += len(chunk)

        if user_sample_count == 0:
            raise DataAuditError(f"User file has no data rows: {path}")
        sample_count += user_sample_count
        excluded_no_modality += user_no_modality
        excluded_no_targets += user_no_targets
        excluded_total += user_excluded
        user_rows.append(
            {
                "user_id": user,
                "fold": fold,
                "sample_count": user_sample_count,
                "timestamp_min": timestamp_min,
                "timestamp_max": timestamp_max,
                "timestamps_sorted": timestamps_sorted,
                "duplicate_timestamp_count": 0,
                "all_modalities_unavailable_rows": user_no_modality,
                "all_selected_labels_unknown_rows": user_no_targets,
                "excluded_rows": user_excluded,
            }
        )
        file_manifest.append(
            {
                "user_id": user,
                "fold": fold,
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "sample_count": user_sample_count,
            }
        )

    expected_users = int(config.get("expected_user_count", len(user_files)))
    sample_range = config.get("expected_sample_count_range", [1, 2**63 - 1])
    if len(user_files) != expected_users:
        raise DataAuditError(f"Expected {expected_users} users, found {len(user_files)}")
    if not int(sample_range[0]) <= sample_count <= int(sample_range[1]):
        raise DataAuditError(
            f"Sample count {sample_count} outside expected range "
            f"[{sample_range[0]}, {sample_range[1]}]"
        )

    return {
        "sample_count": sample_count,
        "eligible_sample_count": sample_count - excluded_total,
        "excluded_total": excluded_total,
        "excluded_no_modality": excluded_no_modality,
        "excluded_no_targets": excluded_no_targets,
        "feature_finite": feature_finite,
        "label_known": label_known,
        "label_positive": label_positive,
        "label_positive_users": label_positive_users,
        "fold_label_positive_users": fold_label_positive_users,
        "fold_known": fold_known,
        "fold_positive": fold_positive,
        "modality_available": modality_available,
        "modality_coavailability": modality_coavailability,
        "label_cooccurrence": label_cooccurrence,
        "label_names": label_names,
        "user_rows": user_rows,
        "file_manifest": file_manifest,
    }


def write_audit_artifacts(
    report_dir: Path,
    manifest_dir: Path,
    schema: FeatureSchema,
    stats: dict[str, Any],
    split: dict[str, list[str] | int],
    config: dict[str, Any],
    archive_records: list[dict[str, Any]],
) -> dict[str, str]:
    report_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    sample_count = int(stats["sample_count"])
    feature_missing = sample_count - stats["feature_finite"]
    summary = {
        "status": "passed",
        "dataset": "ExtraSensory",
        "user_count": len(stats["user_rows"]),
        "sample_count": sample_count,
        "eligible_sample_count": int(stats["eligible_sample_count"]),
        "excluded_row_count": int(stats["excluded_total"]),
        "excluded_all_modalities_unavailable": int(stats["excluded_no_modality"]),
        "excluded_all_selected_labels_unknown": int(stats["excluded_no_targets"]),
        "column_count": len(schema.header),
        "feature_count": len(schema.feature_columns),
        "selected_feature_count": sum(len(values) for values in schema.modality_columns.values()),
        "ignored_feature_count": len(schema.ignored_features),
        "label_count": len(schema.label_columns),
        "selected_label_count": len(schema.selected_label_columns),
        "label_value_contract": "0/1/NaN validated for every label column",
        "timestamp_key_contract": "(user_id, timestamp) unique",
        "official_user_count_expected": int(config.get("expected_user_count", 60)),
        "expected_sample_count_range": config.get("expected_sample_count_range"),
        "modality_feature_counts": {
            name: len(schema.modality_columns[name]) for name in MODALITIES
        },
    }
    summary_path = report_dir / "dataset_summary.json"
    write_json(summary_path, summary)

    feature_rows: list[dict[str, Any]] = []
    feature_index = {column: index for index, column in enumerate(schema.feature_columns)}
    modality_for = {
        column: modality
        for modality, columns in schema.modality_columns.items()
        for column in columns
    }
    for column in schema.feature_columns:
        index = feature_index[column]
        finite_count = int(stats["feature_finite"][index])
        feature_rows.append(
            {
                "column": column,
                "role": "ignored_feature" if column in schema.ignored_features else "feature",
                "modality": modality_for.get(column, ""),
                "finite_count": finite_count,
                "missing_or_nonfinite_count": int(feature_missing[index]),
                "missing_or_nonfinite_fraction": float(feature_missing[index] / sample_count),
            }
        )
    pd.DataFrame(feature_rows).to_csv(report_dir / "feature_schema.csv", index=False)

    missingness_rows: list[dict[str, Any]] = []
    for modality_index, modality in enumerate(MODALITIES):
        indices = [feature_index[column] for column in schema.modality_columns[modality]]
        total_cells = sample_count * len(indices)
        finite_cells = int(stats["feature_finite"][indices].sum())
        available_samples = int(stats["modality_available"][modality_index])
        missingness_rows.append(
            {
                "modality": modality,
                "feature_count": len(indices),
                "finite_feature_cells": finite_cells,
                "missing_or_nonfinite_feature_cells": total_cells - finite_cells,
                "cell_missing_fraction": (total_cells - finite_cells) / total_cells,
                "available_samples": available_samples,
                "unavailable_samples": sample_count - available_samples,
                "unavailable_fraction": (sample_count - available_samples) / sample_count,
            }
        )
    pd.DataFrame(missingness_rows).to_csv(report_dir / "modality_missingness.csv", index=False)

    label_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    for label_index, label in enumerate(stats["label_names"]):
        known = int(stats["label_known"][label_index])
        positive = int(stats["label_positive"][label_index])
        label_rows.append(
            {
                "label": label,
                "known_samples": known,
                "positive_samples": positive,
                "negative_samples": known - positive,
                "positive_fraction": positive / known if known else np.nan,
                "positive_users": len(stats["label_positive_users"][label_index]),
            }
        )
        for fold in range(stats["fold_known"].shape[0]):
            fold_known = int(stats["fold_known"][fold, label_index])
            fold_positive = int(stats["fold_positive"][fold, label_index])
            fold_rows.append(
                {
                    "fold": fold,
                    "label": label,
                    "known_samples": fold_known,
                    "positive_samples": fold_positive,
                    "negative_samples": fold_known - fold_positive,
                    "positive_fraction": fold_positive / fold_known if fold_known else np.nan,
                    "positive_users": len(stats["fold_label_positive_users"][fold][label_index]),
                    "fold_users": sum(row["fold"] == fold for row in stats["user_rows"]),
                }
            )
    pd.DataFrame(label_rows).to_csv(report_dir / "label_prevalence.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(report_dir / "label_by_fold.csv", index=False)
    pd.DataFrame(stats["user_rows"]).to_csv(report_dir / "user_sample_counts.csv", index=False)

    _save_heatmap(
        stats["modality_coavailability"],
        list(MODALITIES),
        report_dir / "modality_coavailability.png",
        "Modality co-availability (sample counts)",
    )
    _save_heatmap(
        stats["label_cooccurrence"],
        list(stats["label_names"]),
        report_dir / "label_cooccurrence.png",
        "Selected-label positive co-occurrence (sample counts)",
    )

    split_payload = dict(split)
    for name in ("train", "val", "test"):
        users = split_payload[name]
        assert isinstance(users, list)
        split_payload[f"{name}_user_sha256"] = _user_list_sha256(users)
    split_path = manifest_dir / f"extrasensory_split_fold{split['test_fold']}.json"
    write_json(split_path, split_payload)
    schema_payload = _schema_payload(schema, config)
    schema_path = manifest_dir / "extrasensory_schema_manifest.json"
    write_json(schema_path, schema_payload)
    data_manifest_path = manifest_dir / "extrasensory_data_manifest.json"
    write_json(
        data_manifest_path,
        {
            "dataset": "ExtraSensory",
            "archives": archive_records,
            "schema_manifest": schema_path.name,
            "user_count": len(stats["user_rows"]),
            "sample_count": sample_count,
            "files": stats["file_manifest"],
        },
    )

    zero_positive = [
        {"fold": row["fold"], "label": row["label"]}
        for row in fold_rows
        if row["positive_samples"] == 0
    ]
    report_path = report_dir / "audit_report.md"
    report_path.write_text(
        "# ExtraSensory 数据审计\n\n"
        "## 结果\n\n"
        "Phase 1 审计已通过全部结构检查。以下计数和缺失情况仅作描述；没有使用模型"
        "测试性能选择特征列或标签。\n\n"
        f"- 用户数：{len(stats['user_rows'])}\n"
        f"- 样本数：{sample_count}\n"
        f"- 总列数：{len(schema.header)}\n"
        f"- 输入特征：{len(schema.feature_columns)} 个"
        f"（选用 {sum(len(values) for values in schema.modality_columns.values())} 个，"
        f"明确忽略 {len(schema.ignored_features)} 个）\n"
        f"- 标签列：{len(schema.label_columns)} 个；选用标签："
        f"{len(schema.selected_label_columns)}\n"
        f"- 可进入后续预处理的记录：{stats['eligible_sample_count']}\n"
        f"- 因全部选用模态不可用而排除的记录："
        f"{stats['excluded_no_modality']}\n"
        f"- 因 15 个选用标签全部未知而排除的记录："
        f"{stats['excluded_no_targets']}\n"
        f"- 唯一键：`(user_id, timestamp)` 已验证；未按时间排序的用户文件："
        f"{sum(not row['timestamps_sorted'] for row in stats['user_rows'])}\n\n"
        "## 六模态映射\n\n"
        + "\n".join(
            f"- `{name}`：{len(schema.modality_columns[name])} 列"
            for name in MODALITIES
        )
        + "\n\n"
        "忽略前缀是六模态实验中的有意排除项，不是未映射列。全部列见 "
        "`feature_schema.csv`。\n\n"
        "## 标签与 fold 检查\n\n"
        "全部源标签列均已验证为 `{0, 1, NaN}`。官方测试 fold 两两互斥，官方训练列表"
        "是对应测试列表的补集，生成的训练/验证/测试用户集相互隔离。\n\n"
        f"正例数为零的 fold/标签单元：{json.dumps(zero_positive)}\n\n"
        "## 排除规则\n\n"
        "后续处理数据只能排除没有可用选定模态的记录，或 15 个选定目标全部未知的记录。"
        "上述计数可能重叠；`eligible_sample_count` 对并集只计算一次。\n",
        encoding="utf-8",
    )
    return {
        "dataset_summary": str(summary_path),
        "audit_report": str(report_path),
        "data_manifest": str(data_manifest_path),
        "schema_manifest": str(schema_path),
        "split_manifest": str(split_path),
    }


def run_extrasensory_audit(
    config: dict[str, Any], project_root: str | Path, fold: int = 0, probe_only: bool = False
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    raw_dir = root / config.get("raw_dir", "data/raw")
    extracted_dir = root / config.get("extracted_dir", "data/extracted")
    manifest_dir = root / config.get("manifest_dir", "data/manifests")
    report_dir = root / config.get("report_dir", "reports/data_audit")
    archives = [
        (
            raw_dir / config["feature_archive"],
            extracted_dir / "features",
            config.get("feature_source_url"),
        ),
        (
            raw_dir / config["fold_archive"],
            extracted_dir / "folds",
            config.get("fold_source_url"),
        ),
    ]
    archive_records: list[dict[str, Any]] = []
    for archive, destination, source_url in archives:
        result = extract_archive_verified(archive, destination)
        archive_records.append(
            {
                "filename": archive.name,
                "bytes": result.archive_bytes,
                "sha256": result.archive_sha256,
                "checksum_scope": "local_integrity_only_not_official",
                "source_url": source_url,
                "member_count": result.member_count,
            }
        )

    user_files = discover_user_files(extracted_dir / "features")
    folds = parse_official_folds(extracted_dir / "folds")
    header = inspect_shared_header(user_files)
    schema = build_feature_schema(header, config)
    split = split_from_official_folds(folds, fold)
    probe_payload = {
        "status": "schema_validated",
        "user_file_count": len(user_files),
        "fold_user_count": len({user for values in folds.values() for user in values["test"]}),
        "split_counts": {name: len(split[name]) for name in ("train", "val", "test")},
        **_schema_payload(schema, config),
    }
    interim_dir = root / config.get("interim_dir", "data/interim")
    probe_path = interim_dir / "extrasensory_schema_probe.json"
    write_json(probe_path, probe_payload)
    if probe_only:
        return {"probe_path": str(probe_path), **probe_payload}

    stats = audit_user_files(user_files, schema, folds, config)
    artifacts = write_audit_artifacts(
        report_dir, manifest_dir, schema, stats, split, config, archive_records
    )
    return {
        "status": "passed",
        "user_count": len(user_files),
        "sample_count": stats["sample_count"],
        "eligible_sample_count": stats["eligible_sample_count"],
        "probe_path": str(probe_path),
        **artifacts,
    }
