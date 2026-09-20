import csv
import gzip
import io
import zipfile
from pathlib import Path

import pytest

from robustsense.constants import LABELS, MODALITIES
from robustsense.data.extrasensory import (
    DataAuditError,
    audit_user_files,
    build_feature_schema,
    parse_official_folds,
    read_csv_header,
    safe_extract_zip,
)

USERS = [f"00000000-0000-0000-0000-{index:012d}" for index in range(10)]
FEATURES = (
    "raw_acc:mean",
    "proc_gyro:mean",
    "watch_acceleration:mean",
    "location:variance",
    "audio_naive:mfcc0:mean",
    "discrete:screen_on",
    "raw_magnet:mean",
    "watch_heading:mean",
    "lf_measurements:battery_level",
)


def phase1_config() -> dict:
    return {
        "timestamp_column": "timestamp",
        "label_prefix": "label:",
        "metadata_columns": ["label_source"],
        "labels": list(LABELS),
        "modalities": {
            "phone_acc": ["^raw_acc:"],
            "phone_gyro": ["^proc_gyro:"],
            "watch_acc": ["^watch_acceleration:"],
            "location": ["^location:"],
            "audio": ["^audio_naive:"],
            "phone_state": ["^discrete:"],
        },
        "ignored_feature_patterns": [
            "^raw_magnet:",
            "^watch_heading:",
            "^lf_measurements:",
        ],
        "expected_user_count": 1,
        "expected_sample_count_range": [1, 10],
    }


def csv_bytes(rows: list[list[object]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["timestamp", *FEATURES, *(f"label:{name}" for name in LABELS), "label_source"])
    writer.writerows(rows)
    return gzip.compress(stream.getvalue().encode("utf-8"))


def valid_row(timestamp: int) -> list[object]:
    labels = [1 if index % 2 == 0 else 0 for index in range(len(LABELS))]
    return [timestamp, *([0.5] * len(FEATURES)), *labels, 0]


def write_user_file(path: Path, rows: list[list[object]]) -> None:
    path.write_bytes(csv_bytes(rows))


def test_safe_extract_rejects_zip_slip(tmp_path: Path):
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.txt", "no")
    with pytest.raises(DataAuditError, match="Unsafe ZIP member"):
        safe_extract_zip(archive, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_schema_is_exhaustive_and_exclusive(tmp_path: Path):
    data = tmp_path / f"{USERS[0]}.features_labels.csv.gz"
    write_user_file(data, [valid_row(1)])
    schema = build_feature_schema(read_csv_header(data), phase1_config())

    assert set(schema.modality_columns) == set(MODALITIES)
    assert sum(map(len, schema.modality_columns.values())) == 6
    assert len(schema.ignored_features) == 3


def test_schema_rejects_unmapped_feature(tmp_path: Path):
    config = phase1_config()
    config["ignored_feature_patterns"] = []
    data = tmp_path / f"{USERS[0]}.features_labels.csv.gz"
    write_user_file(data, [valid_row(1)])

    with pytest.raises(DataAuditError, match="unmapped"):
        build_feature_schema(read_csv_header(data), config)


def test_schema_rejects_multiple_assignments(tmp_path: Path):
    config = phase1_config()
    config["modalities"]["phone_gyro"] = ["^raw_acc:", "^proc_gyro:"]
    data = tmp_path / f"{USERS[0]}.features_labels.csv.gz"
    write_user_file(data, [valid_row(1)])

    with pytest.raises(DataAuditError, match="multiply_mapped"):
        build_feature_schema(read_csv_header(data), config)


def test_row_audit_rejects_illegal_label(tmp_path: Path):
    row = valid_row(1)
    row[len(FEATURES) + 1] = 2
    data = tmp_path / f"{USERS[0]}.features_labels.csv.gz"
    write_user_file(data, [row])
    schema = build_feature_schema(read_csv_header(data), phase1_config())

    with pytest.raises(DataAuditError, match="Illegal label value"):
        audit_user_files(
            {USERS[0]: data},
            schema,
            {0: {"train": [], "test": [USERS[0]]}},
            phase1_config(),
        )


def test_row_audit_rejects_duplicate_timestamp(tmp_path: Path):
    data = tmp_path / f"{USERS[0]}.features_labels.csv.gz"
    write_user_file(data, [valid_row(1), valid_row(1)])
    schema = build_feature_schema(read_csv_header(data), phase1_config())

    with pytest.raises(DataAuditError, match=r"Duplicate \(user, timestamp\)"):
        audit_user_files(
            {USERS[0]: data},
            schema,
            {0: {"train": [], "test": [USERS[0]]}},
            phase1_config(),
        )


def test_fold_parser_rejects_empty_component(tmp_path: Path):
    for fold in range(5):
        for split in ("train", "test"):
            for platform in ("android", "iphone"):
                content = "" if (fold, split, platform) == (2, "test", "iphone") else USERS[0]
                (tmp_path / f"fold_{fold}_{split}_{platform}_uuids.txt").write_text(
                    content, encoding="utf-8"
                )

    with pytest.raises(DataAuditError, match="Empty official fold component"):
        parse_official_folds(tmp_path)
