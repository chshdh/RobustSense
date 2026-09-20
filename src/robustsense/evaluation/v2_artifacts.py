"""Deterministic, run-traceable artifact writers for Phase V2-2."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if value != value:
            return "NaN"
        return format(value, ".12g")
    return value


def stable_csv_bytes(rows: list[dict[str, Any]], columns: list[str]) -> bytes:
    if len(columns) != len(set(columns)) or not columns:
        raise ValueError("CSV columns must be unique and non-empty")
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: _csv_value(row.get(column)) for column in columns})
    return buffer.getvalue().encode("utf-8")


def write_stable_csv(
    path: str | Path, rows: list[dict[str, Any]], columns: list[str]
) -> dict[str, Any]:
    payload = stable_csv_bytes(rows, columns)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(payload)
    return {
        "path": output.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "row_count": len(rows),
    }


def write_artifact_lineage(
    path: str | Path,
    artifacts: list[dict[str, Any]],
    *,
    protocol_id: str,
) -> dict[str, Any]:
    normalized = []
    for artifact in artifacts:
        run_ids = sorted(set(artifact.get("source_run_ids", [])))
        if not artifact.get("artifact_path") or not run_ids:
            raise ValueError("Every artifact must have a path and at least one source run ID")
        normalized.append({**artifact, "source_run_ids": run_ids})
    payload = {
        "protocol_id": protocol_id,
        "artifacts": sorted(normalized, key=lambda item: item["artifact_path"]),
    }
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(encoded)
    return {
        "path": output.as_posix(),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "artifact_count": len(normalized),
    }
