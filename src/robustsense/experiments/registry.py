"""Immutable Phase-5 protocol lock and append-preserving run registry."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.data.extrasensory import sha256_file
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

RUN_FIELDS = [
    "run_key",
    "group",
    "profile",
    "model_name",
    "fold",
    "seed",
    "status",
    "attempt_count",
    "created_at",
    "updated_at",
    "run_dir",
    "protocol_sha256",
    "command",
    "reason",
    "error_log",
]
ATTEMPT_FIELDS = [
    "run_key",
    "attempt",
    "status",
    "started_at",
    "finished_at",
    "return_code",
    "log_path",
    "reason",
]
TERMINAL_STATUSES = {"success", "failed", "skipped"}
ACTIVE_STATUSES = {"pending", "running", *TERMINAL_STATUSES}


class ProtocolMutationError(ValueError):
    """Raised when a frozen protocol input changed after lock creation."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _atomic_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(handle, "w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary_name, path)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def freeze_protocol(project_root: Path, plan_path: Path) -> dict[str, Any]:
    """Create or verify a hash lock before any Phase-5 test run is launched."""
    root = project_root.resolve()
    plan_file = plan_path if plan_path.is_absolute() else root / plan_path
    plan = load_config(plan_file)
    records = []
    for relative in plan["protocol_files"]:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Frozen protocol input is missing: {path}")
        records.append({"path": relative, "sha256": sha256_file(path)})
    payload = {
        "version": 1,
        "plan": str(plan_file.relative_to(root)),
        "frozen_on": plan["frozen_on"],
        "files": records,
    }
    payload["protocol_sha256"] = _canonical_sha256(payload)
    lock_path = root / "reports/phase5_protocol_lock.json"
    if lock_path.is_file():
        existing = read_json(lock_path)
        if existing != payload:
            old = {item["path"]: item["sha256"] for item in existing.get("files", [])}
            new = {item["path"]: item["sha256"] for item in records}
            changed = sorted(path for path in set(old) | set(new) if old.get(path) != new.get(path))
            raise ProtocolMutationError(
                "Frozen Phase-5 protocol changed; record an ADR and explicitly replace the "
                f"lock before running tests. Changed files: {changed}"
            )
        return existing
    write_json(lock_path, payload)
    return payload


def build_run_plan(project_root: Path, plan_path: Path, profile_name: str) -> list[dict[str, Any]]:
    root = project_root.resolve()
    plan_file = plan_path if plan_path.is_absolute() else root / plan_path
    plan = load_config(plan_file)
    if profile_name not in plan["profiles"]:
        raise ValueError(f"Profile is not registered in Phase-5 plan: {profile_name}")
    definition = plan["profiles"][profile_name]
    experiment = load_config(root / definition["experiment_config"])
    evaluation = load_config(root / definition["evaluation_config"])
    if experiment.get("profile") != profile_name:
        raise ValueError(f"Experiment profile mismatch for {profile_name}")
    if evaluation.get("profile") != profile_name:
        raise ValueError(f"Evaluation profile mismatch for {profile_name}")
    if experiment.get("folds") != evaluation.get("folds"):
        raise ValueError(f"Fold matrix mismatch for {profile_name}")
    if experiment.get("seeds") != evaluation.get("seeds"):
        raise ValueError(f"Seed matrix mismatch for {profile_name}")
    if definition["models"] != evaluation.get("models"):
        raise ValueError(f"Model matrix mismatch for {profile_name}")

    created = _now()
    rows = []
    for fold in experiment["folds"]:
        for seed in experiment["seeds"]:
            for model_name in definition["models"]:
                run_key = f"{profile_name}|{model_name}|fold{int(fold)}|seed{int(seed)}"
                rows.append(
                    {
                        "run_key": run_key,
                        "group": profile_name,
                        "profile": profile_name,
                        "model_name": model_name,
                        "fold": str(int(fold)),
                        "seed": str(int(seed)),
                        "status": "pending",
                        "attempt_count": "0",
                        "created_at": created,
                        "updated_at": created,
                        "run_dir": (
                            f"runs/extrasensory-{model_name}-fold{fold}-seed{seed}-"
                            f"{profile_name}"
                        ),
                        "protocol_sha256": "",
                        "command": "",
                        "reason": "registered_before_execution",
                        "error_log": "",
                    }
                )
    return rows


class RunRegistry:
    """Store the latest state plus an append-only attempt ledger."""

    def __init__(self, project_root: Path):
        self.root = project_root.resolve()
        self.path = self.root / "reports/run_registry.csv"
        self.attempt_path = self.root / "reports/run_attempts.csv"

    def rows(self) -> list[dict[str, str]]:
        if not self.path.is_file():
            return []
        with self.path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row.get("status") not in ACTIVE_STATUSES:
                raise ValueError(f"Invalid run-registry status: {row.get('status')}")
        return rows

    def attempts(self) -> list[dict[str, str]]:
        if not self.attempt_path.is_file():
            return []
        with self.attempt_path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))

    def register(self, planned: list[dict[str, Any]], protocol_sha256: str) -> list[dict[str, str]]:
        current = {row["run_key"]: row for row in self.rows()}
        for raw in planned:
            row = {name: str(raw.get(name, "")) for name in RUN_FIELDS}
            row["protocol_sha256"] = protocol_sha256
            existing = current.get(row["run_key"])
            if existing is not None:
                immutable = ("group", "profile", "model_name", "fold", "seed", "run_dir")
                changed = [name for name in immutable if existing[name] != row[name]]
                if changed or existing["protocol_sha256"] != protocol_sha256:
                    raise ProtocolMutationError(
                        f"Registered run definition changed for {row['run_key']}: {changed}"
                    )
                continue
            current[row["run_key"]] = row
        rows = sorted(current.values(), key=lambda item: item["run_key"])
        _atomic_csv(self.path, RUN_FIELDS, rows)
        return rows

    def recover_interrupted(self) -> int:
        rows = self.rows()
        recovered = 0
        for row in rows:
            if row["status"] == "running":
                row["status"] = "failed"
                row["updated_at"] = _now()
                row["reason"] = "previous_process_interrupted; use --retry-failed"
                recovered += 1
        if recovered:
            _atomic_csv(self.path, RUN_FIELDS, rows)
        return recovered

    def update(self, run_key: str, **values: Any) -> dict[str, str]:
        rows = self.rows()
        matches = [row for row in rows if row["run_key"] == run_key]
        if len(matches) != 1:
            raise KeyError(f"Expected one registered run for {run_key}")
        row = matches[0]
        if "status" in values and values["status"] not in ACTIVE_STATUSES:
            raise ValueError(f"Invalid run status: {values['status']}")
        for key, value in values.items():
            if key not in RUN_FIELDS:
                raise KeyError(f"Unknown registry field: {key}")
            row[key] = str(value)
        row["updated_at"] = _now()
        _atomic_csv(self.path, RUN_FIELDS, rows)
        return row

    def append_attempt(self, **values: Any) -> None:
        rows = self.attempts()
        rows.append({name: str(values.get(name, "")) for name in ATTEMPT_FIELDS})
        _atomic_csv(self.attempt_path, ATTEMPT_FIELDS, rows)

    def profile_complete(self, profile_name: str) -> bool:
        rows = [row for row in self.rows() if row["profile"] == profile_name]
        return bool(rows) and all(row["status"] == "success" for row in rows)

    def terminal_counts(self, profile_name: str) -> dict[str, int]:
        counts = {status: 0 for status in sorted(ACTIVE_STATUSES)}
        for row in self.rows():
            if row["profile"] == profile_name:
                counts[row["status"]] += 1
        return counts
