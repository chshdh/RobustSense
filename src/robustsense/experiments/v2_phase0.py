"""Phase V2-0 helpers for freezing and verifying the V1 baseline."""

from __future__ import annotations

import csv
import hashlib
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.utils.io import read_json, write_json

V1_STATIC_PATHS = (
    "README.md",
    "docs/EXPERIMENT_PROTOCOL.md",
    "docs/PHASE5_REPORT.md",
    "docs/PHASE6_REPORT.md",
    "docs/PHASE7_REPORT.md",
    "reports/audit_report.md",
    "reports/phase5_protocol_lock.json",
    "reports/readme_source_map.csv",
    "reports/run_attempts.csv",
    "reports/run_completeness.json",
    "reports/run_registry.csv",
    "reports/source_map.csv",
    "reports/technical_report.md",
)
RUN_IDENTITY_SUFFIXES = {".csv", ".json", ".md", ".pkl", ".pt", ".yaml"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def digest_entries(entries: Iterable[dict[str, Any]]) -> str:
    payload = "\n".join(
        f"{entry['path']}\0{entry['bytes']}\0{entry['sha256']}"
        for entry in sorted(entries, key=lambda item: str(item["path"]))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def verify_file_entries(root: str | Path, entries: Iterable[dict[str, Any]]) -> list[str]:
    project_root = Path(root).resolve()
    problems: list[str] = []
    for entry in entries:
        relative = Path(str(entry["path"]))
        path = project_root / relative
        if not path.is_file():
            problems.append(f"missing:{relative.as_posix()}")
            continue
        size = path.stat().st_size
        if size != int(entry["bytes"]):
            problems.append(f"bytes:{relative.as_posix()}:{entry['bytes']}->{size}")
            continue
        actual = sha256_file(path)
        if actual != entry["sha256"]:
            problems.append(f"sha256:{relative.as_posix()}:{entry['sha256']}->{actual}")
    return problems


def _git_metadata(root: Path) -> dict[str, Any]:
    if not (root / ".git").exists():
        return {"repository_present": False, "history_available": False, "head": None}
    command = [
        "git",
        "-c",
        f"safe.directory={root.as_posix()}",
        "-C",
        str(root),
        "rev-parse",
        "--verify",
        "HEAD",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    head = result.stdout.strip() if result.returncode == 0 else None
    return {
        "repository_present": True,
        "history_available": head is not None,
        "head": head,
        "note": (
            None if head else "Git repository has no commit yet; file hashes are authoritative."
        ),
    }


def _collect_paths(root: Path) -> tuple[list[Path], int]:
    protocol_lock = read_json(root / "reports/phase5_protocol_lock.json")
    paths = {root / relative for relative in V1_STATIC_PATHS}
    paths.update(root / item["path"] for item in protocol_lock["files"])
    for pattern in (
        "data/manifests/*.json",
        "data/processed/extrasensory/fold*/processed_manifest.json",
        "data/processed/extrasensory/fold*/preprocessor.json",
        "reports/data_audit/**/*",
        "reports/figures/*.png",
        "reports/tables/*.csv",
    ):
        paths.update(path for path in root.glob(pattern) if path.is_file())

    success_runs = 0
    with (root / "reports/run_registry.csv").open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["status"] != "success":
                continue
            success_runs += 1
            run_dir = root / row["run_dir"]
            if not run_dir.is_dir():
                raise FileNotFoundError(f"Successful V1 run directory is missing: {run_dir}")
            paths.update(
                path
                for path in run_dir.iterdir()
                if path.is_file() and path.suffix.lower() in RUN_IDENTITY_SUFFIXES
            )
    missing = sorted(path for path in paths if not path.is_file())
    if missing:
        joined = ", ".join(str(path.relative_to(root)) for path in missing)
        raise FileNotFoundError(f"V1 authoritative files are missing: {joined}")
    return sorted(paths), success_runs


def build_v1_manifest(root: str | Path) -> dict[str, Any]:
    project_root = Path(root).resolve()
    paths, success_runs = _collect_paths(project_root)
    entries = [
        {
            "path": path.relative_to(project_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in paths
    ]
    completeness = read_json(project_root / "reports/run_completeness.json")
    protocol = read_json(project_root / "reports/phase5_protocol_lock.json")
    if completeness.get("status") != "passed" or completeness.get("validated_run_count") != 60:
        raise ValueError("V1 credible run completeness is not the expected passed 60/60 state")
    return {
        "version": 1,
        "baseline": "RobustSense V1 Phase 7 accepted state",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git": _git_metadata(project_root),
        "v1_protocol_sha256": protocol["protocol_sha256"],
        "credible_runs": {
            "expected": completeness["expected_run_count"],
            "validated": completeness["validated_run_count"],
            "status": completeness["status"],
        },
        "successful_registry_runs": success_runs,
        "scope": {
            "included": [
                "V1 configs and protocol lock",
                "data and preprocessing manifests",
                "authoritative reports, tables, and PNG figures",
                "successful-run checkpoints and compact identity artifacts",
            ],
            "excluded": [
                "large run Parquet prediction/diagnostic files",
                "logs, caches, environments, and raw/processed array payloads",
            ],
            "reason": (
                "Large payload integrity is represented by frozen manifests and authoritative "
                "aggregates; V2 never overwrites runs/ or reports/ V1 paths."
            ),
        },
        "file_count": len(entries),
        "content_sha256": digest_entries(entries),
        "files": entries,
    }


def freeze_v1_manifest(root: str | Path, output: str | Path) -> dict[str, Any]:
    manifest = build_v1_manifest(root)
    write_json(output, manifest)
    return manifest


def verify_v1_manifest(root: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    problems = verify_file_entries(root, manifest["files"])
    content_sha256 = digest_entries(manifest["files"])
    if content_sha256 != manifest["content_sha256"]:
        problems.append("manifest-content-sha256-mismatch")
    return {
        "status": "passed" if not problems else "failed",
        "verified_file_count": len(manifest["files"]),
        "problems": problems,
    }
