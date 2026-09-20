"""Freeze the V2-4 artifact-verifier correction before reconciliation."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import digest_entries, sha256_file, verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v2/phase_v2_4_verifier_correction.yaml",
    "docs/v2/ADR-0007-V2-4产物校验字段修正.md",
    "reports/v2/phase_v2_4_execution_amendment_lock.json",
    "scripts/lock_v2_phase4_verifier_correction.py",
    "scripts/reconcile_v2_phase4_artifacts.py",
    "scripts/run_v2_phase4_parallel_v2.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    parent = read_json(root / "reports/v2/phase_v2_4_execution_amendment_lock.json")
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"Parent execution amendment is invalid: {problems}")
    config = load_config(root / "configs/v2/phase_v2_4_verifier_correction.yaml")
    if config["parent_amendment_sha256"] != parent["amendment_sha256"]:
        raise RuntimeError("Verifier correction references the wrong parent amendment")
    entries = []
    for relative in PATHS:
        path = root / relative
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V2-4",
        "kind": "artifact_verifier_field_correction",
        "status": "frozen_before_corrected_reconciliation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_amendment_sha256": parent["amendment_sha256"],
        "correction_sha256": digest_entries(entries),
        "metric_values_used": False,
        "files": entries,
    }
    write_json(root / "reports/v2/phase_v2_4_verifier_correction_lock.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
