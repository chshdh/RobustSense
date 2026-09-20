"""Freeze the V2-5 Demo 2.0 implementation and artifact contract."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import digest_entries, sha256_file, verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "app/streamlit_app.py",
    "configs/v2/phase_v2_5_plan.yaml",
    "docs/v2/ADR-0009-Demo2真实产物与失败关闭.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_5.md",
    "reports/v2/phase_v2_4_protocol_lock.json",
    "scripts/accept_v2_phase5_demo.py",
    "scripts/lock_v2_phase5_protocol.py",
    "scripts/run_demo.ps1",
    "src/robustsense/v2_demo.py",
    "tests/unit/test_v2_demo.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    parent = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"V2-4 parent lock is invalid: {problems}")
    plan = load_config(root / "configs/v2/phase_v2_5_plan.yaml")
    if plan.get("allow_random_prediction_fallback") is not False:
        raise RuntimeError("V2-5 must prohibit random prediction fallback")
    core = read_json(root / plan["core_report"])
    extended = read_json(root / plan["extended_report"])
    if core.get("successful_core_unit_count") != 60:
        raise RuntimeError("Cannot freeze V2-5 before the core matrix is complete")
    if extended.get("successful_unit_count") != 60:
        raise RuntimeError("Cannot freeze V2-5 before the extended matrix is complete")
    entries = []
    for relative in PATHS:
        path = root / relative
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V2-5",
        "status": "frozen_before_first_demo_acceptance_run",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "protocol_sha256": digest_entries(entries),
        "random_prediction_fallback": False,
        "contract_mismatch_fallback": False,
        "files": entries,
    }
    write_json(root / "reports/v2/phase_v2_5_protocol_lock.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
