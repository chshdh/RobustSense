"""Create the machine-verifiable implementation lock for Phase V2-2."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import (
    digest_entries,
    sha256_file,
    verify_file_entries,
)
from robustsense.utils.io import read_json, write_json

LOCKED_PATHS = (
    "configs/v2/corruption/exhaustive_masks.yaml",
    "configs/v2/corruption/mixed_failures.yaml",
    "configs/v2/corruption/persistent_failures.yaml",
    "configs/v2/evaluation/extended_dev.yaml",
    "docs/RobustSense_V2_可执行项目总规约.md",
    "docs/v2/ADR-0003-扩展评估合同.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_2.md",
    "reports/v2/phase_v2_1_protocol_lock.json",
    "reports/v2/scenarios/availability_masks.jsonl",
    "reports/v2/scenarios/mixed_failures.jsonl",
    "reports/v2/scenarios/scenario_contract.json",
    "scripts/build_v2_phase2_contract.py",
    "src/robustsense/evaluation/v2_aggregate.py",
    "src/robustsense/evaluation/v2_artifacts.py",
    "src/robustsense/evaluation/v2_metrics.py",
    "src/robustsense/evaluation/v2_scenarios.py",
    "src/robustsense/models/mask_only.py",
    "tests/unit/test_mask_only.py",
    "tests/unit/test_v2_aggregate.py",
    "tests/unit/test_v2_artifacts.py",
    "tests/unit/test_v2_metrics.py",
    "tests/unit/test_v2_scenarios.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_2_protocol_lock.json")
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output

    previous_lock = read_json(root / "reports/v2/phase_v2_1_protocol_lock.json")
    previous_problems = verify_file_entries(root, previous_lock["files"])
    if previous_problems:
        raise RuntimeError(f"V2-1 protocol lock is invalid: {previous_problems}")

    entries = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol input is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 3,
        "phase": "V2-2",
        "status": "frozen_phase_v2_2_extended_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest_entries(entries),
        "parent_protocol_sha256": previous_lock["protocol_sha256"],
        "scope": "extended evaluation scenarios, metrics, diagnostics, and lineage contracts",
        "forbidden": [
            "formal credible training",
            "test-derived model, classification, calibration, or abstention selection",
            "treating scenario contract artifacts as model results",
        ],
        "files": entries,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
