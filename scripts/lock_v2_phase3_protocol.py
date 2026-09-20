"""Create the machine-verifiable protocol lock before the first V2-3 test run."""

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
    "configs/v2/ablations/phase_v2_3.yaml",
    "configs/v2/corruption/train.yaml",
    "configs/v2/evaluation/seed13_credible.yaml",
    "configs/v2/models/reliability_constrained.yaml",
    "configs/v2/phase_v2_3_plan.yaml",
    "docs/RobustSense_V2_可执行项目总规约.md",
    "docs/v2/ADR-0004-seed13五折与消融治理.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_3.md",
    "reports/v2/phase_v2_2_protocol_lock.json",
    "reports/v2/phase_v2_3_performance_preflight.json",
    "scripts/profile_v2_phase3_training.py",
    "scripts/run_v2_phase3.py",
    "scripts/run_v2_phase3_unit.py",
    "src/robustsense/corruption/vectorized.py",
    "src/robustsense/experiments/v2_phase3.py",
    "src/robustsense/models/v2_phase3.py",
    "src/robustsense/training/v2_phase3_trainer.py",
    "tests/integration/test_v2_phase3_fixture.py",
    "tests/unit/test_v2_phase3_governance.py",
    "tests/unit/test_v2_phase3_models.py",
    "tests/unit/test_vectorized_corruption.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_3_protocol_lock.json")
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    parent = read_json(root / "reports/v2/phase_v2_2_protocol_lock.json")
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"V2-2 protocol lock is invalid: {problems}")
    entries = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol input is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 4,
        "phase": "V2-3",
        "status": "frozen_before_first_test_run",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest_entries(entries),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "scope": "seed-13 five-fold P2 and P2-A1 through P2-A4 formal matrix",
        "expected_unit_count": 25,
        "test_results_visible_at_freeze": False,
        "forbidden": [
            "changing training or selection rules after test visibility",
            "using test metrics for V2-4 ablation selection",
            "deleting failed attempts or skipping registered units",
            "running V2-4 extended stress tests in this phase",
        ],
        "files": entries,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
