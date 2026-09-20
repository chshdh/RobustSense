"""Freeze the V2-4 protocol before any new-seed test split is opened."""

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
    "configs/experiment/full.yaml",
    "configs/v2/corruption/train.yaml",
    "configs/v2/evaluation/extended_core.yaml",
    "configs/v2/evaluation/multiseed_core.yaml",
    "configs/v2/models/reliability_constrained.yaml",
    "configs/v2/phase_v2_4_plan.yaml",
    "docs/RobustSense_V2_可执行项目总规约.md",
    "docs/v2/ADR-0005-V2-4多种子复用与压力评估.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_4.md",
    "reports/phase5_protocol_lock.json",
    "reports/v2/phase_v2_2_protocol_lock.json",
    "reports/v2/phase_v2_3_protocol_lock.json",
    "reports/v2/phase_v2_3_selection_for_v2_4.json",
    "reports/v2/phase_v2_4_reuse_manifest.json",
    "reports/v2/scenarios/availability_masks.jsonl",
    "reports/v2/scenarios/mixed_failures.jsonl",
    "reports/v2/scenarios/scenario_contract.json",
    "scripts/build_v2_phase4_reuse_manifest.py",
    "scripts/lock_v2_phase4_protocol.py",
    "scripts/run_v2_phase4.py",
    "scripts/run_v2_phase4_extended.py",
    "scripts/run_v2_phase4_extended_unit.py",
    "scripts/run_v2_phase4_optional.py",
    "scripts/run_v2_phase4_unit.py",
    "src/robustsense/evaluation/v2_phase4.py",
    "src/robustsense/experiments/v2_phase4.py",
    "src/robustsense/training/v2_phase4_trainer.py",
    "tests/integration/test_v2_phase4_fixture.py",
    "tests/unit/test_v2_phase4_governance.py",
)


def _verify_parent(root: Path, path: str) -> dict:
    parent = read_json(root / path)
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"Parent protocol lock is invalid: {path}: {problems}")
    return parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_4_protocol_lock.json")
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    parent = _verify_parent(root, "reports/v2/phase_v2_3_protocol_lock.json")
    reuse = read_json(root / "reports/v2/phase_v2_4_reuse_manifest.json")
    if reuse.get("entry_count") != 20 or reuse.get("metric_values_read") is not False:
        raise RuntimeError("V2-4 reuse manifest is not an unseen-metric 20-unit contract")
    if (root / "reports/v2/phase_v2_4_run_registry.csv").exists():
        raise RuntimeError("V2-4 core registry exists before the protocol freeze")
    entries = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol input is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V2-4",
        "status": "frozen_before_first_test_run",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest_entries(entries),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "scope": "B4/B5/P/P2 five-fold three-seed core and extended pressure matrix",
        "expected_core_unit_count": 60,
        "expected_reused_unit_count": 20,
        "expected_new_training_unit_count": 40,
        "expected_extended_unit_count": 60,
        "expected_optional_ablation_unit_count": 20,
        "test_results_visible_at_freeze": False,
        "reuse_metric_values_read_at_freeze": False,
        "aggregation_order": "seeds_within_fold_then_folds",
        "forbidden": [
            "changing training or threshold rules after new-seed test visibility",
            "replacing validation-selected optional ablations using test results",
            "changing scenario sets, Bootstrap repeats, or aggregation order",
            "deleting failed attempts or aggregating failed units",
            "storing full per-scenario extended predictions",
        ],
        "files": entries,
    }
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
