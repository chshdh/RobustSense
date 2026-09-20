"""Create the machine-verifiable implementation lock for Phase V2-1."""

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
    "configs/v2/models/reliability_constrained.yaml",
    "configs/v2/corruption/train.yaml",
    "configs/v2/evaluation/dev.yaml",
    "docs/RobustSense_V2_可执行项目总规约.md",
    "docs/v2/ADR-0002-P2门控与验证集拒绝.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_1.md",
    "reports/v2/phase_v2_0_protocol_lock.json",
    "src/robustsense/evaluation/suite.py",
    "src/robustsense/models/fusion.py",
    "src/robustsense/pipeline.py",
    "src/robustsense/training/phase3_losses.py",
    "src/robustsense/training/phase3_trainer.py",
    "src/robustsense/training/selective.py",
    "src/robustsense/training/trainer.py",
    "tests/integration/test_phase1_audit.py",
    "tests/unit/test_v2_p2.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_1_protocol_lock.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output

    previous_lock = read_json(root / "reports/v2/phase_v2_0_protocol_lock.json")
    previous_problems = verify_file_entries(root, previous_lock["files"])
    if previous_problems:
        raise RuntimeError(f"V2-0 protocol lock is invalid: {previous_problems}")

    entries = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol input is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 2,
        "phase": "V2-1",
        "status": "frozen_phase_v2_1_development",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest_entries(entries),
        "parent_protocol_sha256": previous_lock["protocol_sha256"],
        "scope": "P2 minimum implementation and train/validation development validation",
        "forbidden": [
            "formal credible training",
            "test-derived model or threshold selection",
            "Phase V2-2 extended evaluation claims",
        ],
        "files": entries,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

