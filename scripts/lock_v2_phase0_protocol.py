"""Create the machine-verifiable protocol lock for Phase V2-0."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import digest_entries, sha256_file, verify_v1_manifest
from robustsense.utils.io import read_json, write_json

LOCKED_PATHS = (
    "configs/v2/phase_v2_0.yaml",
    "docs/RobustSense_V2_可执行项目总规约.md",
    "docs/v2/ADR-0001-V1保护与V2隔离.md",
    "docs/v2/EXPERIMENT_PROTOCOL_DRAFT.md",
    "reports/phase5_protocol_lock.json",
    "reports/v2/v1_baseline_manifest.json",
    "scripts/freeze_v1_baseline.py",
    "scripts/profile_v2_baseline.py",
    "scripts/verify_v1_report_rebuild.py",
    "src/robustsense/experiments/v2_phase0.py",
    "tests/unit/test_v2_phase0.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_0_protocol_lock.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    baseline_path = root / "reports/v2/v1_baseline_manifest.json"
    verification = verify_v1_manifest(root, baseline_path)
    if verification["status"] != "passed":
        raise RuntimeError(f"Cannot lock V2-0 over an invalid V1 baseline: {verification}")
    entries = []
    for relative in LOCKED_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"Protocol input is missing: {relative}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    v1_manifest = read_json(baseline_path)
    result = {
        "version": 1,
        "phase": "V2-0",
        "status": "frozen_phase_v2_0",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": digest_entries(entries),
        "v1_content_sha256": v1_manifest["content_sha256"],
        "v1_protocol_sha256": v1_manifest["v1_protocol_sha256"],
        "scope": "V1 protection, namespace isolation, and performance baseline only",
        "deferred": "P2 model and all V2 result-producing experiment configs",
        "files": entries,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
