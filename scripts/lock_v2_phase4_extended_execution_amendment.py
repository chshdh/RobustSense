"""Freeze the execution-only V2-4 extended-evaluation amendment."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import digest_entries, sha256_file, verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v2/phase_v2_4_extended_execution_amendment.yaml",
    "docs/v2/ADR-0008-V2-4扩展评估双任务并行.md",
    "reports/v2/phase_v2_4_protocol_lock.json",
    "scripts/lock_v2_phase4_extended_execution_amendment.py",
    "scripts/run_v2_phase4_extended_parallel.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    parent = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"Parent V2-4 lock is invalid: {problems}")
    config = load_config(
        root / "configs/v2/phase_v2_4_extended_execution_amendment.yaml"
    )
    if config["parent_protocol_sha256"] != parent["protocol_sha256"]:
        raise RuntimeError("Extended amendment references the wrong parent protocol")
    if config["metric_values_used"] is not False:
        raise RuntimeError("Execution amendment must not use performance metrics")
    entries = []
    for relative in PATHS:
        path = root / relative
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V2-4",
        "kind": "extended_evaluation_parallelism_amendment",
        "status": "frozen_before_parallel_launch",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "amendment_sha256": digest_entries(entries),
        "max_parallel_units": int(config["max_parallel_units"]),
        "metric_values_used": False,
        "files": entries,
    }
    output = root / "reports/v2/phase_v2_4_extended_execution_amendment_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
