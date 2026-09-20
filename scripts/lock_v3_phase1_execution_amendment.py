"""冻结 V3-1 P2 回放 batch size 修正。"""

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
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v3/phase_v3_1_execution_amendment.yaml",
    "docs/v3/ADR-0002-V3-1回放批量修正.md",
    "reports/v3/phase_v3_1_protocol_lock.json",
    "scripts/lock_v3_phase1_execution_amendment.py",
    "scripts/run_v3_phase1_amended.py",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    parent = read_json(root / "reports/v3/phase_v3_1_protocol_lock.json")
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"V3-1 父协议锁无效：{problems}")
    amendment = load_config(root / "configs/v3/phase_v3_1_execution_amendment.yaml")
    if amendment.get("successful_unit_count_before_amendment") != 0:
        raise RuntimeError("修正必须记录为首个成功单元之前")
    existing = list((root / "reports/v3/phase_v3_1/units").glob("*/manifest.json"))
    if existing:
        raise RuntimeError(f"修正前已经存在成功单元：{existing}")
    if amendment.get("original_inference_batch_size") != 2048:
        raise RuntimeError("修正没有准确记录原 batch size")
    if amendment.get("amended_inference_batch_size") != 512:
        raise RuntimeError("修正后的 batch size 必须复现 V2 原评估配置")
    if amendment.get("probability_consistency_atol") != 1.0e-6:
        raise RuntimeError("概率一致性闸门不得放宽")

    entries = []
    for relative in PATHS:
        path = root / relative
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V3-1",
        "status": "execution_amendment_frozen_before_first_successful_unit",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "amendment_sha256": digest_entries(entries),
        "successful_unit_count_before_amendment": 0,
        "probability_consistency_atol": 1.0e-6,
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_1_execution_amendment_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
