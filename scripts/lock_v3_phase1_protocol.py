"""在第一次正式测试评估前冻结 V3-1 选择性预测协议。"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import (
    digest_entries,
    sha256_file,
    verify_file_entries,
    verify_v1_manifest,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v3/phase_v3_1_plan.yaml",
    "docs/v3/ADR-0001-V3-1选择性预测公平对比.md",
    "docs/v3/EXPERIMENT_PROTOCOL_V3_1.md",
    "reports/v2/phase_v2_6_release_lock.json",
    "scripts/lock_v3_phase1_protocol.py",
    "scripts/run_v3_phase1.py",
    "src/robustsense/evaluation/v3_selective.py",
    "tests/unit/test_v3_selective.py",
)


def _validate_plan(plan: dict[str, object]) -> None:
    expected_scores = {
        "primary_generic": "mean_binary_certainty",
        "secondary_generic": "mean_normalized_threshold_margin",
        "proposed": "p2_system_reliability",
    }
    if plan.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-1 计划状态不是首次测试评估前冻结")
    if plan.get("models") != ["B4", "B5", "P", "P2"]:
        raise RuntimeError("V3-1 模型集合发生变化")
    if plan.get("folds") != [0, 1, 2, 3, 4] or plan.get("seeds") != [13, 29, 47]:
        raise RuntimeError("V3-1 fold/seed 矩阵发生变化")
    if plan.get("confidence_scores") != expected_scores:
        raise RuntimeError("V3-1 置信度规则发生变化")
    if plan.get("test_time_model_selection_allowed") is not False:
        raise RuntimeError("V3-1 必须禁止测试时模型选择")
    if plan.get("oracle_selector_allowed") is not False:
        raise RuntimeError("V3-1 必须禁止 oracle 选择器")
    if plan.get("documentation_language") != "zh-CN":
        raise RuntimeError("V3-1 文档必须为中文")


def _validate_registry(root: Path, plan: dict[str, object]) -> int:
    registry_path = root / str(plan["core_registry"])
    with registry_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    expected = {
        (model, int(fold), int(seed))
        for model in plan["models"]
        for fold in plan["folds"]
        for seed in plan["seeds"]
    }
    actual = {
        (row["model_name"], int(row["fold"]), int(row["seed"]))
        for row in rows
        if row["status"] == "success"
    }
    if actual != expected or len(rows) != len(expected):
        raise RuntimeError("V2-4 核心登记表不是 V3-1 所需的 60/60 成功矩阵")
    for row in rows:
        run = root / row["run_dir"]
        required = [run / "predictions.parquet", run / "thresholds.json"]
        if row["model_name"] == "P2":
            required.extend([run / "best_checkpoint.pt", run / "resolved_config.yaml"])
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"V3-1 冻结输入缺失：{missing}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v3/phase_v3_1_plan.yaml")
    _validate_plan(plan)

    parent = read_json(root / str(plan["parent_release_lock"]))
    parent_problems = verify_file_entries(root, parent["files"])
    if parent_problems:
        raise RuntimeError(f"V2-6 父发布锁无效：{parent_problems}")
    if parent.get("release_sha256") != digest_entries(parent["files"]):
        raise RuntimeError("V2-6 父发布锁摘要不一致")

    v1 = verify_v1_manifest(root, root / "reports/v2/v1_baseline_manifest.json")
    if v1["status"] != "passed" or v1["verified_file_count"] != 813:
        raise RuntimeError(f"V1 冻结清单无效：{v1}")
    registry_count = _validate_registry(root, plan)

    entries = []
    for relative in PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V3-1 协议文件缺失：{path}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V3-1",
        "status": "frozen_before_first_test_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_release_sha256": parent["release_sha256"],
        "protocol_sha256": digest_entries(entries),
        "registry_success_count": registry_count,
        "v1_verified_file_count": v1["verified_file_count"],
        "test_time_model_selection_allowed": False,
        "oracle_selector_allowed": False,
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_1_protocol_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
