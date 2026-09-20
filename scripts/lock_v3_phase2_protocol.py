"""在首次正式 test 评估前冻结 V3-2 任务置信度协议。"""

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
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v3/phase_v3_2_plan.yaml",
    "docs/v3/ADR-0003-V3-2任务对齐置信度.md",
    "docs/v3/EXPERIMENT_PROTOCOL_V3_2.md",
    "reports/v3/phase_v3_1_acceptance.json",
    "scripts/lock_v3_phase2_protocol.py",
    "scripts/run_v3_phase2.py",
    "src/robustsense/evaluation/v3_task_confidence.py",
    "tests/unit/test_v3_task_confidence.py",
)


def _validate_plan(plan: dict[str, object]) -> None:
    if plan.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-2 计划状态不正确")
    if plan.get("model") != "P2":
        raise RuntimeError("V3-2 必须只使用 P2 主模型")
    if plan.get("folds") != [0, 1, 2, 3, 4] or plan.get("seeds") != [13, 29, 47]:
        raise RuntimeError("V3-2 fold/seed 矩阵发生变化")
    if plan.get("fit_split") != "val" or plan.get("evaluation_split") != "test":
        raise RuntimeError("V3-2 validation/test 边界不正确")
    selector = plan.get("selector_model", {})
    expected = {
        "type": "standardized_logistic_regression",
        "target": "at_least_one_wrong_known_label",
        "regularization_c": 1.0,
        "solver": "lbfgs",
        "max_iter": 1000,
        "class_weight": None,
        "hyperparameter_search": False,
    }
    if selector != expected:
        raise RuntimeError("V3-2 selector 冻结配置发生变化")
    if plan.get("test_time_model_selection_allowed") is not False:
        raise RuntimeError("V3-2 必须禁止测试时选模")
    if plan.get("oracle_selector_allowed") is not False:
        raise RuntimeError("V3-2 必须禁止 oracle selector")
    if plan.get("inference_batch_size") != 512:
        raise RuntimeError("V3-2 必须复用 V2 原始推理 batch size")


def _validate_registry(root: Path, plan: dict[str, object]) -> int:
    with (root / str(plan["core_registry"])).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = [
            row
            for row in csv.DictReader(stream)
            if row["model_name"] == "P2" and row["status"] == "success"
        ]
    expected = {
        (int(fold), int(seed)) for fold in plan["folds"] for seed in plan["seeds"]
    }
    actual = {(int(row["fold"]), int(row["seed"])) for row in rows}
    if actual != expected or len(rows) != 15:
        raise RuntimeError("V3-2 P2 登记表不是 15/15")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v3/phase_v3_2_plan.yaml")
    _validate_plan(plan)

    parent = read_json(root / str(plan["parent_acceptance"]))
    parent_problems = verify_file_entries(root, parent["files"])
    if parent_problems:
        raise RuntimeError(f"V3-1 父验收无效：{parent_problems}")
    if parent.get("acceptance_sha256") != digest_entries(parent["files"]):
        raise RuntimeError("V3-1 父验收摘要不一致")
    if parent.get("status") != "passed" or parent.get("unit_count") != 15:
        raise RuntimeError("V3-1 尚未完成通过验收")
    registry_count = _validate_registry(root, plan)
    existing = list((root / "reports/v3/phase_v3_2/units").glob("*/manifest.json"))
    if existing:
        raise RuntimeError("V3-2 协议冻结前已经存在正式单元")

    entries = []
    for relative in PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V3-2 协议文件缺失：{path}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V3-2",
        "status": "frozen_before_first_test_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_acceptance_sha256": parent["acceptance_sha256"],
        "protocol_sha256": digest_entries(entries),
        "registry_success_count": registry_count,
        "successful_unit_count_before_lock": 0,
        "test_time_model_selection_allowed": False,
        "oracle_selector_allowed": False,
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_2_protocol_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
