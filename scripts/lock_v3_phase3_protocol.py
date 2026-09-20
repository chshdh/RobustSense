"""在首次 test 评估前冻结 V3-3 用户分组连续风险协议。"""

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
    "configs/v3/phase_v3_3_plan.yaml",
    "docs/v3/ADR-0004-V3-3用户分组连续风险.md",
    "docs/v3/EXPERIMENT_PROTOCOL_V3_3.md",
    "reports/v3/phase_v3_2_acceptance.json",
    "scripts/lock_v3_phase3_protocol.py",
    "scripts/run_v3_phase3.py",
    "src/robustsense/evaluation/v3_grouped_risk.py",
    "tests/unit/test_v3_grouped_risk.py",
)


def _validate_plan(plan: dict[str, object]) -> None:
    if plan.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-3 计划状态不正确")
    if plan.get("fit_split") != "val" or plan.get("evaluation_split") != "test":
        raise RuntimeError("V3-3 validation/test 边界不正确")
    if plan.get("selection_split") != "val_grouped_oof":
        raise RuntimeError("V3-3 必须使用 validation grouped OOF 选择")
    if plan.get("grouping_unit") != "user_id":
        raise RuntimeError("V3-3 GroupKFold 必须按用户划分")
    risk = plan.get("risk_model", {})
    expected = {
        "type": "standardized_ridge_regression",
        "solver": "svd",
        "alpha_candidates": [0.1, 1.0, 10.0, 100.0, 1000.0],
        "grouped_cv_splits": 4,
        "selection_metric": "oof_normalized_aurc",
        "alpha_tie_break": "larger_alpha",
        "feature_family_selection": "lower_oof_normalized_aurc",
        "feature_family_tie_break": "grouped_risk_output_compact",
    }
    if risk != expected:
        raise RuntimeError("V3-3 Ridge 或分组选择合同发生变化")
    if plan.get("risk_target") != "masked_binary_cross_entropy_per_sample":
        raise RuntimeError("V3-3 必须直接拟合连续 masked BCE")
    if plan.get("inference_batch_size") != 512:
        raise RuntimeError("V3-3 必须使用冻结回放 batch size 512")
    if plan.get("test_time_model_selection_allowed") is not False:
        raise RuntimeError("V3-3 必须禁止 test 选模")
    if plan.get("oracle_selector_allowed") is not False:
        raise RuntimeError("V3-3 必须禁止 oracle selector")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v3/phase_v3_3_plan.yaml")
    _validate_plan(plan)
    parent = read_json(root / str(plan["parent_acceptance"]))
    parent_problems = verify_file_entries(root, parent["files"])
    if parent_problems:
        raise RuntimeError(f"V3-2 父验收无效：{parent_problems}")
    if parent.get("acceptance_sha256") != digest_entries(parent["files"]):
        raise RuntimeError("V3-2 父验收摘要不一致")
    if parent.get("status") != "passed" or parent.get("unit_count") != 15:
        raise RuntimeError("V3-2 尚未完整通过验收")
    existing = list((root / "reports/v3/phase_v3_3/units").glob("*/manifest.json"))
    if existing:
        raise RuntimeError("V3-3 协议冻结前已存在正式单元")

    entries = []
    for relative in PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V3-3 协议文件缺失：{path}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V3-3",
        "status": "frozen_before_first_test_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_acceptance_sha256": parent["acceptance_sha256"],
        "protocol_sha256": digest_entries(entries),
        "successful_unit_count_before_lock": 0,
        "grouping_unit": "user_id",
        "test_time_model_selection_allowed": False,
        "oracle_selector_allowed": False,
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_3_protocol_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
