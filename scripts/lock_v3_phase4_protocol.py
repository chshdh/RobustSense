"""在首次 test 评估前冻结 V3-4 双策略固定阈值协议。"""

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
    "configs/v3/phase_v3_4_plan.yaml",
    "docs/v3/ADR-0005-V3-4双策略决策层.md",
    "docs/v3/EXPERIMENT_PROTOCOL_V3_4.md",
    "reports/v3/phase_v3_3_acceptance.json",
    "scripts/lock_v3_phase4_protocol.py",
    "scripts/run_v3_phase4.py",
    "src/robustsense/evaluation/v3_dual_policy.py",
    "tests/unit/test_v3_dual_policy.py",
)


def _validate_plan(plan: dict[str, object]) -> None:
    if plan.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-4 计划状态不正确")
    if (
        plan.get("threshold_fit_split") != "val"
        or plan.get("evaluation_split") != "test"
    ):
        raise RuntimeError("V3-4 validation/test 边界不正确")
    modes = plan.get("policy_modes", {})
    if list(modes) != ["risk_control", "error_alert"]:
        raise RuntimeError("V3-4 必须冻结两个显式策略模式")
    if modes["risk_control"].get("confidence_source") != (
        "v3_3_grouped_risk_selected"
    ):
        raise RuntimeError("V3-4 risk_control 来源不正确")
    if modes["error_alert"].get("confidence_source") != "mean_binary_certainty":
        raise RuntimeError("V3-4 error_alert 来源不正确")
    if plan.get("target_coverages") != [0.8, 0.9, 0.95]:
        raise RuntimeError("V3-4 目标覆盖率发生变化")
    hypotheses = plan.get("primary_hypotheses", {})
    expected = {
        "risk_specialization": ("risk_control", "error_alert", "risk_masked_bce", "lower"),
        "alert_specialization": (
            "error_alert",
            "risk_control",
            "rejected_error_recall",
            "higher",
        ),
    }
    for name, contract in expected.items():
        item = hypotheses.get(name, {})
        actual = (
            item.get("candidate"),
            item.get("reference"),
            item.get("metric"),
            item.get("direction"),
        )
        if actual != contract:
            raise RuntimeError(f"V3-4 首要假设发生变化：{name}")
        if (
            item.get("minimum_winning_target_coverages") != 2
            or item.get("minimum_fold_wins_per_winning_coverage") != 3
        ):
            raise RuntimeError(f"V3-4 成功判据发生变化：{name}")
    readiness = plan.get("coverage_transfer_readiness", {})
    if readiness.get("maximum_per_mode") != 0.05:
        raise RuntimeError("V3-4 覆盖率迁移准备线发生变化")
    if plan.get("inference_batch_size") != 512:
        raise RuntimeError("V3-4 必须使用冻结回放 batch size 512")
    for forbidden in (
        "test_rank_selection_allowed",
        "automatic_mode_selection_allowed",
        "test_time_threshold_tuning_allowed",
    ):
        if plan.get(forbidden) is not False:
            raise RuntimeError(f"V3-4 禁止项没有关闭：{forbidden}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v3/phase_v3_4_plan.yaml")
    _validate_plan(plan)
    parent = read_json(root / str(plan["parent_acceptance"]))
    parent_problems = verify_file_entries(root, parent["files"])
    if parent_problems:
        raise RuntimeError(f"V3-3 父验收无效：{parent_problems}")
    if parent.get("acceptance_sha256") != digest_entries(parent["files"]):
        raise RuntimeError("V3-3 父验收摘要不一致")
    if parent.get("status") != "passed" or parent.get("unit_count") != 15:
        raise RuntimeError("V3-3 尚未完整通过验收")
    existing = list((root / "reports/v3/phase_v3_4/units").glob("*/manifest.json"))
    if existing:
        raise RuntimeError("V3-4 协议冻结前已存在正式单元")

    entries = []
    for relative in PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V3-4 协议文件缺失：{path}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V3-4",
        "status": "frozen_before_first_test_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_acceptance_sha256": parent["acceptance_sha256"],
        "protocol_sha256": digest_entries(entries),
        "successful_unit_count_before_lock": 0,
        "threshold_source_split": "val",
        "test_rank_selection_allowed": False,
        "automatic_mode_selection_allowed": False,
        "test_time_threshold_tuning_allowed": False,
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_4_protocol_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
