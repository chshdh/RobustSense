"""在首次候选 test 评估前冻结 V4-1 最终模型优化协议。"""

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
    "configs/v4/label_conditioned_fusion.yaml",
    "configs/v4/phase_v4_1_plan.yaml",
    "configs/v4/phase_v4_1_execution_amendment.yaml",
    "configs/v4/extended_core.yaml",
    "docs/v4/ADR-0001-V4-1标签级可靠融合.md",
    "docs/v4/EXPERIMENT_PROTOCOL_V4_1.md",
    "reports/v3/phase_v3_4_acceptance.json",
    "scripts/lock_v4_phase1_protocol.py",
    "scripts/run_v4_phase1.py",
    "scripts/run_v4_phase1_parallel.py",
    "src/robustsense/models/v4_label_fusion.py",
    "tests/unit/test_v4_label_fusion.py",
)


def _validate_plan(plan: dict[str, object]) -> None:
    if plan.get("status") != "frozen_before_first_candidate_test_evaluation":
        raise RuntimeError("V4-1 计划状态不正确")
    if plan.get("final_model_optimization") is not True:
        raise RuntimeError("V4-1 必须标记为最后一次模型优化")
    if plan.get("further_model_versions_after_this_phase") is not False:
        raise RuntimeError("V4-1 停止规则没有冻结")
    if plan.get("folds") != [0, 1, 2, 3, 4] or plan.get("seeds") != [13, 29, 47]:
        raise RuntimeError("V4-1 正式矩阵必须是 5 折 × 3 种子")
    training = plan.get("training", {})
    expected_training = {
        "warm_start": "matching_fold_seed_p2_checkpoint",
        "initial_probability_equivalence_atol": 0.000001,
        "initial_weight_equivalence_atol": 0.000001,
        "initial_logit_diagnostic_atol": 0.00001,
        "trainable_scope": "utility_nets_only",
        "loss": "masked_weighted_bce_with_logits",
        "corrupted_training_view": True,
        "clean_consistency_view": False,
        "epochs": 30,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "gradient_clip_norm": 5.0,
        "pos_weight_cap": 20.0,
        "early_stopping_patience": 6,
        "early_stopping_metric": "validation_macro_f1_at_0_5",
        "epoch_zero_checkpoint_allowed": True,
        "device": "auto",
    }
    if training != expected_training:
        raise RuntimeError("V4-1 训练合同发生变化")
    primary = plan.get("primary_success", {})
    if primary != {
        "minimum_absolute_mean_gain_vs_p2": 0.01,
        "minimum_winning_fold_count": 3,
    }:
        raise RuntimeError("V4-1 首要成功线发生变化")
    if plan.get("safety_gates") != {
        "minimum_micro_f1_delta": -0.005,
        "maximum_masked_bce_delta": 0.005,
        "maximum_parameter_count_relative_increase": 0.25,
        "maximum_latency_relative_increase": 0.5,
    }:
        raise RuntimeError("V4-1 安全闸门发生变化")
    if (
        plan.get("threshold_source_split") != "val"
        or plan.get("evaluation_split") != "test"
        or plan.get("test_time_model_selection_allowed") is not False
        or plan.get("test_time_threshold_tuning_allowed") is not False
    ):
        raise RuntimeError("V4-1 validation/test 治理边界错误")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v4/phase_v4_1_plan.yaml")
    _validate_plan(plan)
    parent = read_json(root / str(plan["parent_acceptance"]))
    problems = verify_file_entries(root, parent["files"])
    if problems:
        raise RuntimeError(f"V3-4 父验收无效：{problems}")
    if parent.get("acceptance_sha256") != digest_entries(parent["files"]):
        raise RuntimeError("V3-4 父验收摘要不一致")
    if parent.get("status") != "passed" or parent.get("unit_count") != 15:
        raise RuntimeError("V3-4 尚未完整通过验收")
    existing = list((root / str(plan["run_root"])).glob("*/run_manifest.json"))
    if existing:
        raise RuntimeError("V4-1 协议冻结前已经存在正式候选单元")

    entries = []
    for relative in PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f"V4-1 协议文件缺失：{path}")
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )
    result = {
        "version": 1,
        "phase": "V4-1",
        "status": "frozen_before_first_candidate_test_evaluation",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "parent_acceptance_sha256": parent["acceptance_sha256"],
        "protocol_sha256": digest_entries(entries),
        "successful_candidate_unit_count_before_lock": 0,
        "final_model_optimization": True,
        "further_model_versions_allowed": False,
        "test_time_model_selection_allowed": False,
        "test_time_threshold_tuning_allowed": False,
        "files": entries,
    }
    output = root / "reports/v4/phase_v4_1_protocol_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
