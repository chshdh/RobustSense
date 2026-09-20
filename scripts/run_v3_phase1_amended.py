"""按冻结修正以原 V2 batch size 运行 V3-1。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import run_v3_phase1 as base

from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json


def _validate_amendment(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    plan, parent = base._validate_protocol(root)
    amendment = load_config(root / "configs/v3/phase_v3_1_execution_amendment.yaml")
    lock = read_json(root / "reports/v3/phase_v3_1_execution_amendment_lock.json")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V3-1 执行修正锁校验失败：{problems}")
    if lock.get("parent_protocol_sha256") != parent.get("protocol_sha256"):
        raise RuntimeError("V3-1 执行修正没有连接到当前父协议")
    if amendment.get("scope") != "p2_reinference_batch_size_only":
        raise RuntimeError("V3-1 执行修正范围异常")
    if amendment.get("probability_consistency_atol") != 1.0e-6:
        raise RuntimeError("V3-1 概率一致性闸门被修改")
    if any(
        amendment.get(name) is not False
        for name in (
            "metrics_changed",
            "coverage_grid_changed",
            "confidence_scores_changed",
            "primary_comparisons_changed",
            "test_time_model_selection_allowed",
        )
    ):
        raise RuntimeError("V3-1 执行修正越过了允许范围")
    amended_plan = {**plan}
    amended_plan["inference_batch_size"] = int(amendment["amended_inference_batch_size"])
    return amended_plan, parent, lock


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--device", default="auto")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan, protocol_lock, amendment_lock = _validate_amendment(root)
    registry = base._registry(root, plan)

    if arguments.aggregate_only:
        base.aggregate_results(root, plan, protocol_lock)
        return
    if arguments.all:
        for fold in plan["folds"]:
            for seed in plan["seeds"]:
                base.evaluate_unit(
                    root,
                    plan,
                    protocol_lock,
                    registry,
                    fold=int(fold),
                    seed=int(seed),
                    device_name=arguments.device,
                    force=arguments.force,
                )
        base.aggregate_results(root, plan, protocol_lock)
        return
    if arguments.fold is None or arguments.seed is None:
        parser.error("单元模式必须同时提供 --fold 与 --seed，或使用 --all")
    print(
        "AMENDMENT "
        f"{amendment_lock['amendment_sha256']}：P2 回放 batch_size="
        f"{plan['inference_batch_size']}",
        flush=True,
    )
    base.evaluate_unit(
        root,
        plan,
        protocol_lock,
        registry,
        fold=arguments.fold,
        seed=arguments.seed,
        device_name=arguments.device,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()
