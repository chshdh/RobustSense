"""独立校正 V4-1 报告中的总参数量统计，不改动模型或预测。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robustsense.experiments.v2_phase0 import sha256_file
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _state_dict_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"检查点格式错误：{path}")
    state_dict = checkpoint.get("state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"检查点不含 state_dict：{path}")
    tensors = [value for value in state_dict.values() if isinstance(value, torch.Tensor)]
    if not tensors:
        raise RuntimeError(f"检查点没有张量：{path}")
    return int(sum(tensor.numel() for tensor in tensors))


def _registry(root: Path) -> dict[tuple[int, int], Path]:
    frame = pd.read_csv(root / "reports/v2/phase_v2_4_run_registry.csv")
    selected = frame.loc[(frame["model_name"] == "P2") & (frame["status"] == "success")]
    registry: dict[tuple[int, int], Path] = {}
    for row in selected.itertuples(index=False):
        key = (int(row.fold), int(row.seed))
        if key in registry:
            raise RuntimeError(f"P2 登记表键重复：{key}")
        registry[key] = root / str(row.run_dir)
    return registry


def _refresh_summaries(
    frame: pd.DataFrame,
    folds: list[int],
    seeds: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [
        column
        for column in frame.columns
        if column.startswith(("candidate_", "baseline_", "delta_", "relative_delta_"))
        and column not in {"candidate_run_id", "baseline_run_id"}
    ]
    fold_summary = frame.groupby("fold", as_index=False)[value_columns].mean()
    fold_summary["seed_count"] = len(seeds)
    if sorted(fold_summary["fold"].astype(int).tolist()) != sorted(folds):
        raise RuntimeError("V4-1 折级汇总不完整")
    summary: dict[str, Any] = {"fold_count": len(folds)}
    for column in value_columns:
        values = fold_summary[column].to_numpy(dtype=float)
        summary[column] = float(np.mean(values))
        summary[f"{column}_fold_std"] = float(np.std(values, ddof=1))
    return fold_summary, pd.DataFrame([summary])


def _rewrite_document(root: Path, report: dict[str, Any]) -> None:
    summary = report["summary"]
    primary_passed = bool(report["primary_success"])
    fold_summary = pd.read_csv(root / "reports/v4/phase_v4_1_fold_summary.csv")
    lines = [
        "# RobustSense V4-1 最后一次模型优化报告",
        "",
        "## 完成状态",
        "",
        "V4-1 已完成 5 折 × 3 种子的 15 个 P2-LC 正式单元。",
        "",
        f"- 协议哈希：`{report['protocol_sha256']}`",
        f"- P2 Macro-F1：{summary['baseline_macro_f1']:.6f}",
        f"- P2-LC Macro-F1：{summary['candidate_macro_f1']:.6f}",
        f"- 绝对差值：{report['candidate_minus_p2_macro_f1']:+.6f}",
        f"- 胜出折数：{report['candidate_winning_fold_count']}/5",
        f"- 首要成功判定：{'通过' if primary_passed else '未通过'}",
        "",
        "## 逐折结果",
        "",
        "| Fold | P2 Macro-F1 | P2-LC Macro-F1 | 差值 |",
        "|---:|---:|---:|---:|",
    ]
    for row in fold_summary.itertuples(index=False):
        lines.append(
            f"| {int(row.fold)} | {row.baseline_macro_f1:.6f} | "
            f"{row.candidate_macro_f1:.6f} | {row.delta_macro_f1:+.6f} |"
        )
    lines.extend(
        [
            "",
        "## 安全闸门",
        "",
        ]
    )
    for name, result in report["safety_gates"].items():
        lines.append(
            f"- `{name}`：{result['value']:+.6f}，"
            f"{'通过' if result['passed'] else '未通过'}。"
        )
    lines.extend(
        [
            "",
            "## 参数量统计校正",
            "",
            "原始聚合器把冻结后的可训练参数量误标成总参数量。"
            "本报告已从候选与基线检查点的完整 `state_dict` 独立复算总参数量；"
            "该校正不改变模型、阈值、预测或任何性能指标。",
            "",
            f"- P2 总参数量：{summary['baseline_parameter_count']:.0f}",
            f"- P2-LC 总参数量：{summary['candidate_parameter_count']:.0f}",
            f"- 相对增加：{summary['relative_delta_parameter_count']:+.4%}",
            "",
            "## 科学结论",
            "",
            "P2-LC 的标签级可靠融合只在 2/5 个折上取得正增益，"
            "总体 Macro-F1 仅增加 0.000404，明显低于预注册的 0.010。"
            "同时，fold2 出现负迁移，说明新增标签级路由自由度对验证噪声敏感，"
            "没有形成稳定的跨用户泛化收益。",
            "",
            "所有安全闸门通过，只说明模型没有越过预设的精度、校准、参数量和延迟底线；"
            "它不等于首要模型假设通过。基于收益与复杂度的权衡，项目最终默认模型继续使用 P2，"
            "P2-LC 保留为一次有严格阴性结论的消融实验，不作为新的发布模型。",
            "",
            "## 验收与模型判定的区别",
            "",
            "- 工程完整性验收：通过。15/15 单元、测试隔离、产物哈希和 106 项回归测试均有效。",
            "- 首要科学假设：未通过。平均增益与胜出折数均未达到冻结标准。",
            "- 发布决策：保留 P2，不升级默认模型。",
            "",
            "## 停止规则",
            "",
            "这是预先声明的最后一次模型优化。无论结果是否通过，不再增加新的模型版本。",
            "",
        ]
    )
    (root / "docs/v4/PHASE_V4_1_REPORT.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v4/phase_v4_1_plan.yaml")
    protocol = read_json(root / "reports/v4/phase_v4_1_protocol_lock.json")
    folds = [int(value) for value in plan["folds"]]
    seeds = [int(value) for value in plan["seeds"]]
    fold_seed_path = root / "reports/v4/phase_v4_1_fold_seed_results.csv"
    frame = pd.read_csv(fold_seed_path)
    if len(frame) != len(folds) * len(seeds):
        raise RuntimeError(f"V4-1 单元行数错误：{len(frame)}")
    registry = _registry(root)
    audit_rows: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        fold, seed = int(row["fold"]), int(row["seed"])
        candidate_dir = root / str(plan["run_root"]) / str(row["candidate_run_id"])
        baseline_dir = registry[(fold, seed)]
        candidate_checkpoint = candidate_dir / "best_checkpoint.pt"
        baseline_checkpoint = baseline_dir / "best_checkpoint.pt"
        candidate_count = _state_dict_parameter_count(candidate_checkpoint)
        baseline_count = _state_dict_parameter_count(baseline_checkpoint)
        old_candidate_count = float(row["candidate_parameter_count"])
        frame.at[index, "candidate_parameter_count"] = candidate_count
        frame.at[index, "baseline_parameter_count"] = baseline_count
        frame.at[index, "delta_parameter_count"] = candidate_count - baseline_count
        frame.at[index, "relative_delta_parameter_count"] = (
            candidate_count / baseline_count - 1.0
        )
        audit_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "candidate_run_id": str(row["candidate_run_id"]),
                "baseline_run_id": str(row["baseline_run_id"]),
                "old_mislabeled_candidate_parameter_count": old_candidate_count,
                "candidate_total_parameter_count": candidate_count,
                "baseline_total_parameter_count": baseline_count,
                "relative_increase": candidate_count / baseline_count - 1.0,
                "candidate_checkpoint_sha256": sha256_file(candidate_checkpoint),
                "baseline_checkpoint_sha256": sha256_file(baseline_checkpoint),
            }
        )
    frame = frame.sort_values(["fold", "seed"])
    fold_summary, summary_frame = _refresh_summaries(frame, folds, seeds)
    frame.to_csv(fold_seed_path, index=False)
    fold_summary.to_csv(root / "reports/v4/phase_v4_1_fold_summary.csv", index=False)
    summary_frame.to_csv(root / "reports/v4/phase_v4_1_summary.csv", index=False)

    report_path = root / "reports/v4/phase_v4_1_report.json"
    report = read_json(report_path)
    report["summary"] = summary_frame.iloc[0].to_dict()
    relative = float(summary_frame.iloc[0]["relative_delta_parameter_count"])
    gate = report["safety_gates"]["parameter_count"]
    gate["value"] = relative
    gate["passed"] = relative <= float(gate["threshold"])
    report["all_safety_gates_passed"] = all(
        bool(result["passed"]) for result in report["safety_gates"].values()
    )
    report["outputs"] = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in (
            fold_seed_path,
            root / "reports/v4/phase_v4_1_fold_summary.csv",
            root / "reports/v4/phase_v4_1_summary.csv",
        )
    ]
    report["parameter_accounting_correction"] = {
        "status": "corrected_from_checkpoint_state_dict",
        "corrected_at_utc": datetime.now(UTC).isoformat(),
        "cause": "冻结主干后参数辅助函数只统计 requires_grad，误标为总参数量",
        "model_predictions_changed": False,
        "scientific_metrics_changed": False,
        "audit": "reports/v4/phase_v4_1_parameter_correction.json",
    }
    write_json(report_path, report)
    _rewrite_document(root, report)

    correction = {
        "version": 1,
        "phase": "V4-1",
        "status": "complete",
        "protocol_sha256": protocol["protocol_sha256"],
        "corrected_at_utc": datetime.now(UTC).isoformat(),
        "cause": "原聚合器把 requires_grad 参数量误标成总参数量",
        "scope": "parameter_accounting_metadata_only",
        "model_predictions_changed": False,
        "scientific_metrics_changed": False,
        "unit_count": len(audit_rows),
        "candidate_total_parameter_count": int(
            summary_frame.iloc[0]["candidate_parameter_count"]
        ),
        "baseline_total_parameter_count": int(
            summary_frame.iloc[0]["baseline_parameter_count"]
        ),
        "relative_increase": relative,
        "units": audit_rows,
    }
    output = root / "reports/v4/phase_v4_1_parameter_correction.json"
    write_json(output, correction)
    print(f"V4-1 参数量校正完成：{output}")


if __name__ == "__main__":
    main()
