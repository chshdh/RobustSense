"""运行 V3-4 双策略 validation 固定阈值决策实验。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_v3_phase1 as v31
import run_v3_phase2 as v32

from robustsense.constants import MODALITIES
from robustsense.evaluation.v3_dual_policy import (
    POLICY_MODES,
    DualPolicyDecisionLayer,
    fixed_threshold_policy_metrics,
    tune_coverage_threshold,
)
from robustsense.evaluation.v3_grouped_risk import (
    RidgeRiskSelector,
    build_compact_risk_features,
)
from robustsense.evaluation.v3_selective import (
    mean_binary_certainty,
    sample_error_indicator,
)
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _validate_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_config(root / "configs/v3/phase_v3_4_plan.yaml")
    lock = read_json(root / "reports/v3/phase_v3_4_protocol_lock.json")
    if lock.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-4 协议尚未在首次 test 评估前冻结")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V3-4 协议锁校验失败：{problems}")
    return plan, lock


def _unit_dir(root: Path, plan: dict[str, Any], fold: int, seed: int) -> Path:
    return root / plan["unit_output_root"] / f"fold{fold}_seed{seed}"


def _unit_is_current(output: Path, root: Path, protocol_sha256: str) -> bool:
    manifest_path = output / "manifest.json"
    if not manifest_path.is_file():
        return False
    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "success"
        or manifest.get("protocol_sha256") != protocol_sha256
    ):
        return False
    return not verify_file_entries(
        root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
    )


def _risk_confidences(
    root: Path,
    fold: int,
    seed: int,
    validation: dict[str, Any],
    test: dict[str, Any],
    val_usable: np.ndarray,
    test_usable: np.ndarray,
    thresholds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str, Path, Path]:
    parent_dir = (
        root / "reports/v3/phase_v3_3/units" / f"fold{fold}_seed{seed}"
    )
    selectors_path = parent_dir / "selectors.json"
    manifest_path = parent_dir / "manifest.json"
    selectors = read_json(selectors_path)
    parent_manifest = read_json(manifest_path)
    if (
        selectors.get("phase") != "V3-3"
        or parent_manifest.get("status") != "success"
        or selectors.get("selected_family") != parent_manifest.get("selected_family")
    ):
        raise RuntimeError("V3-3 风险选择器来源无效")
    family = str(selectors["selected_family"])
    if family not in {
        "grouped_risk_output_compact",
        "grouped_risk_full_compact",
    }:
        raise RuntimeError(f"未知 V3-3 风险特征家族：{family}")
    selector = RidgeRiskSelector.from_artifact(
        selectors["selectors"][family]["selector"]
    )
    include_sensor = family == "grouped_risk_full_compact"
    val_features, val_names = build_compact_risk_features(
        validation,
        modality_names=list(MODALITIES),
        thresholds=thresholds,
        include_sensor_features=include_sensor,
    )
    test_features, test_names = build_compact_risk_features(
        test,
        modality_names=list(MODALITIES),
        thresholds=thresholds,
        include_sensor_features=include_sensor,
    )
    if tuple(val_names) != selector.feature_names or val_names != test_names:
        raise RuntimeError("V3-3 风险选择器特征合同与回放特征不一致")
    return (
        selector.predict_confidence(val_features[val_usable]),
        selector.predict_confidence(test_features[test_usable]),
        family,
        selectors_path,
        manifest_path,
    )


def evaluate_unit(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
    *,
    fold: int,
    seed: int,
    device_name: str,
    force: bool,
) -> Path:
    if fold not in plan["folds"] or seed not in plan["seeds"]:
        raise ValueError("fold 或 seed 超出 V3-4 冻结矩阵")
    output = _unit_dir(root, plan, fold, seed)
    if not force and _unit_is_current(output, root, lock["protocol_sha256"]):
        print(f"SKIP fold={fold} seed={seed}: 现有 V3-4 单元通过哈希校验")
        return output

    row = registry[(fold, seed)]
    run_path = root / row["run_dir"]
    frozen = v31._load_prediction_artifact(root, "P2", run_path)
    bundle, validation, test, data_manifest_path, val_path = v32._collect_splits(
        root,
        fold=fold,
        seed=seed,
        device_name=device_name,
        batch_size=int(plan["inference_batch_size"]),
    )
    max_delta = v32._validate_frozen_test(
        frozen, test, float(plan["probability_consistency_atol"])
    )
    val_usable = np.asarray(validation["target_mask"], dtype=bool).any(axis=1)
    test_usable = np.asarray(test["target_mask"], dtype=bool).any(axis=1)
    if not val_usable.any() or not test_usable.any():
        raise RuntimeError("V3-4 validation 或 test 没有可评分样本")

    risk_val, risk_test, selected_family, selector_path, parent_manifest_path = (
        _risk_confidences(
            root,
            fold,
            seed,
            validation,
            test,
            val_usable,
            test_usable,
            bundle.thresholds,
        )
    )
    confidences = {
        "risk_control": (
            risk_val,
            risk_test,
            "v3_3_grouped_risk_selected",
        ),
        "error_alert": (
            mean_binary_certainty(validation["probabilities"][val_usable]),
            mean_binary_certainty(test["probabilities"][test_usable]),
            "mean_binary_certainty",
        ),
    }
    if tuple(plan["policy_modes"]) != POLICY_MODES:
        raise RuntimeError("V3-4 策略模式顺序与实现不一致")

    probabilities = np.asarray(test["probabilities"], dtype=np.float64)[test_usable]
    targets = np.asarray(test["targets"], dtype=np.float64)[test_usable]
    target_mask = np.asarray(test["target_mask"], dtype=bool)[test_usable]
    errors = sample_error_indicator(
        probabilities, targets, target_mask, bundle.thresholds
    )
    result_rows: list[dict[str, Any]] = []
    threshold_details: dict[str, dict[str, dict[str, Any]]] = {}
    layer_thresholds: dict[str, dict[str, float]] = {}
    for mode in POLICY_MODES:
        val_confidence, test_confidence, source = confidences[mode]
        threshold_details[mode] = {}
        layer_thresholds[mode] = {}
        for target_coverage in (float(value) for value in plan["target_coverages"]):
            tuned = tune_coverage_threshold(
                val_confidence,
                target_coverage=target_coverage,
                source_split="val",
            )
            key = f"{target_coverage:.2f}"
            layer_thresholds[mode][key] = tuned["threshold"]
            threshold_details[mode][key] = tuned
            metrics = fixed_threshold_policy_metrics(
                probabilities,
                targets,
                target_mask,
                test_confidence,
                errors,
                labels=bundle.labels,
                classification_thresholds=bundle.thresholds,
                acceptance_threshold=tuned["threshold"],
                target_coverage=target_coverage,
            )
            result_rows.append(
                {
                    "phase": "V3-4",
                    "model_id": "P2",
                    "fold": fold,
                    "seed": seed,
                    "run_id": bundle.run_id,
                    "scenario": plan["scenario"],
                    "policy_mode": mode,
                    "confidence_source": source,
                    "selected_risk_family": selected_family,
                    "threshold_source_split": "val",
                    "test_rank_selection": False,
                    "validation_sample_count": tuned["validation_sample_count"],
                    "validation_accepted_count": tuned["validation_accepted_count"],
                    "validation_achieved_coverage": tuned[
                        "validation_achieved_coverage"
                    ],
                    "validation_coverage_absolute_error": tuned[
                        "validation_coverage_absolute_error"
                    ],
                    "protocol_sha256": lock["protocol_sha256"],
                    **metrics,
                }
            )

    output.mkdir(parents=True, exist_ok=True)
    results_path = output / "results.csv"
    policy_path = output / "policy.json"
    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    policy_artifact = DualPolicyDecisionLayer(layer_thresholds).to_artifact()
    policy_artifact.update(
        {
            "phase": "V3-4",
            "fold": fold,
            "seed": seed,
            "run_id": bundle.run_id,
            "selected_risk_family": selected_family,
            "threshold_details": threshold_details,
        }
    )
    write_json(policy_path, v31._json_safe(policy_artifact))
    source_paths = [
        data_manifest_path,
        val_path,
        bundle.checkpoint_path,
        bundle.run_path / "resolved_config.yaml",
        selector_path,
        parent_manifest_path,
    ]
    sources = [*frozen.source_entries]
    sources.extend(v31._relative_entry(root, path) for path in source_paths)
    manifest = {
        "version": 1,
        "phase": "V3-4",
        "status": "success",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "run_id": bundle.run_id,
        "threshold_fit_split": "val",
        "evaluation_split": "test",
        "test_rank_selection_allowed": False,
        "automatic_mode_selection_allowed": False,
        "policy_modes": list(POLICY_MODES),
        "target_coverages": [float(value) for value in plan["target_coverages"]],
        "selected_risk_family": selected_family,
        "validation_sample_count": int(val_usable.sum()),
        "test_sample_count": int(test_usable.sum()),
        "result_row_count": len(result_rows),
        "device": str(bundle.device),
        "inference_batch_size": int(plan["inference_batch_size"]),
        "validation_inference_seconds": float(validation["elapsed_seconds"]),
        "test_inference_seconds": float(test["elapsed_seconds"]),
        "max_abs_probability_delta_vs_frozen": max_delta,
        "sources": sources,
        "outputs": [
            v31._relative_entry(root, results_path),
            v31._relative_entry(root, policy_path),
        ],
    }
    write_json(output / "manifest.json", manifest)
    coverages = pd.DataFrame(result_rows).groupby("policy_mode")["test_coverage"].mean()
    print(
        f"DONE fold={fold} seed={seed}: risk={coverages['risk_control']:.3f}, "
        f"alert={coverages['error_alert']:.3f}, delta={max_delta:.2e}",
        flush=True,
    )
    return output


def _fold_and_overall(results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_columns = [
        "acceptance_threshold",
        "validation_achieved_coverage",
        "validation_coverage_absolute_error",
        "test_coverage",
        "coverage_absolute_error",
        "risk_masked_bce",
        "macro_f1",
        "micro_f1",
        "accepted_error_prevalence",
        "rejected_error_precision",
        "rejected_error_recall",
    ]
    fold = (
        results.groupby(["policy_mode", "target_coverage", "fold"], as_index=False)[
            value_columns
        ]
        .mean()
        .sort_values(["policy_mode", "target_coverage", "fold"])
    )
    fold["seed_count"] = 3
    rows = []
    for (mode, coverage), group in fold.groupby(
        ["policy_mode", "target_coverage"], sort=True
    ):
        row: dict[str, Any] = {
            "policy_mode": mode,
            "target_coverage": coverage,
            "fold_count": int(group["fold"].nunique()),
        }
        for column in value_columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[column] = float(np.nanmean(values))
            row[f"{column}_fold_std"] = float(np.nanstd(values, ddof=1))
        rows.append(row)
    return fold, pd.DataFrame(rows)


def _comparisons(
    fold: pd.DataFrame, plan: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    status: dict[str, Any] = {}
    for hypothesis_name, hypothesis in plan["primary_hypotheses"].items():
        metric = hypothesis["metric"]
        candidate = hypothesis["candidate"]
        reference = hypothesis["reference"]
        winning_targets = 0
        target_rows = []
        for coverage in (float(value) for value in plan["target_coverages"]):
            candidate_rows = fold[
                (fold["policy_mode"] == candidate)
                & np.isclose(fold["target_coverage"], coverage)
            ][["fold", metric]].rename(columns={metric: "candidate_value"})
            reference_rows = fold[
                (fold["policy_mode"] == reference)
                & np.isclose(fold["target_coverage"], coverage)
            ][["fold", metric]].rename(columns={metric: "reference_value"})
            paired = candidate_rows.merge(reference_rows, on="fold", validate="one_to_one")
            paired["delta"] = paired["candidate_value"] - paired["reference_value"]
            direction = hypothesis["direction"]
            fold_wins = int(
                (paired["delta"] < 0).sum()
                if direction == "lower"
                else (paired["delta"] > 0).sum()
            )
            delta_mean = float(paired["delta"].mean())
            correct_mean = delta_mean < 0 if direction == "lower" else delta_mean > 0
            target_pass = bool(
                correct_mean
                and fold_wins
                >= int(hypothesis["minimum_fold_wins_per_winning_coverage"])
            )
            winning_targets += int(target_pass)
            result = {
                "hypothesis": hypothesis_name,
                "candidate": candidate,
                "reference": reference,
                "metric": metric,
                "direction": direction,
                "target_coverage": coverage,
                "candidate_minus_reference_mean": delta_mean,
                "candidate_minus_reference_fold_std": float(
                    paired["delta"].std(ddof=1)
                ),
                "candidate_win_fold_count": fold_wins,
                "fold_count": int(len(paired)),
                "target_pass": target_pass,
            }
            rows.append(result)
            target_rows.append(result)
        passed = winning_targets >= int(hypothesis["minimum_winning_target_coverages"])
        status[hypothesis_name] = {
            "passed": passed,
            "winning_target_coverage_count": winning_targets,
            "required_winning_target_coverage_count": int(
                hypothesis["minimum_winning_target_coverages"]
            ),
            "targets": target_rows,
        }
    return rows, status


def _write_report(
    root: Path,
    protocol_sha256: str,
    overall: pd.DataFrame,
    comparisons: list[dict[str, Any]],
    hypotheses: dict[str, Any],
    readiness: dict[str, Any],
) -> Path:
    lines = [
        "# RobustSense V3-4 阶段报告",
        "",
        "## 1. 完成状态",
        "",
        "V3-4 已完成 15/15 个固定阈值决策单元；P2 与 V3-3 风险选择器均未重新训练。",
        "",
        f"- 协议哈希：`{protocol_sha256}`",
        "- validation 产生阈值，test 只应用固定阈值。",
        "- 每个单元 2 种模式 × 3 个目标覆盖率，共 90 行正式结果。",
        "",
        "## 2. 五折结果",
        "",
        (
            "| 模式 | 目标覆盖率 | 实际覆盖率 | 覆盖误差 | 接受 BCE↓ | "
            "Macro-F1↑ | 拒绝错误精度↑ | 拒绝错误召回↑ |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values(["policy_mode", "target_coverage"]).to_dict(
        "records"
    ):
        lines.append(
            "| `{mode}` | {target:.2f} | {actual:.4f} | {error:.4f} | "
            "{risk:.6f} | {f1:.6f} | {precision:.6f} | {recall:.6f} |".format(
                mode=row["policy_mode"],
                target=row["target_coverage"],
                actual=row["test_coverage"],
                error=row["coverage_absolute_error"],
                risk=row["risk_masked_bce"],
                f1=row["macro_f1"],
                precision=row["rejected_error_precision"],
                recall=row["rejected_error_recall"],
            )
        )
    lines.extend(["", "## 3. 预声明专长检验", ""])
    labels = {
        "risk_specialization": "风险控制专长",
        "alert_specialization": "错误告警专长",
    }
    for name, result in hypotheses.items():
        lines.append(
            f"- {labels[name]}：{'通过' if result['passed'] else '未通过'}；"
            f"满足 {result['winning_target_coverage_count']}/3 个目标覆盖率。"
        )
        for comparison in comparisons:
            if comparison["hypothesis"] == name:
                lines.append(
                    "  - 目标 {coverage:.2f}：候选减参照 {delta:+.6f}，"
                    "胜出 {wins}/5 折，{status}。".format(
                        coverage=comparison["target_coverage"],
                        delta=comparison["candidate_minus_reference_mean"],
                        wins=comparison["candidate_win_fold_count"],
                        status="通过" if comparison["target_pass"] else "未通过",
                    )
                )
    lines.extend(["", "## 4. 覆盖率迁移", ""])
    for mode, result in readiness.items():
        lines.append(
            f"- `{mode}`：平均绝对误差 {result['mean_coverage_absolute_error']:.6f}，"
            f"部署准备线{'通过' if result['passed'] else '未通过'}。"
        )
    lines.extend(
        [
            "",
            "## 5. 解释边界",
            "",
            (
                "本阶段验证的是固定阈值策略，不是测试集内的理想排序曲线。"
                "实际覆盖率允许偏离目标；模式必须由业务显式指定，不能根据 "
                "test 结果自动择优。完整性验收通过也不等于科学假设必然通过。"
            ),
            "",
        ]
    )
    path = root / "docs/v3/PHASE_V3_4_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aggregate_results(root: Path, plan: dict[str, Any], lock: dict[str, Any]) -> Path:
    frames = []
    manifests = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            unit = _unit_dir(root, plan, int(fold), int(seed))
            if not _unit_is_current(unit, root, lock["protocol_sha256"]):
                raise RuntimeError(f"V3-4 单元缺失或无效：fold={fold}, seed={seed}")
            frames.append(pd.read_csv(unit / "results.csv"))
            manifests.append(read_json(unit / "manifest.json"))
    results = pd.concat(frames, ignore_index=True)
    if len(results) != 90:
        raise RuntimeError(f"V3-4 结果行数不是 90：{len(results)}")
    fold, overall = _fold_and_overall(results)
    comparisons, hypotheses = _comparisons(fold, plan)
    maximum_error = float(plan["coverage_transfer_readiness"]["maximum_per_mode"])
    readiness = {}
    for mode, group in fold.groupby("policy_mode", sort=True):
        mean_error = float(group["coverage_absolute_error"].mean())
        readiness[mode] = {
            "mean_coverage_absolute_error": mean_error,
            "maximum_allowed": maximum_error,
            "passed": mean_error <= maximum_error,
        }

    results_path = root / "reports/v3/phase_v3_4_fold_seed_results.csv"
    fold_path = root / "reports/v3/phase_v3_4_fold_summary.csv"
    overall_path = root / "reports/v3/phase_v3_4_summary.csv"
    comparisons_path = root / "reports/v3/phase_v3_4_comparisons.csv"
    results.to_csv(results_path, index=False)
    fold.to_csv(fold_path, index=False)
    overall.to_csv(overall_path, index=False)
    pd.DataFrame(comparisons).to_csv(comparisons_path, index=False)
    report = {
        "version": 1,
        "phase": "V3-4",
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "unit_count": len(manifests),
        "result_row_count": len(results),
        "policy_modes": list(POLICY_MODES),
        "target_coverages": [float(value) for value in plan["target_coverages"]],
        "all_units_share_protocol": all(
            manifest["protocol_sha256"] == lock["protocol_sha256"]
            for manifest in manifests
        ),
        "primary_hypotheses": hypotheses,
        "dual_specialization_passed": all(
            result["passed"] for result in hypotheses.values()
        ),
        "coverage_transfer_readiness": readiness,
        "coverage_transfer_all_modes_passed": all(
            result["passed"] for result in readiness.values()
        ),
        "overall_summary": overall.to_dict("records"),
        "outputs": [
            v31._relative_entry(root, results_path),
            v31._relative_entry(root, fold_path),
            v31._relative_entry(root, overall_path),
            v31._relative_entry(root, comparisons_path),
        ],
    }
    report_path = root / "reports/v3/phase_v3_4_report.json"
    write_json(report_path, v31._json_safe(report))
    document = _write_report(
        root,
        lock["protocol_sha256"],
        overall,
        comparisons,
        hypotheses,
        readiness,
    )
    print(f"AGGREGATED V3-4 15/15：{report_path}", flush=True)
    print(f"REPORT：{document}", flush=True)
    return report_path


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
    plan, lock = _validate_protocol(root)
    registry = v32._p2_registry(root, plan)
    if arguments.aggregate_only:
        aggregate_results(root, plan, lock)
        return
    if arguments.all:
        for fold in plan["folds"]:
            for seed in plan["seeds"]:
                evaluate_unit(
                    root,
                    plan,
                    lock,
                    registry,
                    fold=int(fold),
                    seed=int(seed),
                    device_name=arguments.device,
                    force=arguments.force,
                )
        aggregate_results(root, plan, lock)
        return
    if arguments.fold is None or arguments.seed is None:
        parser.error("单元模式必须同时提供 --fold 与 --seed，或使用 --all")
    evaluate_unit(
        root,
        plan,
        lock,
        registry,
        fold=arguments.fold,
        seed=arguments.seed,
        device_name=arguments.device,
        force=arguments.force,
    )


if __name__ == "__main__":
    main()
