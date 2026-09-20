"""运行 V3-3 用户分组连续风险选择器实验。"""

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
from robustsense.evaluation.v2_metrics import masked_binary_cross_entropy_per_sample
from robustsense.evaluation.v3_grouped_risk import (
    build_compact_risk_features,
    select_grouped_ridge_risk_selector,
)
from robustsense.evaluation.v3_selective import (
    mean_binary_certainty,
    sample_error_indicator,
    selective_curve,
    summarize_selector,
)
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _validate_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_config(root / "configs/v3/phase_v3_3_plan.yaml")
    lock = read_json(root / "reports/v3/phase_v3_3_protocol_lock.json")
    if lock.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-3 协议尚未在首次 test 评估前冻结")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V3-3 协议锁校验失败：{problems}")
    return plan, lock


def _unit_dir(root: Path, plan: dict[str, Any], fold: int, seed: int) -> Path:
    return root / plan["unit_output_root"] / f"fold{fold}_seed{seed}"


def _unit_is_current(output: Path, root: Path, protocol_sha256: str) -> bool:
    path = output / "manifest.json"
    if not path.is_file():
        return False
    manifest = read_json(path)
    if (
        manifest.get("status") != "success"
        or manifest.get("protocol_sha256") != protocol_sha256
    ):
        return False
    return not verify_file_entries(
        root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
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
        raise ValueError("fold 或 seed 超出 V3-3 冻结矩阵")
    output = _unit_dir(root, plan, fold, seed)
    if not force and _unit_is_current(output, root, lock["protocol_sha256"]):
        print(f"SKIP fold={fold} seed={seed}: 现有 V3-3 单元通过哈希校验", flush=True)
        return output

    row = registry[(fold, seed)]
    run_path = root / row["run_dir"]
    frozen = v31._load_prediction_artifact(root, "P2", run_path)
    bundle, validation, test, manifest_path, val_path = v32._collect_splits(
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
    val_losses = masked_binary_cross_entropy_per_sample(
        validation["probabilities"][val_usable],
        validation["targets"][val_usable],
        validation["target_mask"][val_usable],
    )
    test_errors = sample_error_indicator(
        test["probabilities"][test_usable],
        test["targets"][test_usable],
        test["target_mask"][test_usable],
        bundle.thresholds,
    )
    if not np.isfinite(val_losses).all() or not np.isfinite(test_errors).all():
        raise RuntimeError("V3-3 validation 风险或 test 错误目标含无效值")
    val_groups = np.asarray(validation["user_id"]).astype(str)[val_usable]
    risk_config = plan["risk_model"]
    coverage_grid = np.asarray(plan["coverage_grid"], dtype=np.float64)

    fitted: dict[str, Any] = {}
    test_confidences: dict[str, np.ndarray] = {
        "mean_binary_certainty": mean_binary_certainty(
            test["probabilities"][test_usable]
        ),
        "p2_system_reliability": np.asarray(
            test["system_reliability"], dtype=np.float64
        )[test_usable],
    }
    for selector_name, include_sensor in (
        ("grouped_risk_output_compact", False),
        ("grouped_risk_full_compact", True),
    ):
        val_features, feature_names = build_compact_risk_features(
            validation,
            modality_names=list(MODALITIES),
            thresholds=bundle.thresholds,
            include_sensor_features=include_sensor,
        )
        test_features, test_feature_names = build_compact_risk_features(
            test,
            modality_names=list(MODALITIES),
            thresholds=bundle.thresholds,
            include_sensor_features=include_sensor,
        )
        if feature_names != test_feature_names:
            raise RuntimeError(f"{selector_name} validation/test 特征顺序不一致")
        selector, selection = select_grouped_ridge_risk_selector(
            val_features[val_usable],
            feature_names,
            val_losses,
            val_groups,
            alphas=[float(value) for value in risk_config["alpha_candidates"]],
            n_splits=int(risk_config["grouped_cv_splits"]),
            solver=str(risk_config["solver"]),
            coverage_grid=coverage_grid,
        )
        if any(row["group_overlap_count"] != 0 for row in selection["fold_contract"]):
            raise RuntimeError(f"{selector_name} GroupKFold 出现用户重叠")
        test_confidences[selector_name] = selector.predict_confidence(
            test_features[test_usable]
        )
        fitted[selector_name] = {
            "selector": selector.to_artifact(),
            "selection": selection,
            "feature_count": len(feature_names),
        }

    output_oof = fitted["grouped_risk_output_compact"]["selection"][
        "selected_oof_normalized_aurc"
    ]
    full_oof = fitted["grouped_risk_full_compact"]["selection"][
        "selected_oof_normalized_aurc"
    ]
    selected_family = (
        "grouped_risk_full_compact"
        if full_oof < output_oof
        else "grouped_risk_output_compact"
    )
    test_confidences["grouped_risk_selected"] = test_confidences[selected_family].copy()

    probabilities = np.asarray(test["probabilities"], dtype=np.float64)[test_usable]
    targets = np.asarray(test["targets"], dtype=np.float64)[test_usable]
    target_mask = np.asarray(test["target_mask"], dtype=bool)[test_usable]
    fixed_coverages = tuple(float(value) for value in plan["fixed_coverages"])
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for score_method in plan["selectors"]:
        confidence = test_confidences[score_method]
        curve = selective_curve(
            probabilities,
            targets,
            target_mask,
            confidence,
            labels=bundle.labels,
            thresholds=bundle.thresholds,
            coverage_grid=coverage_grid,
            run_id=bundle.run_id,
            score_method=score_method,
        )
        summary = summarize_selector(
            curve,
            confidence,
            test_errors,
            run_id=bundle.run_id,
            score_method=score_method,
            fixed_coverages=fixed_coverages,
        )
        common = {
            "phase": "V3-3",
            "model_id": "P2",
            "fold": fold,
            "seed": seed,
            "scenario": plan["scenario"],
            "protocol_sha256": lock["protocol_sha256"],
            "selected_family": selected_family,
            "source_prediction_sha256": frozen.source_entries[0]["sha256"],
        }
        curve_rows.extend({**common, **curve_row} for curve_row in curve)
        summary_rows.append({**common, **summary})

    output.mkdir(parents=True, exist_ok=True)
    curves_path = output / "curves.csv"
    summary_path = output / "summary.csv"
    selectors_path = output / "selectors.json"
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    write_json(
        selectors_path,
        {
            "version": 1,
            "phase": "V3-3",
            "fit_split": "val",
            "selection_split": "val_grouped_oof",
            "fold": fold,
            "seed": seed,
            "run_id": bundle.run_id,
            "selected_family": selected_family,
            "selectors": fitted,
        },
    )
    source_paths = [
        manifest_path,
        val_path,
        bundle.checkpoint_path,
        bundle.run_path / "resolved_config.yaml",
    ]
    sources = [*frozen.source_entries]
    sources.extend(v31._relative_entry(root, path) for path in source_paths)
    manifest = {
        "version": 1,
        "phase": "V3-3",
        "status": "success",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "run_id": bundle.run_id,
        "fit_split": "val",
        "selection_split": "val_grouped_oof",
        "evaluation_split": "test",
        "grouping_unit": "user_id",
        "fit_sample_count": int(val_usable.sum()),
        "test_sample_count": int(test_usable.sum()),
        "validation_unique_user_count": int(len(np.unique(val_groups))),
        "selector_count": len(test_confidences),
        "fitted_selector_count": len(fitted),
        "feature_counts": {
            name: artifact["feature_count"] for name, artifact in fitted.items()
        },
        "selected_family": selected_family,
        "selected_family_oof_aurc": min(output_oof, full_oof),
        "device": str(bundle.device),
        "inference_batch_size": int(plan["inference_batch_size"]),
        "validation_inference_seconds": float(validation["elapsed_seconds"]),
        "test_inference_seconds": float(test["elapsed_seconds"]),
        "max_abs_probability_delta_vs_frozen": max_delta,
        "sources": sources,
        "outputs": [
            v31._relative_entry(root, curves_path),
            v31._relative_entry(root, summary_path),
            v31._relative_entry(root, selectors_path),
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"DONE fold={fold} seed={seed}: selected={selected_family}, "
        f"OOF={min(output_oof, full_oof):.6f}, delta={max_delta:.2e}",
        flush=True,
    )
    return output


def _write_report(
    root: Path,
    protocol_sha256: str,
    overall: pd.DataFrame,
    comparisons: list[dict[str, Any]],
    selected_counts: dict[str, int],
) -> Path:
    lines = [
        "# RobustSense V3-3 阶段报告",
        "",
        "## 1. 完成状态",
        "",
        "V3-3 已完成 15 个用户分组连续风险单元；P2 主模型未重新训练。",
        "",
        f"- 协议哈希：`{protocol_sha256}`",
        "- 正式单元：15/15",
        (
            "- validation 选择 output-compact："
            f"{selected_counts.get('grouped_risk_output_compact', 0)} 个单元"
        ),
        (
            "- validation 选择 full-compact："
            f"{selected_counts.get('grouped_risk_full_compact', 0)} 个单元"
        ),
        "",
        "## 2. 五折聚合结果",
        "",
        "| 选择器 | AURC↓ | 错误检测 AUROC↑ | AUPRC↑ | Risk@80%↓ | Macro-F1@80%↑ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values("score_method").to_dict("records"):
        lines.append(
            "| `{score}` | {aurc:.6f} | {auroc:.6f} | {auprc:.6f} | "
            "{risk:.6f} | {f1:.6f} |".format(
                score=row["score_method"],
                aurc=row["normalized_aurc"],
                auroc=row["error_detection_auroc"],
                auprc=row["error_detection_auprc"],
                risk=row["risk_masked_bce_0_80"],
                f1=row["macro_f1_0_80"],
            )
        )
    lines.extend(["", "## 3. 预声明主比较", ""])
    for comparison in comparisons:
        aurc = comparison["metrics"]["normalized_aurc"]
        auroc = comparison["metrics"]["error_detection_auroc"]
        conclusion = "通过" if comparison["primary_success"] else "未同时通过"
        lines.extend(
            [
                f"### {comparison['name']}",
                "",
                f"- 候选：`{comparison['candidate']}`",
                f"- 参照：`{comparison['reference']}`",
                (
                    f"- AURC 差值：{aurc['candidate_minus_reference_mean']:.6f}；"
                    f"候选胜出 {aurc['candidate_win_fold_count']}/5 折。"
                ),
                (
                    "- 错误检测 AUROC 差值："
                    f"{auroc['candidate_minus_reference_mean']:.6f}；候选胜出 "
                    f"{auroc['candidate_win_fold_count']}/5 折。"
                ),
                f"- 双主指标判定：**{conclusion}**。",
                "",
            ]
        )
    lines.extend(
        [
            "## 4. 解释边界",
            "",
            (
                "所有 α 与特征家族选择均来自 validation 用户分组 OOF AURC；test 只用于"
                "最终评分。selected 曲线必须与 output/full 原始曲线一起解释。"
            ),
            "",
            "## 5. 可复核产物",
            "",
            "每个单元均保存 α 候选、四折用户隔离合同、最终 Ridge 参数、曲线和摘要。",
            "",
        ]
    )
    path = root / "docs/v3/PHASE_V3_3_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aggregate_results(root: Path, plan: dict[str, Any], lock: dict[str, Any]) -> Path:
    curve_frames = []
    summary_frames = []
    manifests = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            unit = _unit_dir(root, plan, int(fold), int(seed))
            if not _unit_is_current(unit, root, lock["protocol_sha256"]):
                raise RuntimeError(f"V3-3 单元缺失或无效：fold={fold}, seed={seed}")
            curve_frames.append(pd.read_csv(unit / "curves.csv"))
            summary_frames.append(pd.read_csv(unit / "summary.csv"))
            manifests.append(read_json(unit / "manifest.json"))
    curves = pd.concat(curve_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    if len(curves) != 1500 or len(summaries) != 75:
        raise RuntimeError(f"V3-3 曲线/摘要数量异常：{len(curves)}/{len(summaries)}")
    fold_seed_curves_path = root / "reports/v3/phase_v3_3_fold_seed_curves.csv"
    fold_seed_summary_path = root / "reports/v3/phase_v3_3_fold_seed_summary.csv"
    curves.to_csv(fold_seed_curves_path, index=False)
    summaries.to_csv(fold_seed_summary_path, index=False)
    value_columns = v31._numeric_value_columns(
        summaries, excluded={"fold", "seed", "valid_count", "error_count"}
    )
    fold_summary = (
        summaries.groupby(["model_id", "score_method", "fold"], as_index=False)[
            value_columns
        ]
        .mean()
        .sort_values(["score_method", "fold"])
    )
    fold_summary["seed_count"] = 3
    fold_summary_path = root / "reports/v3/phase_v3_3_fold_summary.csv"
    fold_summary.to_csv(fold_summary_path, index=False)
    overall = v31._mean_with_fold_std(fold_summary, value_columns)
    overall_path = root / "reports/v3/phase_v3_3_summary.csv"
    overall.to_csv(overall_path, index=False)

    curve_values = [
        "coverage",
        "accepted_count",
        "risk_masked_bce",
        "macro_f1",
        "micro_f1",
        "known_target_count",
        "confidence_cutoff",
    ]
    fold_curves = curves.groupby(
        ["model_id", "score_method", "fold", "requested_coverage"], as_index=False
    )[curve_values].mean()
    aggregate_rows = []
    for (model_id, score_method, requested), group in fold_curves.groupby(
        ["model_id", "score_method", "requested_coverage"], sort=True
    ):
        aggregate: dict[str, Any] = {
            "model_id": model_id,
            "score_method": score_method,
            "requested_coverage": requested,
            "fold_count": int(group["fold"].nunique()),
        }
        for column in curve_values:
            values = group[column].to_numpy(dtype=np.float64)
            aggregate[column] = float(np.nanmean(values))
            aggregate[f"{column}_fold_std"] = float(np.nanstd(values, ddof=1))
        aggregate_rows.append(aggregate)
    aggregate_curves = pd.DataFrame(aggregate_rows)
    curves_path = root / "reports/v3/phase_v3_3_curves.csv"
    aggregate_curves.to_csv(curves_path, index=False)

    comparisons = [
        v31._paired_comparison(
            fold_summary,
            name="grouped_continuous_risk_gain",
            candidate_model="P2",
            candidate_score="grouped_risk_selected",
            reference_model="P2",
            reference_score="mean_binary_certainty",
        ),
        v31._paired_comparison(
            fold_summary,
            name="compact_sensor_increment",
            candidate_model="P2",
            candidate_score="grouped_risk_full_compact",
            reference_model="P2",
            reference_score="grouped_risk_output_compact",
        ),
    ]
    selected_counts = pd.Series(
        [manifest["selected_family"] for manifest in manifests]
    ).value_counts().to_dict()
    report = {
        "version": 1,
        "phase": "V3-3",
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "unit_count": len(manifests),
        "selector_summary_count": len(summaries),
        "curve_point_count": len(curves),
        "all_units_share_protocol": all(
            manifest["protocol_sha256"] == lock["protocol_sha256"]
            for manifest in manifests
        ),
        "selected_family_counts": selected_counts,
        "primary_comparisons": comparisons,
        "overall_summary": overall.to_dict("records"),
        "outputs": [
            v31._relative_entry(root, fold_seed_curves_path),
            v31._relative_entry(root, fold_seed_summary_path),
            v31._relative_entry(root, fold_summary_path),
            v31._relative_entry(root, overall_path),
            v31._relative_entry(root, curves_path),
        ],
    }
    report_path = root / "reports/v3/phase_v3_3_report.json"
    write_json(report_path, v31._json_safe(report))
    document = _write_report(
        root, lock["protocol_sha256"], overall, comparisons, selected_counts
    )
    print(f"AGGREGATED V3-3 15/15：{report_path}", flush=True)
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
