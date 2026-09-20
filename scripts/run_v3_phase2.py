"""运行 V3-2 validation 拟合的任务对齐置信度实验。"""

from __future__ import annotations

import argparse
import csv
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_v3_phase1 as v31

from robustsense.constants import MODALITIES
from robustsense.data.dataset import MultiModalDataset
from robustsense.evaluation.v3_selective import (
    error_detection_metrics,
    mean_binary_certainty,
    sample_error_indicator,
    selective_curve,
    summarize_selector,
)
from robustsense.evaluation.v3_task_confidence import (
    build_task_confidence_features,
    fit_task_confidence_selector,
)
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.training.trainer import _collect_predictions, _loader
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json
from robustsense.v2_demo import load_demo_bundle


def _validate_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_config(root / "configs/v3/phase_v3_2_plan.yaml")
    lock = read_json(root / "reports/v3/phase_v3_2_protocol_lock.json")
    if lock.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-2 协议尚未在首次测试评估前冻结")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V3-2 协议锁校验失败：{problems}")
    return plan, lock


def _p2_registry(
    root: Path, plan: dict[str, Any]
) -> dict[tuple[int, int], dict[str, str]]:
    with (root / plan["core_registry"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        (int(row["fold"]), int(row["seed"])): row
        for row in rows
        if row["model_name"] == "P2" and row["status"] == "success"
    }
    expected = len(plan["folds"]) * len(plan["seeds"])
    if len(mapping) != expected:
        raise RuntimeError(f"P2 核心登记表只有 {len(mapping)}/{expected} 个成功单元")
    return mapping


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


def _collect_splits(
    root: Path,
    *,
    fold: int,
    seed: int,
    device_name: str,
    batch_size: int,
) -> tuple[Any, dict[str, Any], dict[str, Any], Path, Path]:
    bundle = load_demo_bundle(root, "P2", fold, seed, device_name=device_name)
    resolved_path = bundle.run_path / "resolved_config.yaml"
    resolved = load_config(resolved_path)
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    val_path = processed_dir / "val.npz"
    val_dataset = MultiModalDataset(val_path, manifest_path)
    val_loader = _loader(val_dataset, batch_size, False, seed, bundle.device)
    test_loader = _loader(bundle.dataset, batch_size, False, seed, bundle.device)
    validation = _collect_predictions(
        bundle.model, val_loader, bundle.device, bundle.modality_slices
    )
    test = _collect_predictions(
        bundle.model, test_loader, bundle.device, bundle.modality_slices
    )
    return bundle, validation, test, manifest_path, val_path


def _validate_frozen_test(
    artifact: Any, collected: dict[str, Any], atol: float
) -> float:
    users = np.asarray(collected["user_id"]).astype(str)
    timestamps = np.asarray(collected["timestamp"], dtype=np.int64)
    if not (
        np.array_equal(users, artifact.users)
        and np.array_equal(timestamps, artifact.timestamps)
    ):
        raise RuntimeError("V3-2 test 样本键与冻结 P2 预测不一致")
    probabilities = np.asarray(collected["probabilities"], dtype=np.float64)
    delta = float(np.max(np.abs(probabilities - artifact.probabilities)))
    if not np.allclose(probabilities, artifact.probabilities, rtol=0.0, atol=atol):
        raise RuntimeError(f"V3-2 test 概率与冻结 P2 预测不一致：{delta}")
    if not np.array_equal(collected["target_mask"], artifact.target_mask):
        raise RuntimeError("V3-2 test 已知标签掩码与冻结产物不一致")
    if not np.allclose(
        collected["targets"], artifact.targets, rtol=0.0, atol=0.0, equal_nan=True
    ):
        raise RuntimeError("V3-2 test 标签与冻结产物不一致")
    return delta


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
        raise ValueError("fold 或 seed 超出 V3-2 冻结矩阵")
    output = _unit_dir(root, plan, fold, seed)
    if not force and _unit_is_current(output, root, lock["protocol_sha256"]):
        print(f"SKIP fold={fold} seed={seed}: 现有 V3-2 单元通过哈希校验", flush=True)
        return output

    row = registry[(fold, seed)]
    run_path = root / row["run_dir"]
    frozen = v31._load_prediction_artifact(root, "P2", run_path)
    bundle, validation, test, manifest_path, val_path = _collect_splits(
        root,
        fold=fold,
        seed=seed,
        device_name=device_name,
        batch_size=int(plan["inference_batch_size"]),
    )
    max_delta = _validate_frozen_test(
        frozen, test, float(plan["probability_consistency_atol"])
    )
    if frozen.labels != bundle.labels:
        raise RuntimeError("V3-2 bundle 标签顺序与冻结预测不一致")

    val_usable = np.asarray(validation["target_mask"], dtype=bool).any(axis=1)
    test_usable = np.asarray(test["target_mask"], dtype=bool).any(axis=1)
    if not val_usable.any() or not test_usable.any():
        raise RuntimeError("V3-2 validation 或 test 没有可评分样本")
    val_errors = sample_error_indicator(
        validation["probabilities"][val_usable],
        validation["targets"][val_usable],
        validation["target_mask"][val_usable],
        bundle.thresholds,
    )
    test_errors = sample_error_indicator(
        test["probabilities"][test_usable],
        test["targets"][test_usable],
        test["target_mask"][test_usable],
        bundle.thresholds,
    )

    selector_config = plan["selector_model"]
    fitted: dict[str, Any] = {}
    test_confidences: dict[str, np.ndarray] = {
        "mean_binary_certainty": mean_binary_certainty(
            test["probabilities"][test_usable]
        ),
        "p2_system_reliability": np.asarray(
            test["system_reliability"], dtype=np.float64
        )[test_usable],
    }
    for name, include_sensor in (
        ("task_confidence_output_only", False),
        ("task_confidence_full", True),
    ):
        val_features, feature_names = build_task_confidence_features(
            validation,
            labels=bundle.labels,
            modality_names=list(MODALITIES),
            thresholds=bundle.thresholds,
            include_sensor_features=include_sensor,
        )
        test_features, test_feature_names = build_task_confidence_features(
            test,
            labels=bundle.labels,
            modality_names=list(MODALITIES),
            thresholds=bundle.thresholds,
            include_sensor_features=include_sensor,
        )
        if feature_names != test_feature_names:
            raise RuntimeError(f"{name} validation/test 特征顺序不一致")
        selector = fit_task_confidence_selector(
            val_features[val_usable],
            feature_names,
            val_errors,
            regularization_c=float(selector_config["regularization_c"]),
            solver=str(selector_config["solver"]),
            max_iter=int(selector_config["max_iter"]),
        )
        if selector.iterations >= selector.max_iter:
            raise RuntimeError(f"{name} 达到最大迭代次数，未满足冻结拟合合同")
        val_confidence = selector.predict_confidence(val_features[val_usable])
        test_confidences[name] = selector.predict_confidence(test_features[test_usable])
        fitted[name] = {
            **selector.to_artifact(),
            "fit_split": "val",
            "fit_sample_count": int(val_usable.sum()),
            "fit_error_count": int(np.sum(val_errors)),
            "fit_error_prevalence": float(np.mean(val_errors)),
            "fit_in_sample_diagnostics": error_detection_metrics(
                val_confidence,
                val_errors,
                run_id=bundle.run_id,
                score_method=name,
            ),
        }

    probabilities = np.asarray(test["probabilities"], dtype=np.float64)[test_usable]
    targets = np.asarray(test["targets"], dtype=np.float64)[test_usable]
    target_mask = np.asarray(test["target_mask"], dtype=bool)[test_usable]
    coverage_grid = np.asarray(plan["coverage_grid"], dtype=np.float64)
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
            "phase": "V3-2",
            "model_id": "P2",
            "fold": fold,
            "seed": seed,
            "scenario": plan["scenario"],
            "protocol_sha256": lock["protocol_sha256"],
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
            "phase": "V3-2",
            "source_split": "val",
            "fold": fold,
            "seed": seed,
            "run_id": bundle.run_id,
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
        "phase": "V3-2",
        "status": "success",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "run_id": bundle.run_id,
        "fit_split": "val",
        "evaluation_split": "test",
        "fit_sample_count": int(val_usable.sum()),
        "test_sample_count": int(test_usable.sum()),
        "fit_error_prevalence": float(np.mean(val_errors)),
        "test_error_prevalence": float(np.mean(test_errors)),
        "selector_count": len(test_confidences),
        "fitted_selector_count": len(fitted),
        "feature_counts": {
            name: len(artifact["feature_names"]) for name, artifact in fitted.items()
        },
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
        f"DONE fold={fold} seed={seed}: val={int(val_usable.sum())}, "
        f"test={int(test_usable.sum())}, 4 条曲线, delta={max_delta:.2e}",
        flush=True,
    )
    return output


def _write_report(
    root: Path,
    protocol_sha256: str,
    overall: pd.DataFrame,
    comparisons: list[dict[str, Any]],
) -> Path:
    lines = [
        "# RobustSense V3-2 阶段报告",
        "",
        "## 1. 完成状态",
        "",
        (
            "V3-2 已完成 5 折 × 3 随机种子的 validation 任务置信度拟合与冻结 test "
            "评估。P2 主模型未重新训练。"
        ),
        "",
        f"- 协议哈希：`{protocol_sha256}`",
        "- 正式单元：15/15",
        "- 拟合选择器：output-only 与 full",
        "- test 曲线：60 条",
        "",
        "## 2. 五折聚合结果",
        "",
        "| 选择器 | AURC↓ | 错误检测 AUROC↑ | 错误检测 AUPRC↑ | Risk@80%↓ | Macro-F1@80%↑ |",
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
                "任务置信度层只使用 validation 标签拟合；test 仅用于最终评分。full 对 "
                "output-only 的差异代表传感器状态和可靠度特征在逐标签输出之外的增量价值。"
            ),
            "",
            "## 5. 可复核产物",
            "",
            "完整 selector 参数、曲线、fold/seed 摘要与机器可读比较均位于 `reports/v3/`。",
            "",
        ]
    )
    path = root / "docs/v3/PHASE_V3_2_REPORT.md"
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
                raise RuntimeError(f"V3-2 单元缺失或无效：fold={fold}, seed={seed}")
            curve_frames.append(pd.read_csv(unit / "curves.csv"))
            summary_frames.append(pd.read_csv(unit / "summary.csv"))
            manifests.append(read_json(unit / "manifest.json"))
    curves = pd.concat(curve_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    if len(curves) != 1200 or len(summaries) != 60:
        raise RuntimeError(f"V3-2 曲线/摘要数量异常：{len(curves)}/{len(summaries)}")

    fold_seed_curves_path = root / "reports/v3/phase_v3_2_fold_seed_curves.csv"
    fold_seed_summary_path = root / "reports/v3/phase_v3_2_fold_seed_summary.csv"
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
    fold_summary_path = root / "reports/v3/phase_v3_2_fold_summary.csv"
    fold_summary.to_csv(fold_summary_path, index=False)
    overall = v31._mean_with_fold_std(fold_summary, value_columns)
    overall_path = root / "reports/v3/phase_v3_2_summary.csv"
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
    curves_path = root / "reports/v3/phase_v3_2_curves.csv"
    aggregate_curves.to_csv(curves_path, index=False)

    comparisons = [
        v31._paired_comparison(
            fold_summary,
            name="task_alignment_gain",
            candidate_model="P2",
            candidate_score="task_confidence_full",
            reference_model="P2",
            reference_score="mean_binary_certainty",
        ),
        v31._paired_comparison(
            fold_summary,
            name="sensor_feature_increment",
            candidate_model="P2",
            candidate_score="task_confidence_full",
            reference_model="P2",
            reference_score="task_confidence_output_only",
        ),
    ]
    report = {
        "version": 1,
        "phase": "V3-2",
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
    report_path = root / "reports/v3/phase_v3_2_report.json"
    write_json(report_path, v31._json_safe(report))
    document = _write_report(root, lock["protocol_sha256"], overall, comparisons)
    print(f"AGGREGATED V3-2 15/15：{report_path}", flush=True)
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
    registry = _p2_registry(root, plan)
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
