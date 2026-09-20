"""运行 V3-1 同覆盖率选择性预测评估并生成中文报告。"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.data.extrasensory import sha256_file
from robustsense.evaluation.v3_selective import (
    mean_binary_certainty,
    mean_normalized_threshold_margin,
    sample_error_indicator,
    selective_curve,
    summarize_selector,
)
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.training.trainer import _collect_predictions, _loader
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json
from robustsense.v2_demo import load_demo_bundle


@dataclass
class PredictionArtifact:
    model_id: str
    run_id: str
    run_path: Path
    labels: list[str]
    thresholds: np.ndarray
    users: np.ndarray
    timestamps: np.ndarray
    probabilities: np.ndarray
    targets: np.ndarray
    target_mask: np.ndarray
    source_entries: list[dict[str, Any]]


def _relative_entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_protocol(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_config(root / "configs/v3/phase_v3_1_plan.yaml")
    lock = read_json(root / "reports/v3/phase_v3_1_protocol_lock.json")
    if lock.get("status") != "frozen_before_first_test_evaluation":
        raise RuntimeError("V3-1 协议尚未在第一次测试评估前冻结")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V3-1 协议锁校验失败：{problems}")
    if plan.get("status") != lock.get("status"):
        raise RuntimeError("V3-1 计划与协议锁状态不一致")
    return plan, lock


def _registry(root: Path, plan: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, str]]:
    with (root / plan["core_registry"]).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        (row["model_name"], int(row["fold"]), int(row["seed"])): row
        for row in rows
        if row["status"] == "success"
    }
    expected = len(plan["models"]) * len(plan["folds"]) * len(plan["seeds"])
    if len(mapping) != expected:
        raise RuntimeError(f"核心登记表只有 {len(mapping)}/{expected} 个唯一成功单元")
    return mapping


def _sample_key_sha256(users: np.ndarray, timestamps: np.ndarray) -> str:
    digest = hashlib.sha256()
    for user, timestamp in zip(users, timestamps, strict=True):
        digest.update(str(user).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(int(timestamp)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _load_prediction_artifact(
    root: Path, model_id: str, run_path: Path
) -> PredictionArtifact:
    prediction_path = run_path / "predictions.parquet"
    threshold_path = run_path / "thresholds.json"
    threshold_artifact = read_json(threshold_path)
    labels = list(threshold_artifact.get("labels", []))
    thresholds = np.asarray(threshold_artifact.get("values", []), dtype=np.float64)
    if threshold_artifact.get("source_split") != "val":
        raise RuntimeError(f"{model_id} 分类阈值不是 validation 来源")
    if not labels or thresholds.shape != (len(labels),):
        raise RuntimeError(f"{model_id} 标签与阈值维度不一致")

    columns = [
        "run_id",
        "fold",
        "user_id",
        "timestamp",
        "scenario",
        "label",
        "target",
        "target_known",
        "probability",
    ]
    frame = pd.read_parquet(prediction_path, columns=columns)
    if frame.empty or set(frame["scenario"].astype(str)) != {"natural_missingness"}:
        raise RuntimeError(f"{model_id} 预测产物不是唯一的自然缺失场景")
    run_ids = set(frame["run_id"].astype(str))
    if run_ids != {run_path.name}:
        raise RuntimeError(f"{model_id} 预测 run_id 与目录名不一致：{run_ids}")

    users: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    for label in labels:
        block = frame.loc[frame["label"].astype(str) == label]
        if block.empty:
            raise RuntimeError(f"{model_id} 缺少标签预测：{label}")
        block_users = block["user_id"].astype(str).to_numpy()
        block_timestamps = block["timestamp"].to_numpy(dtype=np.int64)
        if users is None:
            users = block_users
            timestamps = block_timestamps
            keys = pd.DataFrame({"user": users, "timestamp": timestamps})
            if keys.duplicated().any():
                raise RuntimeError(f"{model_id} 样本键存在重复")
        elif not (
            np.array_equal(users, block_users)
            and np.array_equal(timestamps, block_timestamps)
        ):
            raise RuntimeError(f"{model_id} 不同标签的样本顺序不一致")
        probabilities.append(block["probability"].to_numpy(dtype=np.float64))
        targets.append(block["target"].to_numpy(dtype=np.float64))
        target_masks.append(block["target_known"].to_numpy(dtype=bool))
    if len(frame) != len(labels) * len(probabilities[0]):
        raise RuntimeError(f"{model_id} 预测行数不符合样本数乘标签数")
    del frame
    gc.collect()
    assert users is not None and timestamps is not None
    return PredictionArtifact(
        model_id=model_id,
        run_id=run_path.name,
        run_path=run_path,
        labels=labels,
        thresholds=thresholds,
        users=users,
        timestamps=timestamps,
        probabilities=np.column_stack(probabilities),
        targets=np.column_stack(targets),
        target_mask=np.column_stack(target_masks),
        source_entries=[
            _relative_entry(root, prediction_path),
            _relative_entry(root, threshold_path),
        ],
    )


def _validate_shared_population(artifacts: dict[str, PredictionArtifact]) -> None:
    reference = artifacts["B4"]
    for model_id, artifact in artifacts.items():
        if artifact.labels != reference.labels:
            raise RuntimeError(f"{model_id} 标签顺序与 B4 不一致")
        if not (
            np.array_equal(artifact.users, reference.users)
            and np.array_equal(artifact.timestamps, reference.timestamps)
        ):
            raise RuntimeError(f"{model_id} 样本键或顺序与 B4 不一致")
        if not np.array_equal(artifact.target_mask, reference.target_mask):
            raise RuntimeError(f"{model_id} 已知标签掩码与 B4 不一致")
        if not np.allclose(
            artifact.targets, reference.targets, rtol=0.0, atol=0.0, equal_nan=True
        ):
            raise RuntimeError(f"{model_id} 真实标签与 B4 不一致")


def _p2_system_reliability(
    root: Path,
    artifact: PredictionArtifact,
    *,
    fold: int,
    seed: int,
    device_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    bundle = load_demo_bundle(root, "P2", fold, seed, device_name=device_name)
    loader = _loader(bundle.dataset, batch_size, False, seed, bundle.device)
    collected = _collect_predictions(
        bundle.model, loader, bundle.device, bundle.modality_slices
    )
    users = np.asarray(collected["user_id"]).astype(str)
    timestamps = np.asarray(collected["timestamp"], dtype=np.int64)
    if not (
        np.array_equal(users, artifact.users)
        and np.array_equal(timestamps, artifact.timestamps)
    ):
        raise RuntimeError("P2 重新推理的样本键与冻结预测不一致")
    probabilities = np.asarray(collected["probabilities"], dtype=np.float64)
    max_delta = float(np.max(np.abs(probabilities - artifact.probabilities)))
    if not np.allclose(probabilities, artifact.probabilities, rtol=0.0, atol=1.0e-6):
        raise RuntimeError(f"P2 重新推理概率与冻结产物不一致：max_delta={max_delta}")
    reliability = np.asarray(collected["system_reliability"], dtype=np.float64).reshape(-1)
    if reliability.shape != (len(artifact.probabilities),):
        raise RuntimeError("P2 系统可靠度不是每个样本一个值")
    if not np.isfinite(reliability).all() or ((reliability < 0) | (reliability > 1)).any():
        raise RuntimeError("P2 系统可靠度含非法值")
    metadata = {
        "device": str(bundle.device),
        "batch_size": batch_size,
        "elapsed_seconds": float(collected["elapsed_seconds"]),
        "latency_ms_per_sample": float(collected["latency_ms_per_sample"]),
        "max_abs_probability_delta_vs_frozen": max_delta,
    }
    sources = [
        _relative_entry(root, bundle.checkpoint_path),
        _relative_entry(root, bundle.run_path / "resolved_config.yaml"),
    ]
    return reliability, metadata, sources


def _unit_output_dir(root: Path, plan: dict[str, Any], fold: int, seed: int) -> Path:
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


def evaluate_unit(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[tuple[str, int, int], dict[str, str]],
    *,
    fold: int,
    seed: int,
    device_name: str,
    force: bool,
) -> Path:
    if fold not in plan["folds"] or seed not in plan["seeds"]:
        raise ValueError("fold 或 seed 超出 V3-1 冻结矩阵")
    output = _unit_output_dir(root, plan, fold, seed)
    if not force and _unit_is_current(output, root, lock["protocol_sha256"]):
        print(f"SKIP fold={fold} seed={seed}: 已有通过协议与源文件校验的结果", flush=True)
        return output

    artifacts: dict[str, PredictionArtifact] = {}
    for model_id in plan["models"]:
        row = registry[(model_id, fold, seed)]
        artifacts[model_id] = _load_prediction_artifact(
            root, model_id, root / row["run_dir"]
        )
    _validate_shared_population(artifacts)
    p2_reliability, inference, p2_sources = _p2_system_reliability(
        root,
        artifacts["P2"],
        fold=fold,
        seed=seed,
        device_name=device_name,
        batch_size=int(plan["inference_batch_size"]),
    )

    reference = artifacts["B4"]
    usable = reference.target_mask.any(axis=1)
    if not usable.any():
        raise RuntimeError("该单元没有任何含已知标签的样本")
    coverage_grid = np.asarray(plan["coverage_grid"], dtype=np.float64)
    fixed_coverages = tuple(float(value) for value in plan["fixed_coverages"])
    curve_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for model_id, artifact in artifacts.items():
        probabilities = artifact.probabilities[usable]
        targets = artifact.targets[usable]
        target_mask = artifact.target_mask[usable]
        errors = sample_error_indicator(
            probabilities, targets, target_mask, artifact.thresholds
        )
        scores = {
            "mean_binary_certainty": mean_binary_certainty(probabilities),
            "mean_normalized_threshold_margin": mean_normalized_threshold_margin(
                probabilities, artifact.thresholds
            ),
        }
        if model_id == "P2":
            scores["p2_system_reliability"] = p2_reliability[usable]
        for score_method, confidence in scores.items():
            curve = selective_curve(
                probabilities,
                targets,
                target_mask,
                confidence,
                labels=artifact.labels,
                thresholds=artifact.thresholds,
                coverage_grid=coverage_grid,
                run_id=artifact.run_id,
                score_method=score_method,
            )
            summary = summarize_selector(
                curve,
                confidence,
                errors,
                run_id=artifact.run_id,
                score_method=score_method,
                fixed_coverages=fixed_coverages,
            )
            common = {
                "phase": "V3-1",
                "model_id": model_id,
                "fold": fold,
                "seed": seed,
                "scenario": plan["scenario"],
                "protocol_sha256": lock["protocol_sha256"],
                "source_prediction_sha256": artifact.source_entries[0]["sha256"],
            }
            curve_rows.extend({**common, **row} for row in curve)
            summary_rows.append({**common, **summary})

    output.mkdir(parents=True, exist_ok=True)
    curves_path = output / "curves.csv"
    summary_path = output / "summary.csv"
    pd.DataFrame(curve_rows).to_csv(curves_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    sources = [
        entry
        for artifact in artifacts.values()
        for entry in artifact.source_entries
    ] + p2_sources
    manifest = {
        "version": 1,
        "phase": "V3-1",
        "status": "success",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "model_count": len(artifacts),
        "selector_curve_count": len(summary_rows),
        "sample_count_before_known_target_filter": int(len(reference.probabilities)),
        "sample_count": int(usable.sum()),
        "excluded_no_known_target_count": int((~usable).sum()),
        "sample_key_sha256": _sample_key_sha256(
            reference.users[usable], reference.timestamps[usable]
        ),
        "p2_reinference": inference,
        "sources": sources,
        "outputs": [
            _relative_entry(root, curves_path),
            _relative_entry(root, summary_path),
        ],
    }
    write_json(output / "manifest.json", manifest)
    print(
        f"DONE fold={fold} seed={seed}: {len(summary_rows)} 条曲线，"
        f"{int(usable.sum())} 个样本，P2 推理 {inference['elapsed_seconds']:.2f}s",
        flush=True,
    )
    return output


def _numeric_value_columns(frame: pd.DataFrame, excluded: set[str]) -> list[str]:
    return [
        column
        for column in frame.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(frame[column])
    ]


def _mean_with_fold_std(
    fold_summary: pd.DataFrame, value_columns: list[str]
) -> pd.DataFrame:
    rows = []
    for (model_id, score_method), group in fold_summary.groupby(
        ["model_id", "score_method"], sort=True
    ):
        row: dict[str, Any] = {
            "model_id": model_id,
            "score_method": score_method,
            "fold_count": int(group["fold"].nunique()),
        }
        for column in value_columns:
            values = group[column].to_numpy(dtype=np.float64)
            row[column] = float(np.nanmean(values))
            row[f"{column}_fold_std"] = float(np.nanstd(values, ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def _paired_comparison(
    fold_summary: pd.DataFrame,
    *,
    name: str,
    candidate_model: str,
    candidate_score: str,
    reference_model: str,
    reference_score: str,
) -> dict[str, Any]:
    candidate = fold_summary.loc[
        (fold_summary["model_id"] == candidate_model)
        & (fold_summary["score_method"] == candidate_score)
    ].set_index("fold")
    reference = fold_summary.loc[
        (fold_summary["model_id"] == reference_model)
        & (fold_summary["score_method"] == reference_score)
    ].set_index("fold")
    if set(candidate.index) != set(reference.index) or len(candidate) != 5:
        raise RuntimeError(f"主比较 {name} 没有五个配对 fold")
    metrics = {
        "normalized_aurc": "lower",
        "error_detection_auroc": "higher",
        "error_detection_auprc": "higher",
        "risk_masked_bce_0_80": "lower",
        "risk_masked_bce_0_90": "lower",
        "risk_masked_bce_0_95": "lower",
        "macro_f1_0_80": "higher",
        "macro_f1_0_90": "higher",
        "macro_f1_0_95": "higher",
    }
    result: dict[str, Any] = {
        "name": name,
        "candidate": f"{candidate_model}/{candidate_score}",
        "reference": f"{reference_model}/{reference_score}",
        "fold_count": 5,
        "metrics": {},
    }
    for metric, direction in metrics.items():
        deltas = (candidate.loc[sorted(candidate.index), metric] - reference.loc[
            sorted(reference.index), metric
        ]).to_numpy(dtype=np.float64)
        wins = deltas < 0 if direction == "lower" else deltas > 0
        result["metrics"][metric] = {
            "direction": direction,
            "candidate_minus_reference_mean": float(np.mean(deltas)),
            "candidate_minus_reference_fold_std": float(np.std(deltas, ddof=1)),
            "candidate_win_fold_count": int(wins.sum()),
            "fold_deltas": deltas.tolist(),
        }
    primary = result["metrics"]
    result["primary_success"] = bool(
        primary["normalized_aurc"]["candidate_minus_reference_mean"] < 0
        and primary["error_detection_auroc"]["candidate_minus_reference_mean"] > 0
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def _format_metric(value: float | None) -> str:
    return "NA" if value is None or not np.isfinite(value) else f"{value:.6f}"


def _write_chinese_report(
    root: Path,
    protocol_sha256: str,
    overall: pd.DataFrame,
    comparisons: list[dict[str, Any]],
) -> Path:
    lines = [
        "# RobustSense V3-1 阶段报告",
        "",
        "## 1. 完成状态",
        "",
        (
            "V3-1 已完成 5 折 × 3 随机种子的同覆盖率选择性预测评估。"
            "全过程复用 V2-4 冻结 checkpoint，没有重新训练，也没有根据测试结果修改选择器。"
        ),
        "",
        f"- 协议哈希：`{protocol_sha256}`",
        "- 正式 fold/seed 单元：15/15",
        "- 模型：B4、B5、P、P2",
        "- 每单元选择器：9 条",
        "- 总选择器曲线：135 条",
        "",
        "## 2. 五折聚合结果",
        "",
        (
            "| 模型 | 选择器 | 归一化 AURC↓ | 错误检测 AUROC↑ | "
            "错误检测 AUPRC↑ | Risk@80%↓ | Macro-F1@80%↑ |"
        ),
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in overall.sort_values(["model_id", "score_method"]).to_dict("records"):
        lines.append(
            "| {model_id} | `{score_method}` | {aurc} | {auroc} | {auprc} | {risk} | {f1} |".format(
                model_id=row["model_id"],
                score_method=row["score_method"],
                aurc=_format_metric(row["normalized_aurc"]),
                auroc=_format_metric(row["error_detection_auroc"]),
                auprc=_format_metric(row["error_detection_auprc"]),
                risk=_format_metric(row["risk_masked_bce_0_80"]),
                f1=_format_metric(row["macro_f1_0_80"]),
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
                    "- AURC 差值（候选−参照）："
                    f"{aurc['candidate_minus_reference_mean']:.6f}；候选胜出 "
                    f"{aurc['candidate_win_fold_count']}/5 折。"
                ),
                (
                    "- 错误检测 AUROC 差值（候选−参照）："
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
                "AURC 变小表示在相同覆盖率下，保留下来的样本平均 masked BCE 更低；"
                "这说明风险排序更有效，不等于所有样本的强制分类准确率提高。P2 与自身普通"
                "确定度的比较主要检验选择器，P2 与 B5 的比较还混合了分类器差异。所有未"
                "通过的主比较同样属于正式结果，不能删除。"
            ),
            "",
            "## 5. 可复核产物",
            "",
            "完整曲线、fold/seed 摘要、fold 聚合和机器可读比较均位于 `reports/v3/`。",
            "",
        ]
    )
    path = root / "docs/v3/PHASE_V3_1_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def aggregate_results(root: Path, plan: dict[str, Any], lock: dict[str, Any]) -> Path:
    curve_frames = []
    summary_frames = []
    manifests = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            unit = _unit_output_dir(root, plan, int(fold), int(seed))
            if not _unit_is_current(unit, root, lock["protocol_sha256"]):
                raise RuntimeError(f"单元缺失或已失效：fold={fold}, seed={seed}")
            curve_frames.append(pd.read_csv(unit / "curves.csv"))
            summary_frames.append(pd.read_csv(unit / "summary.csv"))
            manifests.append(read_json(unit / "manifest.json"))
    curves = pd.concat(curve_frames, ignore_index=True)
    summaries = pd.concat(summary_frames, ignore_index=True)
    if len(summaries) != 15 * 9:
        raise RuntimeError(f"选择器摘要不是预期的 135 条：{len(summaries)}")

    fold_seed_curves_path = root / "reports/v3/phase_v3_1_fold_seed_curves.csv"
    fold_seed_summary_path = root / "reports/v3/phase_v3_1_fold_seed_summary.csv"
    curves.to_csv(fold_seed_curves_path, index=False)
    summaries.to_csv(fold_seed_summary_path, index=False)

    summary_values = _numeric_value_columns(
        summaries,
        excluded={"fold", "seed", "valid_count", "error_count"},
    )
    fold_summary = (
        summaries.groupby(["model_id", "score_method", "fold"], as_index=False)[
            summary_values
        ]
        .mean()
        .sort_values(["model_id", "score_method", "fold"])
    )
    fold_summary["seed_count"] = 3
    fold_summary_path = root / "reports/v3/phase_v3_1_fold_summary.csv"
    fold_summary.to_csv(fold_summary_path, index=False)

    overall = _mean_with_fold_std(fold_summary, summary_values)
    overall_path = root / "reports/v3/phase_v3_1_summary.csv"
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
    fold_curves = (
        curves.groupby(
            ["model_id", "score_method", "fold", "requested_coverage"],
            as_index=False,
        )[curve_values]
        .mean()
        .sort_values(["model_id", "score_method", "fold", "requested_coverage"])
    )
    aggregate_curve_rows = []
    for (model_id, score_method, requested), group in fold_curves.groupby(
        ["model_id", "score_method", "requested_coverage"], sort=True
    ):
        row: dict[str, Any] = {
            "model_id": model_id,
            "score_method": score_method,
            "requested_coverage": requested,
            "fold_count": int(group["fold"].nunique()),
        }
        for column in curve_values:
            values = group[column].to_numpy(dtype=np.float64)
            row[column] = float(np.nanmean(values))
            row[f"{column}_fold_std"] = float(np.nanstd(values, ddof=1))
        aggregate_curve_rows.append(row)
    aggregate_curves = pd.DataFrame(aggregate_curve_rows)
    aggregate_curves_path = root / "reports/v3/phase_v3_1_curves.csv"
    aggregate_curves.to_csv(aggregate_curves_path, index=False)

    comparisons = [
        _paired_comparison(
            fold_summary,
            name="selector_isolation",
            candidate_model="P2",
            candidate_score="p2_system_reliability",
            reference_model="P2",
            reference_score="mean_binary_certainty",
        ),
        _paired_comparison(
            fold_summary,
            name="strong_baseline",
            candidate_model="P2",
            candidate_score="p2_system_reliability",
            reference_model="B5",
            reference_score="mean_binary_certainty",
        ),
    ]
    report = {
        "version": 1,
        "phase": "V3-1",
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
        "inputs": [
            _relative_entry(root, _unit_output_dir(root, plan, fold, seed) / "manifest.json")
            for fold in plan["folds"]
            for seed in plan["seeds"]
        ],
        "outputs": [
            _relative_entry(root, fold_seed_curves_path),
            _relative_entry(root, fold_seed_summary_path),
            _relative_entry(root, fold_summary_path),
            _relative_entry(root, overall_path),
            _relative_entry(root, aggregate_curves_path),
        ],
    }
    report_path = root / "reports/v3/phase_v3_1_report.json"
    write_json(report_path, _json_safe(report))
    document_path = _write_chinese_report(
        root, lock["protocol_sha256"], overall, comparisons
    )
    print(f"AGGREGATED 15/15：{report_path}", flush=True)
    print(f"REPORT：{document_path}", flush=True)
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
    registry = _registry(root, plan)

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
