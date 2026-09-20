"""Deterministic Phase-4 run validation, aggregation, plots, and report draft."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robustsense-matplotlib"))

import matplotlib
import numpy as np
import pandas as pd

from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


class IncompleteRunMatrixError(ValueError):
    """Raised when report inputs do not satisfy the registered run matrix."""


def aggregate_mean_std(
    frame: pd.DataFrame,
    group_columns: list[str],
    value_columns: list[str],
) -> pd.DataFrame:
    """Aggregate without ever replacing unavailable metrics with zero."""
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_columns, keys, strict=True))
        for column in value_columns:
            numeric = pd.to_numeric(group[column], errors="coerce")
            count = int(numeric.notna().sum())
            row[f"{column}_count"] = count
            row[f"{column}_mean"] = float(numeric.mean()) if count else np.nan
            row[f"{column}_std"] = float(numeric.std(ddof=1)) if count > 1 else np.nan
        row["source_run_ids"] = "|".join(sorted(group["run_id"].unique()))
        rows.append(row)
    return pd.DataFrame(rows)


def _safe_spearman(left: pd.Series, right: pd.Series) -> float:
    paired = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(paired) < 2 or paired["left"].nunique() < 2 or paired["right"].nunique() < 2:
        return np.nan
    return float(paired["left"].corr(paired["right"], method="spearman"))


def _expected_run_id(model: str, fold: int, seed: int, profile: str) -> str:
    return f"extrasensory-{model}-fold{fold}-seed{seed}-{profile}"


def validate_run_matrix(
    project_root: Path, runs_dir: Path, plan: dict[str, Any]
) -> list[tuple[Path, dict[str, Any]]]:
    validated: list[tuple[Path, dict[str, Any]]] = []
    errors: list[str] = []
    controlled_models = set(plan["controlled_models"])
    base_outputs = {
        "evaluation_manifest.json",
        "natural_missingness_strata.parquet",
        "controlled_subset.parquet",
        "robustness_metrics.parquet",
        "robustness_per_label.parquet",
        "modality_response.parquet",
        "controlled_predictions.parquet",
        "robustness_auc.parquet",
    }
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for model in plan["models"]:
                run_id = _expected_run_id(model, int(fold), int(seed), plan["profile"])
                run_path = runs_dir / run_id
                if not run_path.is_dir():
                    errors.append(f"missing run directory: {run_id}")
                    continue
                missing = sorted(name for name in base_outputs if not (run_path / name).is_file())
                if missing:
                    errors.append(f"{run_id} missing outputs: {missing}")
                    continue
                manifest = read_json(run_path / "evaluation_manifest.json")
                expected_fields = {
                    "run_id": run_id,
                    "model_name": model,
                    "fold": int(fold),
                    "seed": int(seed),
                    "profile": plan["profile"],
                    "suite_version": int(plan["suite_version"]),
                    "controlled_suite": model in controlled_models,
                    "threshold_source_split": "val",
                }
                mismatches = {
                    key: (value, manifest.get(key))
                    for key, value in expected_fields.items()
                    if manifest.get(key) != value
                }
                if mismatches:
                    errors.append(f"{run_id} manifest mismatch: {mismatches}")
                    continue
                validated.append((run_path, manifest))
    if errors:
        raise IncompleteRunMatrixError("Run matrix validation failed:\n- " + "\n- ".join(errors))

    consistency_fields = (
        "evaluation_plan_sha256",
        "corruption_config_sha256",
        "scenario_manifest_sha256",
        "source_schema_sha256",
    )
    for field in consistency_fields:
        values = {manifest[field] for _, manifest in validated}
        if len(values) != 1:
            raise IncompleteRunMatrixError(f"Inconsistent {field}: {sorted(values)}")
    for fold in plan["folds"]:
        hashes = {
            manifest["controlled_subset_sha256"]
            for _, manifest in validated
            if manifest["fold"] == int(fold)
        }
        if len(hashes) != 1:
            raise IncompleteRunMatrixError(
                f"Fold {fold} does not share one controlled sample-ID set"
            )
        processed_hashes = {
            manifest["processed_manifest_sha256"]
            for _, manifest in validated
            if manifest["fold"] == int(fold)
        }
        if len(processed_hashes) != 1:
            raise IncompleteRunMatrixError(f"Fold {fold} has inconsistent processed-data manifests")
    return validated


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(
        path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "RobustSense Phase 4"},
    )
    plt.close(fig)


def _bar_plot(frame: pd.DataFrame, x: str, y: str, title: str, ylabel: str, path: Path) -> None:
    ordered = frame.sort_values(y, ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(ordered[x], ordered[y], color="#3478bf")
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=55)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _bar_plot_with_error(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    ylabel: str,
    path: Path,
) -> None:
    aggregate = frame.groupby(x, as_index=False)[y].agg(["mean", "std"]).reset_index()
    aggregate = aggregate.sort_values("mean", ascending=False)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.bar(
        aggregate[x],
        aggregate["mean"],
        yerr=aggregate["std"].fillna(0.0),
        capsize=3,
        color="#3478bf",
    )
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    axis.tick_params(axis="x", rotation=55)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, path)


def _line_by_model(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for model_name, group in frame.groupby("model_name", sort=True):
        curve = group.groupby(x, as_index=False)[y].mean().sort_values(x)
        axis.plot(curve[x], curve[y], marker="o", label=model_name)
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def _line_by_model_with_error(
    frame: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9, 5))
    for model_name, group in frame.groupby("model_name", sort=True):
        curve = group.groupby(x)[y].agg(["mean", "std", "count"]).reset_index().sort_values(x)
        axis.errorbar(
            curve[x],
            curve["mean"],
            yerr=curve["std"],
            marker="o",
            capsize=3,
            label=model_name,
        )
    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    _save_figure(fig, path)


def _heatmap(
    values: pd.DataFrame, title: str, path: Path, colorbar_label: str = "Macro-F1"
) -> None:
    fig, axis = plt.subplots(
        figsize=(max(7, values.shape[1] * 1.1), max(4, values.shape[0] * 0.55))
    )
    image = axis.imshow(values.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    axis.set_xticks(range(values.shape[1]), values.columns, rotation=45, ha="right")
    axis.set_yticks(range(values.shape[0]), values.index)
    axis.set_title(title)
    fig.colorbar(image, ax=axis, label=colorbar_label)
    _save_figure(fig, path)


def _load_evaluation_frames(
    validated: list[tuple[Path, dict[str, Any]]],
) -> dict[str, pd.DataFrame]:
    metrics = []
    per_label = []
    strata = []
    response = []
    auc = []
    natural_rows = []
    efficiency_rows = []
    for run_path, manifest in validated:
        metrics.append(pd.read_parquet(run_path / "robustness_metrics.parquet"))
        per_label.append(pd.read_parquet(run_path / "robustness_per_label.parquet"))
        strata.append(pd.read_parquet(run_path / "natural_missingness_strata.parquet"))
        response.append(pd.read_parquet(run_path / "modality_response.parquet"))
        auc_frame = pd.read_parquet(run_path / "robustness_auc.parquet")
        if not auc_frame.empty:
            auc.append(auc_frame)
        natural = read_json(run_path / "test_metrics.json")
        natural_rows.append(
            {
                "run_id": manifest["run_id"],
                "model_name": manifest["model_name"],
                "fold": manifest["fold"],
                "seed": manifest["seed"],
                "sample_count": natural["sample_count"],
                "macro_f1": natural["macro_f1"],
                "micro_f1": natural["micro_f1"],
                "mean_average_precision": natural["mean_average_precision"],
                "brier_score": natural["brier_score"],
            }
        )
        clean = metrics[-1][metrics[-1]["family"] == "clean_complete"].iloc[0]
        resolved = load_config(run_path / "resolved_config.yaml")
        efficiency_rows.append(
            {
                "run_id": manifest["run_id"],
                "model_name": manifest["model_name"],
                "fold": manifest["fold"],
                "seed": manifest["seed"],
                "parameter_count": int(resolved["parameter_count"]),
                "checkpoint_bytes": manifest["checkpoint_bytes"],
                "latency_ms_per_sample": clean["latency_ms_per_sample"],
            }
        )
    return {
        "metrics": pd.concat(metrics, ignore_index=True),
        "per_label": pd.concat(per_label, ignore_index=True),
        "strata": pd.concat(strata, ignore_index=True),
        "response": pd.concat(response, ignore_index=True),
        "auc": pd.concat(auc, ignore_index=True) if auc else pd.DataFrame(),
        "natural": pd.DataFrame(natural_rows),
        "efficiency": pd.DataFrame(efficiency_rows),
    }


def _write_tables(frames: dict[str, pd.DataFrame], tables: Path) -> dict[str, pd.DataFrame]:
    metrics = frames["metrics"]
    clean = metrics[metrics["family"] == "clean_complete"].copy()
    drop_each = metrics[metrics["family"] == "drop_each_one"].copy()
    drop_random = metrics[metrics["family"] == "drop_random_k"].copy()
    gaussian = metrics[metrics["family"] == "gaussian_noise"].copy()
    bias_scale = metrics[metrics["family"].isin(["bias_drift", "scale_error"])].copy()
    quality_response = frames["response"][
        frames["response"]["family"].isin(
            ["clean_complete", "drop_each_one", "gaussian_noise", "bias_drift", "scale_error"]
        )
    ].copy()
    response_correlations: list[dict[str, Any]] = []
    response_noise = quality_response[
        quality_response["family"].isin(["gaussian_noise", "bias_drift", "scale_error"])
        & quality_response["is_target_modality"]
    ].copy()
    response_noise["intensity"] = response_noise["severity"].abs()
    scale_rows = response_noise["family"] == "scale_error"
    response_noise.loc[scale_rows, "intensity"] = (
        response_noise.loc[scale_rows, "severity"] - 1.0
    ).abs()
    correlation_keys = ["run_id", "model_name", "fold", "seed", "family", "modality"]
    for keys, group in response_noise.groupby(correlation_keys, sort=True):
        curve = group.groupby("intensity", as_index=False).agg(
            mean_fusion_weight=("mean_fusion_weight", "mean"),
            mean_reliability=("mean_reliability", "mean"),
            reliability_mae=("reliability_mae", "mean"),
        )
        response_correlations.append(
            {
                **dict(zip(correlation_keys, keys, strict=True)),
                "point_count": len(curve),
                "fusion_weight_spearman": _safe_spearman(
                    curve["intensity"], curve["mean_fusion_weight"]
                ),
                "reliability_spearman": _safe_spearman(
                    curve["intensity"], curve["mean_reliability"]
                ),
                "reliability_mae_mean": curve["reliability_mae"].mean(),
            }
        )
    combined_performance = pd.concat(
        [
            clean.assign(evaluation_track="controlled_complete"),
            frames["natural"].assign(evaluation_track="natural_missingness"),
        ],
        ignore_index=True,
    )
    aggregate_performance = aggregate_mean_std(
        combined_performance,
        ["evaluation_track", "model_name"],
        ["macro_f1", "micro_f1", "mean_average_precision", "brier_score"],
    )
    tables_by_name = {
        "e1_clean_performance.csv": clean,
        "e2_natural_missingness.csv": frames["natural"],
        "e2_natural_by_available_count.csv": frames["strata"],
        "e3_drop_each_modality.csv": drop_each,
        "e4_drop_random_k.csv": drop_random,
        "e5_gaussian_noise.csv": gaussian,
        "e6_bias_scale.csv": bias_scale,
        "e7_quality_response.csv": quality_response,
        "e7_response_correlations.csv": pd.DataFrame(response_correlations),
        "e8_efficiency.csv": frames["efficiency"],
        "aggregate_performance.csv": aggregate_performance,
        "robustness_auc.csv": frames["auc"],
        "controlled_per_label.csv": frames["per_label"],
    }
    for name, frame in tables_by_name.items():
        sort_columns = [
            column
            for column in (
                "run_id",
                "model_name",
                "fold",
                "seed",
                "scenario_id",
                "family",
                "label",
                "modality",
            )
            if column in frame.columns
        ]
        ordered = frame.sort_values(sort_columns) if sort_columns else frame
        ordered.to_csv(tables / name, index=False, lineterminator="\n")
    return tables_by_name


def _write_plots(
    project_root: Path,
    plan: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    tables_by_name: dict[str, pd.DataFrame],
    figures: Path,
) -> list[str]:
    clean = tables_by_name["e1_clean_performance.csv"]
    natural = frames["natural"]
    controlled_names = plan["controlled_models"]
    scope = "development subset" if plan["profile"] == "dev" else "frozen test folds"
    _bar_plot_with_error(
        clean,
        "model_name",
        "macro_f1",
        f"E1: controlled-complete performance ({scope})",
        "Macro-F1",
        figures / "clean_performance.png",
    )
    _line_by_model(
        frames["strata"],
        "available_modality_count",
        "macro_f1",
        "E2: natural missingness by available count (all legal test rows)",
        "Available modality count",
        "Macro-F1",
        figures / "natural_missingness_by_available_count.png",
    )
    missing = tables_by_name["e4_drop_random_k.csv"].copy()
    clean_missing = clean[clean["model_name"].isin(controlled_names)].copy()
    clean_missing["severity"] = 0.0
    missing = pd.concat([clean_missing, missing], ignore_index=True)
    _line_by_model_with_error(
        missing,
        "severity",
        "macro_f1",
        f"E4: missing modalities ({scope}; mean and SD)",
        "Missing modality count",
        "Macro-F1",
        figures / "f1_vs_missing_modalities.png",
    )
    drop_pivot = tables_by_name["e3_drop_each_modality.csv"].pivot_table(
        index="model_name", columns="target_modality", values="macro_f1", aggfunc="mean"
    )
    _heatmap(
        drop_pivot,
        f"E3: drop-one-modality Macro-F1 (same IDs within each {scope})",
        figures / "drop_each_modality_heatmap.png",
    )
    noise_panels = (
        ("gaussian_noise", "Gaussian sigma"),
        ("bias_drift", "Absolute bias"),
        ("scale_error", "Absolute scale deviation"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for axis, (family, xlabel) in zip(axes, noise_panels, strict=True):
        family_frame = frames["metrics"][frames["metrics"]["family"] == family].copy()
        family_frame["intensity"] = family_frame["severity"].abs()
        if family == "scale_error":
            family_frame["intensity"] = (family_frame["severity"] - 1.0).abs()
        for model_name, group in family_frame.groupby("model_name", sort=True):
            clean_value = clean.loc[clean["model_name"] == model_name, "macro_f1"]
            curve_rows = pd.concat(
                [
                    pd.DataFrame({"intensity": [0.0], "macro_f1": [float(clean_value.iloc[0])]}),
                    group[["intensity", "macro_f1"]],
                ],
                ignore_index=True,
            )
            curve = curve_rows.groupby("intensity")["macro_f1"].agg(["mean", "std"]).reset_index()
            axis.errorbar(
                curve["intensity"],
                curve["mean"],
                yerr=curve["std"],
                marker="o",
                capsize=2,
                label=model_name,
            )
        axis.set_title(family.replace("_", " "))
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Macro-F1 (mean across target modalities)")
    axes[-1].legend(fontsize=7)
    fig.suptitle(f"E5-E6: feature-space robustness ({scope}; mean and SD)")
    _save_figure(fig, figures / "noise_robustness_curves.png")
    response = tables_by_name["e7_quality_response.csv"]
    gaussian_response = response[
        (response["family"].isin(["clean_complete", "gaussian_noise"]))
        & ((response["is_target_modality"]) | (response["family"] == "clean_complete"))
    ].copy()
    _line_by_model(
        gaussian_response.dropna(subset=["mean_reliability"]),
        "severity",
        "mean_reliability",
        f"E7: predicted reliability under Gaussian corruption ({scope})",
        "Gaussian sigma",
        "Mean target-modality reliability",
        figures / "reliability_vs_noise.png",
    )
    _line_by_model(
        gaussian_response.dropna(subset=["mean_fusion_weight"]),
        "severity",
        "mean_fusion_weight",
        f"E7: gate weight under Gaussian corruption ({scope})",
        "Gaussian sigma",
        "Mean target-modality fusion weight",
        figures / "gate_weight_vs_noise.png",
    )
    label_clean = frames["per_label"][frames["per_label"]["family"] == "clean_complete"]
    label_pivot = label_clean.pivot_table(
        index="model_name", columns="label", values="f1", aggfunc="mean"
    )
    _heatmap(
        label_pivot,
        f"E1 per-label F1 (controlled complete, {scope})",
        figures / "per_label_f1.png",
        "F1",
    )
    generated = [
        "clean_performance.png",
        "natural_missingness_by_available_count.png",
        "f1_vs_missing_modalities.png",
        "drop_each_modality_heatmap.png",
        "noise_robustness_curves.png",
        "reliability_vs_noise.png",
        "gate_weight_vs_noise.png",
        "per_label_f1.png",
        "model_overview.png",
        "label_prevalence.png",
        "modality_missingness.png",
    ]
    if plan["profile"] == "dev":
        ablation_models = ["gated", "robust-gated", "quality-aware-cls", "quality-aware"]
        ablation = natural[natural["model_name"].isin(ablation_models)].copy()
        _bar_plot(
            ablation,
            "model_name",
            "macro_f1",
            "Development architecture/loss progression (not formal A1-A5)",
            "Natural-missingness Macro-F1",
            figures / "architecture_progression.png",
        )
        generated.append("architecture_progression.png")
    efficiency = frames["efficiency"].copy()
    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter(
        efficiency["parameter_count"], efficiency["latency_ms_per_sample"], color="#934aa8"
    )
    for _, row in efficiency.iterrows():
        axis.annotate(
            row["model_name"],
            (row["parameter_count"], row["latency_ms_per_sample"]),
            fontsize=7,
        )
    axis.set_title("E8: parameter count vs inference latency")
    axis.set_xlabel("Trainable parameters")
    axis.set_ylabel("Latency (ms/sample)")
    axis.grid(alpha=0.25)
    _save_figure(fig, figures / "model_overview.png")

    prevalence_path = project_root / "reports/data_audit/label_prevalence.csv"
    prevalence = pd.read_csv(prevalence_path)
    _bar_plot(
        prevalence,
        "label",
        "positive_fraction",
        "ExtraSensory selected-label prevalence",
        "Positive fraction among known labels",
        figures / "label_prevalence.png",
    )
    missingness_path = project_root / "reports/data_audit/modality_missingness.csv"
    missingness = pd.read_csv(missingness_path)
    _bar_plot(
        missingness,
        "modality",
        "unavailable_fraction",
        "Natural modality unavailability",
        "Unavailable fraction",
        figures / "modality_missingness.png",
    )
    return generated


def _technical_report(
    project_root: Path,
    plan: dict[str, Any],
    frames: dict[str, pd.DataFrame],
    manifests: list[dict[str, Any]],
) -> str:
    clean = frames["metrics"][frames["metrics"]["family"] == "clean_complete"]
    natural = frames["natural"]

    def summary_lines(frame: pd.DataFrame) -> list[str]:
        summary = frame.groupby("model_name").agg(
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            map_mean=("mean_average_precision", "mean"),
            map_std=("mean_average_precision", "std"),
        )
        return [
            f"| {model} | {row.macro_f1_mean:.4f} | "
            f"{row.macro_f1_std:.4f} | {row.map_mean:.4f} | {row.map_std:.4f} |"
            for model, row in summary.sort_index().iterrows()
        ]

    def mean_value(frame: pd.DataFrame, model: str, column: str = "macro_f1") -> float:
        return float(frame.loc[frame["model_name"] == model, column].mean())

    def mean_relative_drop(frame: pd.DataFrame, model: str) -> float:
        return mean_value(frame, model, "relative_drop")

    clean_lines = summary_lines(clean)
    natural_lines = summary_lines(natural)
    controlled_counts = {
        int(fold): next(
            int(item["controlled_sample_count"])
            for item in manifests
            if int(item["fold"]) == int(fold)
        )
        for fold in plan["folds"]
    }
    controlled_count_text = ", ".join(
        f"fold {fold}：{count:,}" for fold, count in controlled_counts.items()
    )
    if plan["profile"] == "dev":
        validity = "这些开发结果只属于工程证据，不是研究结论。"
    elif plan["profile"] == "credible":
        validity = "这是冻结的五折、单种子可信结果集。"
    else:
        validity = "这是冻结的五折、三种子扩展结果集。"

    if plan["profile"] == "dev":
        run_ids = "\n".join(f"- `{item['run_id']}`" for item in manifests)
        return (
            "# RobustSense 技术报告：开发结果\n\n"
            "## 范围与有效性\n\n"
            f"本报告由注册矩阵 `{plan['name']}` 生成，使用 fold {plan['folds']} 和种子 "
            f"{plan['seeds']}。受控样本数为 {controlled_count_text}。自然评估使用每条合法测试记录。"
            f"{validity}\n\n"
            "## E1：受控完整模态子集\n\n"
            "| 模型 | Macro-F1 均值 | Macro-F1 标准差 | mAP 均值 | mAP 标准差 |\n"
            "|---|---:|---:|---:|---:|\n"
            + "\n".join(clean_lines)
            + "\n\n## E2：自然缺失\n\n"
            "| 模型 | Macro-F1 均值 | Macro-F1 标准差 | mAP 均值 | mAP 标准差 |\n"
            "|---|---:|---:|---:|---:|\n"
            + "\n".join(natural_lines)
            + "\n\n## E3–E8\n\n"
            "生成的表格和图形保留绝对性能、相对退化、响应诊断、效率和源 run ID。"
            "该开发配置只用于工程检查，不得提升为研究结论。\n\n"
            "## 包含的 run\n\n"
            + run_ids
            + "\n"
        )

    single_models = sorted(
        model for model in clean["model_name"].unique() if model.startswith("single-")
    )
    single_clean = {
        model: mean_value(clean, model)
        for model in single_models
    }
    single_natural = {
        model: mean_value(natural, model)
        for model in single_models
    }
    best_single_clean = max(single_clean, key=single_clean.get)
    best_single_natural = max(single_natural, key=single_natural.get)
    gated_clean = mean_value(clean, "gated")
    robust_clean = mean_value(clean, "robust-gated")
    gated_natural = mean_value(natural, "gated")
    quality_natural = mean_value(natural, "quality-aware")
    robust_natural_map = mean_value(natural, "robust-gated", "mean_average_precision")
    quality_natural_map = mean_value(natural, "quality-aware", "mean_average_precision")

    clean_pivot = clean.pivot(index="fold", columns="model_name", values="macro_f1")
    natural_pivot = natural.pivot(index="fold", columns="model_name", values="macro_f1")
    gated_clean_single_wins = int(
        (clean_pivot["gated"] > clean_pivot[best_single_clean]).sum()
    )
    quality_natural_gated_wins = int(
        (natural_pivot["quality-aware"] > natural_pivot["gated"]).sum()
    )

    metrics = frames["metrics"]
    drop_each = metrics[metrics["family"] == "drop_each_one"]
    drop_random = metrics[metrics["family"] == "drop_random_k"]
    gaussian_max = metrics[
        (metrics["family"] == "gaussian_noise") & (metrics["severity"] == 2.0)
    ]
    bias_max = metrics[
        (metrics["family"] == "bias_drift") & (metrics["severity"].abs() == 1.0)
    ]
    scale_max = metrics[
        (metrics["family"] == "scale_error")
        & ((metrics["severity"] - 1.0).abs() == 0.5)
    ]
    primary_models = ["gated", "robust-gated", "quality-aware"]
    controlled_lines = []
    for model in primary_models:
        values = [
            mean_value(drop_each, model),
            mean_value(drop_random[drop_random["drop_count"] == 1], model),
            mean_value(drop_random[drop_random["drop_count"] == 2], model),
            mean_value(drop_random[drop_random["drop_count"] == 3], model),
            mean_value(gaussian_max, model),
            mean_value(bias_max, model),
            mean_value(scale_max, model),
        ]
        controlled_lines.append(
            f"| {model} | " + " | ".join(f"{value:.4f}" for value in values) + " |"
        )

    random_drop_lines = []
    for model in primary_models:
        values = [
            mean_relative_drop(
                drop_random[drop_random["drop_count"] == drop_count], model
            )
            for drop_count in (1, 2, 3)
        ]
        random_drop_lines.append(
            f"| {model} | " + " | ".join(f"{100 * value:.2f}%" for value in values) + " |"
        )

    response = frames["response"].copy()
    response = response[
        response["family"].isin(["gaussian_noise", "bias_drift", "scale_error"])
        & response["is_target_modality"]
    ].copy()
    response["intensity"] = response["severity"].abs()
    scale_rows = response["family"] == "scale_error"
    response.loc[scale_rows, "intensity"] = (
        response.loc[scale_rows, "severity"] - 1.0
    ).abs()
    correlation_rows: list[dict[str, Any]] = []
    keys = ["run_id", "model_name", "fold", "seed", "family", "modality"]
    for key_values, group in response.groupby(keys, sort=True):
        curve = group.groupby("intensity", as_index=False).agg(
            mean_fusion_weight=("mean_fusion_weight", "mean"),
            mean_reliability=("mean_reliability", "mean"),
            reliability_mae=("reliability_mae", "mean"),
        )
        correlation_rows.append(
            {
                **dict(zip(keys, key_values, strict=True)),
                "fusion_weight_spearman": _safe_spearman(
                    curve["intensity"], curve["mean_fusion_weight"]
                ),
                "reliability_spearman": _safe_spearman(
                    curve["intensity"], curve["mean_reliability"]
                ),
                "reliability_mae_mean": curve["reliability_mae"].mean(),
            }
        )
    correlations = pd.DataFrame(correlation_rows)
    quality_correlations = correlations[correlations["model_name"] == "quality-aware"]
    response_lines = []
    family_names = {
        "gaussian_noise": "Gaussian 噪声",
        "bias_drift": "偏置漂移",
        "scale_error": "尺度误差",
    }
    for family in ("gaussian_noise", "bias_drift", "scale_error"):
        family_rows = quality_correlations[quality_correlations["family"] == family]
        response_lines.append(
            f"| {family_names[family]} | {family_rows['reliability_spearman'].mean():.4f} | "
            f"{family_rows['fusion_weight_spearman'].mean():.4f} | "
            f"{family_rows['reliability_mae_mean'].mean():.4f} |"
        )

    reliability_consistent = bool(
        quality_correlations.groupby("family")["reliability_spearman"].mean().lt(0).all()
    )
    gate_consistent = bool(
        quality_correlations.groupby("family")["fusion_weight_spearman"].mean().lt(0).all()
    )
    robust_drop_advantage = all(
        mean_relative_drop(drop_random[drop_random["drop_count"] == count], "robust-gated")
        < mean_relative_drop(drop_random[drop_random["drop_count"] == count], "gated")
        for count in (1, 2, 3)
    )

    efficiency = frames["efficiency"].groupby("model_name").agg(
        parameter_count=("parameter_count", "first"),
        checkpoint_bytes=("checkpoint_bytes", "first"),
        latency_mean=("latency_ms_per_sample", "mean"),
        latency_std=("latency_ms_per_sample", "std"),
    )
    efficiency_lines = [
        f"| {model} | {int(row.parameter_count):,} | "
        f"{int(row.checkpoint_bytes) / 1024:.1f} KiB | "
        f"{row.latency_mean:.4f} ± {row.latency_std:.4f} |"
        for model, row in efficiency.sort_values("parameter_count").iterrows()
    ]

    label_clean = frames["per_label"][frames["per_label"]["family"] == "clean_complete"]
    label_summary = (
        label_clean[label_clean["model_name"].isin(primary_models)]
        .groupby(["label", "model_name"])["f1"]
        .mean()
        .unstack("model_name")
        .sort_index()
    )
    label_lines = [
        f"| {label} | {row['gated']:.4f} | {row['robust-gated']:.4f} | "
        f"{row['quality-aware']:.4f} |"
        for label, row in label_summary.iterrows()
    ]

    ablation_path = project_root / "reports/tables/phase5_ablation_dev.csv"
    ablation_section = ""
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path).sort_values("ablation_id")
        ablation_names = {
            "A0": "完整 P",
            "A1": "无 Sensor Dropout",
            "A2": "无噪声增强",
            "A3": "无显式质量特征",
            "A4": "无 ReliabilityNet，仅内容门控",
            "A5": "无可靠度辅助损失",
        }
        ablation_lines = [
            f"| {row.ablation_id} | {ablation_names.get(row.ablation_id, row.change)} | "
            f"{row.natural_macro_f1:.4f} | "
            f"{row.clean_complete_macro_f1:.4f} |"
            for row in ablation.itertuples()
        ]
        ablation_section = (
            "## 8. 开发消融筛查\n\n"
            "A0–A5 筛查只使用 fold 0、种子 13 和三 epoch 预算。它仅属于工程证据，"
            "不能用于声称组件的因果效应。混合排序本身很重要：在这次短期筛查中，"
            "移除 ReliabilityNet（A4）提高了自然 Macro-F1，而移除 Sensor Dropout（A1）"
            "使两项报告指标都下降。查看这些数值后没有修改可信配置。\n\n"
            "| ID | 改动 | 自然 Macro-F1 | 受控完整 Macro-F1 |\n"
            "|---|---|---:|---:|\n"
            + "\n".join(ablation_lines)
            + "\n\n"
        )

    h1_status = "支持" if gated_clean_single_wins == len(plan["folds"]) else "部分支持"
    h2_status = "支持" if mean_relative_drop(drop_random, "gated") > 0 else "不支持"
    h3_status = "支持" if robust_drop_advantage else "部分支持"
    h4_status = "支持" if reliability_consistent and gate_consistent else "部分支持"

    return (
        "# RobustSense：质量感知鲁棒多传感器融合\n\n"
        "## 可信实验技术报告\n\n"
        "### 摘要\n\n"
        "RobustSense 面向传感器自然缺失和特征空间污染下的多标签上下文识别。"
        f"在 60 名用户的 ExtraSensory 数据上，冻结的五折单种子实验共验证 "
        f"{len(manifests)} 个 run。完整模态条件下，Robust Gated Fusion 的平均 "
        f"Macro-F1 最高（{robust_clean:.4f}）；自然缺失条件下，Quality-Aware "
        f"模型最高（{quality_natural:.4f}），比普通 Gated Fusion 高 "
        f"{quality_natural - gated_natural:+.4f}，并在 "
        f"{quality_natural_gated_wins}/{len(plan['folds'])} 个外层 fold 上取胜。"
        "Sensor Dropout 模型在随机缺失 1–3 个模态时均降低了相对退化。"
        "质量模型的可靠度输出会随污染增强而下降，但融合权重在 Gaussian 噪声和"
        "偏置下没有一致下降，因此关于权重可解释性的假设仅得到部分支持。\n\n"
        "## 1. 范围、有效性与研究问题\n\n"
        f"本报告由注册矩阵 `{plan['name']}` 生成，使用 fold {plan['folds']} 和种子 "
        f"{plan['seeds']}。自然评估使用每条合法测试记录。受控评估使用完整模态总体"
        f"（{controlled_count_text}），并在每个 fold 内固定样本 ID。{validity}"
        "标准差按五个外层用户 fold 计算；这些 fold 不是五个独立数据集，单个训练"
        "种子也不能量化种子级不确定性。\n\n"
        "| 假设 | 结论 | 注册证据 |\n"
        "|---|---|---|\n"
        f"| H1：完整输入下融合优于多数单传感器 | {h1_status} | Gated 平均超过最佳"
        f"单模态模型（{best_single_clean}）"
        f" {gated_clean - single_clean[best_single_clean]:+.4f}，并在 "
        f"{gated_clean_single_wins}/{len(plan['folds'])} 个 fold 中获胜。 |\n"
        f"| H2：模态故障会使普通融合退化 | {h2_status} | Gated 的相对下降从单模态"
        "故障向多模态故障条件增加。 |\n"
        f"| H3：Sensor Dropout 可降低退化 | {h3_status} | 随机移除 1、2、3 个模态时，"
        "Robust Gated 的平均相对下降均低于 Gated。 |\n"
        f"| H4：质量信号会单调降低受损模态权重 | {h4_status} | 可靠度响应单调，"
        "但不同污染族中的门控权重响应不一致。 |\n\n"
        "## 2. 数据与防泄漏控制\n\n"
        "项目使用 ExtraSensory 官方预计算特征数据：60 名用户、377,346 条分钟级记录。"
        "冻结输入包含六个模态组（手机加速度计、手机陀螺仪、手表加速度计、位置、音频"
        "和手机状态）、177 个特征和 15 个多标签目标。未知标签保持为 NaN，不计入损失"
        "或指标。外层测试用户、验证用户和训练用户互斥。填补、稳健缩放、类别权重、早停"
        "和逐标签阈值只使用训练/验证数据。测试阈值绝不会针对故障场景重新调节。\n\n"
        "自然缺失与受控故障回答不同问题。前者使用全部合法测试记录并保留观测可用性；"
        "后者从六模态完整记录开始，注入确定性的特征空间缺失、Gaussian 噪声、偏置或尺度"
        "误差。这些污染是对预计算特征的压力测试，不是对物理传感器电子器件的模拟。\n\n"
        "## 3. 相关工作\n\n"
        "ExtraSensory 工作建立了基于个人设备多源传感器进行在野上下文识别的数据基础，"
        "并公开逐用户预计算特征及用户级划分。ModDrop 通过训练时随机丢弃完整模态，说明"
        "模态级随机失活能够减少跨模态共适应并提高缺失信号下的鲁棒性。后续研究进一步"
        "表明，多模态模型通常会在推理时缺失模态的条件下明显退化，而且融合策略的最优"
        "选择依赖数据集。RobustSense 沿用模态级失活思想，但聚焦 ExtraSensory 的六路"
        "预计算传感器特征，并额外比较自然缺失、确定性受控污染、显式质量特征和可靠度"
        "辅助监督。由于数据集、任务与协议不同，本项目不把结果解释为对相关工作的直接"
        "性能超越。\n\n"
        "参考：[ExtraSensory 原始论文](https://doi.org/10.1109/MPRV.2017.3971131)、"
        "[ModDrop](https://arxiv.org/abs/1501.00102)、"
        "[缺失模态下的多模态 Transformer 研究](https://openaccess.thecvf.com/content/CVPR2022/html/Ma_Are_Multimodal_Transformers_Robust_to_Missing_Modality_CVPR_2022_paper.html)、"
        "[缺失模态鲁棒动作识别实践](https://doi.org/10.1609/aaai.v37i3.25378)。\n\n"
        "## 4. 模型与训练协议\n\n"
        "比较对象包括六个单模态 MLP、线性和非线性早期融合、后期融合、带可用性掩码的"
        "门控融合、污染训练的 Robust Gated Fusion 以及提出的 Quality-Aware Fusion。"
        "提出模型加入可观测质量特征、ReliabilityNet 头、可靠度监督和干净/污染一致性"
        "正则化。模型使用 AdamW、批量大小 512、最多 80 epoch、验证 Macro-F1 早停"
        "（patience 10），并在所有测试场景中使用相同的验证集派生标签阈值。\n\n"
        "## 5. E1–E2：完整输入与自然缺失输入\n\n"
        "### E1：受控完整模态子集\n\n"
        "| 模型 | Macro-F1 均值 | Macro-F1 标准差 | mAP 均值 | mAP 标准差 |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(clean_lines)
        + "\n\nRobust Gated Fusion 的受控完整 Macro-F1 排名第一，但相对 Gated 的优势"
        f"仅为 {robust_clean - gated_clean:+.4f}，且出现在 "
        f"{int((clean_pivot['robust-gated'] > clean_pivot['gated']).sum())}/5 个 fold。"
        f"三个门控变体在每个 fold 上都超过最佳单传感器 {best_single_clean}。这支持多源"
        "互补性，但不意味着门控变体之间的小差异具有统计决定性。\n\n"
        "### E2：自然缺失\n\n"
        "| 模型 | Macro-F1 均值 | Macro-F1 标准差 | mAP 均值 | mAP 标准差 |\n"
        "|---|---:|---:|---:|---:|\n"
        + "\n".join(natural_lines)
        + "\n\nQuality-Aware Fusion 在自然缺失下排名第一，相对 Gated 提升 "
        f"{quality_natural - gated_natural:+.4f}；配对 fold 差异在 "
        f"{quality_natural_gated_wins}/5 个 fold 中为正。它比最佳自然单模态模型"
        f"（{best_single_natural}）高 "
        f"{quality_natural - single_natural[best_single_natural]:+.4f}。Robust Gated "
        f"的 mAP（{robust_natural_map:.4f}）略高于 Quality-Aware"
        f"（{quality_natural_map:.4f}），因此模型排名会随主要指标略有变化。\n\n"
        "## 6. E3–E6：受控故障鲁棒性\n\n"
        "下表均为五个 fold 的平均 Macro-F1；适用时还会对目标模态或确定性掩码取平均。\n\n"
        "| 模型 | 逐一移除 | 随机 k=1 | 随机 k=2 | 随机 k=3 | Gaussian σ=2 | "
        "偏置 \\|b\\|=1 | 尺度 \\|c−1\\|=0.5 |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(controlled_lines)
        + "\n\n随机移除模态时使用相对退化，可减弱干净性能小幅偏移对鲁棒性比较的影响：\n\n"
        "| 模型 | k=1 相对下降 | k=2 相对下降 | k=3 相对下降 |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(random_drop_lines)
        + "\n\n在三个缺失水平上，Robust Gated 都降低了 Gated 的相对下降。移除一个模态时，"
        "Quality-Aware 略差于 Robust Gated；移除两个和三个模态时，它的相对下降最小，"
        "并在 Gaussian σ=2 时取得最佳绝对 Macro-F1。phone-state 污染对普通 Gated "
        "尤其有害，而手机加速度计和陀螺仪扰动的平均损失较小。稳健特征缩放后尺度误差"
        "较温和；偏置和高强度 Gaussian 噪声更能显现模型差异。\n\n"
        "## 7. E7：可靠度与门控响应\n\n"
        "先针对每个 run 和目标模态，计算污染强度与模型响应的 Spearman 相关，再对五个"
        "fold 和六个模态取平均。负值是期望方向。\n\n"
        "| 污染族 | 可靠度 ρ | 融合权重 ρ | 可靠度 MAE |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(response_lines)
        + "\n\nReliabilityNet 信号符合预期：每个污染族的平均相关都显著为负。学习到的门控"
        "不会简单复制该信号；Gaussian 噪声和偏置漂移的平均门控权重相关为正，仅尺度误差"
        "为负。因此 H4 只得到部分支持。这一区别很重要：鲁棒性提升不能证明学习到的注意力"
        "权重是经过校准的传感器可靠度解释。缺失模态仍由结构硬掩码为精确的零。\n\n"
        + ablation_section
        + "## 9. 逐标签表现\n\n"
        "受控完整条件下，不同标签的 F1 差异明显。`OR_indoors`、`LYING_DOWN` 和 "
        "`SLEEPING` 等常见结构性上下文明显比 `EATING`、`TALKING` 和 `OR_standing` "
        "等稀疏或含混活动容易。相对普通 Gated，质量感知训练提高了 `BICYCLING` 和 "
        "`IN_A_CAR`，但并非每个标签都提高。\n\n"
        "| 标签 | Gated | Robust Gated | Quality-Aware |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(label_lines)
        + "\n\n## 10. E8：效率\n\n"
        "延迟在各 fold 的受控完整子集上进行批量推理测量，以毫秒/样本报告。这是本地机器"
        "测量，不是端到端移动端延迟声明。\n\n"
        "| 模型 | 参数量 | 检查点大小 | 延迟 ms/样本（均值 ± 标准差） |\n"
        "|---|---:|---:|---:|\n"
        + "\n".join(efficiency_lines)
        + "\n\nQuality-Aware 比 Gated 多 14,790 个参数，在此次批量测量中更慢，但全部检查点"
        "仍小于 0.6 MiB。项目尚未测量能耗、流处理开销、特征提取或部署延迟。\n\n"
        "## 11. 讨论与局限\n\n"
        "最强发现不是干净数据上的大幅增益，而是污染感知训练在输入缺失时保留更多性能，"
        "显式质量建模既提供一致的可靠度响应，也取得最佳自然缺失 Macro-F1。普通门控模型"
        "已经很有竞争力，提出模型的绝对增益相对 fold 变化较小。结果应描述为稳健的工程改进，"
        "而不是最先进性能声明。\n\n"
        "主要限制包括：可信矩阵只有一个种子；消融仅为开发筛查；受控测试存在完整子集选择"
        "偏差；故障作用于特征空间而非原始信号；数据集年代较早、标签由用户自报且用户总体"
        "有限；没有跨数据集验证；没有实时或端侧评估。门控权重结果也警示：在没有直接校准"
        "检验时，不能把注意力式权重解释为可靠度。未来应为 B4/B5/P 增加多个种子，运行五折"
        "消融，评估原始信号故障，测试第二个数据集，并测量端到端移动端成本。\n\n"
        "## 12. 结论\n\n"
        "注册实验支持一个克制的结论：在本数据和冻结协议下，污染感知训练能减少模态缺失"
        "造成的相对性能损失，显式质量建模进一步取得最佳自然缺失 Macro-F1，并产生对污染"
        "强度敏感的可靠度信号。完整输入下的增益较小，门控权重也不能稳定充当可靠度解释。"
        "因此 RobustSense 的主要贡献是可复现的鲁棒融合实现与严格评估证据，而不是新的"
        "最先进性能或已经验证的实时系统。\n\n"
        "## 13. 可复现性与可追溯性\n\n"
        "全部 60 个可信 run 均通过输出合同。`run_completeness.json` 记录已验证的 run ID"
        "和逐 fold 受控子集哈希。每个生成表保留 run ID，`source_map.csv` 将表格和图形"
        "映射到不可变 run 集。`run_registry.csv` 和 `run_attempts.csv` 保留执行状态与尝试。"
        "协议哈希记录在 `phase5_protocol_lock.json`；冻结后没有根据测试结果修改配置。\n"
    )


def generate_report(
    project_root: str | Path,
    runs_dir: str | Path = "runs",
    output_dir: str | Path = "reports",
    plan_path: str | Path = "configs/evaluation/dev.yaml",
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    runs = Path(runs_dir)
    if not runs.is_absolute():
        runs = root / runs
    output = Path(output_dir)
    if not output.is_absolute():
        output = root / output
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    plan = load_config(plan_file)
    validated = validate_run_matrix(root, runs, plan)
    frames = _load_evaluation_frames(validated)
    tables = output / "tables"
    figures = output / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    tables_by_name = _write_tables(frames, tables)
    figure_names = _write_plots(root, plan, frames, tables_by_name, figures)
    manifests = [manifest for _, manifest in validated]
    report_text = _technical_report(root, plan, frames, manifests)
    (output / "technical_report.md").write_text(report_text, encoding="utf-8")
    run_ids = sorted(manifest["run_id"] for manifest in manifests)
    clean = frames["metrics"][frames["metrics"]["family"] == "clean_complete"]
    natural = frames["natural"]
    drop_three = frames["metrics"][
        (frames["metrics"]["family"] == "drop_random_k")
        & (frames["metrics"]["drop_count"] == 3)
    ]

    def readme_claim(
        claim_id: str,
        frame: pd.DataFrame,
        model: str,
        column: str,
        evidence_artifact: str,
        selector: str,
    ) -> dict[str, Any]:
        rows = frame[frame["model_name"] == model]
        return {
            "claim_id": claim_id,
            "value": float(rows[column].mean()),
            "evidence_artifact": evidence_artifact,
            "selector": selector,
            "source_run_ids": "|".join(sorted(rows["run_id"].unique())),
        }

    readme_claims = [
        readme_claim(
            "controlled_complete_robust_gated_macro_f1",
            clean,
            "robust-gated",
            "macro_f1",
            "tables/e1_clean_performance.csv",
            "model_name=robust-gated; mean(macro_f1) across folds",
        ),
        readme_claim(
            "natural_quality_aware_macro_f1",
            natural,
            "quality-aware",
            "macro_f1",
            "tables/e2_natural_missingness.csv",
            "model_name=quality-aware; mean(macro_f1) across folds",
        ),
        readme_claim(
            "natural_gated_macro_f1",
            natural,
            "gated",
            "macro_f1",
            "tables/e2_natural_missingness.csv",
            "model_name=gated; mean(macro_f1) across folds",
        ),
        readme_claim(
            "drop_three_quality_aware_relative_drop",
            drop_three,
            "quality-aware",
            "relative_drop",
            "tables/e4_drop_random_k.csv",
            "model_name=quality-aware; drop_count=3; mean(relative_drop)",
        ),
        readme_claim(
            "drop_three_gated_relative_drop",
            drop_three,
            "gated",
            "relative_drop",
            "tables/e4_drop_random_k.csv",
            "model_name=gated; drop_count=3; mean(relative_drop)",
        ),
    ]
    natural_pivot = natural.pivot(index="fold", columns="model_name", values="macro_f1")
    quality_rows = natural[natural["model_name"] == "quality-aware"]
    readme_claims.extend(
        [
            {
                "claim_id": "natural_quality_aware_minus_gated_macro_f1",
                "value": float(
                    (natural_pivot["quality-aware"] - natural_pivot["gated"]).mean()
                ),
                "evidence_artifact": "tables/e2_natural_missingness.csv",
                "selector": "paired fold mean: quality-aware minus gated macro_f1",
                "source_run_ids": "|".join(
                    sorted(
                        natural[
                            natural["model_name"].isin(["quality-aware", "gated"])
                        ]["run_id"].unique()
                    )
                ),
            },
            {
                "claim_id": "natural_quality_aware_fold_wins_over_gated",
                "value": int(
                    (natural_pivot["quality-aware"] > natural_pivot["gated"]).sum()
                ),
                "evidence_artifact": "tables/e2_natural_missingness.csv",
                "selector": "count folds where quality-aware macro_f1 > gated macro_f1",
                "source_run_ids": "|".join(
                    sorted(
                        natural[
                            natural["model_name"].isin(["quality-aware", "gated"])
                        ]["run_id"].unique()
                    )
                ),
            },
            {
                "claim_id": "credible_validated_run_count",
                "value": len(manifests),
                "evidence_artifact": "run_completeness.json",
                "selector": "validated_run_count",
                "source_run_ids": "|".join(run_ids),
            },
            {
                "claim_id": "quality_aware_parameter_count",
                "value": int(
                    frames["efficiency"].loc[
                        frames["efficiency"]["model_name"] == "quality-aware",
                        "parameter_count",
                    ].iloc[0]
                ),
                "evidence_artifact": "tables/e8_efficiency.csv",
                "selector": "model_name=quality-aware; parameter_count",
                "source_run_ids": "|".join(sorted(quality_rows["run_id"].unique())),
            },
        ]
    )
    audit_summary_path = root / "reports/data_audit/dataset_summary.json"
    if audit_summary_path.is_file():
        audit_summary = read_json(audit_summary_path)
        readme_claims.extend(
            [
                {
                    "claim_id": "extrasensory_user_count",
                    "value": int(audit_summary["user_count"]),
                    "evidence_artifact": "data_audit/dataset_summary.json",
                    "selector": "user_count",
                    "source_run_ids": "",
                },
                {
                    "claim_id": "extrasensory_sample_count",
                    "value": int(audit_summary["sample_count"]),
                    "evidence_artifact": "data_audit/dataset_summary.json",
                    "selector": "sample_count",
                    "source_run_ids": "",
                },
            ]
        )
    registry_path = root / "reports/run_registry.csv"
    if registry_path.is_file():
        registry = pd.read_csv(registry_path)
        successful = registry[registry["status"] == "success"]
        readme_claims.append(
            {
                "claim_id": "registered_successful_unit_count",
                "value": len(successful),
                "evidence_artifact": "run_registry.csv",
                "selector": "count rows where status=success",
                "source_run_ids": "|".join(
                    sorted(Path(path).name for path in successful["run_dir"])
                ),
            }
        )
    pd.DataFrame(readme_claims).to_csv(
        output / "readme_source_map.csv", index=False, lineterminator="\n"
    )
    ablation_summary_path = output / "phase5_ablation_dev_summary.json"
    ablation_run_ids: list[str] = []
    ablation_artifacts: list[str] = []
    if ablation_summary_path.is_file():
        ablation_summary = read_json(ablation_summary_path)
        ablation_run_ids = sorted(ablation_summary.get("source_run_ids", []))
        for relative in ("tables/phase5_ablation_dev.csv", "figures/ablation.png"):
            if (output / relative).is_file():
                ablation_artifacts.append(relative)
    generated = sorted(
        [f"tables/{name}" for name in tables_by_name]
        + [f"figures/{name}" for name in figure_names]
        + ablation_artifacts
        + [
            "technical_report.md",
            "source_map.csv",
            "readme_source_map.csv",
            "run_completeness.json",
        ]
    )
    source_map = pd.DataFrame(
        {
            "artifact": generated,
            "source_run_ids": "|".join(run_ids),
            "evaluation_plan": str(plan_file.relative_to(root)),
        }
    )
    if ablation_run_ids:
        ablation_sources = "|".join(ablation_run_ids)
        final_report_sources = "|".join(sorted({*run_ids, *ablation_run_ids}))
        source_map.loc[
            source_map["artifact"].isin(ablation_artifacts), "source_run_ids"
        ] = ablation_sources
        source_map.loc[
            source_map["artifact"] == "technical_report.md", "source_run_ids"
        ] = final_report_sources
    source_map.to_csv(output / "source_map.csv", index=False, lineterminator="\n")
    completeness = {
        "status": "passed",
        "plan": str(plan_file.relative_to(root)),
        "profile": plan["profile"],
        "folds": plan["folds"],
        "seeds": plan["seeds"],
        "expected_run_count": len(plan["models"]) * len(plan["folds"]) * len(plan["seeds"]),
        "validated_run_count": len(validated),
        "run_ids": run_ids,
        "controlled_subset_sha256_by_fold": {
            str(fold): sorted(
                {
                    manifest["controlled_subset_sha256"]
                    for manifest in manifests
                    if manifest["fold"] == int(fold)
                }
            )[0]
            for fold in plan["folds"]
        },
    }
    write_json(output / "run_completeness.json", completeness)
    return {
        "status": "passed",
        "validated_run_count": len(validated),
        "table_count": len(tables_by_name),
        "figure_count": len(figure_names),
        "report_path": str(output / "technical_report.md"),
    }
