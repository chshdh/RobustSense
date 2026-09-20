"""V2-3 run planning, isolated registry, validation-only selection, and aggregation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from robustsense.evaluation.v2_artifacts import write_stable_csv
from robustsense.experiments.registry import RUN_FIELDS, RunRegistry
from robustsense.training.v2_phase3_trainer import v2_phase3_run_id
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


class V2Phase3Registry(RunRegistry):
    """Use V2 paths so the frozen V1 registry is never modified."""

    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.path = self.root / "reports/v2/phase_v2_3_run_registry.csv"
        self.attempt_path = self.root / "reports/v2/phase_v2_3_run_attempts.csv"


def build_v2_phase3_plan(
    project_root: str | Path,
    plan_path: str | Path = "configs/v2/phase_v2_3_plan.yaml",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    plan = load_config(path)
    experiment = load_config(root / plan["experiment_config"])
    ablations = load_config(root / plan["ablation_config"])
    if plan["folds"] != experiment["folds"] or plan["seeds"] != experiment["seeds"]:
        raise ValueError("V2-3 plan and experiment fold/seed matrices do not match")
    if plan["variants"] != list(ablations):
        raise ValueError("V2-3 plan and ablation definitions do not match")
    created = datetime.now(UTC).isoformat()
    rows = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for variant in plan["variants"]:
                run_id = v2_phase3_run_id(variant, int(fold), int(seed))
                raw = {
                    "run_key": f"V2-3|{variant}|fold{fold}|seed{seed}",
                    "group": "V2-3",
                    "profile": experiment["profile"],
                    "model_name": variant,
                    "fold": str(int(fold)),
                    "seed": str(int(seed)),
                    "status": "pending",
                    "attempt_count": "0",
                    "created_at": created,
                    "updated_at": created,
                    "run_dir": f"{experiment['run_root']}/{run_id}",
                    "protocol_sha256": "",
                    "command": "",
                    "reason": "registered_before_execution",
                    "error_log": "",
                }
                rows.append({name: raw.get(name, "") for name in RUN_FIELDS})
    if len(rows) != int(plan["expected_unit_count"]):
        raise AssertionError("V2-3 plan does not contain exactly 25 units")
    return rows


def select_v2_4_ablations(
    project_root: str | Path,
    registry_rows: list[dict[str, str]],
    *,
    maximum_selected: int = 2,
) -> dict[str, Any]:
    """Read validation metrics only; test artifacts are deliberately inaccessible here."""
    root = Path(project_root).resolve()
    validation: dict[tuple[int, int, str], float] = {}
    source_runs = []
    for row in registry_rows:
        if row["status"] != "success":
            raise ValueError("Selection requires every V2-3 unit to succeed")
        val_path = root / row["run_dir"] / "val_metrics.json"
        metrics = read_json(val_path)
        value = metrics.get("macro_f1")
        validation[(int(row["fold"]), int(row["seed"]), row["model_name"])] = (
            float(value) if value is not None else np.nan
        )
        source_runs.append(row["run_key"])
    variants = sorted({row["model_name"] for row in registry_rows} - {"P2"})
    rankings = []
    for variant in variants:
        deltas = []
        for row in registry_rows:
            if row["model_name"] != variant:
                continue
            key = (int(row["fold"]), int(row["seed"]))
            value = validation[(*key, variant)]
            baseline = validation[(*key, "P2")]
            if np.isfinite(value) and np.isfinite(baseline):
                deltas.append(value - baseline)
        rankings.append(
            {
                "variant": variant,
                "paired_fold_count": len(deltas),
                "mean_validation_macro_f1_delta_vs_P2": (
                    float(np.mean(deltas)) if deltas else np.nan
                ),
                "mean_absolute_paired_delta": (
                    float(np.mean(np.abs(deltas))) if deltas else np.nan
                ),
            }
        )
    ordered = sorted(
        rankings,
        key=lambda row: (
            -row["mean_absolute_paired_delta"]
            if np.isfinite(row["mean_absolute_paired_delta"])
            else float("inf"),
            row["variant"],
        ),
    )
    selected = [
        row["variant"]
        for row in ordered
        if np.isfinite(row["mean_absolute_paired_delta"])
    ][:maximum_selected]
    return {
        "phase": "V2-3",
        "decision_for": "V2-4 optional ablation seed expansion",
        "source_split": "val",
        "test_metrics_read": False,
        "rule": "top_two_mean_absolute_paired_validation_macro_f1_delta_vs_P2",
        "tie_break": "variant_id_ascending",
        "maximum_selected": maximum_selected,
        "selected_variants": selected,
        "ranking": ordered,
        "source_run_keys": sorted(source_runs),
    }


def aggregate_v2_phase3(
    project_root: str | Path,
    registry: V2Phase3Registry,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows = [row for row in registry.rows() if row["group"] == "V2-3"]
    if len(rows) != 25 or any(row["status"] != "success" for row in rows):
        raise ValueError("V2-3 aggregation requires 25 successful units")
    selection = select_v2_4_ablations(root, rows)
    write_json(root / "reports/v2/phase_v2_3_selection_for_v2_4.json", selection)
    result_rows = []
    for row in rows:
        run_path = root / row["run_dir"]
        val = read_json(run_path / "val_metrics.json")
        test = read_json(run_path / "test_metrics.json")
        selective = test["selective"]
        result_rows.append(
            {
                "run_id": test["run_id"],
                "variant": row["model_name"],
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                "val_macro_f1": val["macro_f1"],
                "test_forced_macro_f1": test["forced"]["macro_f1"],
                "test_forced_micro_f1": test["forced"]["micro_f1"],
                "test_forced_map": test["forced"]["mean_average_precision"],
                "test_coverage": selective["test_coverage"] if selective else np.nan,
                "test_selective_macro_f1": (
                    selective["accepted_metrics"]["macro_f1"] if selective else np.nan
                ),
            }
        )
    columns = list(result_rows[0])
    table = write_stable_csv(
        root / "reports/v2/phase_v2_3_results.csv",
        sorted(result_rows, key=lambda row: (row["variant"], row["fold"])),
        columns,
    )
    summary_rows = []
    metric_names = (
        "val_macro_f1",
        "test_forced_macro_f1",
        "test_forced_micro_f1",
        "test_forced_map",
        "test_coverage",
        "test_selective_macro_f1",
    )
    for variant in sorted({row["variant"] for row in result_rows}):
        selected_rows = [row for row in result_rows if row["variant"] == variant]
        summary: dict[str, Any] = {"variant": variant, "fold_count": len(selected_rows)}
        for metric in metric_names:
            values = np.asarray([row[metric] for row in selected_rows], dtype=np.float64)
            finite = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(finite.mean()) if len(finite) else np.nan
            summary[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else np.nan
            )
        summary_rows.append(summary)
    summary_table = write_stable_csv(
        root / "reports/v2/phase_v2_3_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )
    payload = {
        "phase": "V2-3",
        "status": "complete",
        "planned_unit_count": 25,
        "successful_unit_count": 25,
        "failed_unit_count": 0,
        "result_table": table,
        "summary_table": summary_table,
        "selection_for_v2_4": selection,
    }
    write_json(root / "reports/v2/phase_v2_3_interim_report.json", payload)
    return payload
