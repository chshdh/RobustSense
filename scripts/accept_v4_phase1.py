"""验收 V4-1 最终模型优化的完整性、隔离性与校正后报告。"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robustsense.experiments.v2_phase0 import (
    digest_entries,
    sha256_file,
    verify_file_entries,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _entry(root: Path, path: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _pytest_summary(path: Path) -> dict[str, int]:
    xml_root = ET.parse(path).getroot()
    node = xml_root if "tests" in xml_root.attrib else xml_root.find("testsuite")
    if node is None:
        raise RuntimeError("pytest XML 中没有测试汇总")
    return {
        name: int(node.attrib.get(name, 0))
        for name in ("tests", "failures", "errors", "skipped")
    }


def _state_dict_parameter_count(path: Path) -> int:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict") if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"检查点不含 state_dict：{path}")
    return int(
        sum(value.numel() for value in state_dict.values() if isinstance(value, torch.Tensor))
    )


def _validate_unit(
    root: Path,
    run_dir: Path,
    protocol: dict[str, Any],
    plan: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = run_dir / "run_manifest.json"
    initialization_path = run_dir / "initialization.json"
    evaluation_path = run_dir / "evaluation_manifest.json"
    metrics_path = run_dir / "test_metrics.json"
    thresholds_path = run_dir / "thresholds.json"
    checkpoint_path = run_dir / "best_checkpoint.pt"
    predictions_path = run_dir / "predictions.parquet"
    resolved_path = run_dir / "resolved_config.yaml"
    required = [
        manifest_path,
        initialization_path,
        evaluation_path,
        metrics_path,
        thresholds_path,
        checkpoint_path,
        predictions_path,
        resolved_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"V4-1 单元产物缺失：{missing}")

    manifest = read_json(manifest_path)
    if (
        manifest.get("status") != "success_test_evaluated_once"
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        or manifest.get("test_split_opened_during_training") is not False
    ):
        raise RuntimeError(f"V4-1 单元状态或测试隔离无效：{run_dir}")
    initialization = read_json(initialization_path)
    training = plan["training"]
    equivalence = {
        "probability": (
            float(initialization["probability_max_abs_delta"]),
            float(training["initial_probability_equivalence_atol"]),
        ),
        "label_weight": (
            float(initialization["label_weight_max_abs_delta"]),
            float(training["initial_weight_equivalence_atol"]),
        ),
        "logit": (
            float(initialization["logit_max_abs_delta"]),
            float(training["initial_logit_diagnostic_atol"]),
        ),
    }
    if initialization.get("equivalent") is not True or any(
        value > tolerance for value, tolerance in equivalence.values()
    ):
        raise RuntimeError(f"V4-1 初始化等价门超限：{run_dir}: {equivalence}")

    thresholds = read_json(thresholds_path)
    if (
        thresholds.get("source_split") != "val"
        or thresholds.get("method") != "per_label_f1"
        or len(thresholds.get("values", [])) != 15
    ):
        raise RuntimeError(f"V4-1 阈值合同错误：{run_dir}")
    metrics = read_json(metrics_path)
    if (
        metrics.get("threshold_source_split") != "val"
        or metrics.get("test_time_model_selection") is not False
        or metrics.get("scenario") != "natural_missingness"
        or metrics.get("model_id") != "P2-LC"
    ):
        raise RuntimeError(f"V4-1 测试合同错误：{run_dir}")
    required_metrics = [
        "macro_f1",
        "micro_f1",
        "mean_average_precision",
        "brier_score",
        "masked_bce",
    ]
    if not all(np.isfinite(float(metrics["metrics"][name])) for name in required_metrics):
        raise RuntimeError(f"V4-1 存在无效测试指标：{run_dir}")

    resolved = load_config(resolved_path)
    source_dir = root / str(resolved["source_p2_run_dir"])
    source_checkpoint = source_dir / "best_checkpoint.pt"
    if not source_checkpoint.is_file():
        raise RuntimeError(f"V4-1 来源 P2 检查点缺失：{source_checkpoint}")
    candidate_checkpoint_sha = sha256_file(checkpoint_path)
    source_checkpoint_sha = sha256_file(source_checkpoint)
    evaluation = read_json(evaluation_path)
    if (
        evaluation.get("status") != "success_test_evaluated_once"
        or evaluation.get("protocol_sha256") != protocol["protocol_sha256"]
        or evaluation.get("test_opened_by_separate_evaluation_process") is not True
        or evaluation.get("threshold_source_split") != "val"
        or evaluation.get("checkpoint_sha256") != candidate_checkpoint_sha
        or manifest.get("checkpoint_sha256") != candidate_checkpoint_sha
        or evaluation.get("source_p2_checkpoint_sha256") != source_checkpoint_sha
        or manifest.get("source_p2_checkpoint_sha256") != source_checkpoint_sha
        or evaluation.get("test_metrics_sha256") != sha256_file(metrics_path)
        or evaluation.get("predictions_sha256") != sha256_file(predictions_path)
        or manifest.get("evaluation_manifest_sha256") != sha256_file(evaluation_path)
    ):
        raise RuntimeError(f"V4-1 评估清单哈希或合同错误：{run_dir}")
    return {
        "fold": int(manifest["fold"]),
        "seed": int(manifest["seed"]),
        "candidate_total_parameter_count": _state_dict_parameter_count(checkpoint_path),
        "max_probability_delta": equivalence["probability"][0],
        "max_label_weight_delta": equivalence["label_weight"][0],
        "max_logit_delta": equivalence["logit"][0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v4/phase_v4_1_plan.yaml")
    protocol_path = root / "reports/v4/phase_v4_1_protocol_lock.json"
    protocol = read_json(protocol_path)
    problems = verify_file_entries(root, protocol["files"])
    if problems:
        raise RuntimeError(f"V4-1 协议锁无效：{problems}")

    run_root = root / str(plan["run_root"])
    run_dirs = [
        run_root / f"extrasensory-p2-lc-fold{fold}-seed{seed}-v4_final"
        for fold in plan["folds"]
        for seed in plan["seeds"]
    ]
    unit_results = [_validate_unit(root, path, protocol, plan) for path in run_dirs]
    if len(unit_results) != 15 or len(
        {(row["fold"], row["seed"]) for row in unit_results}
    ) != 15:
        raise RuntimeError("V4-1 正式矩阵不是 15/15")

    paths = {
        "fold_seed_results": root / "reports/v4/phase_v4_1_fold_seed_results.csv",
        "fold_summary": root / "reports/v4/phase_v4_1_fold_summary.csv",
        "summary": root / "reports/v4/phase_v4_1_summary.csv",
        "report": root / "reports/v4/phase_v4_1_report.json",
        "document": root / "docs/v4/PHASE_V4_1_REPORT.md",
        "parameter_correction": root
        / "reports/v4/phase_v4_1_parameter_correction.json",
        "pytest": root / "reports/v4/phase_v4_1_pytest.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V4-1 汇总产物缺失：{missing}")
    frames = {
        "fold_seed_results": pd.read_csv(paths["fold_seed_results"]),
        "fold_summary": pd.read_csv(paths["fold_summary"]),
        "summary": pd.read_csv(paths["summary"]),
    }
    row_counts = {name: len(frame) for name, frame in frames.items()}
    if row_counts != {"fold_seed_results": 15, "fold_summary": 5, "summary": 1}:
        raise RuntimeError(f"V4-1 汇总行数错误：{row_counts}")

    correction = read_json(paths["parameter_correction"])
    candidate_counts = {row["candidate_total_parameter_count"] for row in unit_results}
    if (
        correction.get("status") != "complete"
        or correction.get("scope") != "parameter_accounting_metadata_only"
        or correction.get("model_predictions_changed") is not False
        or correction.get("scientific_metrics_changed") is not False
        or correction.get("unit_count") != 15
        or candidate_counts != {int(correction["candidate_total_parameter_count"])}
    ):
        raise RuntimeError("V4-1 参数量校正产物无效")
    table_counts = set(
        frames["fold_seed_results"]["candidate_parameter_count"].astype(int)
    )
    if table_counts != candidate_counts:
        raise RuntimeError("V4-1 汇总表没有采用检查点总参数量")

    report = read_json(paths["report"])
    primary = plan["primary_success"]
    macro_delta = float(frames["summary"].iloc[0]["delta_macro_f1"])
    fold_wins = int((frames["fold_summary"]["delta_macro_f1"] > 0).sum())
    expected_primary = bool(
        macro_delta >= float(primary["minimum_absolute_mean_gain_vs_p2"])
        and fold_wins >= int(primary["minimum_winning_fold_count"])
    )
    if (
        report.get("status") != "complete"
        or report.get("unit_count") != 15
        or report.get("final_model_optimization") is not True
        or report.get("further_model_versions_allowed") is not False
        or report.get("primary_success") is not expected_primary
        or report.get("candidate_winning_fold_count") != fold_wins
        or not np.isclose(float(report["candidate_minus_p2_macro_f1"]), macro_delta)
        or report.get("parameter_accounting_correction", {}).get("status")
        != "corrected_from_checkpoint_state_dict"
    ):
        raise RuntimeError("V4-1 机器报告与汇总或停止规则不一致")
    if report.get("outputs") and verify_file_entries(root, report["outputs"]):
        raise RuntimeError("V4-1 机器报告中的汇总哈希失效")

    tests = _pytest_summary(paths["pytest"])
    if tests != {"tests": 106, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"V4-1 回归测试不符合合同：{tests}")

    support_paths = [
        protocol_path,
        root / "configs/v4/phase_v4_1_seed13_compatibility_amendment.yaml",
        *paths.values(),
        *[path / "run_manifest.json" for path in run_dirs],
        *[path / "initialization.json" for path in run_dirs],
        *[path / "evaluation_manifest.json" for path in run_dirs],
        root / "scripts/run_v4_phase1_seed13_compat.py",
        root / "scripts/correct_v4_phase1_parameter_accounting.py",
        root / "scripts/accept_v4_phase1.py",
    ]
    entries = [_entry(root, path) for path in support_paths]
    max_deltas = {
        "probability": max(row["max_probability_delta"] for row in unit_results),
        "label_weight": max(row["max_label_weight_delta"] for row in unit_results),
        "logit": max(row["max_logit_delta"] for row in unit_results),
    }
    result = {
        "version": 1,
        "phase": "V4-1",
        "status": "passed",
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "acceptance_sha256": digest_entries(entries),
        "unit_count": len(unit_results),
        "row_counts": row_counts,
        "initialization_max_abs_deltas": max_deltas,
        "pytest": tests,
        "primary_success": expected_primary,
        "candidate_minus_p2_macro_f1": macro_delta,
        "candidate_winning_fold_count": fold_wins,
        "all_safety_gates_passed": bool(report["all_safety_gates_passed"]),
        "candidate_total_parameter_count": int(next(iter(candidate_counts))),
        "further_model_versions_allowed": False,
        "scientific_outcome_note": (
            "完整性验收通过只证明流程和产物可信，不会把未通过的首要假设改写为成功。"
        ),
        "files": entries,
    }
    output = root / "reports/v4/phase_v4_1_acceptance.json"
    write_json(output, result)
    print(f"V4-1 验收通过：{output}")


if __name__ == "__main__":
    main()
