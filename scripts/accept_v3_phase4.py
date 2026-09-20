"""验收 V3-4 双策略固定阈值矩阵、诊断、测试与报告。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.experiments.v2_phase0 import (
    digest_entries,
    sha256_file,
    verify_file_entries,
)
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


def _validate_unit(root: Path, path: Path, protocol: dict[str, Any]) -> float:
    manifest = read_json(path)
    if (
        manifest.get("status") != "success"
        or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
    ):
        raise RuntimeError(f"V3-4 单元状态或协议无效：{path}")
    problems = verify_file_entries(
        root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
    )
    if problems:
        raise RuntimeError(f"V3-4 单元产物失效：{path}: {problems}")
    expected_contract = {
        "threshold_fit_split": "val",
        "evaluation_split": "test",
        "test_rank_selection_allowed": False,
        "automatic_mode_selection_allowed": False,
        "policy_modes": ["risk_control", "error_alert"],
        "target_coverages": [0.8, 0.9, 0.95],
        "result_row_count": 6,
    }
    for key, value in expected_contract.items():
        if manifest.get(key) != value:
            raise RuntimeError(f"V3-4 单元合同错误 {key}：{path}")

    results = pd.read_csv(path.parent / "results.csv")
    if len(results) != 6:
        raise RuntimeError(f"V3-4 单元结果不是 6 行：{path}")
    if set(results["policy_mode"]) != {"risk_control", "error_alert"}:
        raise RuntimeError(f"V3-4 策略模式错误：{path}")
    if set(np.round(results["target_coverage"], 2)) != {0.8, 0.9, 0.95}:
        raise RuntimeError(f"V3-4 目标覆盖率错误：{path}")
    if set(results["threshold_source_split"]) != {"val"}:
        raise RuntimeError(f"V3-4 阈值不是来自 validation：{path}")
    if set(results["test_rank_selection"].astype(str).str.lower()) != {"false"}:
        raise RuntimeError(f"V3-4 使用了 test 排名：{path}")
    required_metrics = [
        "test_coverage",
        "coverage_absolute_error",
        "risk_masked_bce",
        "macro_f1",
        "micro_f1",
        "accepted_error_prevalence",
        "rejected_error_precision",
        "rejected_error_recall",
    ]
    if not np.isfinite(results[required_metrics].to_numpy(dtype=float)).all():
        raise RuntimeError(f"V3-4 单元存在无效指标：{path}")
    policy = read_json(path.parent / "policy.json")
    if (
        policy.get("policy") != "explicit_dual_objective"
        or policy.get("threshold_source_split") != "val"
        or policy.get("automatic_mode_selection") is not False
        or set(policy.get("thresholds", {})) != {"risk_control", "error_alert"}
        or any(len(values) != 3 for values in policy["thresholds"].values())
    ):
        raise RuntimeError(f"V3-4 固定阈值策略产物错误：{path}")
    return float(manifest["max_abs_probability_delta_vs_frozen"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    protocol = read_json(root / "reports/v3/phase_v3_4_protocol_lock.json")
    problems = verify_file_entries(root, protocol["files"])
    if problems:
        raise RuntimeError(f"V3-4 协议锁无效：{problems}")

    unit_manifests = sorted(
        (root / "reports/v3/phase_v3_4/units").glob("fold*_seed*/manifest.json")
    )
    if len(unit_manifests) != 15:
        raise RuntimeError(f"V3-4 单元不是 15/15：{len(unit_manifests)}")
    max_delta = max(
        _validate_unit(root, path, protocol) for path in unit_manifests
    )
    if max_delta > 1.0e-6:
        raise RuntimeError(f"V3-4 冻结概率复现差超限：{max_delta}")

    paths = {
        "fold_seed_results": root / "reports/v3/phase_v3_4_fold_seed_results.csv",
        "fold_summary": root / "reports/v3/phase_v3_4_fold_summary.csv",
        "summary": root / "reports/v3/phase_v3_4_summary.csv",
        "comparisons": root / "reports/v3/phase_v3_4_comparisons.csv",
        "diagnostics": root / "reports/v3/phase_v3_4_diagnostics.json",
        "report": root / "reports/v3/phase_v3_4_report.json",
        "document": root / "docs/v3/PHASE_V3_4_REPORT.md",
        "pytest": root / "reports/v3/phase_v3_4_pytest.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V3-4 汇总产物缺失：{missing}")
    expected_rows = {
        "fold_seed_results": 90,
        "fold_summary": 30,
        "summary": 6,
        "comparisons": 6,
    }
    actual_rows = {
        name: len(pd.read_csv(paths[name])) for name in expected_rows
    }
    if actual_rows != expected_rows:
        raise RuntimeError(f"V3-4 汇总行数错误：{actual_rows}")

    report = read_json(paths["report"])
    if (
        report.get("status") != "complete"
        or report.get("unit_count") != 15
        or report.get("result_row_count") != 90
        or report.get("policy_modes") != ["risk_control", "error_alert"]
        or report.get("target_coverages") != [0.8, 0.9, 0.95]
        or set(report.get("primary_hypotheses", {}))
        != {"risk_specialization", "alert_specialization"}
    ):
        raise RuntimeError("V3-4 机器报告不完整")
    diagnostics = read_json(paths["diagnostics"])
    if (
        diagnostics.get("status")
        != "complete_posthoc_diagnostic_not_primary_hypothesis"
        or len(diagnostics.get("paired_diagnostics", [])) != 3
    ):
        raise RuntimeError("V3-4 后验诊断边界不清晰")
    tests = _pytest_summary(paths["pytest"])
    if tests != {"tests": 101, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"V3-4 回归测试不符合合同：{tests}")

    source_paths = [
        root / "reports/v3/phase_v3_4_protocol_lock.json",
        *unit_manifests,
        *paths.values(),
        root / "scripts/analyze_v3_phase4.py",
        root / "scripts/accept_v3_phase4.py",
    ]
    entries = [_entry(root, path) for path in source_paths]
    result = {
        "version": 1,
        "phase": "V3-4",
        "status": "passed",
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "acceptance_sha256": digest_entries(entries),
        "unit_count": len(unit_manifests),
        "row_counts": actual_rows,
        "max_abs_probability_delta_vs_frozen": max_delta,
        "pytest": tests,
        "dual_specialization_passed": report["dual_specialization_passed"],
        "primary_hypotheses": report["primary_hypotheses"],
        "coverage_transfer_readiness": report["coverage_transfer_readiness"],
        "coverage_transfer_all_modes_passed": report[
            "coverage_transfer_all_modes_passed"
        ],
        "scientific_outcome_note": (
            "完整性验收通过不改变错误告警首要假设失败的结论。"
        ),
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_4_acceptance.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
