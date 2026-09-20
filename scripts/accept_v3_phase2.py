"""验收 V3-2 任务置信度矩阵、诊断、回归测试与中文报告。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    protocol = read_json(root / "reports/v3/phase_v3_2_protocol_lock.json")
    problems = verify_file_entries(root, protocol["files"])
    if problems:
        raise RuntimeError(f"V3-2 协议锁无效：{problems}")

    unit_manifests = sorted(
        (root / "reports/v3/phase_v3_2/units").glob("fold*_seed*/manifest.json")
    )
    if len(unit_manifests) != 15:
        raise RuntimeError(f"V3-2 单元不是 15/15：{len(unit_manifests)}")
    max_delta = 0.0
    for path in unit_manifests:
        manifest = read_json(path)
        if (
            manifest.get("status") != "success"
            or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        ):
            raise RuntimeError(f"V3-2 单元状态或协议无效：{path}")
        unit_problems = verify_file_entries(
            root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
        )
        if unit_problems:
            raise RuntimeError(f"V3-2 单元产物失效：{path}: {unit_problems}")
        if manifest.get("fit_split") != "val" or manifest.get("evaluation_split") != "test":
            raise RuntimeError(f"V3-2 单元数据边界错误：{path}")
        if manifest.get("feature_counts") != {
            "task_confidence_full": 49,
            "task_confidence_output_only": 30,
        }:
            raise RuntimeError(f"V3-2 单元特征合同错误：{path}")
        if manifest.get("inference_batch_size") != 512:
            raise RuntimeError(f"V3-2 单元 batch size 错误：{path}")
        max_delta = max(
            max_delta, float(manifest["max_abs_probability_delta_vs_frozen"])
        )
        selectors_path = path.parent / "selectors.json"
        selectors = read_json(selectors_path)
        if selectors.get("source_split") != "val" or len(selectors["selectors"]) != 2:
            raise RuntimeError(f"V3-2 selector 来源或数量错误：{selectors_path}")
        if any(
            selector["iterations"] >= selector["max_iter"]
            for selector in selectors["selectors"].values()
        ):
            raise RuntimeError(f"V3-2 selector 未收敛：{selectors_path}")
    if max_delta > 1.0e-6:
        raise RuntimeError("V3-2 test 概率复现超过 1e-6")

    paths = {
        "fold_seed_curves": root / "reports/v3/phase_v3_2_fold_seed_curves.csv",
        "fold_seed_summary": root / "reports/v3/phase_v3_2_fold_seed_summary.csv",
        "fold_summary": root / "reports/v3/phase_v3_2_fold_summary.csv",
        "summary": root / "reports/v3/phase_v3_2_summary.csv",
        "curves": root / "reports/v3/phase_v3_2_curves.csv",
        "fit_generalization": root / "reports/v3/phase_v3_2_fit_generalization.csv",
        "coefficient_stability": (
            root / "reports/v3/phase_v3_2_coefficient_stability.csv"
        ),
        "diagnostics": root / "reports/v3/phase_v3_2_diagnostics.json",
        "report": root / "reports/v3/phase_v3_2_report.json",
        "document": root / "docs/v3/PHASE_V3_2_REPORT.md",
        "pytest": root / "reports/v3/phase_v3_2_pytest.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V3-2 汇总产物缺失：{missing}")
    expected_rows = {
        "fold_seed_curves": 1200,
        "fold_seed_summary": 60,
        "fold_summary": 20,
        "summary": 4,
        "curves": 80,
        "fit_generalization": 30,
        "coefficient_stability": 79,
    }
    actual_rows = {name: len(pd.read_csv(paths[name])) for name in expected_rows}
    if actual_rows != expected_rows:
        raise RuntimeError(f"V3-2 汇总行数错误：{actual_rows}")

    report = read_json(paths["report"])
    if (
        report.get("status") != "complete"
        or report.get("unit_count") != 15
        or report.get("selector_summary_count") != 60
        or report.get("curve_point_count") != 1200
        or report.get("all_units_share_protocol") is not True
        or len(report.get("primary_comparisons", [])) != 2
    ):
        raise RuntimeError("V3-2 机器报告不完整")
    diagnostics = read_json(paths["diagnostics"])
    if (
        diagnostics.get("status")
        != "complete_posthoc_diagnostic_not_primary_model_selection"
        or len(diagnostics.get("selector_summary", [])) != 2
    ):
        raise RuntimeError("V3-2 后验诊断缺少边界标记")
    tests = _pytest_summary(paths["pytest"])
    if tests != {"tests": 91, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"V3-2 回归测试不符合合同：{tests}")

    source_paths = [
        root / "reports/v3/phase_v3_2_protocol_lock.json",
        *unit_manifests,
        *paths.values(),
        root / "scripts/analyze_v3_phase2_selectors.py",
        root / "scripts/accept_v3_phase2.py",
    ]
    entries = [_entry(root, path) for path in source_paths]
    result = {
        "version": 1,
        "phase": "V3-2",
        "status": "passed",
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "acceptance_sha256": digest_entries(entries),
        "unit_count": len(unit_manifests),
        "row_counts": actual_rows,
        "max_abs_probability_delta_vs_frozen": max_delta,
        "pytest": tests,
        "primary_comparisons": report["primary_comparisons"],
        "posthoc_diagnostics": diagnostics["selector_summary"],
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_2_acceptance.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
