"""验收 V3-3 用户分组连续风险矩阵、诊断、测试与报告。"""

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
    protocol = read_json(root / "reports/v3/phase_v3_3_protocol_lock.json")
    problems = verify_file_entries(root, protocol["files"])
    if problems:
        raise RuntimeError(f"V3-3 协议锁无效：{problems}")

    unit_manifests = sorted(
        (root / "reports/v3/phase_v3_3/units").glob("fold*_seed*/manifest.json")
    )
    if len(unit_manifests) != 15:
        raise RuntimeError(f"V3-3 单元不是 15/15：{len(unit_manifests)}")
    selected_counts: dict[str, int] = {}
    max_delta = 0.0
    max_group_overlap = 0
    for path in unit_manifests:
        manifest = read_json(path)
        if (
            manifest.get("status") != "success"
            or manifest.get("protocol_sha256") != protocol["protocol_sha256"]
        ):
            raise RuntimeError(f"V3-3 单元状态或协议无效：{path}")
        unit_problems = verify_file_entries(
            root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
        )
        if unit_problems:
            raise RuntimeError(f"V3-3 单元产物失效：{path}: {unit_problems}")
        if (
            manifest.get("fit_split") != "val"
            or manifest.get("selection_split") != "val_grouped_oof"
            or manifest.get("evaluation_split") != "test"
            or manifest.get("grouping_unit") != "user_id"
        ):
            raise RuntimeError(f"V3-3 数据边界或分组单位错误：{path}")
        if manifest.get("feature_counts") != {
            "grouped_risk_full_compact": 11,
            "grouped_risk_output_compact": 6,
        }:
            raise RuntimeError(f"V3-3 特征合同错误：{path}")
        max_delta = max(
            max_delta, float(manifest["max_abs_probability_delta_vs_frozen"])
        )
        selectors = read_json(path.parent / "selectors.json")
        if len(selectors.get("selectors", {})) != 2:
            raise RuntimeError(f"V3-3 selector 数量错误：{path}")
        oof_values = {}
        for name, artifact in selectors["selectors"].items():
            selection = artifact["selection"]
            if len(selection["candidates"]) != 5 or len(selection["fold_contract"]) != 4:
                raise RuntimeError(f"V3-3 α 网格或 GroupKFold 数量错误：{path}")
            overlap = max(
                fold_contract["group_overlap_count"]
                for fold_contract in selection["fold_contract"]
            )
            max_group_overlap = max(max_group_overlap, overlap)
            oof_values[name] = selection["selected_oof_normalized_aurc"]
        expected_family = (
            "grouped_risk_full_compact"
            if oof_values["grouped_risk_full_compact"]
            < oof_values["grouped_risk_output_compact"]
            else "grouped_risk_output_compact"
        )
        if selectors["selected_family"] != expected_family:
            raise RuntimeError(f"V3-3 family 选择没有遵循 OOF AURC：{path}")
        selected_counts[expected_family] = selected_counts.get(expected_family, 0) + 1
    if max_delta > 1.0e-6 or max_group_overlap != 0:
        raise RuntimeError("V3-3 概率复现或用户隔离合同失败")

    paths = {
        "fold_seed_curves": root / "reports/v3/phase_v3_3_fold_seed_curves.csv",
        "fold_seed_summary": root / "reports/v3/phase_v3_3_fold_seed_summary.csv",
        "fold_summary": root / "reports/v3/phase_v3_3_fold_summary.csv",
        "summary": root / "reports/v3/phase_v3_3_summary.csv",
        "curves": root / "reports/v3/phase_v3_3_curves.csv",
        "selection_diagnostics": root / "reports/v3/phase_v3_3_selection_diagnostics.csv",
        "diagnostics": root / "reports/v3/phase_v3_3_diagnostics.json",
        "report": root / "reports/v3/phase_v3_3_report.json",
        "document": root / "docs/v3/PHASE_V3_3_REPORT.md",
        "pytest": root / "reports/v3/phase_v3_3_pytest.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V3-3 汇总产物缺失：{missing}")
    expected_rows = {
        "fold_seed_curves": 1500,
        "fold_seed_summary": 75,
        "fold_summary": 25,
        "summary": 5,
        "curves": 100,
        "selection_diagnostics": 30,
    }
    actual_rows = {name: len(pd.read_csv(paths[name])) for name in expected_rows}
    if actual_rows != expected_rows:
        raise RuntimeError(f"V3-3 汇总行数错误：{actual_rows}")
    report = read_json(paths["report"])
    if (
        report.get("status") != "complete"
        or report.get("unit_count") != 15
        or report.get("selector_summary_count") != 75
        or report.get("curve_point_count") != 1500
        or report.get("selected_family_counts") != selected_counts
        or len(report.get("primary_comparisons", [])) != 2
    ):
        raise RuntimeError("V3-3 机器报告不完整")
    diagnostics = read_json(paths["diagnostics"])
    if (
        diagnostics.get("status")
        != "complete_posthoc_diagnostic_not_primary_model_selection"
        or diagnostics.get("max_group_overlap_count") != 0
    ):
        raise RuntimeError("V3-3 诊断缺少用户隔离或后验边界标记")
    tests = _pytest_summary(paths["pytest"])
    if tests != {"tests": 96, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"V3-3 回归测试不符合合同：{tests}")

    source_paths = [
        root / "reports/v3/phase_v3_3_protocol_lock.json",
        *unit_manifests,
        *paths.values(),
        root / "scripts/analyze_v3_phase3_selection.py",
        root / "scripts/accept_v3_phase3.py",
    ]
    entries = [_entry(root, path) for path in source_paths]
    result = {
        "version": 1,
        "phase": "V3-3",
        "status": "passed",
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "acceptance_sha256": digest_entries(entries),
        "unit_count": len(unit_manifests),
        "row_counts": actual_rows,
        "selected_family_counts": selected_counts,
        "max_group_overlap_count": max_group_overlap,
        "max_abs_probability_delta_vs_frozen": max_delta,
        "pytest": tests,
        "primary_comparisons": report["primary_comparisons"],
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_3_acceptance.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
