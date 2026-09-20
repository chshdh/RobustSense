"""验收 V3-1 完整矩阵、协议链、回归测试与最终报告。"""

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
    if "tests" in xml_root.attrib:
        node = xml_root
    else:
        node = xml_root.find("testsuite")
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

    protocol = read_json(root / "reports/v3/phase_v3_1_protocol_lock.json")
    protocol_problems = verify_file_entries(root, protocol["files"])
    if protocol_problems:
        raise RuntimeError(f"V3-1 原协议锁无效：{protocol_problems}")
    amendment = read_json(root / "reports/v3/phase_v3_1_execution_amendment_lock.json")
    amendment_problems = verify_file_entries(root, amendment["files"])
    if amendment_problems:
        raise RuntimeError(f"V3-1 执行修正锁无效：{amendment_problems}")
    if amendment["parent_protocol_sha256"] != protocol["protocol_sha256"]:
        raise RuntimeError("V3-1 协议链断裂")

    unit_manifests = sorted(
        (root / "reports/v3/phase_v3_1/units").glob("fold*_seed*/manifest.json")
    )
    if len(unit_manifests) != 15:
        raise RuntimeError(f"V3-1 单元不是 15/15：{len(unit_manifests)}")
    max_probability_delta = 0.0
    for path in unit_manifests:
        manifest = read_json(path)
        if manifest.get("status") != "success":
            raise RuntimeError(f"V3-1 单元未成功：{path}")
        if manifest.get("protocol_sha256") != protocol["protocol_sha256"]:
            raise RuntimeError(f"V3-1 单元协议不一致：{path}")
        problems = verify_file_entries(
            root, [*manifest.get("sources", []), *manifest.get("outputs", [])]
        )
        if problems:
            raise RuntimeError(f"V3-1 单元产物失效：{path}: {problems}")
        replay = manifest["p2_reinference"]
        if replay["batch_size"] != 512:
            raise RuntimeError(f"V3-1 单元没有使用修正后的 batch size：{path}")
        max_probability_delta = max(
            max_probability_delta,
            float(replay["max_abs_probability_delta_vs_frozen"]),
        )
    if max_probability_delta > amendment["probability_consistency_atol"]:
        raise RuntimeError("V3-1 P2 重新推理超过概率一致性闸门")

    paths = {
        "fold_seed_curves": root / "reports/v3/phase_v3_1_fold_seed_curves.csv",
        "fold_seed_summary": root / "reports/v3/phase_v3_1_fold_seed_summary.csv",
        "fold_summary": root / "reports/v3/phase_v3_1_fold_summary.csv",
        "summary": root / "reports/v3/phase_v3_1_summary.csv",
        "curves": root / "reports/v3/phase_v3_1_curves.csv",
        "report": root / "reports/v3/phase_v3_1_report.json",
        "document": root / "docs/v3/PHASE_V3_1_REPORT.md",
        "pytest": root / "reports/v3/phase_v3_1_pytest.xml",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"V3-1 汇总产物缺失：{missing}")
    expected_rows = {
        "fold_seed_curves": 2700,
        "fold_seed_summary": 135,
        "fold_summary": 45,
        "summary": 9,
        "curves": 180,
    }
    actual_rows = {
        name: len(pd.read_csv(paths[name])) for name in expected_rows
    }
    if actual_rows != expected_rows:
        raise RuntimeError(f"V3-1 汇总行数不符合合同：{actual_rows}")

    report = read_json(paths["report"])
    if (
        report.get("status") != "complete"
        or report.get("unit_count") != 15
        or report.get("selector_summary_count") != 135
        or report.get("curve_point_count") != 2700
        or report.get("all_units_share_protocol") is not True
    ):
        raise RuntimeError("V3-1 机器可读报告不完整")
    if len(report.get("primary_comparisons", [])) != 2:
        raise RuntimeError("V3-1 缺少两个预声明主比较")

    tests = _pytest_summary(paths["pytest"])
    if tests != {"tests": 85, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"V3-1 回归测试不符合合同：{tests}")

    source_paths = [
        root / "reports/v3/phase_v3_1_protocol_lock.json",
        root / "reports/v3/phase_v3_1_execution_amendment_lock.json",
        *unit_manifests,
        *paths.values(),
        root / "scripts/accept_v3_phase1.py",
    ]
    entries = [_entry(root, path) for path in source_paths]
    result = {
        "version": 1,
        "phase": "V3-1",
        "status": "passed",
        "accepted_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": protocol["protocol_sha256"],
        "amendment_sha256": amendment["amendment_sha256"],
        "acceptance_sha256": digest_entries(entries),
        "unit_count": len(unit_manifests),
        "row_counts": actual_rows,
        "max_abs_probability_delta_vs_frozen": max_probability_delta,
        "probability_consistency_atol": amendment["probability_consistency_atol"],
        "pytest": tests,
        "primary_comparisons": report["primary_comparisons"],
        "files": entries,
    }
    output = root / "reports/v3/phase_v3_1_acceptance.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
