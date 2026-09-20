"""校验 V2-6 最终材料并生成内部发布锁。"""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageSequence

from robustsense.experiments.v2_phase0 import (
    digest_entries,
    sha256_file,
    verify_file_entries,
    verify_v1_manifest,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

PATHS = (
    "configs/v2/phase_v2_6_plan.yaml",
    "docs/v2/DEMO_RECORDING_SCRIPT.md",
    "docs/v2/EXPERIMENT_PROTOCOL_V2_FINAL.md",
    "docs/v2/INTERVIEW_QA_V2.md",
    "docs/v2/MODEL_CARD_V2.md",
    "docs/v2/ORAL_SCRIPT_2MIN_V2.md",
    "docs/v2/ORAL_SCRIPT_8MIN_V2.md",
    "docs/v2/PERFORMANCE_OPTIMIZATION_CASE.md",
    "docs/v2/PHASE_V2_6_REPORT.md",
    "docs/v2/README.md",
    "docs/v2/RELEASE_REVIEW_V2.md",
    "docs/v2/TECHNICAL_REPORT_V2.md",
    "reports/v2/demo_v2_walkthrough.gif",
    "reports/v2/phase_v2_5_acceptance.json",
    "reports/v2/phase_v2_5_streamlit_health.json",
    "reports/v2/phase_v2_6_pytest.xml",
    "reports/v2/source_map.csv",
    "scripts/build_v2_demo_gif.py",
    "scripts/lock_v2_phase6_release.py",
)


def test_summary(path: Path) -> dict[str, int]:
    root = ET.parse(path).getroot()
    suite = root if "tests" in root.attrib else root.find("testsuite")
    if suite is None:
        raise RuntimeError("pytest XML 中没有 testsuite")
    return {
        "tests": int(suite.attrib["tests"]),
        "failures": int(suite.attrib["failures"]),
        "errors": int(suite.attrib["errors"]),
        "skipped": int(suite.attrib["skipped"]),
    }


def gif_summary(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        durations = [frame.info.get("duration", 0) for frame in ImageSequence.Iterator(image)]
        return {"frames": image.n_frames, "duration_ms": sum(durations)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()

    plan = load_config(root / "configs/v2/phase_v2_6_plan.yaml")
    parent = read_json(root / plan["parent_protocol_lock"])
    parent_problems = verify_file_entries(root, parent["files"])
    if parent_problems:
        raise RuntimeError(f"V2-5 父协议锁无效：{parent_problems}")

    core = read_json(root / "reports/v2/phase_v2_4_core_report.json")
    extended = read_json(root / "reports/v2/phase_v2_4_extended_report.json")
    acceptance = read_json(root / plan["demo_acceptance"])
    health = read_json(root / plan["demo_health"])
    if core["successful_core_unit_count"] != plan["core_success_required"]:
        raise RuntimeError("V2-4 核心矩阵不完整")
    if extended["successful_unit_count"] != plan["extended_success_required"]:
        raise RuntimeError("V2-4 扩展矩阵不完整")
    if acceptance["status"] != "passed" or health["status"] != "passed":
        raise RuntimeError("V2-5 Demo 验收或健康检查未通过")

    v1 = verify_v1_manifest(root, root / "reports/v2/v1_baseline_manifest.json")
    if v1["status"] != "passed":
        raise RuntimeError(f"V1 冻结清单被改写：{v1['problems']}")

    tests = test_summary(root / "reports/v2/phase_v2_6_pytest.xml")
    if tests != {"tests": 78, "failures": 0, "errors": 0, "skipped": 0}:
        raise RuntimeError(f"pytest 结果不符合最终合同：{tests}")
    demo_gif = gif_summary(root / plan["demo_gif"])
    if demo_gif != {"frames": 7, "duration_ms": 45000}:
        raise RuntimeError(f"Demo 动图不符合 7 帧、45 秒合同：{demo_gif}")

    entries = []
    for relative in PATHS:
        path = root / relative
        entries.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": sha256_file(path)}
        )

    result = {
        "version": 1,
        "phase": "V2-6",
        "status": "internal_release_complete_public_release_blocked",
        "locked_at_utc": datetime.now(UTC).isoformat(),
        "parent_protocol_sha256": parent["protocol_sha256"],
        "release_sha256": digest_entries(entries),
        "core_success": core["successful_core_unit_count"],
        "extended_success": extended["successful_unit_count"],
        "pytest": tests,
        "v1_verified_files": v1["verified_file_count"],
        "demo_gif": demo_gif,
        "documentation_language": plan["documentation_language"],
        "ppt_required": plan["ppt_required"],
        "environment_recreation_required": plan["environment_recreation_required"],
        "public_release_blockers": plan["public_release_blockers"],
        "files": entries,
    }
    output = root / "reports/v2/phase_v2_6_release_lock.json"
    write_json(output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
