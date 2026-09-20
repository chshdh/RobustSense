"""Rebuild the frozen V1 report twice and verify byte-level determinism."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.evaluation.report import generate_report
from robustsense.experiments.v2_phase0 import sha256_file
from robustsense.utils.io import write_json

ROOT_REPORTS = (
    "readme_source_map.csv",
    "run_completeness.json",
    "source_map.csv",
    "technical_report.md",
)


def _snapshot(root: Path) -> dict[str, str]:
    reports = root / "reports"
    paths = [reports / name for name in ROOT_REPORTS]
    paths.extend(sorted((reports / "tables").glob("*.csv")))
    paths.extend(sorted((reports / "figures").glob("*.png")))
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"V1 report artifacts are missing: {missing}")
    return {path.relative_to(root).as_posix(): sha256_file(path) for path in paths}


def _changes(before: dict[str, str], after: dict[str, str]) -> list[dict[str, Any]]:
    paths = sorted(set(before) | set(after))
    return [
        {"path": path, "before": before.get(path), "after": after.get(path)}
        for path in paths
        if before.get(path) != after.get(path)
    ]


def verify_rebuild(root: Path) -> dict[str, Any]:
    before = _snapshot(root)
    generate_report(root, "runs", "reports", "configs/evaluation/credible.yaml")
    first = _snapshot(root)
    generate_report(root, "runs", "reports", "configs/evaluation/credible.yaml")
    second = _snapshot(root)
    initial_changes = _changes(before, first)
    repeat_changes = _changes(first, second)
    return {
        "version": 1,
        "phase": "V2-0",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "artifact_count": len(second),
        "initial_rebuild_changed_count": len(initial_changes),
        "repeat_rebuild_changed_count": len(repeat_changes),
        "initial_rebuild_changes": initial_changes,
        "repeat_rebuild_changes": repeat_changes,
        "status": "passed" if not initial_changes and not repeat_changes else "failed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/v1_report_rebuild.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = verify_rebuild(root)
    write_json(output, result)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

