"""Run the resumable 60-unit V2-4 extended pressure-test matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.data.extrasensory import sha256_file
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.experiments.v2_phase4 import (
    V2Phase4ExtendedRegistry,
    V2Phase4Registry,
    aggregate_v2_phase4_extended,
    build_v2_phase4_extended_plan,
)
from robustsense.utils.io import read_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_key(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _verified_lock(root: Path) -> dict[str, Any]:
    lock = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    if lock.get("phase") != "V2-4" or lock.get("status") != "frozen_before_first_test_run":
        raise ValueError("V2-4 protocol lock is absent or not frozen")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise ValueError(f"V2-4 protocol lock is invalid: {problems}")
    return lock


def _run_logged(command: list[str], root: Path, log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{_now()}] extended evaluation\n")
        stream.write(subprocess.list2cmdline(command) + "\n")
        stream.flush()
        completed = subprocess.run(
            command,
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        stream.write(f"[{_now()}] return_code={completed.returncode}\n")
    return completed.returncode


def _verify_output(root: Path, row: dict[str, str]) -> None:
    output = root / row["run_dir"] / "extended_v2_4"
    required = (
        "mask_metrics.csv",
        "mask_summary.json",
        "mixed_metrics.csv",
        "modality_response.csv",
        "fault_detection.json",
        "persistent_episode_manifest.jsonl",
        "persistent_classification.csv",
        "persistent_response.csv",
        "evaluation_manifest.json",
    )
    missing = [name for name in required if not (output / name).is_file()]
    if missing:
        raise ValueError(f"Extended output is incomplete: {missing}")
    manifest = read_json(output / "evaluation_manifest.json")
    if (
        manifest.get("model_id") != row["model_name"]
        or int(manifest.get("fold", -1)) != int(row["fold"])
        or int(manifest.get("seed", -1)) != int(row["seed"])
    ):
        raise ValueError("Extended manifest identity mismatch")
    if manifest.get("threshold_source_split") != "val":
        raise ValueError("Extended evaluation thresholds are not validation-derived")
    if manifest.get("mask_scenario_count") != 63:
        raise ValueError("Extended mask matrix is incomplete")
    if manifest.get("mixed_scenario_count") != 60:
        raise ValueError("Extended mixed-failure matrix is incomplete")
    if manifest.get("persistent_scenario_count") != 12:
        raise ValueError("Extended persistent matrix is incomplete")
    if manifest.get("per_sample_predictions_stored") is not False:
        raise ValueError("Extended evaluator unexpectedly stored per-scenario predictions")
    if sha256_file(output / "evaluation_manifest.json") == "":
        raise AssertionError("Unreachable empty manifest digest")


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.project_root).resolve()
    lock = _verified_lock(root)
    core_registry = V2Phase4Registry(root)
    planned = build_v2_phase4_extended_plan(root, core_registry)
    registry = V2Phase4ExtendedRegistry(root)
    recovered = registry.recover_interrupted()
    registry.register(planned, lock["protocol_sha256"])
    if arguments.plan_only:
        return {
            "phase": "V2-4",
            "profile": "v2_multiseed_extended",
            "registered_unit_count": len(planned),
            "recovered_interrupted_units": recovered,
            "launched": 0,
        }
    desired = "failed" if arguments.retry_failed else "pending"
    current = {row["run_key"]: row for row in registry.rows()}
    rows = [current[row["run_key"]] for row in planned]
    rows = [row for row in rows if row["status"] == desired]
    if arguments.model:
        rows = [row for row in rows if row["model_name"] in set(arguments.model)]
    if arguments.fold:
        rows = [row for row in rows if int(row["fold"]) in set(arguments.fold)]
    if arguments.seed:
        rows = [row for row in rows if int(row["seed"]) in set(arguments.seed)]
    if arguments.max_units is not None:
        rows = rows[: arguments.max_units]
    outcomes = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{_now()}] V2-4 extended {index}/{len(rows)}: {row['run_key']}",
            flush=True,
        )
        attempt = int(row["attempt_count"]) + 1
        log_relative = (
            Path("runs/v2/_phase_v2_4_extended_logs")
            / _safe_key(row["run_key"])
            / f"attempt-{attempt}.log"
        )
        command = [
            sys.executable,
            str(root / "scripts/run_v2_phase4_extended_unit.py"),
            "--run-dir",
            str(root / row["run_dir"]),
            "--model-id",
            row["model_name"],
        ]
        started = _now()
        registry.update(
            row["run_key"],
            status="running",
            attempt_count=attempt,
            command=subprocess.list2cmdline(command),
            reason="",
            error_log=str(log_relative),
        )
        return_code = _run_logged(command, root, root / log_relative)
        reason = f"evaluation_exit_{return_code}" if return_code else ""
        if return_code == 0:
            try:
                _verify_output(root, row)
            except Exception as exc:
                return_code = 1
                reason = f"output_contract_error: {exc}"
        status = "success" if return_code == 0 else "failed"
        registry.update(row["run_key"], status=status, reason=reason)
        registry.append_attempt(
            run_key=row["run_key"],
            attempt=attempt,
            status=status,
            started_at=started,
            finished_at=_now(),
            return_code=return_code,
            log_path=str(log_relative),
            reason=reason,
        )
        outcomes.append({"run_key": row["run_key"], "status": status, "reason": reason})
    aggregate = None
    if registry.profile_complete("v2_multiseed_extended"):
        aggregate = aggregate_v2_phase4_extended(root, registry)
    failures = [row for row in outcomes if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} V2-4 extended units failed")
    return {
        "phase": "V2-4",
        "profile": "v2_multiseed_extended",
        "launched": len(rows),
        "outcomes": outcomes,
        "status_counts": registry.terminal_counts("v2_multiseed_extended"),
        "aggregate": aggregate,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--model", action="append", choices=("B4", "B5", "P", "P2"))
    parser.add_argument("--fold", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    result = execute(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
