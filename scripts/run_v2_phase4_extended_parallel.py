"""Run the resumable V2-4 extended matrix with two independent workers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from run_v2_phase4_extended import _run_logged, _verified_lock, _verify_output

from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.experiments.v2_phase4 import (
    V2Phase4ExtendedRegistry,
    V2Phase4Registry,
    aggregate_v2_phase4_extended,
    build_v2_phase4_extended_plan,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_key(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _verified_amendment(root: Path, parent: dict[str, Any]) -> dict[str, Any]:
    amendment = read_json(
        root / "reports/v2/phase_v2_4_extended_execution_amendment_lock.json"
    )
    problems = verify_file_entries(root, amendment["files"])
    if problems:
        raise ValueError(f"Extended execution amendment is invalid: {problems}")
    if (
        amendment.get("status") != "frozen_before_parallel_launch"
        or amendment.get("parent_protocol_sha256") != parent["protocol_sha256"]
    ):
        raise ValueError("Extended execution amendment does not match the protocol")
    return amendment


def _worker(
    root: Path,
    row: dict[str, str],
    attempt: int,
    log_relative: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(root / "scripts/run_v2_phase4_extended_unit.py"),
        "--run-dir",
        str(root / row["run_dir"]),
        "--model-id",
        row["model_name"],
    ]
    reason = ""
    return_code = 1
    try:
        return_code = _run_logged(command, root, root / log_relative)
        reason = f"evaluation_exit_{return_code}" if return_code else ""
        if return_code == 0:
            _verify_output(root, row)
    except Exception as exc:
        return_code = 1
        reason = f"output_contract_error: {exc}"
    return {
        "attempt": attempt,
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "reason": reason,
        "log_relative": str(log_relative),
    }


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.project_root).resolve()
    parent = _verified_lock(root)
    amendment = _verified_amendment(root, parent)
    config = load_config(
        root / "configs/v2/phase_v2_4_extended_execution_amendment.yaml"
    )
    workers = int(config["max_parallel_units"])
    if workers != 2 or workers != int(amendment["max_parallel_units"]):
        raise ValueError("Extended worker count differs from the frozen amendment")
    core_registry = V2Phase4Registry(root)
    planned = build_v2_phase4_extended_plan(root, core_registry)
    registry = V2Phase4ExtendedRegistry(root)
    recovered = registry.recover_interrupted()
    registry.register(planned, parent["protocol_sha256"])
    current = {row["run_key"]: row for row in registry.rows()}
    rows = [current[row["run_key"]] for row in planned]
    desired = "failed" if arguments.retry_failed else "pending"
    rows = [row for row in rows if row["status"] == desired]
    if arguments.model:
        rows = [row for row in rows if row["model_name"] in set(arguments.model)]
    if arguments.fold:
        rows = [row for row in rows if int(row["fold"]) in set(arguments.fold)]
    if arguments.seed:
        rows = [row for row in rows if int(row["seed"]) in set(arguments.seed)]
    if arguments.max_units is not None:
        rows = rows[: arguments.max_units]
    if arguments.plan_only:
        return {
            "phase": "V2-4",
            "profile": "v2_multiseed_extended",
            "parallel_workers": workers,
            "recovered_interrupted_units": recovered,
            "eligible_units": len(rows),
            "launched": 0,
        }

    outcomes: list[dict[str, Any]] = []
    pending = iter(rows)
    active: dict[Future[dict[str, Any]], tuple[dict[str, str], str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while True:
            while len(active) < workers:
                try:
                    row = next(pending)
                except StopIteration:
                    break
                attempt = int(row["attempt_count"]) + 1
                log_relative = (
                    Path("runs/v2/_phase_v2_4_extended_parallel_logs")
                    / _safe_key(row["run_key"])
                    / f"attempt-{attempt}.log"
                )
                started = _now()
                command = [
                    sys.executable,
                    str(root / "scripts/run_v2_phase4_extended_unit.py"),
                    "--run-dir",
                    str(root / row["run_dir"]),
                    "--model-id",
                    row["model_name"],
                ]
                registry.update(
                    row["run_key"],
                    status="running",
                    attempt_count=attempt,
                    command=subprocess.list2cmdline(command),
                    reason=f"parallel_worker_limit_{workers}",
                    error_log=str(log_relative),
                )
                future = pool.submit(_worker, root, row, attempt, log_relative)
                active[future] = (row, started)
                print(f"[{_now()}] extended start: {row['run_key']}", flush=True)
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                row, started = active.pop(future)
                result = future.result()
                registry.update(
                    row["run_key"], status=result["status"], reason=result["reason"]
                )
                registry.append_attempt(
                    run_key=row["run_key"],
                    attempt=result["attempt"],
                    status=result["status"],
                    started_at=started,
                    finished_at=_now(),
                    return_code=result["return_code"],
                    log_path=result["log_relative"],
                    reason=result["reason"],
                )
                outcomes.append({"run_key": row["run_key"], **result})
                print(
                    f"[{_now()}] extended finish: {row['run_key']} -> "
                    f"{result['status']}",
                    flush=True,
                )
    aggregate = None
    if registry.profile_complete("v2_multiseed_extended"):
        aggregate = aggregate_v2_phase4_extended(root, registry)
    failures = [row for row in outcomes if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} parallel V2-4 extended units failed")
    return {
        "phase": "V2-4",
        "profile": "v2_multiseed_extended",
        "parallel_workers": workers,
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
    parser.add_argument("--max-units", type=int)
    parser.add_argument("--model", action="append", choices=("B4", "B5", "P", "P2"))
    parser.add_argument("--fold", action="append", type=int)
    parser.add_argument("--seed", action="append", type=int)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
