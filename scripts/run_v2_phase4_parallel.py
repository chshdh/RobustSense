"""Run remaining V2-4 core units with two workers and one registry writer."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.experiments.v2_phase4 import (
    CORE_MODELS,
    V2Phase4Registry,
    aggregate_v2_phase4_core,
    bootstrap_v2_phase4_natural,
    build_v2_phase4_core_plan,
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


def _verified_locks(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    parent = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    parent_problems = verify_file_entries(root, parent["files"])
    if parent.get("status") != "frozen_before_first_test_run" or parent_problems:
        raise ValueError(f"V2-4 parent lock is invalid: {parent_problems}")
    amendment = read_json(
        root / "reports/v2/phase_v2_4_execution_amendment_lock.json"
    )
    amendment_problems = verify_file_entries(root, amendment["files"])
    if amendment.get("status") != "frozen_before_parallel_launch" or amendment_problems:
        raise ValueError(f"Parallel amendment lock is invalid: {amendment_problems}")
    if amendment.get("parent_protocol_sha256") != parent["protocol_sha256"]:
        raise ValueError("Parallel amendment does not reference the active parent lock")
    return parent, amendment


def _p2_command(
    root: Path,
    row: dict[str, str],
    protocol_sha256: str,
    mode: str,
) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts/run_v2_phase4_unit.py"),
        mode,
        "--project-root",
        str(root),
        "--variant",
        "P2",
        "--fold",
        row["fold"],
        "--seed",
        row["seed"],
        "--protocol-sha256",
        protocol_sha256,
    ]


def _v1_command(
    root: Path,
    row: dict[str, str],
    protocol_sha256: str,
    model_name: str,
    mode: str,
) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts/run_v2_phase4_unit.py"),
        mode,
        "--project-root",
        str(root),
        "--v1-model",
        model_name,
        "--fold",
        row["fold"],
        "--seed",
        row["seed"],
        "--protocol-sha256",
        protocol_sha256,
    ]


def _commands(
    root: Path, row: dict[str, str], protocol_sha256: str
) -> tuple[list[str], list[str]]:
    model_id = row["model_name"]
    if model_id == "P2":
        return (
            _p2_command(root, row, protocol_sha256, "train"),
            _p2_command(root, row, protocol_sha256, "evaluate"),
        )
    model_name = str(CORE_MODELS[model_id]["model_name"])
    return (
        _v1_command(root, row, protocol_sha256, model_name, "train"),
        _v1_command(root, row, protocol_sha256, model_name, "evaluate"),
    )


def _verify_outputs(root: Path, row: dict[str, str]) -> None:
    run_path = root / row["run_dir"]
    required = [
        "best_checkpoint.pt",
        "resolved_config.yaml",
        "thresholds.json",
        "val_metrics.json",
        "test_metrics.json",
        "predictions.parquet",
        "evaluation_manifest.json",
    ]
    if row["model_name"] == "P2":
        required.extend(("abstention_threshold.json", "risk_coverage.json", "run_manifest.json"))
    else:
        required.extend(("v2_run_manifest.json", "modality_diagnostics.parquet"))
    missing = [name for name in required if not (run_path / name).is_file()]
    if missing:
        raise ValueError(f"V2-4 output contract is incomplete: {missing}")
    resolved = load_config(run_path / "resolved_config.yaml")
    evaluation = read_json(run_path / "evaluation_manifest.json")
    if int(resolved["fold"]) != int(row["fold"]) or int(resolved["seed"]) != int(
        row["seed"]
    ):
        raise ValueError("Run identity does not match the registry")
    if evaluation.get("threshold_source_split") != "val":
        raise ValueError("Test thresholds are not validation-derived")
    if row["model_name"] == "P2":
        if resolved.get("phase") != "V2-4" or resolved.get("variant") != "P2":
            raise ValueError("P2 output identity mismatch")
    else:
        expected = CORE_MODELS[row["model_name"]]["model_name"]
        if resolved.get("model_name") != expected or resolved.get("profile") != "full":
            raise ValueError("V1-model output identity mismatch")


def _run_logged(command: list[str], root: Path, stream: Any, heading: str) -> int:
    stream.write(f"\n[{_now()}] {heading}\n")
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
    stream.flush()
    return completed.returncode


def _worker(
    root: Path,
    row: dict[str, str],
    protocol_sha256: str,
    attempt: int,
    log_relative: Path,
) -> dict[str, Any]:
    train_command, evaluation_command = _commands(root, row, protocol_sha256)
    log_path = root / log_relative
    log_path.parent.mkdir(parents=True, exist_ok=True)
    return_code = 0
    reason = ""
    with log_path.open("a", encoding="utf-8") as stream:
        return_code = _run_logged(train_command, root, stream, "parallel train")
        if return_code:
            reason = f"training_exit_{return_code}"
        else:
            return_code = _run_logged(
                evaluation_command, root, stream, "parallel evaluate"
            )
            if return_code:
                reason = f"evaluation_exit_{return_code}"
    if return_code == 0:
        try:
            _verify_outputs(root, row)
        except Exception as exc:
            return_code = 1
            reason = f"output_contract_error: {exc}"
    return {
        "row": row,
        "attempt": attempt,
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "reason": reason,
        "log_relative": str(log_relative),
    }


def _handoff_worker(
    root: Path,
    row: dict[str, str],
    protocol_sha256: str,
    attempt: int,
    log_relative: Path,
    poll_seconds: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Wait for the already-running serial train process, then evaluate it once."""
    run_path = root / row["run_dir"]
    manifest_path = run_path / "v2_run_manifest.json"
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if manifest_path.is_file():
            manifest = read_json(manifest_path)
            if manifest.get("status") == "trained_validation_frozen_test_not_opened":
                break
        time.sleep(poll_seconds)
    else:
        return {
            "row": row,
            "attempt": attempt,
            "return_code": 1,
            "status": "failed",
            "reason": "handoff_training_timeout",
            "log_relative": str(log_relative),
        }
    time.sleep(poll_seconds)
    _, evaluation_command = _commands(root, row, protocol_sha256)
    log_path = root / log_relative
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        return_code = _run_logged(
            evaluation_command, root, stream, "parallel handoff evaluate"
        )
    reason = f"evaluation_exit_{return_code}" if return_code else ""
    if return_code == 0:
        try:
            _verify_outputs(root, row)
        except Exception as exc:
            return_code = 1
            reason = f"output_contract_error: {exc}"
    return {
        "row": row,
        "attempt": attempt,
        "return_code": return_code,
        "status": "success" if return_code == 0 else "failed",
        "reason": reason,
        "log_relative": str(log_relative),
    }


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.project_root).resolve()
    parent, amendment = _verified_locks(root)
    config = load_config(root / "configs/v2/phase_v2_4_execution_amendment.yaml")
    workers = int(config["max_parallel_units"])
    if workers != 2 or workers != int(amendment["max_parallel_units"]):
        raise ValueError("Parallel worker count differs from the frozen amendment")
    planned = build_v2_phase4_core_plan(root, arguments.plan)
    registry = V2Phase4Registry(root)
    registry.register(planned, parent["protocol_sha256"])
    running = [row for row in registry.rows() if row["status"] == "running"]
    handoff_key = str(config["handoff_run_key"])
    if len(running) > 1 or (running and running[0]["run_key"] != handoff_key):
        raise RuntimeError(f"Unexpected running units at parallel handoff: {running}")
    current = {row["run_key"]: row for row in registry.rows()}
    rows = [current[row["run_key"]] for row in planned]
    desired = "failed" if arguments.retry_failed else "pending"
    rows = [row for row in rows if row["status"] == desired]
    if arguments.max_units is not None:
        rows = rows[: arguments.max_units]
    if arguments.plan_only:
        return {
            "phase": "V2-4",
            "parallel_workers": workers,
            "eligible_units": len(rows),
            "launched": 0,
        }

    outcomes = []
    pending = iter(rows)
    active: dict[Future[dict[str, Any]], tuple[dict[str, str], str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        if running:
            handoff = running[0]
            attempt = int(handoff["attempt_count"])
            started = handoff["updated_at"]
            log_relative = Path(handoff["error_log"])
            future = pool.submit(
                _handoff_worker,
                root,
                handoff,
                parent["protocol_sha256"],
                attempt,
                log_relative,
                int(config["handoff_poll_seconds"]),
                int(config["handoff_timeout_seconds"]),
            )
            active[future] = (handoff, started)
            print(f"[{_now()}] parallel adopt: {handoff['run_key']}", flush=True)
        while True:
            while len(active) < workers:
                try:
                    row = next(pending)
                except StopIteration:
                    break
                attempt = int(row["attempt_count"]) + 1
                log_relative = (
                    Path("runs/v2/_phase_v2_4_parallel_logs")
                    / _safe_key(row["run_key"])
                    / f"attempt-{attempt}.log"
                )
                train_command, _ = _commands(root, row, parent["protocol_sha256"])
                started = _now()
                registry.update(
                    row["run_key"],
                    status="running",
                    attempt_count=attempt,
                    command=subprocess.list2cmdline(train_command),
                    reason=f"parallel_worker_limit_{workers}",
                    error_log=str(log_relative),
                )
                future = pool.submit(
                    _worker,
                    root,
                    row,
                    parent["protocol_sha256"],
                    attempt,
                    log_relative,
                )
                active[future] = (row, started)
                print(f"[{_now()}] parallel start: {row['run_key']}", flush=True)
            if not active:
                break
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                row, started = active.pop(future)
                result = future.result()
                registry.update(
                    row["run_key"],
                    status=result["status"],
                    reason=result["reason"],
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
                outcomes.append(
                    {
                        "run_key": row["run_key"],
                        "status": result["status"],
                        "reason": result["reason"],
                    }
                )
                print(
                    f"[{_now()}] parallel finish: {row['run_key']} -> "
                    f"{result['status']}",
                    flush=True,
                )
    aggregate = None
    bootstrap = None
    if registry.profile_complete("v2_multiseed_core"):
        aggregate = aggregate_v2_phase4_core(root, registry)
        bootstrap = bootstrap_v2_phase4_natural(root, registry)
    failures = [row for row in outcomes if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} parallel V2-4 units failed")
    return {
        "phase": "V2-4",
        "parallel_workers": workers,
        "launched": len(rows),
        "outcomes": outcomes,
        "status_counts": registry.terminal_counts("v2_multiseed_core"),
        "aggregate": aggregate,
        "bootstrap": bootstrap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--plan", default="configs/v2/phase_v2_4_plan.yaml")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-units", type=int)
    result = execute(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
