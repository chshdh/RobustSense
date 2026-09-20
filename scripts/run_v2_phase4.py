"""Run the frozen, resumable 60-unit V2-4 multiseed core matrix."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
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


def _run_logged(command: list[str], root: Path, log_path: Path, heading: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
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
    return completed.returncode


def _verified_lock(root: Path) -> dict[str, Any]:
    lock = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    if lock.get("phase") != "V2-4" or lock.get("status") != "frozen_before_first_test_run":
        raise ValueError("V2-4 protocol lock is absent or not frozen")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise ValueError(f"V2-4 protocol lock is invalid: {problems}")
    return lock


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


def _verify_outputs(root: Path, row: dict[str, str]) -> None:
    run_path = root / row["run_dir"]
    common = (
        "best_checkpoint.pt",
        "resolved_config.yaml",
        "thresholds.json",
        "val_metrics.json",
        "test_metrics.json",
        "predictions.parquet",
        "evaluation_manifest.json",
    )
    missing = [name for name in common if not (run_path / name).is_file()]
    if row["model_name"] == "P2":
        missing.extend(
            name
            for name in ("abstention_threshold.json", "risk_coverage.json", "run_manifest.json")
            if not (run_path / name).is_file()
        )
    else:
        missing.extend(
            name
            for name in ("v2_run_manifest.json", "modality_diagnostics.parquet")
            if not (run_path / name).is_file()
        )
    if missing:
        raise ValueError(f"V2-4 output contract is incomplete: {sorted(set(missing))}")
    resolved = load_config(run_path / "resolved_config.yaml")
    evaluation = read_json(run_path / "evaluation_manifest.json")
    if int(resolved["fold"]) != int(row["fold"]) or int(resolved["seed"]) != int(
        row["seed"]
    ):
        raise ValueError("V2-4 run identity does not match the registry")
    if evaluation.get("threshold_source_split") != "val":
        raise ValueError("V2-4 test thresholds must be validation-derived")
    if row["model_name"] == "P2":
        if resolved.get("phase") != "V2-4" or resolved.get("variant") != "P2":
            raise ValueError("V2-4 P2 output identity mismatch")
    else:
        expected = CORE_MODELS[row["model_name"]]["model_name"]
        if resolved.get("model_name") != expected or resolved.get("profile") != "full":
            raise ValueError("V2-4 V1-model output identity mismatch")


def execute(arguments: argparse.Namespace) -> dict[str, Any]:
    root = Path(arguments.project_root).resolve()
    lock = _verified_lock(root)
    planned = build_v2_phase4_core_plan(root, arguments.plan)
    registry = V2Phase4Registry(root)
    recovered = registry.recover_interrupted()
    registry.register(planned, lock["protocol_sha256"])
    if arguments.plan_only:
        return {
            "phase": "V2-4",
            "protocol_sha256": lock["protocol_sha256"],
            "registered_core_unit_count": len(planned),
            "reused_success_count": sum(row["status"] == "success" for row in planned),
            "recovered_interrupted_units": recovered,
            "registry": str(registry.path),
            "launched": 0,
        }
    desired_status = "failed" if arguments.retry_failed else "pending"
    by_key = {row["run_key"]: row for row in registry.rows()}
    rows = [by_key[row["run_key"]] for row in planned]
    rows = [row for row in rows if row["status"] == desired_status]
    if arguments.model:
        rows = [row for row in rows if row["model_name"] in set(arguments.model)]
    if arguments.fold:
        rows = [row for row in rows if int(row["fold"]) in set(arguments.fold)]
    if arguments.seed:
        rows = [row for row in rows if int(row["seed"]) in set(arguments.seed)]
    if arguments.max_units is not None:
        if arguments.max_units < 0:
            raise ValueError("--max-units must be non-negative")
        rows = rows[: arguments.max_units]
    outcomes = []
    for index, row in enumerate(rows, start=1):
        print(
            f"[{_now()}] V2-4 core unit {index}/{len(rows)} starting: {row['run_key']}",
            flush=True,
        )
        attempt = int(row["attempt_count"]) + 1
        log_relative = (
            Path("runs/v2/_phase_v2_4_logs")
            / _safe_key(row["run_key"])
            / f"attempt-{attempt}.log"
        )
        log_path = root / log_relative
        model_id = row["model_name"]
        if model_id == "P2":
            train_command = _p2_command(root, row, lock["protocol_sha256"], "train")
        else:
            train_command = _v1_command(
                root,
                row,
                lock["protocol_sha256"],
                str(CORE_MODELS[model_id]["model_name"]),
                "train",
            )
        started = _now()
        registry.update(
            row["run_key"],
            status="running",
            attempt_count=attempt,
            command=subprocess.list2cmdline(train_command),
            reason="",
            error_log=str(log_relative),
        )
        return_code = _run_logged(train_command, root, log_path, "train")
        reason = f"training_exit_{return_code}" if return_code else ""
        if return_code == 0:
            if model_id == "P2":
                evaluation_command = _p2_command(
                    root, row, lock["protocol_sha256"], "evaluate"
                )
            else:
                evaluation_command = _v1_command(
                    root,
                    row,
                    lock["protocol_sha256"],
                    str(CORE_MODELS[model_id]["model_name"]),
                    "evaluate",
                )
            return_code = _run_logged(evaluation_command, root, log_path, "evaluate")
            if return_code:
                reason = f"evaluation_exit_{return_code}"
        if return_code == 0:
            try:
                _verify_outputs(root, row)
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
        print(f"[{_now()}] V2-4 core unit finished: {row['run_key']} -> {status}", flush=True)
    aggregate = None
    bootstrap = None
    if registry.profile_complete("v2_multiseed_core"):
        aggregate = aggregate_v2_phase4_core(root, registry)
        bootstrap = bootstrap_v2_phase4_natural(root, registry)
    failures = [row for row in outcomes if row["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} V2-4 units failed; logs and successful units were retained"
        )
    return {
        "phase": "V2-4",
        "protocol_sha256": lock["protocol_sha256"],
        "registered_core_unit_count": len(planned),
        "recovered_interrupted_units": recovered,
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
    parser.add_argument("--max-units", type=int, default=None)
    parser.add_argument("--model", action="append", choices=tuple(CORE_MODELS), default=None)
    parser.add_argument("--fold", action="append", type=int, default=None)
    parser.add_argument("--seed", action="append", type=int, default=None)
    result = execute(parser.parse_args())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
