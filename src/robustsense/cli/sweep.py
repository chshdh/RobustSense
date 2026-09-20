"""Registered, resumable Phase-5 experiment sweep command."""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.cli._common import run_cli
from robustsense.experiments.aggregate import aggregate_ablation_development_screen
from robustsense.experiments.preflight import build_preflight
from robustsense.experiments.registry import RunRegistry, build_run_plan, freeze_protocol
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a frozen, registered and resumable RobustSense Phase-5 sweep"
    )
    parser.add_argument("--profile", choices=("ablation_dev", "credible", "full"), required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--plan", default="configs/experiment/phase5_plan.yaml")
    parser.add_argument(
        "--prepare-folds",
        action="store_true",
        help="Prepare any missing registered fold before training",
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Freeze/register/preflight without launching a training run",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed units only; successful and pending units are untouched",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=None,
        help="Bound this invocation; remaining registered units stay pending",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Run only this registered model (repeatable; other units remain registered)",
    )
    return parser


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


def _safe_key(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in value
    )


def _run_logged(command: list[str], root: Path, log_path: Path, heading: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as stream:
        stream.write(f"\n[{_utc_now()}] {heading}\n")
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
        stream.write(f"[{_utc_now()}] return_code={completed.returncode}\n")
    return completed.returncode


def _prepare_registered_folds(
    root: Path, planned: list[dict[str, Any]], data_config: str
) -> list[int]:
    prepared = []
    folds = sorted({int(row["fold"]) for row in planned})
    for fold in folds:
        manifest = root / f"data/processed/extrasensory/fold{fold}/processed_manifest.json"
        if manifest.is_file():
            prepared.append(fold)
            continue
        log_path = root / f"runs/_phase5_logs/preparation/fold{fold}.log"
        command = [
            sys.executable,
            "-m",
            "robustsense.cli.prepare",
            "--config",
            data_config,
            "--fold",
            str(fold),
        ]
        return_code = _run_logged(command, root, log_path, f"prepare fold {fold}")
        if return_code or not manifest.is_file():
            raise RuntimeError(f"Fold {fold} preparation failed; see {log_path}")
        prepared.append(fold)
    return prepared


def _verify_run_outputs(root: Path, row: dict[str, str]) -> None:
    run_path = root / row["run_dir"]
    required = (
        "best_checkpoint.pt",
        "resolved_config.yaml",
        "thresholds.json",
        "test_metrics.json",
        "evaluation_manifest.json",
    )
    missing = [name for name in required if not (run_path / name).is_file()]
    if missing:
        raise ValueError(f"Run output contract is incomplete: {missing}")
    resolved = load_config(run_path / "resolved_config.yaml")
    manifest = read_json(run_path / "evaluation_manifest.json")
    expected = {
        "profile": row["profile"],
        "model_name": row["model_name"],
        "fold": int(row["fold"]),
        "seed": int(row["seed"]),
    }
    for name, value in expected.items():
        if resolved.get(name) != value or manifest.get(name) != value:
            raise ValueError(f"{name} mismatch in registered run outputs")


def execute_sweep(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.project_root).resolve()
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    lock = freeze_protocol(root, plan_path)
    planned = build_run_plan(root, plan_path, args.profile)
    registry = RunRegistry(root)
    recovered = registry.recover_interrupted()
    registry.register(planned, lock["protocol_sha256"])
    plan = load_config(plan_path)
    definition = plan["profiles"][args.profile]
    prerequisite = definition.get("requires_complete_profile")
    if prerequisite and not registry.profile_complete(prerequisite):
        raise ValueError(
            f"Profile {args.profile} is gated until every {prerequisite} unit succeeds"
        )
    preflight = build_preflight(root, plan_path, args.profile)
    prepared_folds = preflight["prepared_folds"]
    if args.prepare_folds:
        prepared_folds = _prepare_registered_folds(root, planned, plan["data_config"])

    if args.plan_only:
        return {
            "profile": args.profile,
            "protocol_sha256": lock["protocol_sha256"],
            "registered_run_count": len(planned),
            "recovered_interrupted_runs": recovered,
            "prepared_folds": prepared_folds,
            "preflight": preflight,
            "registry": str(registry.path),
            "launched": 0,
        }

    rows = [row for row in registry.rows() if row["profile"] == args.profile]
    if args.model:
        unknown = sorted(set(args.model) - {row["model_name"] for row in rows})
        if unknown:
            raise ValueError(f"Models are not registered for {args.profile}: {unknown}")
        rows = [row for row in rows if row["model_name"] in set(args.model)]
    desired_status = "failed" if args.retry_failed else "pending"
    rows = [row for row in rows if row["status"] == desired_status]
    if args.max_runs is not None:
        if args.max_runs < 0:
            raise ValueError("--max-runs must be non-negative")
        rows = rows[: args.max_runs]

    outcomes = []
    for row in rows:
        attempt = int(row["attempt_count"]) + 1
        log_relative = (
            Path("runs/_phase5_logs")
            / _safe_key(row["run_key"])
            / f"attempt-{attempt}.log"
        )
        log_path = root / log_relative
        train_command = [
            sys.executable,
            "-m",
            "robustsense.cli.train",
            "--project-root",
            str(root),
            "--data-config",
            plan["data_config"],
            "--model",
            row["model_name"],
            "--fold",
            row["fold"],
            "--seed",
            row["seed"],
            "--profile",
            row["profile"],
        ]
        command_text = subprocess.list2cmdline(train_command)
        started = _utc_now()
        registry.update(
            row["run_key"],
            status="running",
            attempt_count=attempt,
            command=command_text,
            reason="",
            error_log=str(log_relative),
        )
        return_code = _run_logged(train_command, root, log_path, "training")
        reason = ""
        if return_code == 0:
            evaluate_command = [
                sys.executable,
                "-m",
                "robustsense.cli.evaluate",
                "--run-dir",
                str(root / row["run_dir"]),
                "--suite",
                definition["evaluation_suite"],
            ]
            return_code = _run_logged(evaluate_command, root, log_path, "evaluation")
            if return_code:
                reason = f"evaluation_exit_{return_code}"
        else:
            reason = f"training_exit_{return_code}"
        if return_code == 0:
            try:
                _verify_run_outputs(root, row)
            except Exception as exc:
                return_code = 1
                reason = f"output_contract_error: {exc}"
        status = "success" if return_code == 0 else "failed"
        finished = _utc_now()
        registry.update(row["run_key"], status=status, reason=reason)
        registry.append_attempt(
            run_key=row["run_key"],
            attempt=attempt,
            status=status,
            started_at=started,
            finished_at=finished,
            return_code=return_code,
            log_path=str(log_relative),
            reason=reason,
        )
        outcomes.append({"run_key": row["run_key"], "status": status, "reason": reason})

    aggregate = None
    if registry.profile_complete(args.profile):
        if args.profile == "ablation_dev":
            aggregate = aggregate_ablation_development_screen(root, plan_path)
        else:
            from robustsense.evaluation.report import generate_report

            aggregate = generate_report(
                root,
                "runs",
                "reports",
                definition["evaluation_config"],
            )
    failures = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    if failures:
        raise RuntimeError(
            f"{len(failures)} registered units failed; successful units were retained and "
            "only failed units are eligible for --retry-failed"
        )
    return {
        "profile": args.profile,
        "protocol_sha256": lock["protocol_sha256"],
        "registered_run_count": len(planned),
        "recovered_interrupted_runs": recovered,
        "prepared_folds": prepared_folds,
        "launched": len(rows),
        "outcomes": outcomes,
        "status_counts": registry.terminal_counts(args.profile),
        "aggregate": aggregate,
    }


def main() -> None:
    args = build_parser().parse_args()
    run_cli(lambda: execute_sweep(args))


if __name__ == "__main__":
    main()
