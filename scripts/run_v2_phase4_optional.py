"""Run the validation-selected, non-core V2-4 ablation extension."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.experiments.v2_phase4 import (
    V2Phase4OptionalRegistry,
    build_v2_phase4_optional_plan,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def execute(arguments: argparse.Namespace) -> dict:
    root = Path(arguments.project_root).resolve()
    lock = read_json(root / "reports/v2/phase_v2_4_protocol_lock.json")
    problems = verify_file_entries(root, lock["files"])
    if lock.get("phase") != "V2-4" or problems:
        raise ValueError(f"V2-4 protocol lock is invalid: {problems}")
    planned = build_v2_phase4_optional_plan(root, arguments.plan)
    registry = V2Phase4OptionalRegistry(root)
    recovered = registry.recover_interrupted()
    registry.register(planned, lock["protocol_sha256"])
    if arguments.plan_only:
        return {"registered": len(planned), "recovered": recovered, "launched": 0}
    desired = "failed" if arguments.retry_failed else "pending"
    current = {row["run_key"]: row for row in registry.rows()}
    rows = [current[row["run_key"]] for row in planned]
    rows = [row for row in rows if row["status"] == desired]
    if arguments.max_units is not None:
        rows = rows[: arguments.max_units]
    outcomes = []
    for index, row in enumerate(rows, start=1):
        attempt = int(row["attempt_count"]) + 1
        log_relative = (
            Path("runs/v2/_phase_v2_4_optional_logs")
            / _safe(row["run_key"])
            / f"attempt-{attempt}.log"
        )
        base = [
            sys.executable,
            str(root / "scripts/run_v2_phase4_unit.py"),
            "--project-root",
            str(root),
            "--variant",
            row["model_name"],
            "--fold",
            row["fold"],
            "--seed",
            row["seed"],
            "--protocol-sha256",
            lock["protocol_sha256"],
        ]
        started = _now()
        registry.update(
            row["run_key"],
            status="running",
            attempt_count=attempt,
            command=subprocess.list2cmdline([*base[:2], "train", *base[2:]]),
            reason="",
            error_log=str(log_relative),
        )
        (root / log_relative).parent.mkdir(parents=True, exist_ok=True)
        return_code = 0
        with (root / log_relative).open("a", encoding="utf-8") as stream:
            for mode in ("train", "evaluate"):
                command = [*base[:2], mode, *base[2:]]
                stream.write(f"\n[{_now()}] {mode}\n")
                stream.flush()
                completed = subprocess.run(
                    command,
                    cwd=root,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
                return_code = completed.returncode
                if return_code:
                    break
        reason = f"subprocess_exit_{return_code}" if return_code else ""
        if return_code == 0:
            run = root / row["run_dir"]
            required = (
                "best_checkpoint.pt",
                "thresholds.json",
                "test_metrics.json",
                "predictions.parquet",
                "evaluation_manifest.json",
            )
            missing = [name for name in required if not (run / name).is_file()]
            resolved = load_config(run / "resolved_config.yaml")
            if missing or resolved.get("variant") != row["model_name"]:
                return_code = 1
                reason = f"output_contract_error:{missing}"
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
        outcomes.append({"run_key": row["run_key"], "status": status})
        print(f"[{_now()}] optional {index}/{len(rows)} -> {status}", flush=True)
    failures = [row for row in outcomes if row["status"] == "failed"]
    if failures:
        raise RuntimeError(f"{len(failures)} optional ablation units failed")
    return {
        "registered": len(planned),
        "recovered": recovered,
        "launched": len(rows),
        "status_counts": registry.terminal_counts("v2_multiseed_optional"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--plan", default="configs/v2/phase_v2_4_plan.yaml")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--max-units", type=int)
    print(json.dumps(execute(parser.parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
