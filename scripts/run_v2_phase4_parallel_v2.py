"""Resume V2-4 safely, then launch the corrected parallel scheduler."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import run_v2_phase4_parallel as parallel
from reconcile_v2_phase4_artifacts import (
    reconcile_failed,
    reconcile_running,
    verify_outputs,
)

from robustsense.experiments.v2_phase4 import V2Phase4Registry
from robustsense.utils.config import load_config


def _resume_interrupted_workers(root: Path) -> None:
    """Wait for adopted workers to finish and reconcile their complete artifacts."""
    registry = V2Phase4Registry(root)
    corrected = reconcile_failed(root, registry)
    if corrected:
        print(f"reconciled {len(corrected)} verifier false failures", flush=True)
    config = load_config(root / "configs/v2/phase_v2_4_execution_amendment.yaml")
    poll_seconds = int(config["handoff_poll_seconds"])
    deadline = time.monotonic() + int(config["handoff_timeout_seconds"])
    while any(row["status"] == "running" for row in registry.rows()):
        completed, not_ready = reconcile_running(root, registry)
        if completed:
            print(f"reconciled {len(completed)} interrupted workers", flush=True)
        if not not_ready:
            continue
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Interrupted workers did not finish: {not_ready}")
        time.sleep(poll_seconds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--project-root", default=".")
    arguments, _ = parser.parse_known_args()
    _resume_interrupted_workers(Path(arguments.project_root).resolve())
    parallel._verify_outputs = verify_outputs
    parallel.main()
