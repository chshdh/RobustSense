"""Reconcile valid V2-4 artifacts after correcting the threshold-source lookup."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from robustsense.experiments.v2_phase4 import CORE_MODELS, V2Phase4Registry
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json


def _now() -> str:
    return datetime.now(UTC).isoformat()


def verify_outputs(root: Path, row: dict[str, str]) -> None:
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
    thresholds = read_json(run_path / "thresholds.json")
    evaluation = read_json(run_path / "evaluation_manifest.json")
    if int(resolved["fold"]) != int(row["fold"]) or int(resolved["seed"]) != int(
        row["seed"]
    ):
        raise ValueError("Run identity does not match the registry")
    if thresholds.get("source_split") != "val":
        raise ValueError("Threshold artifact is not validation-derived")
    if evaluation.get("phase") != "V2-4":
        raise ValueError("Evaluation manifest does not belong to V2-4")
    if row["model_name"] == "P2":
        if resolved.get("phase") != "V2-4" or resolved.get("variant") != "P2":
            raise ValueError("P2 output identity mismatch")
        abstention = read_json(run_path / "abstention_threshold.json")
        if abstention.get("source_split") != "val":
            raise ValueError("P2 abstention threshold is not validation-derived")
    else:
        expected = CORE_MODELS[row["model_name"]]["model_name"]
        if resolved.get("model_name") != expected or resolved.get("profile") != "full":
            raise ValueError("V1-model output identity mismatch")


def reconcile_failed(root: Path, registry: V2Phase4Registry) -> list[dict[str, Any]]:
    corrected = []
    expected_reason = "output_contract_error: Test thresholds are not validation-derived"
    for row in registry.rows():
        if row["status"] != "failed" or row["reason"] != expected_reason:
            continue
        verify_outputs(root, row)
        registry.update(
            row["run_key"],
            status="success",
            reason="artifact_reconciliation_no_rerun:thresholds.source_split=val",
        )
        timestamp = _now()
        registry.append_attempt(
            run_key=row["run_key"],
            attempt=f"{row['attempt_count']}-reconcile",
            status="success",
            started_at=timestamp,
            finished_at=timestamp,
            return_code=0,
            log_path=row["error_log"],
            reason="artifact_reconciliation_no_rerun",
        )
        corrected.append({"run_key": row["run_key"], "status": "success"})
    return corrected


def reconcile_running(
    root: Path, registry: V2Phase4Registry
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    corrected = []
    not_ready = []
    for row in registry.rows():
        if row["status"] != "running":
            continue
        try:
            verify_outputs(root, row)
        except Exception as exc:
            not_ready.append({"run_key": row["run_key"], "reason": str(exc)})
            continue
        registry.update(
            row["run_key"],
            status="success",
            reason="controller_handoff_artifacts_verified",
        )
        timestamp = _now()
        registry.append_attempt(
            run_key=row["run_key"],
            attempt=f"{row['attempt_count']}-handoff",
            status="success",
            started_at=row["updated_at"],
            finished_at=timestamp,
            return_code=0,
            log_path=row["error_log"],
            reason="controller_handoff_artifacts_verified",
        )
        corrected.append({"run_key": row["run_key"], "status": "success"})
    return corrected, not_ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    registry = V2Phase4Registry(root)
    corrected = reconcile_failed(root, registry)
    running, not_ready = reconcile_running(root, registry)
    print(
        json.dumps(
            {
                "corrected_failed": corrected,
                "corrected_running": running,
                "not_ready": not_ready,
                "corrected_count": len(corrected) + len(running),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
