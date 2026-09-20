"""Resource and wall-time preflight based on measured Phase-2/3 dev epochs."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

from robustsense.experiments.registry import build_run_plan
from robustsense.utils.config import load_config
from robustsense.utils.io import write_json


def _training_observation(
    root: Path, model_name: str, profile_name: str
) -> tuple[float | None, float | None, str | None]:
    source_model = model_name
    if model_name.startswith("ablation-"):
        source_model = "quality-aware"
    candidates = sorted(
        root.glob(
            f"runs/extrasensory-{source_model}-fold*-seed*-{profile_name}/train_log.csv"
        )
    )
    source = profile_name
    if not candidates:
        candidates = [
            root / f"runs/extrasensory-{source_model}-fold0-seed13-dev/train_log.csv"
        ]
        source = "dev"
    epoch_seconds = []
    completed_epochs = []
    for path in candidates:
        if not path.is_file():
            continue
        with path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        epoch_seconds.extend(float(row["epoch_seconds"]) for row in rows)
        completed_epochs.append(len(rows))
    if not epoch_seconds:
        return None, None, None
    return (
        sum(epoch_seconds) / len(epoch_seconds),
        sum(completed_epochs) / len(completed_epochs),
        source,
    )


def build_preflight(
    project_root: Path, plan_path: Path, profile_name: str
) -> dict[str, Any]:
    root = project_root.resolve()
    plan_file = plan_path if plan_path.is_absolute() else root / plan_path
    plan = load_config(plan_file)
    definition = plan["profiles"][profile_name]
    experiment = load_config(root / definition["experiment_config"])
    planned = build_run_plan(root, plan_file, profile_name)
    raw_observations = {
        model: _training_observation(root, model, profile_name)
        for model in dict.fromkeys(row["model_name"] for row in planned)
    }
    missing = sorted(
        model for model, (seconds, _, _) in raw_observations.items() if seconds is None
    )
    fold_seed_count = len(experiment["folds"]) * len(experiment["seeds"])
    epochs = int(experiment["epochs"])
    patience = int(experiment["early_stopping_patience"])
    minimum_epochs = min(epochs, patience + 1)
    paired_ratios = []
    if profile_name != "dev":
        for model, (seconds, _, source) in raw_observations.items():
            if source != profile_name or seconds is None:
                continue
            dev_seconds, _, _ = _training_observation(root, model, "dev")
            if dev_seconds:
                paired_ratios.append(seconds / dev_seconds)
    batch_adjustment = (
        sorted(paired_ratios)[len(paired_ratios) // 2]
        if paired_ratios
        else max(1.0, (1024.0 / float(experiment["batch_size"])) ** 0.5)
    )
    observed = {}
    projected_lower_seconds = 0.0
    projected_upper_seconds = 0.0
    for model, (seconds, completed_epochs, source) in raw_observations.items():
        adjusted_seconds = seconds
        if seconds is not None and source == "dev" and profile_name != "dev":
            adjusted_seconds *= batch_adjustment
        observed[model] = {
            "mean_epoch_seconds": seconds,
            "completed_epochs_mean": completed_epochs,
            "source_profile": source,
            "projected_epoch_seconds": adjusted_seconds,
        }
        if adjusted_seconds is not None:
            projected_epochs = (
                completed_epochs
                if source == profile_name and completed_epochs is not None
                else minimum_epochs
            )
            projected_lower_seconds += adjusted_seconds * projected_epochs
            projected_upper_seconds += adjusted_seconds * epochs * 1.15
    lower_hours = projected_lower_seconds * fold_seed_count / 3600.0
    upper_hours = projected_upper_seconds * fold_seed_count / 3600.0
    disk = shutil.disk_usage(root)
    payload = {
        "profile": profile_name,
        "planned_run_count": len(planned),
        "folds": experiment["folds"],
        "seeds": experiment["seeds"],
        "epochs_max": epochs,
        "early_stopping_patience": patience,
        "batch_size": experiment["batch_size"],
        "training_observations": observed,
        "fallback_batch_adjustment": batch_adjustment,
        "missing_timing_sources": missing,
        "estimated_training_hours": {
            "early_stop_floor": lower_hours if not missing else None,
            "max_epoch_upper": upper_hours if not missing else None,
            "basis": (
                "completed same-profile epoch counts are projected when available; other "
                "models use the early-stop floor and a measured batch-size adjustment; "
                "the upper bound uses all epochs plus 15% fold variation; evaluation time "
                "is additional"
            ),
        },
        "free_disk_gib": disk.free / (1024**3),
        "prepared_folds": [
            fold
            for fold in experiment["folds"]
            if (root / f"data/processed/extrasensory/fold{fold}/processed_manifest.json").is_file()
        ],
        "ready_to_launch": not missing,
    }
    output = root / f"reports/phase5_preflight_{profile_name}.json"
    write_json(output, payload)
    return payload
