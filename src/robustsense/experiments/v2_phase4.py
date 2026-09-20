"""V2-4 reuse contracts, 60-unit core plan, registry, and natural aggregation."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.data.extrasensory import sha256_file
from robustsense.evaluation.v2_aggregate import validate_shared_scenario_samples
from robustsense.evaluation.v2_artifacts import write_stable_csv
from robustsense.experiments.registry import RUN_FIELDS, RunRegistry
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.training.v2_phase4_trainer import v2_phase4_run_id
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

CORE_MODELS = {
    "B4": {"engine": "v1", "model_name": "gated", "config": "gated.yaml"},
    "B5": {
        "engine": "v1",
        "model_name": "robust-gated",
        "config": "robust_gated.yaml",
    },
    "P": {
        "engine": "v1",
        "model_name": "quality-aware",
        "config": "quality_aware.yaml",
    },
    "P2": {"engine": "v2", "variant": "P2"},
}

TRAINING_KEYS = (
    "epochs",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "gradient_clip_norm",
    "pos_weight_cap",
    "early_stopping_patience",
    "threshold_grid",
)


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _training_contract(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: profile[key] for key in TRAINING_KEYS}


def _verified_lock(root: Path, relative: str, expected_phase: str | None = None) -> dict:
    lock = read_json(root / relative)
    if expected_phase is not None and lock.get("phase") != expected_phase:
        raise ValueError(f"Unexpected parent lock phase: {relative}")
    if all("bytes" in entry for entry in lock["files"]):
        problems = verify_file_entries(root, lock["files"])
    else:
        problems = []
        for entry in lock["files"]:
            path = root / entry["path"]
            if not path.is_file():
                problems.append(f"missing:{entry['path']}")
            elif sha256_file(path) != entry["sha256"]:
                problems.append(f"sha256:{entry['path']}")
    if problems:
        raise ValueError(f"Parent protocol lock is invalid: {relative}: {problems}")
    return lock


def build_v2_phase4_reuse_manifest(
    project_root: str | Path,
    plan_path: str | Path = "configs/v2/phase_v2_4_plan.yaml",
) -> dict[str, Any]:
    """Verify the 20 seed-13 sources without reading their performance values."""
    root = Path(project_root).resolve()
    plan_file = Path(plan_path)
    if not plan_file.is_absolute():
        plan_file = root / plan_file
    plan = load_config(plan_file)
    v1_lock = _verified_lock(root, plan["v1_protocol_lock"])
    v2_lock = _verified_lock(root, plan["v2_3_protocol_lock"], "V2-3")
    profile = load_config(root / plan["experiment_config"])
    v1_full = load_config(root / plan["v1_experiment_config"])
    expected_training = _training_contract(profile)
    if _training_contract(v1_full) != expected_training:
        raise ValueError("V1 full and V2-4 training hyperparameters do not match")
    p2_model = load_config(root / plan["model_config"])
    p2_corruption = load_config(root / plan["corruption_config"])
    entries: list[dict[str, Any]] = []
    for fold in plan["folds"]:
        processed = root / "data/processed/extrasensory" / f"fold{fold}"
        expected_processed = sha256_file(processed / "processed_manifest.json")
        expected_preprocessor = sha256_file(processed / "preprocessor.json")
        for model_id, definition in CORE_MODELS.items():
            if model_id == "P2":
                run_id = f"extrasensory-p2-fold{fold}-seed13-v2_seed13"
                run_dir = root / "runs/v2/phase_v2_3" / run_id
            else:
                model_name = definition["model_name"]
                run_id = f"extrasensory-{model_name}-fold{fold}-seed13-credible"
                run_dir = root / "runs" / run_id
            required = (
                "best_checkpoint.pt",
                "resolved_config.yaml",
                "thresholds.json",
                "val_metrics.json",
                "test_metrics.json",
                "predictions.parquet",
                "evaluation_manifest.json",
            )
            missing = [name for name in required if not (run_dir / name).is_file()]
            if missing:
                raise FileNotFoundError(f"Reuse source is incomplete: {run_id}: {missing}")
            resolved = load_config(run_dir / "resolved_config.yaml")
            evaluation = read_json(run_dir / "evaluation_manifest.json")
            thresholds = read_json(run_dir / "thresholds.json")
            contract = resolved["checkpoint_contract"]
            if int(resolved["fold"]) != int(fold) or int(resolved["seed"]) != 13:
                raise ValueError(f"Reuse source identity mismatch: {run_id}")
            if thresholds.get("source_split") != "val":
                raise ValueError(f"Reuse thresholds are not validation-derived: {run_id}")
            if contract["processed_manifest_sha256"] != expected_processed:
                raise ValueError(f"Processed manifest mismatch: {run_id}")
            if contract["preprocessor_sha256"] != expected_preprocessor:
                raise ValueError(f"Preprocessor mismatch: {run_id}")
            if _training_contract(resolved["experiment"]) != expected_training:
                raise ValueError(f"Training hyperparameter mismatch: {run_id}")
            if model_id == "P2":
                if resolved.get("variant") != "P2" or resolved.get("phase") != "V2-3":
                    raise ValueError(f"P2 reuse identity mismatch: {run_id}")
                if resolved["model_config"] != p2_model:
                    raise ValueError(f"P2 model configuration mismatch: {run_id}")
                if resolved["corruption_config"] != p2_corruption:
                    raise ValueError(f"P2 corruption configuration mismatch: {run_id}")
                if resolved.get("abstention_source_split") != "val":
                    raise ValueError(f"P2 abstention source mismatch: {run_id}")
            else:
                expected_model = definition["model_name"]
                expected_config = load_config(
                    root / "configs/model" / str(definition["config"])
                )
                if resolved.get("model_name") != expected_model:
                    raise ValueError(f"V1 model identity mismatch: {run_id}")
                if resolved["model_config"] != expected_config:
                    raise ValueError(f"V1 model configuration mismatch: {run_id}")
                if evaluation.get("threshold_source_split") != "val":
                    raise ValueError(f"V1 evaluation threshold source mismatch: {run_id}")
            entries.append(
                {
                    "run_key": f"V2-4|{model_id}|fold{fold}|seed13",
                    "model_id": model_id,
                    "fold": int(fold),
                    "seed": 13,
                    "source_run_id": run_id,
                    "source_run_dir": str(run_dir.relative_to(root)).replace("\\", "/"),
                    "checkpoint_sha256": sha256_file(run_dir / "best_checkpoint.pt"),
                    "predictions_sha256": sha256_file(run_dir / "predictions.parquet"),
                    "thresholds_sha256": sha256_file(run_dir / "thresholds.json"),
                    "processed_manifest_sha256": expected_processed,
                    "preprocessor_sha256": expected_preprocessor,
                    "training_contract_sha256": _canonical_sha256(expected_training),
                }
            )
    if len(entries) != int(plan["expected_reused_unit_count"]):
        raise AssertionError("V2-4 reuse manifest must contain exactly 20 entries")
    return {
        "version": 1,
        "phase": "V2-4",
        "status": "verified_before_first_v2_4_test_run",
        "metric_values_read": False,
        "v1_protocol_sha256": v1_lock["protocol_sha256"],
        "v2_3_protocol_sha256": v2_lock["protocol_sha256"],
        "training_contract_sha256": _canonical_sha256(expected_training),
        "entry_count": len(entries),
        "entries": sorted(entries, key=lambda row: row["run_key"]),
    }


class V2Phase4Registry(RunRegistry):
    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.path = self.root / "reports/v2/phase_v2_4_run_registry.csv"
        self.attempt_path = self.root / "reports/v2/phase_v2_4_run_attempts.csv"


class V2Phase4ExtendedRegistry(RunRegistry):
    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.path = self.root / "reports/v2/phase_v2_4_extended_registry.csv"
        self.attempt_path = self.root / "reports/v2/phase_v2_4_extended_attempts.csv"


class V2Phase4OptionalRegistry(RunRegistry):
    def __init__(self, project_root: Path):
        super().__init__(project_root)
        self.path = self.root / "reports/v2/phase_v2_4_optional_registry.csv"
        self.attempt_path = self.root / "reports/v2/phase_v2_4_optional_attempts.csv"


def build_v2_phase4_core_plan(
    project_root: str | Path,
    plan_path: str | Path = "configs/v2/phase_v2_4_plan.yaml",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    plan = load_config(path)
    reuse = read_json(root / plan["reuse_manifest"])
    if reuse.get("status") != "verified_before_first_v2_4_test_run":
        raise ValueError("V2-4 reuse manifest is not verified")
    reused = {row["run_key"]: row for row in reuse["entries"]}
    created = _now()
    rows = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            for model_id, definition in CORE_MODELS.items():
                run_key = f"V2-4|{model_id}|fold{fold}|seed{seed}"
                source = reused.get(run_key)
                if seed in plan["reuse_seeds"] and source is None:
                    raise ValueError(f"Missing frozen reuse source: {run_key}")
                if source is not None:
                    run_dir = source["source_run_dir"]
                    status = "success"
                    reason = f"reused_contract_verified:{source['source_run_id']}"
                elif definition["engine"] == "v1":
                    model_name = definition["model_name"]
                    run_dir = f"runs/extrasensory-{model_name}-fold{fold}-seed{seed}-full"
                    status = "pending"
                    reason = "registered_before_execution"
                else:
                    run_id = v2_phase4_run_id("P2", int(fold), int(seed))
                    run_dir = f"{plan['run_root']}/{run_id}"
                    status = "pending"
                    reason = "registered_before_execution"
                raw = {
                    "run_key": run_key,
                    "group": "V2-4",
                    "profile": "v2_multiseed_core",
                    "model_name": model_id,
                    "fold": str(int(fold)),
                    "seed": str(int(seed)),
                    "status": status,
                    "attempt_count": "0",
                    "created_at": created,
                    "updated_at": created,
                    "run_dir": run_dir,
                    "protocol_sha256": "",
                    "command": "",
                    "reason": reason,
                    "error_log": "",
                }
                rows.append({name: raw.get(name, "") for name in RUN_FIELDS})
    if len(rows) != int(plan["expected_core_unit_count"]):
        raise AssertionError("V2-4 core plan must contain exactly 60 units")
    if sum(row["status"] == "success" for row in rows) != int(
        plan["expected_reused_unit_count"]
    ):
        raise AssertionError("V2-4 core plan must reuse exactly 20 units")
    return rows


def _natural_metrics(root: Path, row: dict[str, str]) -> dict[str, float]:
    metrics = read_json(root / row["run_dir"] / "test_metrics.json")
    if row["model_name"] == "P2":
        values = metrics["forced"]
    else:
        values = metrics
    return {
        "macro_f1": float(values["macro_f1"]),
        "micro_f1": float(values["micro_f1"]),
        "mean_average_precision": float(values["mean_average_precision"]),
        "brier_score": float(values["brier_score"]),
    }


def aggregate_v2_phase4_core(
    project_root: str | Path, registry: V2Phase4Registry
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows = [row for row in registry.rows() if row["profile"] == "v2_multiseed_core"]
    if len(rows) != 60 or any(row["status"] != "success" for row in rows):
        raise ValueError("V2-4 aggregation requires 60 successful core units")
    results = []
    for row in rows:
        metrics = _natural_metrics(root, row)
        results.append(
            {
                "run_key": row["run_key"],
                "run_dir": row["run_dir"],
                "model_id": row["model_name"],
                "fold": int(row["fold"]),
                "seed": int(row["seed"]),
                **metrics,
            }
        )
    results.sort(key=lambda item: (item["model_id"], item["fold"], item["seed"]))
    result_path = root / "reports/v2/phase_v2_4_core_results.csv"
    write_stable_csv(result_path, results, list(results[0]))
    fold_rows = []
    for model_id in CORE_MODELS:
        for fold in range(5):
            group = [
                row for row in results if row["model_id"] == model_id and row["fold"] == fold
            ]
            if len(group) != 3:
                raise ValueError("Every V2-4 model/fold group must contain three seeds")
            values = np.asarray([row["macro_f1"] for row in group], dtype=np.float64)
            fold_rows.append(
                {
                    "model_id": model_id,
                    "fold": fold,
                    "seed_count": len(values),
                    "macro_f1_seed_mean": float(values.mean()),
                    "macro_f1_seed_std": float(values.std(ddof=1)),
                }
            )
    fold_path = root / "reports/v2/phase_v2_4_fold_seed_aggregation.csv"
    write_stable_csv(fold_path, fold_rows, list(fold_rows[0]))
    summary = []
    for model_id in CORE_MODELS:
        values = np.asarray(
            [row["macro_f1_seed_mean"] for row in fold_rows if row["model_id"] == model_id],
            dtype=np.float64,
        )
        summary.append(
            {
                "model_id": model_id,
                "fold_count": len(values),
                "seed_count_per_fold": 3,
                "natural_macro_f1_mean": float(values.mean()),
                "natural_macro_f1_fold_std": float(values.std(ddof=1)),
            }
        )
    summary_path = root / "reports/v2/phase_v2_4_core_summary.csv"
    write_stable_csv(summary_path, summary, list(summary[0]))
    report = {
        "phase": "V2-4",
        "status": "core_natural_matrix_complete",
        "successful_core_unit_count": 60,
        "aggregation_order": "seeds_within_fold_then_folds",
        "results": {"path": str(result_path), "sha256": sha256_file(result_path)},
        "fold_seed_aggregation": {
            "path": str(fold_path),
            "sha256": sha256_file(fold_path),
        },
        "summary": {"path": str(summary_path), "sha256": sha256_file(summary_path)},
    }
    write_json(root / "reports/v2/phase_v2_4_core_report.json", report)
    return report


def _paired_user_confusion_counts(
    p2_path: Path,
    p_path: Path,
    labels: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    columns = (
        "user_id",
        "timestamp",
        "scenario",
        "label",
        "target",
        "target_known",
        "prediction",
    )
    p2 = pd.read_parquet(p2_path, columns=list(columns))
    baseline = pd.read_parquet(p_path, columns=list(columns))
    p2 = p2[p2["scenario"] == "natural_missingness"].drop(columns="scenario")
    baseline = baseline[baseline["scenario"] == "natural_missingness"].drop(
        columns="scenario"
    )
    keys = ["user_id", "timestamp", "label"]
    p2 = p2.sort_values(keys, kind="stable").reset_index(drop=True)
    baseline = baseline.sort_values(keys, kind="stable").reset_index(drop=True)
    if len(p2) != len(baseline) or not p2[keys].equals(baseline[keys]):
        raise ValueError("P2 and P natural predictions do not share ordered samples")
    for column in ("target", "target_known"):
        first = p2[column].to_numpy()
        second = baseline[column].to_numpy()
        if not np.array_equal(first, second, equal_nan=True):
            raise ValueError(f"P2 and P prediction targets differ: {column}")
    users = np.asarray(sorted(set(p2["user_id"].astype(str).tolist())))
    user_lookup = {user: index for index, user in enumerate(users)}
    label_lookup = {label: index for index, label in enumerate(labels)}
    if set(p2["label"].astype(str)) != set(labels):
        raise ValueError("Prediction labels do not match the frozen label order")
    user_index = np.asarray(
        [user_lookup[user] for user in p2["user_id"].astype(str)], dtype=np.int64
    )
    label_index = np.asarray(
        [label_lookup[label] for label in p2["label"].astype(str)], dtype=np.int64
    )
    known = p2["target_known"].to_numpy(dtype=bool)
    truth = np.nan_to_num(p2["target"].to_numpy(), nan=0.0).astype(bool)

    def counts(frame: pd.DataFrame) -> np.ndarray:
        prediction = frame["prediction"].to_numpy(dtype=bool)
        result = np.zeros((len(users), len(labels), 3), dtype=np.int64)
        np.add.at(result[:, :, 0], (user_index, label_index), known & prediction & truth)
        np.add.at(result[:, :, 1], (user_index, label_index), known & prediction & ~truth)
        np.add.at(result[:, :, 2], (user_index, label_index), known & ~prediction & truth)
        return result

    return users, counts(p2), counts(baseline)


def _macro_f1_from_counts(counts: np.ndarray) -> float:
    totals = counts.sum(axis=0)
    denominator = 2 * totals[:, 0] + totals[:, 1] + totals[:, 2]
    valid = denominator > 0
    if not valid.any():
        return np.nan
    return float(np.mean(2 * totals[valid, 0] / denominator[valid]))


def bootstrap_v2_phase4_natural(
    project_root: str | Path,
    registry: V2Phase4Registry,
    config_path: str | Path = "configs/v2/evaluation/extended_core.yaml",
) -> dict[str, Any]:
    """Paired, fold-stratified user bootstrap for the natural P2-minus-P endpoint."""
    root = Path(project_root).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    rows = [row for row in registry.rows() if row["profile"] == "v2_multiseed_core"]
    if len(rows) != 60 or any(row["status"] != "success" for row in rows):
        raise ValueError("Natural Bootstrap requires 60 successful V2-4 core units")
    by_identity = {
        (row["model_name"], int(row["fold"]), int(row["seed"])): row for row in rows
    }
    labels = list(
        read_json(root / "data/processed/extrasensory/fold0/processed_manifest.json")[
            "labels"
        ]
    )
    counts: dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    source_runs = []
    for fold in config["folds"]:
        reference_users: np.ndarray | None = None
        for seed in config["seeds"]:
            p2_row = by_identity[("P2", int(fold), int(seed))]
            p_row = by_identity[("P", int(fold), int(seed))]
            users, p2_counts, p_counts = _paired_user_confusion_counts(
                root / p2_row["run_dir"] / "predictions.parquet",
                root / p_row["run_dir"] / "predictions.parquet",
                labels,
            )
            if reference_users is None:
                reference_users = users
            elif not np.array_equal(reference_users, users):
                raise ValueError("Seeds within a fold do not share the same test users")
            counts[(int(fold), int(seed))] = (users, p2_counts, p_counts)
            source_runs.extend([p2_row["run_key"], p_row["run_key"]])
    repeats = int(config["bootstrap_repeats"])
    random_seed = int(config["bootstrap_seed"])
    generator = np.random.default_rng(random_seed)
    index_hasher = hashlib.sha256()
    draw_rows = []
    for repeat in range(repeats):
        fold_p2 = []
        fold_p = []
        fold_delta = []
        for fold in config["folds"]:
            users = counts[(int(fold), int(config["seeds"][0]))][0]
            selected = generator.integers(0, len(users), size=len(users))
            index_hasher.update(
                f"repeat={repeat}|fold={fold}|".encode()
                + ",".join(map(str, selected.tolist())).encode()
                + b"\n"
            )
            seed_p2 = []
            seed_p = []
            for seed in config["seeds"]:
                _, p2_counts, p_counts = counts[(int(fold), int(seed))]
                seed_p2.append(_macro_f1_from_counts(p2_counts[selected]))
                seed_p.append(_macro_f1_from_counts(p_counts[selected]))
            p2_value = float(np.nanmean(seed_p2))
            p_value = float(np.nanmean(seed_p))
            fold_p2.append(p2_value)
            fold_p.append(p_value)
            fold_delta.append(p2_value - p_value)
        draw_rows.append(
            {
                "repeat": repeat,
                "p2_macro_f1_seed_then_fold_mean": float(np.nanmean(fold_p2)),
                "p_macro_f1_seed_then_fold_mean": float(np.nanmean(fold_p)),
                "delta_p2_minus_p": float(np.nanmean(fold_delta)),
            }
        )
    values = np.asarray([row["delta_p2_minus_p"] for row in draw_rows])
    finite = values[np.isfinite(values)]
    draws_path = root / "reports/v2/phase_v2_4_natural_bootstrap_draws.csv"
    draw_artifact = write_stable_csv(draws_path, draw_rows, list(draw_rows[0]))
    summary = {
        "phase": "V2-4",
        "comparison": "P2_minus_P",
        "scenario": "natural_missingness",
        "sampling_unit": "user_stratified_by_outer_fold",
        "paired_dimensions": ["fold", "seed", "user_id", "timestamp", "label"],
        "aggregation_order": "seeds_within_fold_then_folds",
        "bootstrap_seed": random_seed,
        "repeats": repeats,
        "valid_repeat_count": len(finite),
        "sample_index_sha256": index_hasher.hexdigest(),
        "mean_delta_p2_minus_p": float(finite.mean()),
        "ci_2_5": float(np.quantile(finite, 0.025)),
        "ci_97_5": float(np.quantile(finite, 0.975)),
        "source_run_keys": sorted(set(source_runs)),
        "draws": draw_artifact,
    }
    write_json(root / "reports/v2/phase_v2_4_natural_bootstrap.json", summary)
    return summary


def build_v2_phase4_extended_plan(
    _project_root: str | Path,
    core_registry: V2Phase4Registry,
) -> list[dict[str, Any]]:
    core_rows = [
        row for row in core_registry.rows() if row["profile"] == "v2_multiseed_core"
    ]
    if len(core_rows) != 60 or any(row["status"] != "success" for row in core_rows):
        raise ValueError("Extended evaluation requires 60 successful core units")
    created = _now()
    rows = []
    for core in core_rows:
        run_key = (
            f"V2-4-EXT|{core['model_name']}|fold{core['fold']}|seed{core['seed']}"
        )
        raw = {
            "run_key": run_key,
            "group": "V2-4-extended",
            "profile": "v2_multiseed_extended",
            "model_name": core["model_name"],
            "fold": core["fold"],
            "seed": core["seed"],
            "status": "pending",
            "attempt_count": "0",
            "created_at": created,
            "updated_at": created,
            "run_dir": core["run_dir"],
            "protocol_sha256": "",
            "command": "",
            "reason": "registered_after_core_success",
            "error_log": "",
        }
        rows.append({name: raw.get(name, "") for name in RUN_FIELDS})
    if len(rows) != 60 or len({row["run_key"] for row in rows}) != 60:
        raise AssertionError("V2-4 extended plan must contain 60 unique units")
    return rows


def build_v2_phase4_optional_plan(
    project_root: str | Path,
    plan_path: str | Path = "configs/v2/phase_v2_4_plan.yaml",
) -> list[dict[str, Any]]:
    root = Path(project_root).resolve()
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    plan = load_config(path)
    created = _now()
    rows = []
    for variant in plan["optional_ablation_variants"]:
        for fold in plan["folds"]:
            for seed in plan["optional_ablation_seeds"]:
                run_id = v2_phase4_run_id(variant, int(fold), int(seed))
                raw = {
                    "run_key": f"V2-4-OPT|{variant}|fold{fold}|seed{seed}",
                    "group": "V2-4-optional",
                    "profile": "v2_multiseed_optional",
                    "model_name": variant,
                    "fold": str(int(fold)),
                    "seed": str(int(seed)),
                    "status": "pending",
                    "attempt_count": "0",
                    "created_at": created,
                    "updated_at": created,
                    "run_dir": f"{plan['run_root']}/{run_id}",
                    "protocol_sha256": "",
                    "command": "",
                    "reason": "validation_selected_optional_ablation",
                    "error_log": "",
                }
                rows.append({name: raw.get(name, "") for name in RUN_FIELDS})
    if len(rows) != int(plan["expected_optional_ablation_unit_count"]):
        raise AssertionError("V2-4 optional plan must contain exactly 20 units")
    return rows


def _seed_then_fold_summary(
    frame: pd.DataFrame,
    identity_columns: list[str],
    metric_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fold_group = ["model_id", "fold", *identity_columns]
    fold = (
        frame.groupby(fold_group, as_index=False)[metric_columns]
        .mean(numeric_only=True)
        .sort_values(fold_group, kind="stable")
    )
    fold["seed_count"] = 3
    overall_group = ["model_id", *identity_columns]
    means = fold.groupby(overall_group, as_index=False)[metric_columns].mean(
        numeric_only=True
    )
    standard = fold.groupby(overall_group, as_index=False)[metric_columns].std(ddof=1)
    standard = standard.rename(
        columns={column: f"{column}_fold_std" for column in metric_columns}
    )
    overall = means.merge(standard, on=overall_group, validate="one_to_one")
    overall["fold_count"] = 5
    overall["seed_count_per_fold"] = 3
    return fold, overall


def aggregate_v2_phase4_extended(
    project_root: str | Path,
    registry: V2Phase4ExtendedRegistry,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    rows = [
        row for row in registry.rows() if row["profile"] == "v2_multiseed_extended"
    ]
    if len(rows) != 60 or any(row["status"] != "success" for row in rows):
        raise ValueError("Extended aggregation requires 60 successful units")
    mask_frames = []
    mixed_frames = []
    persistent_frames = []
    manifests = []
    for row in rows:
        output = root / row["run_dir"] / "extended_v2_4"
        mask_frames.append(pd.read_csv(output / "mask_metrics.csv", keep_default_na=False))
        mixed_frames.append(pd.read_csv(output / "mixed_metrics.csv", keep_default_na=False))
        persistent_frames.append(
            pd.read_csv(output / "persistent_classification.csv", keep_default_na=False)
        )
        manifests.append(read_json(output / "evaluation_manifest.json"))
    masks = pd.concat(mask_frames, ignore_index=True)
    mixed = pd.concat(mixed_frames, ignore_index=True)
    persistent = pd.concat(persistent_frames, ignore_index=True)
    expected_models = set(CORE_MODELS)
    for family in (masks, mixed):
        for (_, _), group in family.groupby(["fold", "seed"]):
            validate_shared_scenario_samples(
                group.to_dict("records"), expected_models=expected_models
            )
    metric_columns = [
        "macro_f1",
        "micro_f1",
        "mean_average_precision",
        "brier_score",
    ]
    mask_fold, mask_summary = _seed_then_fold_summary(
        masks,
        ["scenario_id", "available_count", "available_modalities", "drop_modalities"],
        metric_columns,
    )
    mixed_fold, mixed_summary = _seed_then_fold_summary(
        mixed,
        ["scenario_id", "drop_modality", "noisy_modality", "gaussian_sigma"],
        metric_columns,
    )
    artifacts = {}
    for name, frame in (
        ("mask_fold_seed_mean", mask_fold),
        ("mask_summary", mask_summary),
        ("mixed_fold_seed_mean", mixed_fold),
        ("mixed_summary", mixed_summary),
    ):
        artifacts[name] = write_stable_csv(
            root / "reports/v2" / f"phase_v2_4_{name}.csv",
            frame.to_dict("records"),
            frame.columns.tolist(),
        )
    if not persistent.empty:
        persistent_fold, persistent_summary = _seed_then_fold_summary(
            persistent,
            ["scenario_id", "fault", "target_modality", "fault_length", "phase"],
            ["macro_f1", "micro_f1", "masked_bce"],
        )
        for name, frame in (
            ("persistent_fold_seed_mean", persistent_fold),
            ("persistent_summary", persistent_summary),
        ):
            artifacts[name] = write_stable_csv(
                root / "reports/v2" / f"phase_v2_4_{name}.csv",
                frame.to_dict("records"),
                frame.columns.tolist(),
            )
    result = {
        "phase": "V2-4",
        "status": "extended_matrix_complete",
        "successful_unit_count": 60,
        "aggregation_order": "seeds_within_fold_then_folds",
        "mask_row_count": len(masks),
        "mixed_row_count": len(mixed),
        "persistent_classification_row_count": len(persistent),
        "all_scenarios_share_samples_across_models": True,
        "scenario_sample_hashes": sorted(
            {manifest["scenario_sample_sha256"] for manifest in manifests}
        ),
        "artifacts": artifacts,
    }
    write_json(root / "reports/v2/phase_v2_4_extended_report.json", result)
    return result
