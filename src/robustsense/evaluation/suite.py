"""Phase-4 natural-missingness and controlled-complete evaluation suite."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Subset

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.data.extrasensory import sha256_file
from robustsense.evaluation.scenarios import (
    ScenarioSpec,
    build_scenarios,
    scenario_manifest_sha256,
)
from robustsense.models.fusion import build_phase3_model
from robustsense.models.torch_baselines import build_phase2_model
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.trainer import (
    _collect_predictions,
    _device_from_profile,
    _loader,
    _move_batch,
    _verify_checkpoint_contract,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _sample_identifier(user_id: str, timestamp: int) -> str:
    return f"{user_id}|{timestamp}"


def _identifier_sha256(users: list[str], timestamps: np.ndarray) -> str:
    lines = [
        _sample_identifier(user, int(timestamp))
        for user, timestamp in zip(users, timestamps, strict=True)
    ]
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def select_controlled_subset(
    dataset: MultiModalDataset, limit: int | None, seed: int
) -> tuple[np.ndarray, str, int]:
    complete = np.flatnonzero(dataset.availability.all(axis=1))
    full_count = len(complete)
    if limit is not None and len(complete) > limit:
        ranked = sorted(
            complete.tolist(),
            key=lambda index: hashlib.sha256(
                f"{seed}|{dataset.user_id[index]}|{int(dataset.timestamp[index])}".encode()
            ).digest(),
        )[:limit]
        selected = np.asarray(sorted(ranked), dtype=np.int64)
    else:
        selected = complete.astype(np.int64)
    users = dataset.user_id[selected].astype(str).tolist()
    timestamps = dataset.timestamp[selected]
    return selected, _identifier_sha256(users, timestamps), full_count


def _load_model(
    run_path: Path,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[int, int]],
    torch.device,
]:
    resolved = load_config(run_path / "resolved_config.yaml")
    root = next(
        (
            candidate
            for candidate in run_path.parents
            if (candidate / resolved["processed_dir"] / "processed_manifest.json").is_file()
        ),
        None,
    )
    if root is None:
        raise ValueError(f"Cannot locate project root for run: {run_path}")
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    if int(resolved["phase"]) == 2:
        model = build_phase2_model(
            resolved["model_name"],
            modality_dims,
            len(manifest["labels"]),
            resolved["model_config"],
        )
    elif int(resolved["phase"]) == 3:
        model = build_phase3_model(
            resolved["model_name"],
            modality_dims,
            len(manifest["labels"]),
            resolved["model_config"],
        )
    else:
        raise ValueError("Full suite requires a Phase-2 or Phase-3 ExtraSensory run")
    device = _device_from_profile({"device": resolved["device"]})
    model = model.to(device)
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    if resolved["model_name"] == "reliability-constrained":
        abstention_artifact = read_json(run_path / "abstention_threshold.json")
        if abstention_artifact.get("source_split") != "val":
            raise ValueError("P2 abstention threshold is not validation-derived")
        model.set_abstention_threshold(float(abstention_artifact["threshold"]))
    model.eval()
    return model, resolved, manifest, modality_slices, device


def _prediction_frame(
    *,
    run_id: str,
    fold: int,
    seed: int,
    scenario: ScenarioSpec,
    users: list[str],
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    targets: np.ndarray,
    target_mask: np.ndarray,
    labels: list[str],
    thresholds: np.ndarray,
) -> pd.DataFrame:
    values: dict[str, Any] = {
        "run_id": run_id,
        "fold": fold,
        "seed": seed,
        "user_id": users,
        "timestamp": timestamps,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "target_modality": scenario.target_modality,
        "severity": scenario.severity,
        "drop_modalities": "|".join(scenario.drop_modalities),
        "replicate": scenario.replicate,
    }
    for index, label in enumerate(labels):
        values[f"probability__{label}"] = probabilities[:, index]
        values[f"prediction__{label}"] = probabilities[:, index] >= thresholds[index]
        values[f"target__{label}"] = targets[:, index]
        values[f"target_known__{label}"] = target_mask[:, index]
    return pd.DataFrame(values)


def _scenario_rows(
    scenario: ScenarioSpec,
    metrics: dict[str, Any],
    fixed_metrics: dict[str, Any],
    run_id: str,
    model_name: str,
    fold: int,
    seed: int,
    latency_ms_per_sample: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = {
        "run_id": run_id,
        "model_name": model_name,
        "fold": fold,
        "seed": seed,
        "scenario_id": scenario.scenario_id,
        "family": scenario.family,
        "target_modality": scenario.target_modality,
        "severity": scenario.severity,
        "drop_count": len(scenario.drop_modalities),
        "drop_modalities": "|".join(scenario.drop_modalities),
        "replicate": scenario.replicate,
        "sample_count": metrics["sample_count"],
    }
    overall = {
        **common,
        "macro_f1": metrics["macro_f1"],
        "micro_f1": metrics["micro_f1"],
        "mean_average_precision": metrics["mean_average_precision"],
        "brier_score": metrics["brier_score"],
        "macro_f1_fixed_0_5": fixed_metrics["macro_f1"],
        "micro_f1_fixed_0_5": fixed_metrics["micro_f1"],
        "latency_ms_per_sample": latency_ms_per_sample,
    }
    per_label = [
        {
            **common,
            **label_metrics,
        }
        for label_metrics in metrics["per_label"]
    ]
    return overall, per_label


def _response_rows(
    scenario: ScenarioSpec,
    output_weights: np.ndarray | None,
    output_reliability: np.ndarray | None,
    availability: np.ndarray,
    reliability_target: np.ndarray,
    artificial_fault: np.ndarray,
    run_id: str,
    model_name: str,
    fold: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows = []
    for modality_index, modality in enumerate(MODALITIES):
        predicted_reliability = (
            output_reliability[:, modality_index] if output_reliability is not None else None
        )
        target = reliability_target[:, modality_index]
        rows.append(
            {
                "run_id": run_id,
                "model_name": model_name,
                "fold": fold,
                "seed": seed,
                "scenario_id": scenario.scenario_id,
                "family": scenario.family,
                "target_modality": scenario.target_modality,
                "severity": scenario.severity,
                "modality": modality,
                "is_target_modality": modality == scenario.target_modality,
                "available_fraction": float(availability[:, modality_index].mean()),
                "artificial_fault_fraction": float(artificial_fault[:, modality_index].mean()),
                "mean_fusion_weight": (
                    float(output_weights[:, modality_index].mean())
                    if output_weights is not None
                    else np.nan
                ),
                "mean_reliability": (
                    float(predicted_reliability.mean())
                    if predicted_reliability is not None
                    else np.nan
                ),
                "reliability_target": float(target.mean()),
                "reliability_mae": (
                    float(np.abs(predicted_reliability - target).mean())
                    if predicted_reliability is not None
                    else np.nan
                ),
            }
        )
    return rows


def _collect_scenario(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    modality_slices: dict[str, tuple[int, int]],
    registry: CorruptionRegistry,
    scenario: ScenarioSpec,
    scenario_seed: int,
) -> dict[str, Any]:
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    weight_rows: list[np.ndarray] = []
    reliability_rows: list[np.ndarray] = []
    reliability_targets: list[np.ndarray] = []
    artificial_rows: list[np.ndarray] = []
    users: list[str] = []
    timestamps: list[np.ndarray] = []
    inference_seconds = 0.0
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            corruption_view = f"{scenario.scenario_id}|batch{batch_index}"
            if scenario.corruption == "gaussian":
                corruption_view = (
                    f"{scenario.family}|{scenario.target_modality}|batch{batch_index}"
                )
            corrupted, metadata = registry.apply_controlled(
                batch,
                scenario.corruption,
                seed=scenario_seed,
                view=corruption_view,
                target_modality=scenario.target_modality,
                severity=scenario.severity,
                drop_modalities=scenario.drop_modalities,
            )
            corrupted = _move_batch(corrupted, device, modality_slices)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            output = model(corrupted)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.perf_counter() - started
            valid = model.valid_sample_mask(corrupted)
            if not valid.all():
                raise AssertionError("Controlled complete subset became invalid for model")
            weights = output["fusion_weights"]
            if weights is not None:
                if torch.any(weights[~corrupted["availability"]] != 0):
                    raise AssertionError("Unavailable modality received nonzero weight")
                if not torch.allclose(
                    weights.sum(dim=1), torch.ones_like(weights[:, 0]), atol=1e-6
                ):
                    raise AssertionError("Available modality weights do not sum to one")
                weight_rows.append(weights.cpu().numpy())
            reliability = output["reliability"]
            if reliability is not None:
                reliability_rows.append(reliability.cpu().numpy())
            probabilities.append(torch.sigmoid(output["logits"]).cpu().numpy())
            targets.append(corrupted["targets"].cpu().numpy())
            target_masks.append(corrupted["target_mask"].cpu().numpy())
            availability_rows.append(corrupted["availability"].cpu().numpy())
            reliability_targets.append(metadata["reliability_target"].numpy())
            artificial_rows.append(metadata["artificial_fault"].numpy())
            users.extend(batch["user_id"])
            timestamps.append(corrupted["timestamp"].cpu().numpy())
    probability_array = np.concatenate(probabilities)
    return {
        "probabilities": probability_array,
        "targets": np.concatenate(targets),
        "target_mask": np.concatenate(target_masks),
        "availability": np.concatenate(availability_rows),
        "fusion_weights": np.concatenate(weight_rows) if weight_rows else None,
        "reliability": np.concatenate(reliability_rows) if reliability_rows else None,
        "reliability_target": np.concatenate(reliability_targets),
        "artificial_fault": np.concatenate(artificial_rows),
        "user_id": users,
        "timestamp": np.concatenate(timestamps),
        "latency_ms_per_sample": inference_seconds * 1000.0 / len(probability_array),
    }


def _natural_strata(
    collected: dict[str, Any], labels: list[str], thresholds: np.ndarray, run: dict[str, Any]
) -> pd.DataFrame:
    rows = []
    counts = collected["availability"].sum(axis=1)
    for available_count in sorted(np.unique(counts).tolist()):
        selected = counts == available_count
        metrics = phase2_multilabel_metrics(
            collected["probabilities"][selected],
            collected["targets"][selected],
            collected["target_mask"][selected],
            labels,
            thresholds,
        )
        rows.append(
            {
                "run_id": run["run_id"],
                "model_name": run["model_name"],
                "fold": run["fold"],
                "seed": run["seed"],
                "available_modality_count": int(available_count),
                "sample_count": int(selected.sum()),
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "mean_average_precision": metrics["mean_average_precision"],
                "brier_score": metrics["brier_score"],
            }
        )
    return pd.DataFrame(rows)


def _robustness_auc(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    clean = metrics.loc[metrics["family"] == "clean_complete", "macro_f1"]
    if clean.empty:
        return pd.DataFrame()
    clean_value = float(clean.iloc[0])
    for family in ("drop_random_k", "gaussian_noise", "bias_drift", "scale_error"):
        family_rows = metrics[metrics["family"] == family].copy()
        if family_rows.empty:
            continue
        if family == "scale_error":
            family_rows["intensity"] = (family_rows["severity"] - 1.0).abs()
        else:
            family_rows["intensity"] = family_rows["severity"].abs()
        grouped = (
            [(None, family_rows)]
            if family == "drop_random_k"
            else family_rows.groupby("target_modality")
        )
        for modality, group in grouped:
            curve = group.groupby("intensity", as_index=False)["macro_f1"].mean()
            curve = (
                pd.concat(
                    [pd.DataFrame({"intensity": [0.0], "macro_f1": [clean_value]}), curve],
                    ignore_index=True,
                )
                .groupby("intensity", as_index=False)["macro_f1"]
                .mean()
            )
            curve = curve.sort_values("intensity")
            maximum = float(curve["intensity"].max())
            auc = (
                float(np.trapezoid(curve["macro_f1"], curve["intensity"]) / maximum)
                if maximum > 0
                else np.nan
            )
            rows.append(
                {
                    "family": family,
                    "target_modality": modality,
                    "robustness_auc": auc,
                    "max_intensity": maximum,
                    "point_count": len(curve),
                }
            )
    return pd.DataFrame(rows)


def evaluate_full_suite(
    run_dir: str | Path,
    evaluation_config_path: str | Path | None = None,
) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    model, resolved, manifest, modality_slices, device = _load_model(run_path)
    root = run_path.parents[1]
    profile = str(resolved["profile"])
    plan_path = (
        Path(evaluation_config_path)
        if evaluation_config_path is not None
        else root / f"configs/evaluation/{profile}.yaml"
    )
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan = load_config(plan_path)
    model_name = str(resolved["model_name"])
    if model_name not in plan["models"]:
        raise ValueError(f"Model {model_name} is not registered in {plan_path}")
    controlled = model_name in plan["controlled_models"]
    corruption_path = root / "configs/corruption/eval.yaml"
    corruption_config = load_config(corruption_path)
    scenarios = build_scenarios(corruption_config)
    if not controlled:
        scenarios = scenarios[:1]
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    dataset = MultiModalDataset(processed_dir / "test.npz", manifest_path)
    labels = list(manifest["labels"])
    thresholds_artifact = read_json(run_path / "thresholds.json")
    if thresholds_artifact.get("source_split") != "val":
        raise ValueError("Phase-4 evaluation requires validation-derived thresholds")
    thresholds = np.asarray(thresholds_artifact["values"], dtype=np.float32)
    batch_size = int(resolved["experiment"].get("batch_size", 512))

    natural_loader = _loader(dataset, batch_size, False, int(resolved["seed"]), device)
    natural = _collect_predictions(model, natural_loader, device, modality_slices)
    natural_strata = _natural_strata(natural, labels, thresholds, resolved)
    natural_strata.to_parquet(run_path / "natural_missingness_strata.parquet", index=False)

    configured_limit = plan.get(
        "controlled_complete_sample_limit",
        corruption_config["dev_complete_sample_limit"] if profile == "dev" else None,
    )
    limit = int(configured_limit) if configured_limit is not None else None
    selected, subset_sha, complete_population = select_controlled_subset(
        dataset, limit, int(corruption_config["scenario_seed"])
    )
    subset_frame = pd.DataFrame(
        {
            "user_id": dataset.user_id[selected].astype(str),
            "timestamp": dataset.timestamp[selected],
        }
    )
    subset_frame.to_parquet(run_path / "controlled_subset.parquet", index=False)
    controlled_loader = _loader(
        Subset(dataset, selected.tolist()),
        batch_size,
        False,
        int(resolved["seed"]),
        device,
    )
    preprocessor = read_json(processed_dir / "preprocessor.json")
    registry = CorruptionRegistry(
        corruption_config,
        modality_slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    metric_rows: list[dict[str, Any]] = []
    per_label_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    clean_macro_f1: float | None = None
    for scenario in scenarios:
        collected = _collect_scenario(
            model,
            controlled_loader,
            device,
            modality_slices,
            registry,
            scenario,
            int(corruption_config["scenario_seed"]),
        )
        if _identifier_sha256(collected["user_id"], collected["timestamp"]) != subset_sha:
            raise AssertionError("Controlled scenario sample IDs changed")
        metrics = phase2_multilabel_metrics(
            collected["probabilities"],
            collected["targets"],
            collected["target_mask"],
            labels,
            thresholds,
        )
        fixed_metrics = phase2_multilabel_metrics(
            collected["probabilities"],
            collected["targets"],
            collected["target_mask"],
            labels,
            0.5,
        )
        overall, per_label = _scenario_rows(
            scenario,
            metrics,
            fixed_metrics,
            resolved["run_id"],
            model_name,
            int(resolved["fold"]),
            int(resolved["seed"]),
            float(collected["latency_ms_per_sample"]),
        )
        if clean_macro_f1 is None:
            clean_macro_f1 = float(metrics["macro_f1"])
        overall["delta_macro_f1"] = clean_macro_f1 - float(metrics["macro_f1"])
        overall["relative_drop"] = overall["delta_macro_f1"] / max(clean_macro_f1, 1e-12)
        metric_rows.append(overall)
        per_label_rows.extend(per_label)
        response_rows.extend(
            _response_rows(
                scenario,
                collected["fusion_weights"],
                collected["reliability"],
                collected["availability"],
                collected["reliability_target"],
                collected["artificial_fault"],
                resolved["run_id"],
                model_name,
                int(resolved["fold"]),
                int(resolved["seed"]),
            )
        )
        prediction_frames.append(
            _prediction_frame(
                run_id=resolved["run_id"],
                fold=int(resolved["fold"]),
                seed=int(resolved["seed"]),
                scenario=scenario,
                users=collected["user_id"],
                timestamps=collected["timestamp"],
                probabilities=collected["probabilities"],
                targets=collected["targets"],
                target_mask=collected["target_mask"],
                labels=labels,
                thresholds=thresholds,
            )
        )

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_parquet(run_path / "robustness_metrics.parquet", index=False)
    pd.DataFrame(per_label_rows).to_parquet(run_path / "robustness_per_label.parquet", index=False)
    pd.DataFrame(response_rows).to_parquet(run_path / "modality_response.parquet", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_parquet(
        run_path / "controlled_predictions.parquet", index=False
    )
    auc_frame = _robustness_auc(metrics_frame)
    if not auc_frame.empty:
        auc_frame.insert(0, "run_id", resolved["run_id"])
        auc_frame.insert(1, "model_name", model_name)
        auc_frame.insert(2, "fold", int(resolved["fold"]))
        auc_frame.insert(3, "seed", int(resolved["seed"]))
    auc_frame.to_parquet(run_path / "robustness_auc.parquet", index=False)

    natural_metrics = phase2_multilabel_metrics(
        natural["probabilities"],
        natural["targets"],
        natural["target_mask"],
        labels,
        thresholds,
    )
    manifest_value = {
        "suite_version": int(plan["suite_version"]),
        "evaluation_plan": str(plan_path.relative_to(root)),
        "evaluation_plan_sha256": sha256_file(plan_path),
        "corruption_config": str(corruption_path.relative_to(root)),
        "corruption_config_sha256": sha256_file(corruption_path),
        "scenario_manifest_sha256": scenario_manifest_sha256(build_scenarios(corruption_config)),
        "run_id": resolved["run_id"],
        "model_name": model_name,
        "fold": int(resolved["fold"]),
        "seed": int(resolved["seed"]),
        "profile": profile,
        "controlled_suite": controlled,
        "natural_sample_count": int(natural_metrics["sample_count"]),
        "natural_macro_f1": natural_metrics["macro_f1"],
        "complete_population_count": int(complete_population),
        "controlled_sample_count": int(len(selected)),
        "controlled_subset_sha256": subset_sha,
        "scenario_count": len(scenarios),
        "threshold_source_split": "val",
        "processed_manifest_sha256": sha256_file(manifest_path),
        "source_schema_sha256": manifest["source_schema_sha256"],
        "checkpoint_bytes": (run_path / "best_checkpoint.pt").stat().st_size,
        "outputs": [
            "natural_missingness_strata.parquet",
            "controlled_subset.parquet",
            "robustness_metrics.parquet",
            "robustness_per_label.parquet",
            "modality_response.parquet",
            "controlled_predictions.parquet",
            "robustness_auc.parquet",
        ],
    }
    write_json(run_path / "evaluation_manifest.json", manifest_value)
    return manifest_value
