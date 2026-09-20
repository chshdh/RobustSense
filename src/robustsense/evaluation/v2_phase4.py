"""Metrics-only, GPU-batched extended evaluation for the V2-4 core matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Subset

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.data.extrasensory import sha256_file
from robustsense.evaluation.suite import _load_model, select_controlled_subset
from robustsense.evaluation.v2_aggregate import aggregate_mask_metrics
from robustsense.evaluation.v2_artifacts import write_stable_csv
from robustsense.evaluation.v2_metrics import (
    fault_detection_metrics,
    masked_binary_cross_entropy_per_sample,
    persistent_episode_response,
    reliability_calibration,
)
from robustsense.evaluation.v2_scenarios import (
    AvailabilityMaskSpec,
    MixedFailureSpec,
    apply_mixed_failure,
    build_exhaustive_masks,
    build_mixed_failures,
    build_persistent_episodes,
    persistent_episode_manifest_bytes,
    scenario_sample_sha256,
)
from robustsense.models.v2_phase3 import build_v2_phase3_model
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.trainer import (
    _device_from_profile,
    _loader,
    _move_batch,
    _verify_checkpoint_contract,
)
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json


def _root_from_run(run_path: Path, resolved: dict[str, Any]) -> Path:
    for candidate in run_path.parents:
        if (candidate / resolved["processed_dir"] / "processed_manifest.json").is_file():
            return candidate
    raise ValueError(f"Cannot locate project root for run: {run_path}")


def _load_p2(
    run_path: Path,
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[int, int]],
    torch.device,
]:
    resolved = load_config(run_path / "resolved_config.yaml")
    if resolved.get("variant") != "P2" or resolved.get("phase") not in {"V2-3", "V2-4"}:
        raise ValueError("P2 extended evaluation requires a V2-3 or V2-4 P2 run")
    root = _root_from_run(run_path, resolved)
    processed = root / resolved["processed_dir"]
    manifest = read_json(processed / "processed_manifest.json")
    slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    dimensions = {name: end - start for name, (start, end) in slices.items()}
    model = build_v2_phase3_model(
        "P2", dimensions, len(manifest["labels"]), resolved["model_config"]
    )
    device = _device_from_profile({"device": resolved["device"]})
    model = model.to(device)
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    abstention = read_json(run_path / "abstention_threshold.json")
    if abstention.get("source_split") != "val":
        raise ValueError("P2 abstention threshold is not validation-derived")
    model.set_abstention_threshold(float(abstention["threshold"]))
    model.eval()
    return model, resolved, manifest, slices, device


def _load_extended_model(
    run_path: Path, model_id: str
) -> tuple[
    torch.nn.Module,
    dict[str, Any],
    dict[str, Any],
    dict[str, tuple[int, int]],
    torch.device,
]:
    if model_id == "P2":
        return _load_p2(run_path)
    return _load_model(run_path)


def _empty_collector() -> dict[str, list[np.ndarray]]:
    return {
        "probabilities": [],
        "targets": [],
        "target_mask": [],
        "availability": [],
        "fusion_weights": [],
        "reliability": [],
        "reliability_target": [],
        "artificial_fault": [],
    }


def _merge_transformed(batches: list[dict[str, Any]]) -> dict[str, Any]:
    tensor_keys = (
        "features_flat",
        "feature_masks_flat",
        "availability",
        "quality_features",
        "targets",
        "target_mask",
        "timestamp",
    )
    merged = {key: torch.cat([batch[key] for batch in batches]) for key in tensor_keys}
    merged["user_id"] = [user for batch in batches for user in batch["user_id"]]
    return merged


def _finalize_collector(rows: dict[str, list[np.ndarray]]) -> dict[str, Any]:
    required = ("probabilities", "targets", "target_mask", "availability")
    if any(not rows[name] for name in required):
        raise ValueError("Extended scenario produced no predictions")
    result: dict[str, Any] = {
        name: np.concatenate(values) if values else None for name, values in rows.items()
    }
    return result


def _apply_scenario(
    registry: CorruptionRegistry,
    batch: dict[str, Any],
    scenario: dict[str, Any],
    *,
    seed: int,
    batch_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    kind = scenario["kind"]
    spec = scenario["spec"]
    view = f"{scenario['scenario_id']}|batch{batch_index}"
    if kind == "mask":
        mask = spec
        if not isinstance(mask, AvailabilityMaskSpec):
            raise TypeError("Mask scenario has an invalid specification")
        corruption = "missing" if mask.drop_modalities else "clean"
        return registry.apply_controlled(
            batch,
            corruption,
            seed=seed,
            view=view,
            drop_modalities=mask.drop_modalities,
        )
    if kind == "mixed":
        mixed = spec
        if not isinstance(mixed, MixedFailureSpec):
            raise TypeError("Mixed scenario has an invalid specification")
        return apply_mixed_failure(registry, batch, mixed, seed=seed, view=view)
    if kind == "persistent":
        fault = str(scenario["fault"])
        modality = str(scenario["target_modality"])
        if fault == "drop":
            return registry.apply_controlled(
                batch,
                "missing",
                seed=seed,
                view=view,
                drop_modalities=(modality,),
            )
        if fault == "gaussian_sigma_2":
            return registry.apply_controlled(
                batch,
                "gaussian",
                seed=seed,
                view=view,
                target_modality=modality,
                severity=2.0,
            )
    raise ValueError(f"Unknown extended scenario kind: {kind}")


def _collect_scenarios(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    modality_slices: dict[str, tuple[int, int]],
    registry: CorruptionRegistry,
    scenarios: list[dict[str, Any]],
    *,
    scenario_seed: int,
    scenario_group_size: int,
) -> dict[str, dict[str, Any]]:
    """Fuse several corruption scenarios into each inference batch for GPU occupancy."""
    if scenario_group_size <= 0:
        raise ValueError("scenario_group_size must be positive")
    completed: dict[str, dict[str, Any]] = {}
    for group_start in range(0, len(scenarios), scenario_group_size):
        group = scenarios[group_start : group_start + scenario_group_size]
        collectors = {item["scenario_id"]: _empty_collector() for item in group}
        with torch.inference_mode():
            for batch_index, batch in enumerate(loader):
                transformed = []
                metadata = []
                row_count = len(batch["user_id"])
                for scenario in group:
                    altered, details = _apply_scenario(
                        registry,
                        batch,
                        scenario,
                        seed=scenario_seed,
                        batch_index=batch_index,
                    )
                    transformed.append(altered)
                    metadata.append(details)
                merged = _move_batch(
                    _merge_transformed(transformed), device, modality_slices
                )
                valid = model.valid_sample_mask(merged)
                if not bool(valid.all()):
                    raise AssertionError("Complete-subset scenario produced an invalid sample")
                output = model(merged)
                probabilities = torch.sigmoid(output["logits"]).cpu().numpy()
                targets = merged["targets"].cpu().numpy()
                target_mask = merged["target_mask"].cpu().numpy()
                availability = merged["availability"].cpu().numpy()
                weights = output.get("fusion_weights")
                reliability = output.get("reliability")
                for offset, scenario in enumerate(group):
                    start = offset * row_count
                    end = start + row_count
                    collector = collectors[scenario["scenario_id"]]
                    collector["probabilities"].append(probabilities[start:end])
                    collector["targets"].append(targets[start:end])
                    collector["target_mask"].append(target_mask[start:end])
                    collector["availability"].append(availability[start:end])
                    if weights is not None:
                        collector["fusion_weights"].append(
                            weights[start:end].cpu().numpy()
                        )
                    if reliability is not None:
                        collector["reliability"].append(
                            reliability[start:end].cpu().numpy()
                        )
                    collector["reliability_target"].append(
                        metadata[offset]["reliability_target"].numpy()
                    )
                    collector["artificial_fault"].append(
                        metadata[offset]["artificial_fault"].numpy()
                    )
        completed.update(
            {identifier: _finalize_collector(rows) for identifier, rows in collectors.items()}
        )
    return completed


def _metrics(
    collected: dict[str, Any], labels: list[str], thresholds: np.ndarray
) -> dict[str, Any]:
    return phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        thresholds,
    )


def _common(
    *,
    run_id: str,
    model_id: str,
    fold: int,
    seed: int,
    scenario_id: str,
    sample_hash: str,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "model_id": model_id,
        "fold": fold,
        "seed": seed,
        "scenario_id": scenario_id,
        "scenario_sample_sha256": sample_hash,
    }


def _response_rows(
    collected: dict[str, Any], common: dict[str, Any], family: str
) -> list[dict[str, Any]]:
    weights = collected["fusion_weights"]
    reliability = collected["reliability"]
    rows = []
    for index, modality in enumerate(MODALITIES):
        rows.append(
            {
                **common,
                "family": family,
                "modality": modality,
                "mean_fusion_weight": (
                    float(weights[:, index].mean()) if weights is not None else np.nan
                ),
                "mean_reliability": (
                    float(reliability[:, index].mean())
                    if reliability is not None
                    else np.nan
                ),
                "mean_reliability_target": float(
                    collected["reliability_target"][:, index].mean()
                ),
                "artificial_fault_fraction": float(
                    collected["artificial_fault"][:, index].mean()
                ),
            }
        )
    return rows


def _natural_mask_frequencies(dataset: MultiModalDataset) -> dict[str, int]:
    frequencies: dict[str, int] = {}
    for availability in dataset.availability:
        identifier = "mask." + "".join("1" if value else "0" for value in availability)
        if identifier != "mask.000000":
            frequencies[identifier] = frequencies.get(identifier, 0) + 1
    return frequencies


def _write_rows(
    path: Path,
    rows: list[dict[str, Any]],
    empty_columns: list[str] | None = None,
) -> dict[str, Any]:
    if not rows and not empty_columns:
        raise ValueError(f"Cannot infer columns for empty artifact: {path.name}")
    return write_stable_csv(path, rows, list(rows[0]) if rows else empty_columns or [])


def _persistent_classification_rows(
    *,
    clean: dict[str, Any],
    faulted: dict[str, Any],
    episodes: list[Any],
    labels: list[str],
    thresholds: np.ndarray,
    common: dict[str, Any],
    fault: str,
    target_modality: str,
) -> list[dict[str, Any]]:
    rows = []
    for length in sorted({episode.fault_length for episode in episodes}):
        selected = [episode for episode in episodes if episode.fault_length == length]
        for phase in ("pre", "fault", "post"):
            indices = np.asarray(
                [
                    index
                    for episode in selected
                    for index in getattr(episode, f"{phase}_indices")
                ],
                dtype=np.int64,
            )
            source = faulted if phase == "fault" else clean
            metrics = phase2_multilabel_metrics(
                source["probabilities"][indices],
                source["targets"][indices],
                source["target_mask"][indices],
                labels,
                thresholds,
            )
            losses = masked_binary_cross_entropy_per_sample(
                source["probabilities"][indices],
                source["targets"][indices],
                source["target_mask"][indices],
            )
            rows.append(
                {
                    **common,
                    "fault": fault,
                    "target_modality": target_modality,
                    "fault_length": length,
                    "phase": phase,
                    "episode_count": len(selected),
                    "sample_count": len(indices),
                    "macro_f1": metrics["macro_f1"],
                    "micro_f1": metrics["micro_f1"],
                    "masked_bce": float(np.nanmean(losses)),
                }
            )
    return rows


def _persistent_response_rows(
    *,
    clean: dict[str, Any],
    faulted: dict[str, Any],
    episodes: list[Any],
    common: dict[str, Any],
    fault: str,
    target_modality: str,
    detection_threshold: float,
    recovery_tolerance: float,
) -> list[dict[str, Any]]:
    modality_index = MODALITIES.index(target_modality)
    if clean["reliability"] is None:
        clean_scores = np.full(len(clean["probabilities"]), np.nan)
        fault_scores = clean_scores.copy()
    else:
        clean_scores = 1.0 - clean["reliability"][:, modality_index]
        fault_scores = 1.0 - faulted["reliability"][:, modality_index]
    combined = clean_scores.copy()
    fault_indices = np.asarray(
        [index for episode in episodes for index in episode.fault_indices], dtype=np.int64
    )
    combined[fault_indices] = fault_scores[fault_indices]
    rows = []
    for episode in episodes:
        response = persistent_episode_response(
            combined,
            pre_indices=episode.pre_indices,
            fault_indices=episode.fault_indices,
            post_indices=episode.post_indices,
            detection_threshold=detection_threshold,
            recovery_tolerance=recovery_tolerance,
            episode_id=episode.episode_id,
            run_id=str(common["run_id"]),
        )
        rows.append(
            {
                **common,
                "fault": fault,
                "target_modality": target_modality,
                "fault_length": episode.fault_length,
                "user_id": episode.user_id,
                **{
                    key: value
                    for key, value in response.items()
                    if key not in {"run_id", "episode_id"}
                },
                "episode_id": episode.episode_id,
                "reliability_available": clean["reliability"] is not None,
            }
        )
    return rows


def evaluate_v2_phase4_extended(
    run_dir: str | Path,
    *,
    model_id: str,
    config_path: str | Path = "configs/v2/evaluation/extended_core.yaml",
) -> dict[str, Any]:
    """Run 63 masks, 60 mixed faults, and persistent episodes without predictions."""
    run_path = Path(run_dir).resolve()
    model, resolved, manifest, slices, device = _load_extended_model(run_path, model_id)
    root = _root_from_run(run_path, resolved)
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root / config_file
    config = load_config(config_file)
    fold = int(resolved["fold"])
    seed = int(resolved["seed"])
    if model_id not in config["models"] or fold not in config["folds"]:
        raise ValueError("Run is outside the V2-4 extended evaluation matrix")
    if seed not in config["seeds"]:
        raise ValueError("Run seed is outside the V2-4 extended evaluation matrix")
    threshold_artifact = read_json(run_path / "thresholds.json")
    if threshold_artifact.get("source_split") != "val":
        raise ValueError("Extended evaluation requires validation-derived thresholds")
    labels = list(manifest["labels"])
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float32)
    processed = root / resolved["processed_dir"]
    dataset = MultiModalDataset(
        processed / "test.npz", processed / "processed_manifest.json"
    )
    selected, selected_hash, complete_population = select_controlled_subset(
        dataset, None, int(config["scenario_seed"])
    )
    selected_users = dataset.user_id[selected].astype(str)
    selected_timestamps = dataset.timestamp[selected]
    if scenario_sample_sha256(selected_users, selected_timestamps) != selected_hash:
        raise AssertionError("Controlled sample hash implementations disagree")
    loader = _loader(
        Subset(dataset, selected.tolist()),
        int(resolved["experiment"].get("batch_size", 512)),
        False,
        seed,
        device,
    )
    preprocessor = read_json(processed / "preprocessor.json")
    registry = CorruptionRegistry(
        resolved["corruption_config"],
        slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    output_dir = run_path / "extended_v2_4"
    output_dir.mkdir(parents=True, exist_ok=True)
    group_size = int(config["scenario_group_size"])
    run_id = str(resolved["run_id"])

    masks = build_exhaustive_masks()
    mask_scenarios = [
        {"scenario_id": mask.mask_id, "kind": "mask", "spec": mask} for mask in masks
    ]
    mask_outputs = _collect_scenarios(
        model,
        loader,
        device,
        slices,
        registry,
        mask_scenarios,
        scenario_seed=int(config["scenario_seed"]),
        scenario_group_size=group_size,
    )
    mask_rows = []
    response_rows = []
    calibration_predictions = []
    calibration_targets = []
    calibration_valid = []
    fault_labels = []
    for mask in masks:
        collected = mask_outputs[mask.mask_id]
        metrics = _metrics(collected, labels, thresholds)
        common = _common(
            run_id=run_id,
            model_id=model_id,
            fold=fold,
            seed=seed,
            scenario_id=mask.mask_id,
            sample_hash=selected_hash,
        )
        mask_rows.append(
            {
                **common,
                "mask_id": mask.mask_id,
                "available_count": mask.available_count,
                "available_modalities": "|".join(mask.available_modalities),
                "drop_modalities": "|".join(mask.drop_modalities),
                "sample_count": metrics["sample_count"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "mean_average_precision": metrics["mean_average_precision"],
                "brier_score": metrics["brier_score"],
            }
        )
        response_rows.extend(_response_rows(collected, common, "availability_mask"))
        if collected["reliability"] is not None:
            calibration_predictions.append(collected["reliability"])
            calibration_targets.append(collected["reliability_target"])
            calibration_valid.append(np.ones_like(collected["reliability"], dtype=bool))
            fault_labels.append(collected["artificial_fault"])
    mask_artifact = _write_rows(output_dir / "mask_metrics.csv", mask_rows)
    mask_summary = aggregate_mask_metrics(mask_rows, _natural_mask_frequencies(dataset))
    write_json(output_dir / "mask_summary.json", mask_summary)
    clean = mask_outputs["mask.111111"]
    del mask_outputs

    mixed_specs = build_mixed_failures()
    mixed_scenarios = [
        {"scenario_id": item.scenario_id, "kind": "mixed", "spec": item}
        for item in mixed_specs
    ]
    mixed_outputs = _collect_scenarios(
        model,
        loader,
        device,
        slices,
        registry,
        mixed_scenarios,
        scenario_seed=int(config["scenario_seed"]),
        scenario_group_size=group_size,
    )
    mixed_rows = []
    for scenario in mixed_specs:
        collected = mixed_outputs[scenario.scenario_id]
        metrics = _metrics(collected, labels, thresholds)
        common = _common(
            run_id=run_id,
            model_id=model_id,
            fold=fold,
            seed=seed,
            scenario_id=scenario.scenario_id,
            sample_hash=selected_hash,
        )
        mixed_rows.append(
            {
                **common,
                "drop_modality": scenario.drop_modality,
                "noisy_modality": scenario.noisy_modality,
                "gaussian_sigma": scenario.gaussian_sigma,
                "sample_count": metrics["sample_count"],
                "macro_f1": metrics["macro_f1"],
                "micro_f1": metrics["micro_f1"],
                "mean_average_precision": metrics["mean_average_precision"],
                "brier_score": metrics["brier_score"],
            }
        )
        response_rows.extend(_response_rows(collected, common, "mixed_failure"))
        if collected["reliability"] is not None:
            calibration_predictions.append(collected["reliability"])
            calibration_targets.append(collected["reliability_target"])
            calibration_valid.append(np.ones_like(collected["reliability"], dtype=bool))
            fault_labels.append(collected["artificial_fault"])
    mixed_artifact = _write_rows(output_dir / "mixed_metrics.csv", mixed_rows)
    response_artifact = _write_rows(output_dir / "modality_response.csv", response_rows)
    del mixed_outputs

    if calibration_predictions:
        predicted = np.concatenate(calibration_predictions)
        expected = np.concatenate(calibration_targets)
        valid = np.concatenate(calibration_valid)
        faults = np.concatenate(fault_labels)
        calibration = reliability_calibration(
            predicted,
            expected,
            valid,
            modality_names=list(MODALITIES),
            n_bins=int(config["calibration_bins"]),
            run_id=run_id,
        )
        calibration_bins = _write_rows(
            output_dir / "reliability_calibration_bins.csv", calibration["bins"]
        )
        calibration_summary = _write_rows(
            output_dir / "reliability_calibration_summary.csv", calibration["summary"]
        )
        detection = fault_detection_metrics(
            predicted, faults, valid, run_id=run_id
        )
        detection["reliability_available"] = True
    else:
        calibration_bins = None
        calibration_summary = None
        detection = {
            "run_id": run_id,
            "reliability_available": False,
            "valid_count": 0,
            "positive_count": 0,
            "fault_auroc": np.nan,
            "fault_auprc": np.nan,
        }
    write_json(output_dir / "fault_detection.json", detection)

    episodes, episode_summary = build_persistent_episodes(
        selected_users,
        selected_timestamps,
        tuple(int(value) for value in config["persistent_episode_lengths"]),
        max_gap_seconds=int(config["max_gap_seconds"]),
    )
    episode_bytes = persistent_episode_manifest_bytes(episodes)
    episode_path = output_dir / "persistent_episode_manifest.jsonl"
    episode_path.write_bytes(episode_bytes)
    episode_hash = hashlib.sha256(episode_bytes).hexdigest()
    persistent_scenarios = [
        {
            "scenario_id": f"persistent.{fault}.{modality}",
            "kind": "persistent",
            "spec": None,
            "fault": fault,
            "target_modality": modality,
        }
        for fault in config["persistent_faults"]
        for modality in config["persistent_target_modalities"]
    ]
    persistent_outputs = _collect_scenarios(
        model,
        loader,
        device,
        slices,
        registry,
        persistent_scenarios,
        scenario_seed=int(config["scenario_seed"]),
        scenario_group_size=group_size,
    )
    persistent_classification = []
    persistent_response = []
    for scenario in persistent_scenarios:
        identifier = scenario["scenario_id"]
        fault = str(scenario["fault"])
        modality = str(scenario["target_modality"])
        common = _common(
            run_id=run_id,
            model_id=model_id,
            fold=fold,
            seed=seed,
            scenario_id=identifier,
            sample_hash=selected_hash,
        )
        persistent_classification.extend(
            _persistent_classification_rows(
                clean=clean,
                faulted=persistent_outputs[identifier],
                episodes=episodes,
                labels=labels,
                thresholds=thresholds,
                common=common,
                fault=fault,
                target_modality=modality,
            )
        )
        persistent_response.extend(
            _persistent_response_rows(
                clean=clean,
                faulted=persistent_outputs[identifier],
                episodes=episodes,
                common=common,
                fault=fault,
                target_modality=modality,
                detection_threshold=float(config["fault_detection_threshold"]),
                recovery_tolerance=float(config["recovery_tolerance"]),
            )
        )
    persistent_classification_artifact = _write_rows(
        output_dir / "persistent_classification.csv",
        persistent_classification,
        [
            "run_id",
            "model_id",
            "fold",
            "seed",
            "scenario_id",
            "scenario_sample_sha256",
            "fault",
            "target_modality",
            "fault_length",
            "phase",
            "episode_count",
            "sample_count",
            "macro_f1",
            "micro_f1",
            "masked_bce",
        ],
    )
    persistent_response_artifact = _write_rows(
        output_dir / "persistent_response.csv",
        persistent_response,
        [
            "run_id",
            "model_id",
            "fold",
            "seed",
            "scenario_id",
            "scenario_sample_sha256",
            "fault",
            "target_modality",
            "fault_length",
            "user_id",
            "pre_mean_fault_score",
            "fault_mean_fault_score",
            "post_mean_fault_score",
            "detection_delay_steps",
            "recovery_steps",
            "recovery_reference_upper",
            "detected",
            "recovered",
            "episode_id",
            "reliability_available",
        ],
    )

    manifest_path = output_dir / "evaluation_manifest.json"
    result = {
        "phase": "V2-4",
        "profile": config["profile"],
        "run_id": run_id,
        "model_id": model_id,
        "fold": fold,
        "seed": seed,
        "protocol_sha256": resolved.get(
            "v2_protocol_sha256", resolved.get("protocol_sha256")
        ),
        "threshold_source_split": threshold_artifact["source_split"],
        "complete_population_count": complete_population,
        "selected_sample_count": len(selected),
        "scenario_sample_sha256": selected_hash,
        "scenario_group_size": group_size,
        "per_sample_predictions_stored": False,
        "mask_scenario_count": len(mask_rows),
        "mixed_scenario_count": len(mixed_rows),
        "persistent_scenario_count": len(persistent_scenarios),
        "persistent_episode_count": len(episodes),
        "persistent_episode_summary": episode_summary,
        "persistent_episode_manifest_sha256": episode_hash,
        "reliability_available": bool(calibration_predictions),
        "artifacts": {
            "mask_metrics": mask_artifact,
            "mixed_metrics": mixed_artifact,
            "modality_response": response_artifact,
            "persistent_classification": persistent_classification_artifact,
            "persistent_response": persistent_response_artifact,
            "calibration_bins": calibration_bins,
            "calibration_summary": calibration_summary,
        },
    }
    write_json(manifest_path, result)
    result["evaluation_manifest_sha256"] = sha256_file(manifest_path)
    return result
