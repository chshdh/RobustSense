"""Phase-3 gated, corruption-robust and quality-aware training."""

from __future__ import annotations

import csv
import platform
import sys
import time
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.data.extrasensory import sha256_file
from robustsense.models.fusion import build_phase3_model
from robustsense.models.torch_baselines import parameter_count
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.phase3_losses import (
    consistency_mse,
    reliability_mse,
    reliability_ranking_loss,
)
from robustsense.training.selective import tune_abstention_threshold
from robustsense.training.thresholds import tune_per_label_thresholds
from robustsense.training.torch_losses import masked_weighted_bce_with_logits, training_pos_weight
from robustsense.training.trainer import (
    _collect_predictions,
    _device_from_profile,
    _git_commit,
    _loader,
    _move_batch,
    _save_predictions,
    _verify_checkpoint_contract,
)
from robustsense.utils.config import dump_config, load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed

ABLATION_MODELS = {
    "ablation-a1-no-sensor-dropout",
    "ablation-a2-no-noise",
    "ablation-a3-no-quality",
    "ablation-a4-no-reliability",
    "ablation-a5-no-reliability-loss",
}
PHASE3_MODELS = {
    "gated",
    "robust-gated",
    "quality-aware-cls",
    "quality-aware",
    "reliability-constrained",
    *ABLATION_MODELS,
}


def _model_config_path(root: Path, model_name: str) -> Path:
    if model_name == "reliability-constrained":
        return root / "configs/v2/models/reliability_constrained.yaml"
    mapping = {
        "gated": "gated.yaml",
        "robust-gated": "robust_gated.yaml",
        "quality-aware-cls": "quality_aware.yaml",
        "quality-aware": "quality_aware.yaml",
        **{name: "quality_aware.yaml" for name in ABLATION_MODELS},
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported Phase-3 model: {model_name}")
    return root / "configs/model" / mapping[model_name]


def _profile_config_path(root: Path, model_name: str, profile_name: str) -> Path:
    if model_name == "reliability-constrained":
        return root / "configs/v2/evaluation" / f"{profile_name}.yaml"
    return root / "configs/experiment" / f"{profile_name}.yaml"


def _corruption_config_path(root: Path, model_name: str) -> Path:
    if model_name == "reliability-constrained":
        return root / "configs/v2/corruption/train.yaml"
    return root / "configs/corruption/train.yaml"


def _project_root_from_run(run_path: Path, resolved: dict[str, Any]) -> Path:
    for candidate in run_path.parents:
        manifest = candidate / resolved["processed_dir"] / "processed_manifest.json"
        if manifest.is_file():
            return candidate
    raise ValueError(f"Cannot locate project root for run: {run_path}")


def _checkpoint_contract(
    model_name: str,
    processed_manifest: dict[str, Any],
    processed_manifest_path: Path,
    preprocessor_path: Path,
    labels: list[str],
    modality_dims: dict[str, int],
) -> dict[str, Any]:
    return {
        "version": 2 if model_name == "reliability-constrained" else 1,
        "phase": 3,
        "model_name": model_name,
        "processed_manifest_sha256": sha256_file(processed_manifest_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "source_schema_sha256": processed_manifest["source_schema_sha256"],
        "labels": labels,
        "modality_dims": modality_dims,
    }


def _training_policy(model_name: str, model_config: dict[str, Any]) -> dict[str, Any]:
    reliability_weight = float(model_config.get("reliability_loss_weight", 0.1))
    consistency_weight = float(model_config.get("consistency_loss_weight", 0.1))
    policies = {
        "gated": (False, 0.0, 0.0, None),
        "robust-gated": (True, 0.0, 0.0, None),
        "quality-aware-cls": (True, 0.0, 0.0, None),
        "quality-aware": (True, reliability_weight, consistency_weight, None),
        "reliability-constrained": (True, reliability_weight, consistency_weight, None),
        "ablation-a1-no-sensor-dropout": (
            True,
            reliability_weight,
            consistency_weight,
            "disable_sensor_dropout",
        ),
        "ablation-a2-no-noise": (
            True,
            reliability_weight,
            consistency_weight,
            "disable_noise",
        ),
        "ablation-a3-no-quality": (True, reliability_weight, consistency_weight, None),
        "ablation-a4-no-reliability": (True, 0.0, consistency_weight, None),
        "ablation-a5-no-reliability-loss": (True, 0.0, consistency_weight, None),
    }
    corruption_enabled, reliability_loss, consistency_loss, override = policies[model_name]
    policy = {
        "corruption_enabled": corruption_enabled,
        "reliability_loss_weight": reliability_loss,
        "consistency_loss_weight": consistency_loss,
        "corruption_override": override,
    }
    if model_name == "reliability-constrained":
        policy["ranking_loss_weight"] = float(model_config.get("ranking_loss_weight", 0.05))
        policy["ranking_margin"] = float(model_config.get("ranking_margin", 0.1))
    return policy


def _resolved_corruption_config(
    base_config: dict[str, Any], policy: dict[str, Any]
) -> dict[str, Any]:
    """Return the frozen corruption recipe for a Phase-5 ablation."""
    config = deepcopy(base_config)
    override = policy.get("corruption_override")
    if override == "disable_sensor_dropout":
        config["missing_count_probs"] = {"0": 1.0, "1": 0.0, "2": 0.0}
    elif override == "disable_noise":
        config["noisy_modality_probability"] = 0.0
    elif override is not None:
        raise ValueError(f"Unknown corruption override: {override}")
    return config


def _diagnostic_frame(
    *,
    users: list[str],
    timestamps: np.ndarray,
    availability: np.ndarray,
    natural_availability: np.ndarray,
    weights: np.ndarray,
    reliability: np.ndarray | None,
    scenario: str,
    metadata: dict[str, Any] | None = None,
) -> pd.DataFrame:
    sample_count = len(timestamps)
    modality_count = len(MODALITIES)
    if metadata is None:
        artificial = np.zeros_like(availability, dtype=bool)
        dropped = np.zeros_like(availability, dtype=bool)
        severity = np.zeros_like(weights, dtype=np.float32)
        fault_types = np.where(availability, "natural_available", "natural_unavailable")
    else:
        artificial = metadata["artificial_fault"].detach().cpu().numpy()
        dropped = metadata["dropped"].detach().cpu().numpy()
        severity = metadata["severity"].detach().cpu().numpy()
        fault_types = np.asarray(metadata["fault_type"], dtype=object)
    if reliability is None:
        reliability = np.full_like(weights, np.nan, dtype=np.float32)
    return pd.DataFrame(
        {
            "user_id": np.repeat(np.asarray(users, dtype=object), modality_count),
            "timestamp": np.repeat(timestamps, modality_count),
            "scenario": scenario,
            "modality": np.tile(np.asarray(MODALITIES, dtype=object), sample_count),
            "naturally_available": natural_availability.reshape(-1),
            "available_after_corruption": availability.reshape(-1),
            "fusion_weight": weights.reshape(-1),
            "reliability": reliability.reshape(-1),
            "artificial_fault": artificial.reshape(-1),
            "fault_type": fault_types.reshape(-1),
            "severity": severity.reshape(-1),
            "dropped": dropped.reshape(-1),
        }
    )


def _collect_corruption_probe(
    model: torch.nn.Module,
    loader: Any,
    device: torch.device,
    modality_slices: dict[str, tuple[int, int]],
    registry: CorruptionRegistry,
    seed: int,
    max_samples: int,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    seen = 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            natural_availability = batch["availability"].clone()
            corrupted, metadata = registry.apply_training(
                batch, seed=seed, epoch=0, view="phase3_probe"
            )
            corrupted = _move_batch(corrupted, device, modality_slices)
            output = model(corrupted)
            take = min(len(batch["user_id"]), max_samples - seen)
            if take <= 0:
                break
            frames.append(
                _diagnostic_frame(
                    users=batch["user_id"][:take],
                    timestamps=corrupted["timestamp"][:take].cpu().numpy(),
                    availability=corrupted["availability"][:take].cpu().numpy(),
                    natural_availability=natural_availability[:take].numpy(),
                    weights=output["fusion_weights"][:take].cpu().numpy(),
                    reliability=(
                        output["reliability"][:take].cpu().numpy()
                        if output["reliability"] is not None
                        else None
                    ),
                    scenario="phase3_deterministic_probe",
                    metadata={
                        "artificial_fault": metadata["artificial_fault"][:take],
                        "dropped": metadata["dropped"][:take],
                        "severity": metadata["severity"][:take],
                        "fault_type": metadata["fault_type"][:take],
                    },
                )
            )
            seen += take
            if seen >= max_samples:
                break
    if not frames:
        raise ValueError("No samples available for Phase-3 corruption probe")
    return pd.concat(frames, ignore_index=True)


def train_phase3(
    project_root: str | Path,
    data_config_path: str | Path,
    model_name: str,
    fold: int,
    seed: int,
    profile_name: str,
) -> Path:
    if model_name not in PHASE3_MODELS:
        raise ValueError(f"Unsupported Phase-3 model: {model_name}")
    started_at = datetime.now(UTC).isoformat()
    root = Path(project_root).resolve()
    data_config_file = Path(data_config_path)
    if not data_config_file.is_absolute():
        data_config_file = root / data_config_file
    data_config = load_config(data_config_file)
    profile = load_config(_profile_config_path(root, model_name, profile_name))
    model_config = load_config(_model_config_path(root, model_name))
    corruption_config = load_config(_corruption_config_path(root, model_name))
    policy = _training_policy(model_name, model_config)
    corruption_config = _resolved_corruption_config(corruption_config, policy)
    processed_dir = (
        root / data_config.get("processed_dir", "data/processed") / "extrasensory" / f"fold{fold}"
    )
    manifest_path = processed_dir / "processed_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared fold not found: {manifest_path}")
    manifest = read_json(manifest_path)
    train_dataset = MultiModalDataset(processed_dir / "train.npz", manifest_path)
    val_dataset = MultiModalDataset(processed_dir / "val.npz", manifest_path)
    labels = list(manifest["labels"])
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    preprocessor_metadata = read_json(processed_dir / "preprocessor.json")
    registry = CorruptionRegistry(
        corruption_config,
        modality_slices,
        outlier_threshold=float(preprocessor_metadata["outlier_threshold"]),
        clip_value=float(preprocessor_metadata["clip_value"]),
    )
    device = _device_from_profile(profile)
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_phase3_model(model_name, modality_dims, len(labels), model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(profile.get("learning_rate", 0.001)),
        weight_decay=float(profile.get("weight_decay", 0.0001)),
    )
    class_weight_sample_mask = train_dataset.availability.any(axis=1)
    pos_weight = training_pos_weight(
        torch.from_numpy(train_dataset.targets),
        torch.from_numpy(train_dataset.target_mask & class_weight_sample_mask[:, None]),
        cap=float(profile.get("pos_weight_cap", 20.0)),
    ).to(device)
    batch_size = int(profile.get("batch_size", 512))
    train_loader = _loader(train_dataset, batch_size, True, seed, device)
    val_loader = _loader(val_dataset, batch_size, False, seed, device)
    epochs = int(profile.get("epochs", 1))
    patience = int(profile.get("early_stopping_patience", max(1, epochs)))
    run_id = f"extrasensory-{model_name}-fold{fold}-seed{seed}-{profile_name}"
    run_dir = root / profile.get("run_root", "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    contract = _checkpoint_contract(
        model_name,
        manifest,
        manifest_path,
        processed_dir / "preprocessor.json",
        labels,
        modality_dims,
    )
    best_metric = float("-inf")
    epochs_without_improvement = 0
    train_log: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        classification_loss = 0.0
        reliability_loss = 0.0
        consistency_loss = 0.0
        ranking_loss = 0.0
        batch_count = 0
        started = time.perf_counter()
        for batch in train_loader:
            if policy["corruption_enabled"]:
                corrupted_cpu, fault_metadata = registry.apply_training(
                    batch, seed=seed, epoch=epoch, view="train"
                )
                train_batch = _move_batch(corrupted_cpu, device, modality_slices)
                fault_metadata["reliability_target"] = fault_metadata[
                    "reliability_target"
                ].to(device, non_blocking=True)
                fault_metadata["artificial_fault"] = fault_metadata["artificial_fault"].to(
                    device, non_blocking=True
                )
            else:
                train_batch = _move_batch(batch, device, modality_slices)
                fault_metadata = None
            valid = model.valid_sample_mask(train_batch)
            effective_mask = train_batch["target_mask"] & valid.unsqueeze(1)
            if not torch.any(effective_mask):
                continue
            clean_output = None
            needs_clean_view = (
                policy["consistency_loss_weight"] > 0
                or policy.get("ranking_loss_weight", 0.0) > 0
            )
            if needs_clean_view:
                clean_batch = _move_batch(batch, device, modality_slices)
                model.eval()
                if policy.get("ranking_loss_weight", 0.0) > 0:
                    clean_output = model(clean_batch)
                else:
                    with torch.no_grad():
                        clean_output = model(clean_batch)
                model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(train_batch)
            classification = masked_weighted_bce_with_logits(
                output["logits"], train_batch["targets"], effective_mask, pos_weight
            )
            reliability = classification.new_zeros(())
            if policy["reliability_loss_weight"] > 0:
                if output["reliability"] is None or fault_metadata is None:
                    raise AssertionError(
                        "Reliability supervision requires P and corruption metadata"
                    )
                reliability = reliability_mse(
                    output["reliability"], fault_metadata["reliability_target"]
                )
            consistency = classification.new_zeros(())
            if policy["consistency_loss_weight"] > 0:
                if clean_output is None:
                    raise AssertionError("Consistency supervision requires a clean view")
                consistency = consistency_mse(
                    clean_output["logits"], output["logits"], effective_mask
                )
            ranking = classification.new_zeros(())
            if policy.get("ranking_loss_weight", 0.0) > 0:
                if (
                    clean_output is None
                    or clean_output["reliability"] is None
                    or output["reliability"] is None
                    or fault_metadata is None
                ):
                    raise AssertionError("Ranking supervision requires paired reliability views")
                ranking = reliability_ranking_loss(
                    clean_output["reliability"],
                    output["reliability"],
                    fault_metadata["artificial_fault"],
                    margin=policy["ranking_margin"],
                )
            loss = (
                classification
                + policy["reliability_loss_weight"] * reliability
                + policy["consistency_loss_weight"] * consistency
                + policy.get("ranking_loss_weight", 0.0) * ranking
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), float(profile.get("gradient_clip_norm", 5.0)))
            optimizer.step()
            epoch_loss += float(loss.item())
            classification_loss += float(classification.item())
            reliability_loss += float(reliability.item())
            consistency_loss += float(consistency.item())
            ranking_loss += float(ranking.item())
            batch_count += 1

        validation = _collect_predictions(model, val_loader, device, modality_slices)
        validation_fixed = phase2_multilabel_metrics(
            validation["probabilities"],
            validation["targets"],
            validation["target_mask"],
            labels,
            0.5,
        )
        metric = float(validation_fixed["macro_f1"] or 0.0)
        log_row = {
            "epoch": epoch,
            "train_total_loss": epoch_loss / batch_count,
            "train_classification_loss": classification_loss / batch_count,
            "train_reliability_loss": reliability_loss / batch_count,
            "train_consistency_loss": consistency_loss / batch_count,
            "val_macro_f1_at_0_5": metric,
            "epoch_seconds": time.perf_counter() - started,
        }
        if model_name == "reliability-constrained":
            log_row["train_ranking_loss"] = ranking_loss / batch_count
        train_log.append(log_row)
        if metric > best_metric:
            best_metric = metric
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "contract": contract}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _verify_checkpoint_contract(contract, checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    validation = _collect_predictions(model, val_loader, device, modality_slices)
    threshold_grid = profile.get(
        "threshold_grid", [round(value, 2) for value in np.arange(0.05, 1.0, 0.05)]
    )
    thresholds = tune_per_label_thresholds(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        threshold_grid,
        source_split="val",
    )
    val_metrics = phase2_multilabel_metrics(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        labels,
        thresholds,
    )
    val_metrics["fixed_0_5"] = phase2_multilabel_metrics(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        labels,
        0.5,
    )
    write_json(
        run_dir / "thresholds.json",
        {
            "source_split": "val",
            "method": "per_label_f1",
            "grid": threshold_grid,
            "labels": labels,
            "values": thresholds.tolist(),
        },
    )
    write_json(run_dir / "val_metrics.json", val_metrics)
    abstention_artifact = None
    if model_name == "reliability-constrained":
        if validation["system_reliability"] is None:
            raise AssertionError("P2 validation must emit system reliability")
        abstention_artifact = tune_abstention_threshold(
            validation["system_reliability"],
            target_coverage=float(model_config.get("target_coverage", 0.9)),
            source_split="val",
        )
        write_json(run_dir / "abstention_threshold.json", abstention_artifact)
        model.set_abstention_threshold(float(abstention_artifact["threshold"]))
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=train_log[0].keys())
        writer.writeheader()
        writer.writerows(train_log)

    write_json(
        run_dir / "split_manifest.json",
        read_json(root / "data/manifests" / f"extrasensory_split_fold{fold}.json"),
    )
    write_json(
        run_dir / "data_manifest.json",
        read_json(root / "data/manifests/extrasensory_data_manifest.json"),
    )
    resolved = {
        "phase": 3,
        "dataset": "extrasensory",
        "data_config": str(data_config_file.relative_to(root)),
        "model_name": model_name,
        "model_config": model_config,
        "corruption_config": corruption_config,
        "training_policy": policy,
        "profile": profile_name,
        "experiment": profile,
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "run_id": run_id,
        "processed_dir": str(processed_dir.relative_to(root)),
        "checkpoint_contract": contract,
        "parameter_count": parameter_count(model),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "class_weight_source_split": "train",
        "class_weight_sample_count": int(class_weight_sample_mask.sum()),
        "phase3_probe_samples": min(4096, len(val_dataset)),
    }
    if model_name == "reliability-constrained":
        resolved.update(
            {
                "v2_phase": 1,
                "abstention_threshold_artifact": "abstention_threshold.json",
                "abstention_threshold": abstention_artifact,
                "learned_beta": float(model.beta.detach().cpu()),
            }
        )
    dump_config(resolved, run_dir / "resolved_config.yaml")
    write_json(
        run_dir / "environment.json",
        {
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "git_commit": _git_commit(root),
            "seed": seed,
        },
    )
    if model_name == "reliability-constrained":
        (run_dir / "run_summary.md").write_text(
            "# Phase V2-1 P2 development run\n\n"
            f"- Run: `{run_id}`\n"
            f"- Fold: {fold}\n"
            f"- Seed: {seed}\n"
            f"- Validation Macro-F1: {val_metrics['macro_f1']:.6f}\n"
            f"- Validation target coverage: {abstention_artifact['target_coverage']:.6f}\n"
            f"- Validation achieved coverage: {abstention_artifact['validation_coverage']:.6f}\n"
            f"- Learned beta: {float(model.beta.detach().cpu()):.6f}\n\n"
            "Development-only: no test split was opened and no research conclusion is drawn.\n",
            encoding="utf-8",
        )
    else:
        evaluate_phase3(run_dir, suite="natural")
    return run_dir


def evaluate_phase3(run_dir: str | Path, suite: str = "natural") -> dict[str, object]:
    run_path = Path(run_dir).resolve()
    resolved = load_config(run_path / "resolved_config.yaml")
    if resolved.get("phase") != 3:
        raise ValueError("Not a Phase-3 run")
    root = _project_root_from_run(run_path, resolved)
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    labels = list(manifest["labels"])
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    model = build_phase3_model(
        resolved["model_name"], modality_dims, len(labels), resolved["model_config"]
    )
    device = _device_from_profile({"device": resolved["device"]})
    model = model.to(device)
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    expected_contract = _checkpoint_contract(
        resolved["model_name"],
        manifest,
        manifest_path,
        processed_dir / "preprocessor.json",
        labels,
        modality_dims,
    )
    _verify_checkpoint_contract(expected_contract, checkpoint["contract"])
    _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    if resolved["model_name"] == "reliability-constrained":
        abstention_artifact = read_json(run_path / "abstention_threshold.json")
        if abstention_artifact.get("source_split") != "val":
            raise ValueError("P2 abstention threshold is not validation-derived")
        model.set_abstention_threshold(float(abstention_artifact["threshold"]))
    test_dataset = MultiModalDataset(processed_dir / "test.npz", manifest_path)
    loader = _loader(
        test_dataset,
        int(resolved["experiment"].get("batch_size", 512)),
        False,
        int(resolved["seed"]),
        device,
    )
    collected = _collect_predictions(model, loader, device, modality_slices)
    threshold_artifact = read_json(run_path / "thresholds.json")
    if (
        threshold_artifact.get("source_split") != "val"
        or threshold_artifact.get("labels") != labels
    ):
        raise ValueError("Threshold artifact is not validation-derived or labels do not match")
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float32)
    metrics = phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        thresholds,
    )
    metrics.update(
        {
            "suite": suite,
            "scenario": "natural_missingness",
            "fold": int(resolved["fold"]),
            "model_name": resolved["model_name"],
            "parameter_count": int(resolved["parameter_count"]),
            "latency_ms_per_sample": collected["latency_ms_per_sample"],
            "dev_only": resolved["profile"] in {"dev", "ablation_dev"},
        }
    )
    metrics["fixed_0_5"] = phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        0.5,
    )
    write_json(run_path / "test_metrics.json", metrics)
    _save_predictions(
        run_path / "predictions.parquet",
        collected,
        labels,
        thresholds,
        resolved["run_id"],
        int(resolved["fold"]),
    )

    natural_diagnostics = _diagnostic_frame(
        users=collected["user_id"],
        timestamps=collected["timestamp"],
        availability=collected["availability"],
        natural_availability=collected["availability"],
        weights=collected["fusion_weights"],
        reliability=collected["reliability"],
        scenario="natural_missingness",
    )
    preprocessor_metadata = read_json(processed_dir / "preprocessor.json")
    registry = CorruptionRegistry(
        resolved["corruption_config"],
        modality_slices,
        outlier_threshold=float(preprocessor_metadata["outlier_threshold"]),
        clip_value=float(preprocessor_metadata["clip_value"]),
    )
    probe = _collect_corruption_probe(
        model,
        loader,
        device,
        modality_slices,
        registry,
        int(resolved["seed"]),
        int(resolved["phase3_probe_samples"]),
    )
    diagnostics = pd.concat([natural_diagnostics, probe], ignore_index=True)
    diagnostics.insert(0, "run_id", resolved["run_id"])
    diagnostics.to_parquet(run_path / "modality_diagnostics.parquet", index=False)

    development_only = resolved["profile"] in {"dev", "ablation_dev"}
    scope_note = (
        "This is a development-only engineering run, not a research conclusion."
        if development_only
        else "This run is part of the frozen formal experiment registry."
    )
    (run_path / "run_summary.md").write_text(
        "# Phase-3 ExtraSensory run\n\n"
        f"- Run: `{resolved['run_id']}`\n"
        f"- Model: `{resolved['model_name']}`\n"
        f"- Profile: `{resolved['profile']}`\n"
        f"- Fold: {resolved['fold']}\n"
        f"- Epochs completed: {len(pd.read_csv(run_path / 'train_log.csv'))}\n"
        f"- Natural-missingness samples: {metrics['sample_count']}\n"
        f"- Parameters: {metrics['parameter_count']}\n"
        f"- Macro-F1: {metrics['macro_f1']:.6f}\n"
        f"- Micro-F1: {metrics['micro_f1']:.6f}\n"
        f"- mAP: {metrics['mean_average_precision']:.6f}\n"
        f"- Deterministic diagnostic probe samples: {resolved['phase3_probe_samples']}\n\n"
        "The corrupted probe is diagnostic-only and did not tune thresholds. "
        f"{scope_note}\n",
        encoding="utf-8",
    )
    return metrics
