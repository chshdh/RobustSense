"""Leakage-safe Phase-2 training and natural-missingness evaluation."""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader

from robustsense.constants import MODALITIES
from robustsense.data.dataset import MultiModalDataset, add_modality_views, collate_multimodal
from robustsense.data.extrasensory import sha256_file
from robustsense.models.torch_baselines import build_phase2_model, parameter_count
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.thresholds import tune_per_label_thresholds
from robustsense.training.torch_losses import masked_weighted_bce_with_logits, training_pos_weight
from robustsense.utils.config import dump_config, load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed


def _git_commit(project_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _model_config_path(root: Path, model_name: str) -> Path:
    if model_name.startswith("single-"):
        return root / "configs/model/single.yaml"
    mapping = {
        "linear": "early.yaml",
        "mlp": "mlp_early.yaml",
        "late": "late.yaml",
    }
    if model_name not in mapping:
        raise ValueError(f"Unsupported Phase-2 model: {model_name}")
    return root / "configs/model" / mapping[model_name]


def _device_from_profile(profile: dict[str, Any]) -> torch.device:
    requested = str(profile.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)


def _move_batch(
    batch: dict[str, Any], device: torch.device, modality_slices: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    for key in (
        "features_flat",
        "feature_masks_flat",
        "availability",
        "quality_features",
        "targets",
        "target_mask",
        "timestamp",
    ):
        batch[key] = batch[key].to(device, non_blocking=True)
    return add_modality_views(batch, modality_slices)


def _loader(
    dataset: MultiModalDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=device.type == "cuda",
        collate_fn=collate_multimodal,
        generator=generator,
    )


def _collect_predictions(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    modality_slices: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    model.eval()
    probabilities: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    target_masks: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    fusion_weight_rows: list[np.ndarray] = []
    reliability_rows: list[np.ndarray] = []
    utility_rows: list[np.ndarray] = []
    system_reliability_rows: list[np.ndarray] = []
    abstain_rows: list[np.ndarray] = []
    users: list[str] = []
    timestamps: list[np.ndarray] = []
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for batch in loader:
            batch = _move_batch(batch, device, modality_slices)
            valid = model.valid_sample_mask(batch)
            if not torch.any(valid):
                continue
            output = model(batch)
            probabilities.append(torch.sigmoid(output["logits"][valid]).cpu().numpy())
            targets.append(batch["targets"][valid].cpu().numpy())
            target_masks.append(batch["target_mask"][valid].cpu().numpy())
            availability_rows.append(batch["availability"][valid].cpu().numpy())
            if output["fusion_weights"] is not None:
                fusion_weight_rows.append(output["fusion_weights"][valid].cpu().numpy())
            if output["reliability"] is not None:
                reliability_rows.append(output["reliability"][valid].cpu().numpy())
            if output.get("utility_scores") is not None:
                utility_rows.append(output["utility_scores"][valid].cpu().numpy())
            if output.get("system_reliability") is not None:
                system_reliability_rows.append(
                    output["system_reliability"][valid].cpu().numpy()
                )
            if output.get("abstain") is not None:
                abstain_rows.append(output["abstain"][valid].cpu().numpy())
            valid_cpu = valid.cpu().numpy().astype(bool)
            users.extend(
                user
                for user, keep in zip(batch["user_id"], valid_cpu, strict=True)
                if keep
            )
            timestamps.append(batch["timestamp"][valid].cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    if not probabilities:
        raise ValueError("Model has no valid samples for this split")
    result = {
        "probabilities": np.concatenate(probabilities),
        "targets": np.concatenate(targets),
        "target_mask": np.concatenate(target_masks),
        "availability": np.concatenate(availability_rows),
        "user_id": users,
        "timestamp": np.concatenate(timestamps),
        "fusion_weights": (
            np.concatenate(fusion_weight_rows) if fusion_weight_rows else None
        ),
        "reliability": np.concatenate(reliability_rows) if reliability_rows else None,
        "utility_scores": np.concatenate(utility_rows) if utility_rows else None,
        "system_reliability": (
            np.concatenate(system_reliability_rows) if system_reliability_rows else None
        ),
        "abstain": np.concatenate(abstain_rows) if abstain_rows else None,
    }
    result["elapsed_seconds"] = elapsed
    result["latency_ms_per_sample"] = elapsed * 1000.0 / len(result["timestamp"])
    return result


def _checkpoint_contract(
    model_name: str,
    processed_manifest: dict[str, Any],
    processed_manifest_path: Path,
    preprocessor_path: Path,
    labels: list[str],
    modality_dims: dict[str, int],
) -> dict[str, Any]:
    return {
        "version": 1,
        "phase": 2,
        "model_name": model_name,
        "processed_manifest_sha256": sha256_file(processed_manifest_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "source_schema_sha256": processed_manifest["source_schema_sha256"],
        "labels": labels,
        "modality_dims": modality_dims,
    }


def _verify_checkpoint_contract(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    mismatches = {
        key: (expected.get(key), actual.get(key))
        for key in expected
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise ValueError(f"Checkpoint artifact contract mismatch: {mismatches}")


def _save_predictions(
    path: Path,
    collected: dict[str, Any],
    labels: list[str],
    thresholds: np.ndarray,
    run_id: str,
    fold: int,
) -> None:
    rows: list[pd.DataFrame] = []
    for label_index, label in enumerate(labels):
        known = collected["target_mask"][:, label_index].astype(bool)
        target = collected["targets"][:, label_index]
        probability = collected["probabilities"][:, label_index]
        rows.append(
            pd.DataFrame(
                {
                    "run_id": run_id,
                    "fold": fold,
                    "user_id": collected["user_id"],
                    "timestamp": collected["timestamp"],
                    "scenario": "natural_missingness",
                    "label": label,
                    "target": np.where(known, target, np.nan),
                    "target_known": known,
                    "probability": probability,
                    "prediction": probability >= thresholds[label_index],
                    "available_modality_count": collected["availability"].sum(axis=1),
                }
            )
        )
    pd.concat(rows, ignore_index=True).to_parquet(path, index=False)


def train_phase2(
    project_root: str | Path,
    data_config_path: str | Path,
    model_name: str,
    fold: int,
    seed: int,
    profile_name: str,
) -> Path:
    started_at = datetime.now(UTC).isoformat()
    root = Path(project_root).resolve()
    data_config_file = Path(data_config_path)
    if not data_config_file.is_absolute():
        data_config_file = root / data_config_file
    data_config = load_config(data_config_file)
    profile = load_config(root / f"configs/experiment/{profile_name}.yaml")
    model_config_path = _model_config_path(root, model_name)
    model_config = load_config(model_config_path)
    processed_dir = (
        root
        / data_config.get("processed_dir", "data/processed")
        / "extrasensory"
        / f"fold{fold}"
    )
    processed_manifest_path = processed_dir / "processed_manifest.json"
    if not processed_manifest_path.is_file():
        raise FileNotFoundError(
            f"Prepared fold not found: {processed_manifest_path}; run robustsense prepare first"
        )
    processed_manifest = read_json(processed_manifest_path)
    metadata_path = processed_manifest_path
    train_dataset = MultiModalDataset(processed_dir / "train.npz", metadata_path)
    val_dataset = MultiModalDataset(processed_dir / "val.npz", metadata_path)
    labels = list(processed_manifest["labels"])
    modality_slices = {
        name: tuple(processed_manifest["modality_slices"][name]) for name in MODALITIES
    }
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    device = _device_from_profile(profile)
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_phase2_model(model_name, modality_dims, len(labels), model_config).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(profile.get("learning_rate", 0.001)),
        weight_decay=float(profile.get("weight_decay", 0.0001)),
    )
    if model_name.startswith("single-"):
        modality_index = MODALITIES.index(model_name.removeprefix("single-"))
        class_weight_sample_mask = train_dataset.availability[:, modality_index]
    else:
        class_weight_sample_mask = train_dataset.availability.any(axis=1)
    class_weight_target_mask = train_dataset.target_mask & class_weight_sample_mask[:, None]
    pos_weight = training_pos_weight(
        torch.from_numpy(train_dataset.targets),
        torch.from_numpy(class_weight_target_mask),
        cap=float(profile.get("pos_weight_cap", 20.0)),
    ).to(device)
    batch_size = int(profile.get("batch_size", 512))
    train_loader = _loader(train_dataset, batch_size, True, seed, device)
    val_loader = _loader(val_dataset, batch_size, False, seed, device)
    epochs = int(profile.get("epochs", 1))
    patience = int(profile.get("early_stopping_patience", max(1, epochs)))
    best_metric = float("-inf")
    epochs_without_improvement = 0
    train_log: list[dict[str, Any]] = []
    run_id = f"extrasensory-{model_name}-fold{fold}-seed{seed}-{profile_name}"
    run_dir = root / profile.get("run_root", "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    preprocessor_path = processed_dir / "preprocessor.json"
    contract = _checkpoint_contract(
        model_name,
        processed_manifest,
        processed_manifest_path,
        preprocessor_path,
        labels,
        modality_dims,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        known_entries = 0
        started = time.perf_counter()
        for batch in train_loader:
            batch = _move_batch(batch, device, modality_slices)
            valid = model.valid_sample_mask(batch)
            effective_mask = batch["target_mask"] & valid.unsqueeze(1)
            if not torch.any(effective_mask):
                continue
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss = masked_weighted_bce_with_logits(
                output["logits"], batch["targets"], effective_mask, pos_weight
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), float(profile.get("gradient_clip_norm", 5.0)))
            optimizer.step()
            entries = int(effective_mask.sum().item())
            epoch_loss += float(loss.item()) * entries
            known_entries += entries

        validation = _collect_predictions(model, val_loader, device, modality_slices)
        validation_fixed = phase2_multilabel_metrics(
            validation["probabilities"],
            validation["targets"],
            validation["target_mask"],
            labels,
            0.5,
        )
        metric = float(validation_fixed["macro_f1"] or 0.0)
        train_log.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / known_entries,
                "val_macro_f1_at_0_5": metric,
                "epoch_seconds": time.perf_counter() - started,
            }
        )
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
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=train_log[0].keys())
        writer.writeheader()
        writer.writerows(train_log)

    split_manifest = read_json(root / "data/manifests" / f"extrasensory_split_fold{fold}.json")
    data_manifest = read_json(root / "data/manifests/extrasensory_data_manifest.json")
    write_json(run_dir / "split_manifest.json", split_manifest)
    write_json(run_dir / "data_manifest.json", data_manifest)
    resolved = {
        "phase": 2,
        "dataset": "extrasensory",
        "data_config": str(data_config_file.relative_to(root)),
        "model_name": model_name,
        "model_config": model_config,
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
    }
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
    evaluate_phase2(run_dir, suite="natural")
    return run_dir


def evaluate_phase2(run_dir: str | Path, suite: str = "natural") -> dict[str, object]:
    run_path = Path(run_dir).resolve()
    resolved = load_config(run_path / "resolved_config.yaml")
    if resolved.get("phase") != 2:
        raise ValueError("Not a Phase-2 run")
    root = run_path.parents[1]
    processed_dir = root / resolved["processed_dir"]
    processed_manifest_path = processed_dir / "processed_manifest.json"
    processed_manifest = read_json(processed_manifest_path)
    labels = list(processed_manifest["labels"])
    modality_slices = {
        name: tuple(processed_manifest["modality_slices"][name]) for name in MODALITIES
    }
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    model = build_phase2_model(
        resolved["model_name"], modality_dims, len(labels), resolved["model_config"]
    )
    device = _device_from_profile({"device": resolved["device"]})
    model = model.to(device)
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    expected_contract = _checkpoint_contract(
        resolved["model_name"],
        processed_manifest,
        processed_manifest_path,
        processed_dir / "preprocessor.json",
        labels,
        modality_dims,
    )
    _verify_checkpoint_contract(expected_contract, checkpoint["contract"])
    _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    test_dataset = MultiModalDataset(
        processed_dir / "test.npz", processed_manifest_path
    )
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
            "dev_only": resolved["profile"] == "dev",
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
    (run_path / "run_summary.md").write_text(
        "# Phase-2 ExtraSensory dev run\n\n"
        f"- Run: `{resolved['run_id']}`\n"
        f"- Model: `{resolved['model_name']}`\n"
        f"- Fold: {resolved['fold']}\n"
        f"- Natural-missingness samples: {metrics['sample_count']}\n"
        f"- Parameters: {metrics['parameter_count']}\n"
        f"- Macro-F1: {metrics['macro_f1']:.6f}\n"
        f"- Micro-F1: {metrics['micro_f1']:.6f}\n"
        f"- mAP: {metrics['mean_average_precision']:.6f}\n\n"
        "This is a one-fold, one-seed, short dev run for engineering validation only. "
        "It is not a research conclusion.\n",
        encoding="utf-8",
    )
    return metrics
