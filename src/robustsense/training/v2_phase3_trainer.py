"""Leakage-safe formal training and natural-missingness evaluation for V2-3."""

from __future__ import annotations

import csv
import hashlib
import platform
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from robustsense.constants import MODALITIES
from robustsense.corruption.vectorized import VectorizedTrainingCorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.data.extrasensory import sha256_file
from robustsense.evaluation.v2_metrics import risk_coverage_curve
from robustsense.models.torch_baselines import parameter_count
from robustsense.models.v2_phase3 import V2_PHASE3_VARIANTS, build_v2_phase3_model
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.phase3_losses import (
    consistency_mse,
    reliability_mse,
    reliability_ranking_loss,
)
from robustsense.training.selective import tune_abstention_threshold
from robustsense.training.thresholds import tune_per_label_thresholds
from robustsense.training.torch_losses import (
    masked_weighted_bce_with_logits,
    training_pos_weight,
)
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

VARIANT_SLUGS = {
    "P2": "p2",
    "P2-A1": "p2-a1-utility-only",
    "P2-A2": "p2-a2-no-ranking",
    "P2-A3": "p2-a3-no-quality",
    "P2-A4": "p2-a4-forced-only",
}


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def v2_phase3_run_id(variant: str, fold: int, seed: int) -> str:
    if variant not in V2_PHASE3_VARIANTS:
        raise ValueError(f"Unknown V2-3 variant: {variant}")
    return f"extrasensory-{VARIANT_SLUGS[variant]}-fold{fold}-seed{seed}-v2_seed13"


def _project_paths(root: Path, plan_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    return load_config(path), path


def _checkpoint_contract(
    *,
    variant: str,
    protocol_sha256: str,
    manifest: dict[str, Any],
    manifest_path: Path,
    preprocessor_path: Path,
    labels: list[str],
    modality_dims: dict[str, int],
) -> dict[str, Any]:
    return {
        "version": 3,
        "phase": "V2-3",
        "variant": variant,
        "protocol_sha256": protocol_sha256,
        "processed_manifest_sha256": sha256_file(manifest_path),
        "preprocessor_sha256": sha256_file(preprocessor_path),
        "source_schema_sha256": manifest["source_schema_sha256"],
        "labels": labels,
        "modality_dims": modality_dims,
    }


def _root_from_run(run_path: Path, resolved: dict[str, Any]) -> Path:
    for candidate in run_path.parents:
        if (candidate / resolved["processed_dir"] / "processed_manifest.json").is_file():
            return candidate
    raise ValueError(f"Cannot locate project root for V2-3 run: {run_path}")


def _variant_policy(variant: str, model_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "reliability_loss_weight": float(model_config["reliability_loss_weight"]),
        "consistency_loss_weight": float(model_config["consistency_loss_weight"]),
        "ranking_loss_weight": (
            0.0 if variant == "P2-A2" else float(model_config["ranking_loss_weight"])
        ),
        "ranking_margin": float(model_config["ranking_margin"]),
        "selective_prediction": variant != "P2-A4",
        "corruption_implementation": "sample_stable_vectorized_batch_writes",
    }


def train_v2_phase3(
    project_root: str | Path,
    *,
    variant: str,
    fold: int,
    seed: int,
    protocol_sha256: str,
    plan_path: str | Path = "configs/v2/phase_v2_3_plan.yaml",
) -> Path:
    """Train with train/validation only. The test archive is never constructed here."""
    if variant == "P2-A4":
        raise ValueError("P2-A4 must be derived from its matching P2 checkpoint")
    if variant not in V2_PHASE3_VARIANTS:
        raise ValueError(f"Unknown V2-3 variant: {variant}")
    started_at = datetime.now(UTC).isoformat()
    root = Path(project_root).resolve()
    plan, plan_file = _project_paths(root, plan_path)
    if fold not in plan["folds"] or seed not in plan["seeds"]:
        raise ValueError("Fold or seed is outside the frozen V2-3 plan")
    profile = load_config(root / plan["experiment_config"])
    model_config = load_config(root / plan["model_config"])
    corruption_config = load_config(root / plan["corruption_config"])
    policy = _variant_policy(variant, model_config)
    processed_dir = root / "data/processed/extrasensory" / f"fold{fold}"
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    train_dataset = MultiModalDataset(processed_dir / "train.npz", manifest_path)
    val_dataset = MultiModalDataset(processed_dir / "val.npz", manifest_path)
    labels = list(manifest["labels"])
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    preprocessor_path = processed_dir / "preprocessor.json"
    preprocessor = read_json(preprocessor_path)
    registry = VectorizedTrainingCorruptionRegistry(
        corruption_config,
        modality_slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    device = _device_from_profile(profile)
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = build_v2_phase3_model(variant, modality_dims, len(labels), model_config).to(device)
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(profile["learning_rate"]),
        weight_decay=float(profile["weight_decay"]),
    )
    class_weight_sample_mask = train_dataset.availability.any(axis=1)
    pos_weight = training_pos_weight(
        torch.from_numpy(train_dataset.targets),
        torch.from_numpy(train_dataset.target_mask & class_weight_sample_mask[:, None]),
        cap=float(profile["pos_weight_cap"]),
    ).to(device)
    batch_size = int(profile["batch_size"])
    train_loader = _loader(train_dataset, batch_size, True, seed, device)
    val_loader = _loader(val_dataset, batch_size, False, seed, device)
    run_id = v2_phase3_run_id(variant, fold, seed)
    run_dir = root / profile["run_root"] / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    contract = _checkpoint_contract(
        variant=variant,
        protocol_sha256=protocol_sha256,
        manifest=manifest,
        manifest_path=manifest_path,
        preprocessor_path=preprocessor_path,
        labels=labels,
        modality_dims=modality_dims,
    )
    best_metric = float("-inf")
    epochs_without_improvement = 0
    train_log: list[dict[str, Any]] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(profile["epochs"]) + 1):
        model.train()
        loss_names = (
            "total",
            "classification",
            "reliability",
            "consistency",
            "ranking",
        )
        sums = {name: 0.0 for name in loss_names}
        batch_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            corrupted_cpu, fault_metadata = registry.apply_training(
                batch, seed=seed, epoch=epoch, view="train"
            )
            clean_batch = _move_batch(batch, device, modality_slices)
            train_batch = _move_batch(corrupted_cpu, device, modality_slices)
            reliability_target = fault_metadata["reliability_target"].to(
                device, non_blocking=True
            )
            artificial_fault = fault_metadata["artificial_fault"].to(
                device, non_blocking=True
            )
            valid = model.valid_sample_mask(train_batch)
            effective_mask = train_batch["target_mask"] & valid.unsqueeze(1)
            if not torch.any(effective_mask):
                continue
            model.eval()
            clean_output = model(clean_batch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(train_batch)
            classification = masked_weighted_bce_with_logits(
                output["logits"], train_batch["targets"], effective_mask, pos_weight
            )
            reliability = reliability_mse(output["reliability"], reliability_target)
            consistency = consistency_mse(
                clean_output["logits"], output["logits"], effective_mask
            )
            ranking = classification.new_zeros(())
            if policy["ranking_loss_weight"] > 0:
                ranking = reliability_ranking_loss(
                    clean_output["reliability"],
                    output["reliability"],
                    artificial_fault,
                    margin=policy["ranking_margin"],
                )
            loss = (
                classification
                + policy["reliability_loss_weight"] * reliability
                + policy["consistency_loss_weight"] * consistency
                + policy["ranking_loss_weight"] * ranking
            )
            loss.backward()
            clip_grad_norm_(model.parameters(), float(profile["gradient_clip_norm"]))
            optimizer.step()
            for name, value in (
                ("total", loss),
                ("classification", classification),
                ("reliability", reliability),
                ("consistency", consistency),
                ("ranking", ranking),
            ):
                sums[name] += float(value.detach())
            batch_count += 1
        if not batch_count:
            raise RuntimeError("No train batch contained a known target")
        validation = _collect_predictions(model, val_loader, device, modality_slices)
        validation_fixed = phase2_multilabel_metrics(
            validation["probabilities"],
            validation["targets"],
            validation["target_mask"],
            labels,
            0.5,
        )
        metric = float(validation_fixed["macro_f1"] or 0.0)
        epoch_seconds = time.perf_counter() - epoch_started
        train_log.append(
            {
                "epoch": epoch,
                "train_total_loss": sums["total"] / batch_count,
                "train_classification_loss": sums["classification"] / batch_count,
                "train_reliability_loss": sums["reliability"] / batch_count,
                "train_consistency_loss": sums["consistency"] / batch_count,
                "train_ranking_loss": sums["ranking"] / batch_count,
                "val_macro_f1_at_0_5": metric,
                "epoch_seconds": epoch_seconds,
                "train_samples_per_second": len(train_dataset) / epoch_seconds,
            }
        )
        if metric > best_metric:
            best_metric = metric
            epochs_without_improvement = 0
            torch.save({"state_dict": model.state_dict(), "contract": contract}, checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(profile["early_stopping_patience"]):
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _verify_checkpoint_contract(contract, checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    validation = _collect_predictions(model, val_loader, device, modality_slices)
    thresholds = tune_per_label_thresholds(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        profile["threshold_grid"],
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
            "grid": profile["threshold_grid"],
            "labels": labels,
            "values": thresholds.tolist(),
        },
    )
    write_json(run_dir / "val_metrics.json", val_metrics)
    abstention = tune_abstention_threshold(
        validation["system_reliability"],
        target_coverage=float(model_config["target_coverage"]),
        source_split="val",
    )
    write_json(run_dir / "abstention_threshold.json", abstention)
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
        "phase": "V2-3",
        "variant": variant,
        "profile": profile["profile"],
        "fold": fold,
        "seed": seed,
        "run_id": run_id,
        "run_dir": str(run_dir.relative_to(root)),
        "processed_dir": str(processed_dir.relative_to(root)),
        "plan": str(plan_file.relative_to(root)),
        "protocol_sha256": protocol_sha256,
        "model_config": model_config,
        "corruption_config": corruption_config,
        "training_policy": policy,
        "experiment": profile,
        "device": str(device),
        "checkpoint_contract": contract,
        "parameter_count": parameter_count(model),
        "learned_beta": (
            float(model.beta.detach().cpu()) if model.use_reliability_constraint else None
        ),
        "class_weight_source_split": "train",
        "class_weight_sample_count": int(class_weight_sample_mask.sum()),
        "pos_weight": pos_weight.detach().cpu().tolist(),
        "threshold_source_split": "val",
        "abstention_source_split": "val",
        "test_split_opened_during_training": False,
    }
    dump_config(resolved, run_dir / "resolved_config.yaml")
    peak_memory = (
        torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
    )
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
            "cuda_peak_allocated_bytes": int(peak_memory),
            "git_commit": _git_commit(root),
            "seed": seed,
        },
    )
    write_json(
        run_dir / "run_manifest.json",
        {
            "run_id": run_id,
            "variant": variant,
            "fold": fold,
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "status": "trained_validation_frozen_test_not_opened",
            "epochs_completed": len(train_log),
            "best_val_macro_f1_at_0_5": best_metric,
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": state_dict_sha256(checkpoint["state_dict"]),
        },
    )
    (run_dir / "run_summary.md").write_text(
        "# V2-3 训练摘要\n\n"
        f"- Run：`{run_id}`\n"
        f"- 变体：`{variant}`\n"
        f"- Fold：{fold}\n"
        f"- Seed：{seed}\n"
        f"- 完成 epoch：{len(train_log)}\n"
        f"- 最佳 validation Macro-F1@0.5：{best_metric:.6f}\n\n"
        "训练阶段未打开 test；分类阈值和拒绝阈值均来自 validation。\n",
        encoding="utf-8",
    )
    return run_dir


def derive_p2_a4_run(
    project_root: str | Path,
    *,
    fold: int,
    seed: int,
    protocol_sha256: str,
    plan_path: str | Path = "configs/v2/phase_v2_3_plan.yaml",
) -> Path:
    """Materialize inference-only A4 from the predetermined matching P2 checkpoint."""
    root = Path(project_root).resolve()
    plan, plan_file = _project_paths(root, plan_path)
    profile = load_config(root / plan["experiment_config"])
    source_dir = root / profile["run_root"] / v2_phase3_run_id("P2", fold, seed)
    target_dir = root / profile["run_root"] / v2_phase3_run_id("P2-A4", fold, seed)
    required = (
        "best_checkpoint.pt",
        "resolved_config.yaml",
        "thresholds.json",
        "val_metrics.json",
        "train_log.csv",
        "split_manifest.json",
        "data_manifest.json",
    )
    missing = [name for name in required if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Matching P2 source is incomplete: {missing}")
    source_resolved = load_config(source_dir / "resolved_config.yaml")
    source_checkpoint = torch.load(
        source_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    source_state_sha256 = state_dict_sha256(source_checkpoint["state_dict"])
    processed_dir = root / source_resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    labels = list(manifest["labels"])
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    target_contract = _checkpoint_contract(
        variant="P2-A4",
        protocol_sha256=protocol_sha256,
        manifest=manifest,
        manifest_path=manifest_path,
        preprocessor_path=processed_dir / "preprocessor.json",
        labels=labels,
        modality_dims=modality_dims,
    )
    target_dir.mkdir(parents=True, exist_ok=True)
    for name in required[2:]:
        shutil.copy2(source_dir / name, target_dir / name)
    torch.save(
        {"state_dict": source_checkpoint["state_dict"], "contract": target_contract},
        target_dir / "best_checkpoint.pt",
    )
    target_run_id = v2_phase3_run_id("P2-A4", fold, seed)
    resolved = {
        **source_resolved,
        "variant": "P2-A4",
        "run_id": target_run_id,
        "run_dir": str(target_dir.relative_to(root)),
        "plan": str(plan_file.relative_to(root)),
        "protocol_sha256": protocol_sha256,
        "checkpoint_contract": target_contract,
        "training_policy": _variant_policy("P2-A4", source_resolved["model_config"]),
        "derived_from_run_id": source_resolved["run_id"],
        "derived_from_checkpoint_sha256": sha256_file(source_dir / "best_checkpoint.pt"),
        "abstention_source_split": None,
        "test_split_opened_during_training": False,
    }
    dump_config(resolved, target_dir / "resolved_config.yaml")
    write_json(
        target_dir / "environment.json",
        {
            "derived_at": datetime.now(UTC).isoformat(),
            "source_run_id": source_resolved["run_id"],
            "source_checkpoint_sha256": sha256_file(source_dir / "best_checkpoint.pt"),
            "git_commit": _git_commit(root),
        },
    )
    write_json(
        target_dir / "run_manifest.json",
        {
            "run_id": target_run_id,
            "variant": "P2-A4",
            "fold": fold,
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "status": "derived_validation_frozen_test_not_opened",
            "derived_from_run_id": source_resolved["run_id"],
            "checkpoint_sha256": sha256_file(target_dir / "best_checkpoint.pt"),
            "state_dict_sha256": source_state_sha256,
            "source_state_dict_sha256": source_state_sha256,
        },
    )
    (target_dir / "run_summary.md").write_text(
        "# V2-3 P2-A4 派生摘要\n\n"
        f"- Run：`{target_run_id}`\n"
        f"- 源 Run：`{source_resolved['run_id']}`\n"
        "- 改动：移除推理时选择性拒绝，只报告强制预测。\n"
        "- 训练参数：与同 fold/seed 的 P2 完全相同，因此不重复训练。\n"
        "- 当前状态：test 尚未打开。\n",
        encoding="utf-8",
    )
    return target_dir


def evaluate_v2_phase3(run_dir: str | Path) -> dict[str, Any]:
    run_path = Path(run_dir).resolve()
    resolved = load_config(run_path / "resolved_config.yaml")
    if resolved.get("phase") != "V2-3":
        raise ValueError("Not a V2-3 run")
    root = _root_from_run(run_path, resolved)
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    labels = list(manifest["labels"])
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    model = build_v2_phase3_model(
        resolved["variant"], modality_dims, len(labels), resolved["model_config"]
    )
    device = _device_from_profile({"device": resolved["device"]})
    model = model.to(device)
    checkpoint = torch.load(
        run_path / "best_checkpoint.pt", map_location=device, weights_only=False
    )
    _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"])
    abstention = None
    if resolved["variant"] != "P2-A4":
        abstention = read_json(run_path / "abstention_threshold.json")
        if abstention.get("source_split") != "val":
            raise ValueError("Abstention threshold must be validation-derived")
        model.set_abstention_threshold(float(abstention["threshold"]))
    test_dataset = MultiModalDataset(processed_dir / "test.npz", manifest_path)
    loader = _loader(
        test_dataset,
        int(resolved["experiment"]["batch_size"]),
        False,
        int(resolved["seed"]),
        device,
    )
    collected = _collect_predictions(model, loader, device, modality_slices)
    threshold_artifact = read_json(run_path / "thresholds.json")
    if threshold_artifact.get("source_split") != "val" or threshold_artifact.get(
        "labels"
    ) != labels:
        raise ValueError("Classification thresholds must be validation-derived")
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float32)
    forced = phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        thresholds,
    )
    forced["fixed_0_5"] = phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        0.5,
    )
    selective = None
    risk_curve = None
    if abstention is not None:
        accepted = ~collected["abstain"].astype(bool)
        accepted_metrics = phase2_multilabel_metrics(
            collected["probabilities"][accepted],
            collected["targets"][accepted],
            collected["target_mask"][accepted],
            labels,
            thresholds,
        )
        selective = {
            "threshold_source_split": "val",
            "threshold": abstention["threshold"],
            "target_coverage": abstention["target_coverage"],
            "test_coverage": float(accepted.mean()),
            "accepted_sample_count": int(accepted.sum()),
            "rejected_sample_count": int((~accepted).sum()),
            "accepted_metrics": accepted_metrics,
        }
        risk_curve = risk_coverage_curve(
            collected["probabilities"],
            collected["targets"],
            collected["target_mask"],
            collected["system_reliability"],
            labels=labels,
            thresholds=thresholds,
            coverage_grid=np.asarray(
                resolved["experiment"]["risk_coverage_grid"], dtype=np.float64
            ),
            run_id=resolved["run_id"],
        )
        write_json(run_path / "risk_coverage.json", risk_curve)
    result = {
        "phase": "V2-3",
        "run_id": resolved["run_id"],
        "variant": resolved["variant"],
        "fold": int(resolved["fold"]),
        "seed": int(resolved["seed"]),
        "scenario": "natural_missingness",
        "protocol_sha256": resolved["protocol_sha256"],
        "threshold_source_split": "val",
        "abstention_source_split": "val" if abstention is not None else None,
        "forced": forced,
        "selective": selective,
        "parameter_count": int(resolved["parameter_count"]),
        "latency_ms_per_sample": collected["latency_ms_per_sample"],
    }
    write_json(run_path / "test_metrics.json", result)
    _save_predictions(
        run_path / "predictions.parquet",
        collected,
        labels,
        thresholds,
        resolved["run_id"],
        int(resolved["fold"]),
    )
    evaluation_manifest = {
        "phase": "V2-3",
        "run_id": resolved["run_id"],
        "variant": resolved["variant"],
        "fold": int(resolved["fold"]),
        "seed": int(resolved["seed"]),
        "profile": resolved["profile"],
        "protocol_sha256": resolved["protocol_sha256"],
        "processed_manifest_sha256": sha256_file(manifest_path),
        "checkpoint_sha256": sha256_file(run_path / "best_checkpoint.pt"),
        "threshold_artifact_sha256": sha256_file(run_path / "thresholds.json"),
        "abstention_artifact_sha256": (
            sha256_file(run_path / "abstention_threshold.json")
            if abstention is not None
            else None
        ),
        "test_sample_count": len(test_dataset),
        "outputs": [
            "test_metrics.json",
            "predictions.parquet",
            *( ["risk_coverage.json"] if risk_curve is not None else [] ),
        ],
    }
    write_json(run_path / "evaluation_manifest.json", evaluation_manifest)
    write_json(
        run_path / "run_manifest.json",
        {
            **read_json(run_path / "run_manifest.json"),
            "status": "success_test_evaluated_once",
            "evaluation_manifest_sha256": sha256_file(
                run_path / "evaluation_manifest.json"
            ),
        },
    )
    with (run_path / "run_summary.md").open("a", encoding="utf-8") as stream:
        stream.write(
            "\n## 冻结测试评估\n\n"
            f"- 自然缺失 Macro-F1：{forced['macro_f1']:.6f}\n"
            f"- 自然缺失 Micro-F1：{forced['micro_f1']:.6f}\n"
            f"- 样本数：{forced['sample_count']}\n"
        )
        if selective is None:
            stream.write("- 选择性预测：按 A4 协议移除。\n")
        else:
            stream.write(f"- 测试覆盖率：{selective['test_coverage']:.6f}\n")
    return result
