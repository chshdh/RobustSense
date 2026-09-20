"""训练、评估并汇总 V4-1 标签条件可靠融合模型。"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import run_v3_phase1 as v31
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from robustsense.constants import MODALITIES
from robustsense.corruption.vectorized import VectorizedTrainingCorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.evaluation.v2_metrics import masked_binary_cross_entropy_per_sample
from robustsense.experiments.v2_phase0 import sha256_file, verify_file_entries
from robustsense.models.torch_baselines import parameter_count
from robustsense.models.v2_phase3 import build_v2_phase3_model
from robustsense.models.v4_label_fusion import (
    build_label_conditioned_model,
    initialize_label_conditioned_from_p2,
)
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.thresholds import tune_per_label_thresholds
from robustsense.training.torch_losses import (
    masked_weighted_bce_with_logits,
    training_pos_weight,
)
from robustsense.training.trainer import (
    _collect_predictions,
    _device_from_profile,
    _loader,
    _move_batch,
    _save_predictions,
)
from robustsense.training.v2_phase3_trainer import state_dict_sha256
from robustsense.utils.config import dump_config, load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed


def _load_governance(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    plan = load_config(root / "configs/v4/phase_v4_1_plan.yaml")
    lock = read_json(root / "reports/v4/phase_v4_1_protocol_lock.json")
    if lock.get("status") != "frozen_before_first_candidate_test_evaluation":
        raise RuntimeError("V4-1 协议尚未冻结")
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise RuntimeError(f"V4-1 协议锁失效：{problems}")
    return plan, lock


def _p2_registry(
    root: Path, plan: dict[str, Any]
) -> dict[tuple[int, int], dict[str, str]]:
    with (root / plan["baseline_registry"]).open(
        encoding="utf-8-sig", newline=""
    ) as stream:
        rows = list(csv.DictReader(stream))
    mapping = {
        (int(row["fold"]), int(row["seed"])): row
        for row in rows
        if row["model_name"] == "P2" and row["status"] == "success"
    }
    expected = len(plan["folds"]) * len(plan["seeds"])
    if len(mapping) != expected:
        raise RuntimeError(f"V4-1 P2 来源只有 {len(mapping)}/{expected} 个单元")
    return mapping


def _run_dir(root: Path, plan: dict[str, Any], fold: int, seed: int) -> Path:
    return root / plan["run_root"] / f"extrasensory-p2-lc-fold{fold}-seed{seed}-v4_final"


def _checkpoint_contract(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    manifest_path: Path,
    labels: list[str],
    modality_dims: dict[str, int],
    source_checkpoint: Path,
) -> dict[str, Any]:
    return {
        "version": 1,
        "phase": "V4-1",
        "model_id": "P2-LC",
        "protocol_sha256": lock["protocol_sha256"],
        "processed_manifest_sha256": sha256_file(manifest_path),
        "model_config_sha256": sha256_file(root / plan["model_config"]),
        "source_p2_checkpoint_sha256": sha256_file(source_checkpoint),
        "labels": labels,
        "modality_dims": modality_dims,
    }


def _assert_contract(expected: dict[str, Any], actual: dict[str, Any]) -> None:
    if expected != actual:
        differences = {
            key: (expected.get(key), actual.get(key))
            for key in set(expected) | set(actual)
            if expected.get(key) != actual.get(key)
        }
        raise RuntimeError(f"V4-1 检查点合同不一致：{differences}")


def _source_context(
    root: Path,
    plan: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
    fold: int,
    seed: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    row = registry[(fold, seed)]
    source_dir = root / row["run_dir"]
    resolved = load_config(source_dir / "resolved_config.yaml")
    if resolved.get("variant") != "P2":
        raise RuntimeError("V4-1 来源检查点不是 P2")
    checkpoint = torch.load(
        source_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    if row["protocol_sha256"] != resolved.get("protocol_sha256"):
        raise RuntimeError("V4-1 P2 登记表和来源配置协议不一致")
    return source_dir, resolved, checkpoint


def _routing_specialization(
    model: torch.nn.Module,
    probe_batch: dict[str, Any],
) -> dict[str, Any]:
    model.eval()
    with torch.inference_mode():
        output = model(probe_batch)
    weights = output["label_fusion_weights"].detach().cpu().numpy()
    centered = np.abs(weights - weights.mean(axis=1, keepdims=True))
    per_modality = centered.mean(axis=(0, 1))
    return {
        "mean_absolute_label_weight_deviation": float(centered.mean()),
        "maximum_absolute_label_weight_deviation": float(centered.max()),
        "per_modality_mean_absolute_label_weight_deviation": {
            name: float(per_modality[index])
            for index, name in enumerate(MODALITIES)
        },
    }


def train_unit(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
    *,
    fold: int,
    seed: int,
    force: bool,
) -> Path:
    run_dir = _run_dir(root, plan, fold, seed)
    manifest_path_out = run_dir / "run_manifest.json"
    if manifest_path_out.is_file() and not force:
        existing = read_json(manifest_path_out)
        if existing.get("status") in {
            "trained_validation_frozen_test_not_opened",
            "success_test_evaluated_once",
        } and existing.get("protocol_sha256") == lock["protocol_sha256"]:
            print(f"SKIP TRAIN fold={fold} seed={seed}: 已存在有效训练", flush=True)
            return run_dir
    source_dir, source_resolved, source_checkpoint = _source_context(
        root, plan, registry, fold, seed
    )
    source_checkpoint_path = source_dir / "best_checkpoint.pt"
    processed_dir = root / source_resolved["processed_dir"]
    data_manifest_path = processed_dir / "processed_manifest.json"
    data_manifest = read_json(data_manifest_path)
    labels = list(data_manifest["labels"])
    slices = {
        name: tuple(data_manifest["modality_slices"][name]) for name in MODALITIES
    }
    dimensions = {name: end - start for name, (start, end) in slices.items()}
    training = plan["training"]
    device = _device_from_profile({"device": training["device"]})
    set_global_seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    model_config = load_config(root / plan["model_config"])
    candidate = build_label_conditioned_model(
        dimensions, len(labels), model_config
    ).to(device)
    initialize_label_conditioned_from_p2(candidate, source_checkpoint["state_dict"])
    source_model = build_v2_phase3_model(
        "P2", dimensions, len(labels), source_resolved["model_config"]
    ).to(device)
    source_model.load_state_dict(source_checkpoint["state_dict"], strict=True)
    source_model.eval()
    candidate.eval()

    train_dataset = MultiModalDataset(processed_dir / "train.npz", data_manifest_path)
    val_dataset = MultiModalDataset(processed_dir / "val.npz", data_manifest_path)
    batch_size = int(training["batch_size"])
    val_loader = _loader(val_dataset, batch_size, False, seed, device)
    probe_batch = _move_batch(next(iter(val_loader)), device, slices)
    with torch.inference_mode():
        source_output = source_model(probe_batch)
        candidate_output = candidate(probe_batch)
    logit_delta = float(
        torch.max(torch.abs(source_output["logits"] - candidate_output["logits"])).cpu()
    )
    probability_delta = float(
        torch.max(
            torch.abs(
                torch.sigmoid(source_output["logits"])
                - torch.sigmoid(candidate_output["logits"])
            )
        ).cpu()
    )
    label_weights = candidate_output["label_fusion_weights"]
    source_weights = source_output["fusion_weights"][:, None, :].expand_as(label_weights)
    weight_delta = float(torch.max(torch.abs(label_weights - source_weights)).cpu())
    probability_tolerance = float(training["initial_probability_equivalence_atol"])
    weight_tolerance = float(training["initial_weight_equivalence_atol"])
    logit_tolerance = float(training["initial_logit_diagnostic_atol"])
    if (
        probability_delta > probability_tolerance
        or weight_delta > weight_tolerance
        or logit_delta > logit_tolerance
    ):
        raise RuntimeError(
            "V4-1 P2 等价初始化失败："
            f"probability={probability_delta}, logit={logit_delta}, "
            f"weight={weight_delta}"
        )
    del source_model, source_output, candidate_output, label_weights, source_weights

    candidate.freeze_p2_backbone_for_adapter_training()
    trainable_parameters = [
        parameter for parameter in candidate.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    sample_mask = train_dataset.availability.any(axis=1)
    pos_weight = training_pos_weight(
        torch.from_numpy(train_dataset.targets),
        torch.from_numpy(train_dataset.target_mask & sample_mask[:, None]),
        cap=float(training["pos_weight_cap"]),
    ).to(device)
    train_loader = _loader(train_dataset, batch_size, True, seed, device)
    preprocessor = read_json(processed_dir / "preprocessor.json")
    registry_corruption = VectorizedTrainingCorruptionRegistry(
        load_config(root / plan["corruption_config"]),
        slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    contract = _checkpoint_contract(
        root,
        plan,
        lock,
        data_manifest_path,
        labels,
        dimensions,
        source_checkpoint_path,
    )
    initial_validation = _collect_predictions(candidate, val_loader, device, slices)
    initial_metrics = phase2_multilabel_metrics(
        initial_validation["probabilities"],
        initial_validation["targets"],
        initial_validation["target_mask"],
        labels,
        0.5,
    )
    best_metric = float(initial_metrics["macro_f1"] or 0.0)
    torch.save(
        {"state_dict": candidate.state_dict(), "contract": contract, "best_epoch": 0},
        checkpoint_path,
    )
    rows = [
        {
            "epoch": 0,
            "train_classification_loss": np.nan,
            "val_macro_f1_at_0_5": best_metric,
            "epoch_seconds": 0.0,
            "train_samples_per_second": np.nan,
        }
    ]
    epochs_without_improvement = 0
    completed_epochs = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        candidate.train()
        loss_sum = 0.0
        batch_count = 0
        epoch_started = time.perf_counter()
        for batch in train_loader:
            corrupted, _ = registry_corruption.apply_training(
                batch, seed=seed, epoch=epoch, view="v4_label_routing"
            )
            moved = _move_batch(corrupted, device, slices)
            valid = candidate.valid_sample_mask(moved)
            effective_mask = moved["target_mask"] & valid.unsqueeze(1)
            if not torch.any(effective_mask):
                continue
            optimizer.zero_grad(set_to_none=True)
            output = candidate(moved)
            loss = masked_weighted_bce_with_logits(
                output["logits"], moved["targets"], effective_mask, pos_weight
            )
            loss.backward()
            clip_grad_norm_(trainable_parameters, float(training["gradient_clip_norm"]))
            optimizer.step()
            loss_sum += float(loss.detach())
            batch_count += 1
        if not batch_count:
            raise RuntimeError("V4-1 没有包含已知标签的训练批次")
        validation = _collect_predictions(candidate, val_loader, device, slices)
        metrics = phase2_multilabel_metrics(
            validation["probabilities"],
            validation["targets"],
            validation["target_mask"],
            labels,
            0.5,
        )
        metric = float(metrics["macro_f1"] or 0.0)
        elapsed = time.perf_counter() - epoch_started
        rows.append(
            {
                "epoch": epoch,
                "train_classification_loss": loss_sum / batch_count,
                "val_macro_f1_at_0_5": metric,
                "epoch_seconds": elapsed,
                "train_samples_per_second": len(train_dataset) / elapsed,
            }
        )
        completed_epochs = epoch
        if metric > best_metric:
            best_metric = metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": candidate.state_dict(),
                    "contract": contract,
                    "best_epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= int(training["early_stopping_patience"]):
                break
        print(
            f"TRAIN fold={fold} seed={seed} epoch={epoch} val={metric:.6f} "
            f"best={best_metric:.6f} sec={elapsed:.1f}",
            flush=True,
        )

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _assert_contract(contract, checkpoint["contract"])
    candidate.load_state_dict(checkpoint["state_dict"], strict=True)
    candidate.freeze_p2_backbone_for_adapter_training()
    validation = _collect_predictions(candidate, val_loader, device, slices)
    thresholds = tune_per_label_thresholds(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        plan["threshold_grid"],
        source_split="val",
    )
    validation_metrics = phase2_multilabel_metrics(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        labels,
        thresholds,
    )
    validation_metrics["fixed_0_5"] = phase2_multilabel_metrics(
        validation["probabilities"],
        validation["targets"],
        validation["target_mask"],
        labels,
        0.5,
    )
    pd.DataFrame(rows).to_csv(run_dir / "train_log.csv", index=False)
    write_json(
        run_dir / "thresholds.json",
        {
            "source_split": "val",
            "method": "per_label_f1",
            "grid": plan["threshold_grid"],
            "labels": labels,
            "values": thresholds.tolist(),
        },
    )
    write_json(run_dir / "val_metrics.json", v31._json_safe(validation_metrics))
    specialization = _routing_specialization(candidate, probe_batch)
    write_json(run_dir / "routing_specialization.json", specialization)
    initialization = {
        "source_run_id": source_resolved["run_id"],
        "source_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "probability_max_abs_delta": probability_delta,
        "logit_max_abs_delta": logit_delta,
        "label_weight_max_abs_delta": weight_delta,
        "probability_atol": probability_tolerance,
        "label_weight_atol": weight_tolerance,
        "logit_diagnostic_atol": logit_tolerance,
        "equivalent": True,
    }
    write_json(run_dir / "initialization.json", initialization)
    resolved = {
        "phase": "V4-1",
        "model_id": "P2-LC",
        "fold": fold,
        "seed": seed,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.relative_to(root)),
        "processed_dir": str(processed_dir.relative_to(root)),
        "protocol_sha256": lock["protocol_sha256"],
        "model_config": model_config,
        "corruption_config": load_config(root / plan["corruption_config"]),
        "training": training,
        "device": str(device),
        "checkpoint_contract": contract,
        "source_p2_run_dir": str(source_dir.relative_to(root)),
        "source_p2_run_id": source_resolved["run_id"],
        "parameter_count": parameter_count(candidate),
        "trainable_parameter_count": int(
            sum(value.numel() for value in trainable_parameters)
        ),
        "threshold_source_split": "val",
        "test_split_opened_during_training": False,
    }
    dump_config(resolved, run_dir / "resolved_config.yaml")
    manifest = {
        "version": 1,
        "phase": "V4-1",
        "status": "trained_validation_frozen_test_not_opened",
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "run_id": run_dir.name,
        "source_p2_run_id": source_resolved["run_id"],
        "source_p2_checkpoint_sha256": sha256_file(source_checkpoint_path),
        "initial_probability_max_abs_delta": probability_delta,
        "initial_logit_max_abs_delta": logit_delta,
        "initial_label_weight_max_abs_delta": weight_delta,
        "completed_epochs": completed_epochs,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_val_macro_f1_at_0_5": best_metric,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "state_dict_sha256": state_dict_sha256(checkpoint["state_dict"]),
        "test_split_opened_during_training": False,
    }
    write_json(manifest_path_out, manifest)
    (run_dir / "run_summary.md").write_text(
        "# V4-1 标签条件融合训练摘要\n\n"
        f"- Run：`{run_dir.name}`\n"
        f"- 来源 P2：`{source_resolved['run_id']}`\n"
        f"- Fold / seed：{fold} / {seed}\n"
        f"- 最佳 epoch：{checkpoint['best_epoch']}\n"
        f"- 最佳 validation Macro-F1@0.5：{best_metric:.6f}\n"
        f"- 标签路由平均偏离：{specialization['mean_absolute_label_weight_deviation']:.6f}\n\n"
        "训练阶段未打开 test；分类阈值仅来自 validation。\n",
        encoding="utf-8",
    )
    print(
        f"TRAINED fold={fold} seed={seed}: best_epoch={checkpoint['best_epoch']} "
        f"val={best_metric:.6f}",
        flush=True,
    )
    return run_dir


def evaluate_unit(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
    *,
    fold: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    run_dir = _run_dir(root, plan, fold, seed)
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise FileNotFoundError(f"V4-1 尚未训练：{run_dir}")
    run_manifest = read_json(run_manifest_path)
    if run_manifest.get("status") == "success_test_evaluated_once" and not force:
        print(f"SKIP EVAL fold={fold} seed={seed}: 已完成", flush=True)
        return read_json(run_dir / "test_metrics.json")
    if run_manifest.get("status") != "trained_validation_frozen_test_not_opened":
        raise RuntimeError("V4-1 训练单元状态不允许打开 test")
    resolved = load_config(run_dir / "resolved_config.yaml")
    processed_dir = root / resolved["processed_dir"]
    manifest_path = processed_dir / "processed_manifest.json"
    manifest = read_json(manifest_path)
    labels = list(manifest["labels"])
    slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    dimensions = {name: end - start for name, (start, end) in slices.items()}
    device = _device_from_profile({"device": resolved["device"]})
    model = build_label_conditioned_model(
        dimensions, len(labels), resolved["model_config"]
    ).to(device)
    checkpoint_path = run_dir / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    _assert_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    test_dataset = MultiModalDataset(processed_dir / "test.npz", manifest_path)
    test_loader = _loader(
        test_dataset, int(resolved["training"]["batch_size"]), False, seed, device
    )
    collected = _collect_predictions(model, test_loader, device, slices)
    threshold_artifact = read_json(run_dir / "thresholds.json")
    if (
        threshold_artifact.get("source_split") != "val"
        or threshold_artifact.get("labels") != labels
    ):
        raise RuntimeError("V4-1 分类阈值来源或标签顺序错误")
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float64)
    metrics = phase2_multilabel_metrics(
        collected["probabilities"],
        collected["targets"],
        collected["target_mask"],
        labels,
        thresholds,
    )
    metrics["masked_bce"] = float(
        np.nanmean(
            masked_binary_cross_entropy_per_sample(
                collected["probabilities"],
                collected["targets"],
                collected["target_mask"],
            )
        )
    )
    result = {
        "version": 1,
        "phase": "V4-1",
        "model_id": "P2-LC",
        "run_id": run_dir.name,
        "fold": fold,
        "seed": seed,
        "scenario": "natural_missingness",
        "protocol_sha256": lock["protocol_sha256"],
        "threshold_source_split": "val",
        "test_time_model_selection": False,
        "metrics": metrics,
        "parameter_count": int(resolved["parameter_count"]),
        "trainable_parameter_count": int(resolved["trainable_parameter_count"]),
        "latency_ms_per_sample": float(collected["latency_ms_per_sample"]),
    }
    write_json(run_dir / "test_metrics.json", v31._json_safe(result))
    _save_predictions(
        run_dir / "predictions.parquet",
        collected,
        labels,
        thresholds,
        run_dir.name,
        fold,
    )
    source_dir, _, _ = _source_context(root, plan, registry, fold, seed)
    evaluation_manifest = {
        "phase": "V4-1",
        "status": "success_test_evaluated_once",
        "protocol_sha256": lock["protocol_sha256"],
        "fold": fold,
        "seed": seed,
        "threshold_source_split": "val",
        "test_opened_by_separate_evaluation_process": True,
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "source_p2_checkpoint_sha256": sha256_file(
            source_dir / "best_checkpoint.pt"
        ),
        "test_metrics_sha256": sha256_file(run_dir / "test_metrics.json"),
        "predictions_sha256": sha256_file(run_dir / "predictions.parquet"),
        "test_sample_count": len(test_dataset),
    }
    write_json(run_dir / "evaluation_manifest.json", evaluation_manifest)
    run_manifest.update(
        {
            "status": "success_test_evaluated_once",
            "test_split_opened_during_training": False,
            "evaluation_manifest_sha256": sha256_file(
                run_dir / "evaluation_manifest.json"
            ),
        }
    )
    write_json(run_manifest_path, run_manifest)
    print(
        f"EVALUATED fold={fold} seed={seed}: "
        f"macro_f1={metrics['macro_f1']:.6f}",
        flush=True,
    )
    return result


def _baseline_metrics(root: Path, source_dir: Path) -> dict[str, float]:
    result = read_json(source_dir / "test_metrics.json")
    forced = result["forced"]
    artifact = v31._load_prediction_artifact(root, "P2", source_dir)
    masked_bce = float(
        np.nanmean(
            masked_binary_cross_entropy_per_sample(
                artifact.probabilities, artifact.targets, artifact.target_mask
            )
        )
    )
    return {
        "macro_f1": float(forced["macro_f1"]),
        "micro_f1": float(forced["micro_f1"]),
        "mean_average_precision": float(forced["mean_average_precision"]),
        "brier_score": float(forced["brier_score"]),
        "masked_bce": masked_bce,
        "parameter_count": float(result["parameter_count"]),
        "latency_ms_per_sample": float(result["latency_ms_per_sample"]),
    }


def aggregate_results(
    root: Path,
    plan: dict[str, Any],
    lock: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
) -> Path:
    rows = []
    for fold in plan["folds"]:
        for seed in plan["seeds"]:
            run_dir = _run_dir(root, plan, int(fold), int(seed))
            run_manifest = read_json(run_dir / "run_manifest.json")
            if (
                run_manifest.get("status") != "success_test_evaluated_once"
                or run_manifest.get("protocol_sha256") != lock["protocol_sha256"]
            ):
                raise RuntimeError(f"V4-1 单元未完成：fold={fold}, seed={seed}")
            candidate_result = read_json(run_dir / "test_metrics.json")
            candidate = candidate_result["metrics"]
            source_dir = root / registry[(int(fold), int(seed))]["run_dir"]
            baseline = _baseline_metrics(root, source_dir)
            row: dict[str, Any] = {
                "fold": int(fold),
                "seed": int(seed),
                "candidate_run_id": run_dir.name,
                "baseline_run_id": read_json(source_dir / "test_metrics.json")[
                    "run_id"
                ],
                "protocol_sha256": lock["protocol_sha256"],
            }
            for metric in (
                "macro_f1",
                "micro_f1",
                "mean_average_precision",
                "brier_score",
                "masked_bce",
                "parameter_count",
                "latency_ms_per_sample",
            ):
                candidate_value = float(
                    candidate_result[metric]
                    if metric in {"parameter_count", "latency_ms_per_sample"}
                    else candidate[metric]
                )
                baseline_value = baseline[metric]
                row[f"candidate_{metric}"] = candidate_value
                row[f"baseline_{metric}"] = baseline_value
                row[f"delta_{metric}"] = candidate_value - baseline_value
                if metric in {"parameter_count", "latency_ms_per_sample"}:
                    row[f"relative_delta_{metric}"] = (
                        candidate_value / baseline_value - 1.0
                    )
            rows.append(row)
    fold_seed = pd.DataFrame(rows).sort_values(["fold", "seed"])
    value_columns = [
        column
        for column in fold_seed.columns
        if column.startswith(("candidate_", "baseline_", "delta_", "relative_delta_"))
        and column not in {"candidate_run_id", "baseline_run_id"}
    ]
    fold_summary = fold_seed.groupby("fold", as_index=False)[value_columns].mean()
    fold_summary["seed_count"] = len(plan["seeds"])
    summary: dict[str, Any] = {"fold_count": len(plan["folds"])}
    for column in value_columns:
        values = fold_summary[column].to_numpy(dtype=float)
        summary[column] = float(np.mean(values))
        summary[f"{column}_fold_std"] = float(np.std(values, ddof=1))
    summary_frame = pd.DataFrame([summary])
    macro_delta = float(summary["delta_macro_f1"])
    fold_wins = int((fold_summary["delta_macro_f1"] > 0).sum())
    primary = plan["primary_success"]
    primary_passed = bool(
        macro_delta >= float(primary["minimum_absolute_mean_gain_vs_p2"])
        and fold_wins >= int(primary["minimum_winning_fold_count"])
    )
    safety = plan["safety_gates"]
    safety_results = {
        "micro_f1": {
            "value": float(summary["delta_micro_f1"]),
            "threshold": float(safety["minimum_micro_f1_delta"]),
            "passed": float(summary["delta_micro_f1"])
            >= float(safety["minimum_micro_f1_delta"]),
        },
        "masked_bce": {
            "value": float(summary["delta_masked_bce"]),
            "threshold": float(safety["maximum_masked_bce_delta"]),
            "passed": float(summary["delta_masked_bce"])
            <= float(safety["maximum_masked_bce_delta"]),
        },
        "parameter_count": {
            "value": float(summary["relative_delta_parameter_count"]),
            "threshold": float(safety["maximum_parameter_count_relative_increase"]),
            "passed": float(summary["relative_delta_parameter_count"])
            <= float(safety["maximum_parameter_count_relative_increase"]),
        },
        "latency": {
            "value": float(summary["relative_delta_latency_ms_per_sample"]),
            "threshold": float(safety["maximum_latency_relative_increase"]),
            "passed": float(summary["relative_delta_latency_ms_per_sample"])
            <= float(safety["maximum_latency_relative_increase"]),
        },
    }
    fold_seed_path = root / "reports/v4/phase_v4_1_fold_seed_results.csv"
    fold_path = root / "reports/v4/phase_v4_1_fold_summary.csv"
    summary_path = root / "reports/v4/phase_v4_1_summary.csv"
    fold_seed_path.parent.mkdir(parents=True, exist_ok=True)
    fold_seed.to_csv(fold_seed_path, index=False)
    fold_summary.to_csv(fold_path, index=False)
    summary_frame.to_csv(summary_path, index=False)
    report = {
        "version": 1,
        "phase": "V4-1",
        "status": "complete",
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": lock["protocol_sha256"],
        "unit_count": len(fold_seed),
        "primary_metric": "natural_missingness_macro_f1",
        "candidate_minus_p2_macro_f1": macro_delta,
        "candidate_winning_fold_count": fold_wins,
        "required_mean_gain": float(primary["minimum_absolute_mean_gain_vs_p2"]),
        "required_winning_fold_count": int(primary["minimum_winning_fold_count"]),
        "primary_success": primary_passed,
        "safety_gates": safety_results,
        "all_safety_gates_passed": all(
            result["passed"] for result in safety_results.values()
        ),
        "final_model_optimization": True,
        "further_model_versions_allowed": False,
        "summary": summary,
        "outputs": [
            v31._relative_entry(root, fold_seed_path),
            v31._relative_entry(root, fold_path),
            v31._relative_entry(root, summary_path),
        ],
    }
    report_path = root / "reports/v4/phase_v4_1_report.json"
    write_json(report_path, v31._json_safe(report))
    lines = [
        "# RobustSense V4-1 最后一次模型优化报告",
        "",
        "## 完成状态",
        "",
        "V4-1 已完成 5 折 × 3 种子的 15 个 P2-LC 正式单元。",
        "",
        f"- 协议哈希：`{lock['protocol_sha256']}`",
        f"- P2 Macro-F1：{summary['baseline_macro_f1']:.6f}",
        f"- P2-LC Macro-F1：{summary['candidate_macro_f1']:.6f}",
        f"- 绝对差值：{macro_delta:+.6f}",
        f"- 胜出折数：{fold_wins}/5",
        f"- 首要成功判定：{'通过' if primary_passed else '未通过'}",
        "",
        "## 安全闸门",
        "",
    ]
    for name, result in safety_results.items():
        lines.append(
            f"- `{name}`：{result['value']:+.6f}，"
            f"{'通过' if result['passed'] else '未通过'}。"
        )
    lines.extend(
        [
            "",
            "## 停止规则",
            "",
            "这是预先声明的最后一次模型优化。无论结果是否通过，不再增加新的模型版本。",
            "",
        ]
    )
    document = root / "docs/v4/PHASE_V4_1_REPORT.md"
    document.write_text("\n".join(lines), encoding="utf-8")
    print(f"AGGREGATED V4-1：{report_path}", flush=True)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate", "aggregate"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan, lock = _load_governance(root)
    registry = _p2_registry(root, plan)
    if arguments.mode == "aggregate":
        aggregate_results(root, plan, lock, registry)
        return
    if arguments.fold is None or arguments.seed is None:
        parser.error("train/evaluate 必须提供 --fold 和 --seed")
    if arguments.fold not in plan["folds"] or arguments.seed not in plan["seeds"]:
        parser.error("fold/seed 不在冻结矩阵中")
    if arguments.mode == "train":
        train_unit(
            root,
            plan,
            lock,
            registry,
            fold=arguments.fold,
            seed=arguments.seed,
            force=arguments.force,
        )
    else:
        result = evaluate_unit(
            root,
            plan,
            lock,
            registry,
            fold=arguments.fold,
            seed=arguments.seed,
            force=arguments.force,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
