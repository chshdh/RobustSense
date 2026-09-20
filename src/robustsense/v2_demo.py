"""Fail-closed helpers for the V2-5 offline replay demo."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset, collate_multimodal
from robustsense.data.extrasensory import sha256_file
from robustsense.experiments.v2_phase0 import verify_file_entries
from robustsense.models.fusion import build_phase3_model
from robustsense.models.v2_phase3 import build_v2_phase3_model
from robustsense.training.trainer import _move_batch, _verify_checkpoint_contract
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json

DEMO_MODELS = ("B4", "B5", "P", "P2")
DEMO_FOLDS = (0, 1, 2, 3, 4)
DEMO_SEEDS = (13, 29, 47)
DEMO_FAULTS = ("clean", "gaussian", "bias", "scale")
MODEL_NAMES = {
    "B4": "gated",
    "B5": "robust-gated",
    "P": "quality-aware",
    "P2": "P2",
}
QUALITY_COLUMNS = (
    "availability",
    "observed_fraction",
    "outlier_fraction",
    "mean_abs_robust_z",
    "max_abs_robust_z",
)


class V2DemoArtifactError(ValueError):
    """Raised when a V2-5 demo dependency is absent or inconsistent."""


@dataclass
class V2DemoBundle:
    root: Path
    run_path: Path
    run_key: str
    run_id: str
    model_id: str
    source_model_name: str
    source_phase: str
    fold: int
    seed: int
    protocol_sha256: str
    device: torch.device
    model: torch.nn.Module
    dataset: MultiModalDataset
    labels: list[str]
    thresholds: np.ndarray
    abstention_threshold: float | None
    modality_slices: dict[str, tuple[int, int]]
    corruption: CorruptionRegistry
    checkpoint_path: Path
    test_split_sha256: str


def _registry_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise V2DemoArtifactError(f"缺少 V2-4 核心登记表：{path}")
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def validate_v2_demo_context(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    plan_path = root / "configs/v2/phase_v2_5_plan.yaml"
    if not plan_path.is_file():
        raise V2DemoArtifactError(f"缺少 V2-5 冻结计划：{plan_path}")
    plan = load_config(plan_path)
    lock = read_json(root / plan["parent_protocol_lock"])
    problems = verify_file_entries(root, lock["files"])
    if problems:
        raise V2DemoArtifactError(f"V2-4 冻结协议校验失败：{problems}")
    core = read_json(root / plan["core_report"])
    extended = read_json(root / plan["extended_report"])
    bootstrap = read_json(root / plan["natural_bootstrap"])
    if core.get("successful_core_unit_count") != 60:
        raise V2DemoArtifactError("V2-4 核心矩阵不是 60/60 success")
    if extended.get("successful_unit_count") != 60:
        raise V2DemoArtifactError("V2-4 扩展矩阵不是 60/60 success")
    if extended.get("all_scenarios_share_samples_across_models") is not True:
        raise V2DemoArtifactError("扩展场景未通过跨模型同样本合同")
    if bootstrap.get("valid_repeat_count") != bootstrap.get("repeats"):
        raise V2DemoArtifactError("自然缺失 Bootstrap 存在无效重复")
    rows = _registry_rows(root / plan["core_registry"])
    expected = len(DEMO_MODELS) * len(DEMO_FOLDS) * len(DEMO_SEEDS)
    if len(rows) != expected or any(row["status"] != "success" for row in rows):
        raise V2DemoArtifactError("V2-4 核心登记表不完整或包含非成功单元")
    if any(row["protocol_sha256"] != lock["protocol_sha256"] for row in rows):
        raise V2DemoArtifactError("核心登记表协议哈希与 V2-4 锁不一致")
    return {
        "root": root,
        "plan": plan,
        "protocol_sha256": lock["protocol_sha256"],
        "rows": rows,
        "core": core,
        "extended": extended,
        "bootstrap": bootstrap,
    }


def resolve_demo_run(
    project_root: str | Path, model_id: str, fold: int, seed: int
) -> tuple[Path, dict[str, str], dict[str, Any]]:
    if model_id not in DEMO_MODELS:
        raise V2DemoArtifactError(f"不支持的 Demo 模型：{model_id}")
    if int(fold) not in DEMO_FOLDS or int(seed) not in DEMO_SEEDS:
        raise V2DemoArtifactError("fold 或 seed 超出 V2-4 冻结范围")
    context = validate_v2_demo_context(project_root)
    matches = [
        row
        for row in context["rows"]
        if row["model_name"] == model_id
        and int(row["fold"]) == int(fold)
        and int(row["seed"]) == int(seed)
    ]
    if len(matches) != 1:
        raise V2DemoArtifactError(
            f"核心登记表无法唯一解析 {model_id}/fold{fold}/seed{seed}"
        )
    row = matches[0]
    run_path = context["root"] / row["run_dir"]
    if not run_path.is_dir():
        raise V2DemoArtifactError(f"登记的 run 目录不存在：{run_path}")
    return run_path, row, context


def validate_demo_artifacts(run_path: str | Path, model_id: str) -> dict[str, Path]:
    run = Path(run_path).resolve()
    required = {
        "checkpoint": run / "best_checkpoint.pt",
        "resolved_config": run / "resolved_config.yaml",
        "thresholds": run / "thresholds.json",
        "test_metrics": run / "test_metrics.json",
        "evaluation_manifest": run / "evaluation_manifest.json",
        "extended_manifest": run / "extended_v2_4/evaluation_manifest.json",
        "persistent_response": run / "extended_v2_4/persistent_response.csv",
    }
    if model_id == "P2":
        required.update(
            {
                "abstention_threshold": run / "abstention_threshold.json",
                "risk_coverage": run / "risk_coverage.json",
            }
        )
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise V2DemoArtifactError("真实 run 产物不完整：" + ", ".join(missing))
    return required


def _device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise V2DemoArtifactError("选择了 CUDA，但当前环境不可用")
    return device


def load_demo_bundle(
    project_root: str | Path,
    model_id: str,
    fold: int,
    seed: int,
    *,
    device_name: str = "auto",
) -> V2DemoBundle:
    run_path, row, context = resolve_demo_run(project_root, model_id, fold, seed)
    artifacts = validate_demo_artifacts(run_path, model_id)
    resolved = load_config(artifacts["resolved_config"])
    evaluation = read_json(artifacts["evaluation_manifest"])
    extended = read_json(artifacts["extended_manifest"])
    expected_source = MODEL_NAMES[model_id]
    source_name = str(resolved.get("variant", resolved.get("model_name", "")))
    if source_name != expected_source:
        raise V2DemoArtifactError(
            f"模型身份不匹配：登记为 {model_id}，run 内为 {source_name}"
        )
    expected_identity = {
        "fold": int(fold),
        "seed": int(seed),
        "run_id": run_path.name,
    }
    mismatches = {
        key: (expected, resolved.get(key))
        for key, expected in expected_identity.items()
        if resolved.get(key) != expected
    }
    if mismatches:
        raise V2DemoArtifactError(f"run 身份合同不匹配：{mismatches}")
    if (
        int(evaluation.get("fold", -1)) != int(fold)
        or int(evaluation.get("seed", -1)) != int(seed)
        or evaluation.get("run_id") != run_path.name
    ):
        raise V2DemoArtifactError("自然缺失评估清单与登记身份不一致")
    if (
        extended.get("phase") != "V2-4"
        or extended.get("model_id") != model_id
        or int(extended.get("fold", -1)) != int(fold)
        or int(extended.get("seed", -1)) != int(seed)
        or extended.get("threshold_source_split") != "val"
    ):
        raise V2DemoArtifactError("扩展评估清单与 Demo 选择不一致")

    root = context["root"]
    processed_dir = root / str(resolved.get("processed_dir", ""))
    processed_manifest = processed_dir / "processed_manifest.json"
    preprocessor_path = processed_dir / "preprocessor.json"
    test_split = processed_dir / "test.npz"
    for path in (processed_manifest, preprocessor_path, test_split):
        if not path.is_file():
            raise V2DemoArtifactError(f"run 对应的数据产物缺失：{path}")
    manifest = read_json(processed_manifest)
    labels = list(manifest["labels"])
    modality_slices = {
        name: tuple(manifest["modality_slices"][name]) for name in MODALITIES
    }
    modality_dims = {
        name: end - start for name, (start, end) in modality_slices.items()
    }
    if model_id == "P2":
        model = build_v2_phase3_model(
            "P2", modality_dims, len(labels), resolved["model_config"]
        )
    else:
        model = build_phase3_model(
            expected_source, modality_dims, len(labels), resolved["model_config"]
        )
    device = _device(device_name)
    checkpoint = torch.load(
        artifacts["checkpoint"], map_location=device, weights_only=False
    )
    try:
        _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    except (AssertionError, KeyError, ValueError) as exc:
        raise V2DemoArtifactError(f"checkpoint 合同校验失败：{exc}") from exc
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)

    threshold_artifact = read_json(artifacts["thresholds"])
    if (
        threshold_artifact.get("source_split") != "val"
        or threshold_artifact.get("labels") != labels
    ):
        raise V2DemoArtifactError("分类阈值不是 validation 来源或标签顺序不一致")
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float32)
    if thresholds.shape != (len(labels),):
        raise V2DemoArtifactError("分类阈值维度与标签数量不一致")

    abstention_threshold = None
    if model_id == "P2":
        abstention = read_json(artifacts["abstention_threshold"])
        if abstention.get("source_split") != "val":
            raise V2DemoArtifactError("P2 拒绝阈值不是 validation 来源")
        abstention_threshold = float(abstention["threshold"])
        model.set_abstention_threshold(abstention_threshold)
    model.eval()

    preprocessor = read_json(preprocessor_path)
    corruption = CorruptionRegistry(
        load_config(root / "configs/corruption/eval.yaml"),
        modality_slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    dataset = MultiModalDataset(test_split, processed_manifest)
    return V2DemoBundle(
        root=root,
        run_path=run_path,
        run_key=row["run_key"],
        run_id=run_path.name,
        model_id=model_id,
        source_model_name=expected_source,
        source_phase=str(resolved.get("phase")),
        fold=int(fold),
        seed=int(seed),
        protocol_sha256=context["protocol_sha256"],
        device=device,
        model=model,
        dataset=dataset,
        labels=labels,
        thresholds=thresholds,
        abstention_threshold=abstention_threshold,
        modality_slices=modality_slices,
        corruption=corruption,
        checkpoint_path=artifacts["checkpoint"],
        test_split_sha256=sha256_file(test_split),
    )


def load_demo_matrix(
    project_root: str | Path, fold: int, seed: int, *, device_name: str = "auto"
) -> dict[str, V2DemoBundle]:
    bundles = {
        model_id: load_demo_bundle(
            project_root, model_id, fold, seed, device_name=device_name
        )
        for model_id in DEMO_MODELS
    }
    split_hashes = {bundle.test_split_sha256 for bundle in bundles.values()}
    label_orders = {tuple(bundle.labels) for bundle in bundles.values()}
    sample_counts = {len(bundle.dataset) for bundle in bundles.values()}
    if len(split_hashes) != 1 or len(label_orders) != 1 or len(sample_counts) != 1:
        raise V2DemoArtifactError("四模型没有共享同一测试 split、标签顺序和样本数量")
    return bundles


def _pseudonymous_user(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def infer_sample(
    bundle: V2DemoBundle,
    sample_index: int,
    *,
    disabled_modalities: tuple[str, ...] = (),
    fault_type: str = "clean",
    target_modality: str | None = None,
    severity: float = 0.0,
) -> dict[str, Any]:
    if not 0 <= int(sample_index) < len(bundle.dataset):
        raise IndexError(f"样本索引超出范围：{sample_index}")
    unknown = sorted(set(disabled_modalities) - set(MODALITIES))
    if unknown:
        raise ValueError(f"未知模态：{unknown}")
    if len(disabled_modalities) >= len(MODALITIES):
        raise ValueError("至少必须保留一个可用模态")
    if fault_type not in DEMO_FAULTS:
        raise ValueError(f"不支持的故障类型：{fault_type}")
    if fault_type != "clean" and target_modality not in MODALITIES:
        raise ValueError("特征污染必须指定合法目标模态")
    if target_modality in disabled_modalities:
        raise ValueError("污染模态不能同时被关闭")

    row = bundle.dataset[int(sample_index)]
    batch = collate_multimodal([row])
    batch, _ = bundle.corruption.apply_controlled(
        batch,
        "missing" if disabled_modalities else "clean",
        seed=bundle.seed,
        view=f"v2-demo-mask-{sample_index}",
        drop_modalities=tuple(disabled_modalities),
    )
    fault_metadata = None
    if fault_type != "clean":
        batch, fault_metadata = bundle.corruption.apply_controlled(
            batch,
            fault_type,
            seed=bundle.seed,
            view=f"v2-demo-{sample_index}-{fault_type}",
            target_modality=target_modality,
            severity=float(severity),
        )
    batch = _move_batch(batch, bundle.device, bundle.modality_slices)
    with torch.inference_mode():
        output = bundle.model(batch)
        probabilities = torch.sigmoid(output["logits"])[0].detach().cpu().numpy()
        weights = output["fusion_weights"][0].detach().cpu().numpy()
        reliability_tensor = output.get("reliability")
        utility_tensor = output.get("utility_scores")
        system_tensor = output.get("system_reliability")
        abstain_tensor = output.get("abstain")
        reliability = (
            reliability_tensor[0].detach().cpu().numpy()
            if reliability_tensor is not None
            else np.full(len(MODALITIES), np.nan, dtype=np.float32)
        )
        utility = (
            utility_tensor[0].detach().cpu().numpy()
            if utility_tensor is not None
            else np.full(len(MODALITIES), np.nan, dtype=np.float32)
        )
        if system_tensor is not None:
            system_reliability = float(system_tensor[0].detach().cpu())
        elif reliability_tensor is not None:
            system_reliability = float(np.sum(weights * reliability))
        else:
            system_reliability = None
        abstain = (
            bool(abstain_tensor[0].detach().cpu())
            if abstain_tensor is not None
            else None
        )
    availability = batch["availability"][0].detach().cpu().numpy().astype(bool)
    quality = batch["quality_features"][0].detach().cpu().numpy()
    targets = batch["targets"][0].detach().cpu().numpy()
    target_mask = batch["target_mask"][0].detach().cpu().numpy()
    if np.any(weights[~availability] != 0.0):
        raise AssertionError("不可用模态的融合权重必须严格为 0")
    if not np.isclose(float(weights.sum()), 1.0, atol=1.0e-6):
        raise AssertionError("可用模态的最终融合权重之和必须为 1")

    return {
        "run_key": bundle.run_key,
        "run_id": bundle.run_id,
        "model_id": bundle.model_id,
        "fold": bundle.fold,
        "seed": bundle.seed,
        "device": str(bundle.device),
        "checkpoint": str(bundle.checkpoint_path),
        "protocol_sha256": bundle.protocol_sha256,
        "sample_index": int(sample_index),
        "user": _pseudonymous_user(str(row["user_id"])),
        "timestamp": int(row["timestamp"]),
        "labels": bundle.labels,
        "probabilities": probabilities,
        "thresholds": bundle.thresholds.copy(),
        "predictions": probabilities >= bundle.thresholds,
        "targets": targets,
        "target_mask": target_mask,
        "modalities": list(MODALITIES),
        "availability": availability,
        "quality_features": quality,
        "reliability": reliability,
        "utility_scores": utility,
        "fusion_weights": weights,
        "system_reliability": system_reliability,
        "abstention_threshold": bundle.abstention_threshold,
        "abstain": abstain,
        "acceptance": (
            "不适用" if abstain is None else ("拒绝" if abstain else "接受")
        ),
        "disabled_modalities": list(disabled_modalities),
        "fault_type": fault_type,
        "target_modality": target_modality,
        "severity": float(severity),
        "fault_metadata": fault_metadata,
    }


def _verified_report_frame(project_root: str | Path, artifact: str) -> pd.DataFrame:
    context = validate_v2_demo_context(project_root)
    metadata = context["extended"]["artifacts"].get(artifact)
    if not metadata:
        raise V2DemoArtifactError(f"扩展报告没有声明产物：{artifact}")
    path = Path(metadata["path"])
    if not path.is_absolute():
        path = context["root"] / path
    if not path.is_file() or sha256_file(path) != metadata["sha256"]:
        raise V2DemoArtifactError(f"报告产物缺失或哈希不匹配：{path}")
    frame = pd.read_csv(path)
    if len(frame) != int(metadata["row_count"]):
        raise V2DemoArtifactError(f"报告产物行数不匹配：{path}")
    return frame


def mask_background(project_root: str | Path) -> pd.DataFrame:
    return _verified_report_frame(project_root, "mask_summary")


def mixed_background(project_root: str | Path) -> pd.DataFrame:
    return _verified_report_frame(project_root, "mixed_summary")


def persistent_phase_timeline(
    project_root: str | Path,
    *,
    fault: str,
    target_modality: str,
    fault_length: int,
) -> pd.DataFrame:
    frame = _verified_report_frame(project_root, "persistent_summary")
    selected = frame[
        (frame["fault"] == fault)
        & (frame["target_modality"] == target_modality)
        & (frame["fault_length"] == int(fault_length))
    ].copy()
    if len(selected) != len(DEMO_MODELS) * 3:
        raise V2DemoArtifactError("持续故障三阶段聚合行不完整")
    order = {"pre": 0, "fault": 1, "post": 2}
    selected["phase_order"] = selected["phase"].map(order)
    return selected.sort_values(["model_id", "phase_order"]).reset_index(drop=True)


def persistent_episode_timeline(
    bundle: V2DemoBundle,
    *,
    fault: str,
    target_modality: str,
    fault_length: int,
    episode_index: int = 0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    path = bundle.run_path / "extended_v2_4/persistent_response.csv"
    manifest = read_json(bundle.run_path / "extended_v2_4/evaluation_manifest.json")
    declared = manifest["artifacts"]["persistent_response"]
    if sha256_file(path) != declared["sha256"]:
        raise V2DemoArtifactError("持续故障 episode 文件哈希不匹配")
    frame = pd.read_csv(path)
    selected = frame[
        (frame["fault"] == fault)
        & (frame["target_modality"] == target_modality)
        & (frame["fault_length"] == int(fault_length))
        & frame["reliability_available"].astype(bool)
    ].sort_values("episode_id")
    if selected.empty:
        raise V2DemoArtifactError(f"{bundle.model_id} 没有可用的真实可靠度 episode")
    row = selected.iloc[int(episode_index) % len(selected)]
    timeline = pd.DataFrame(
        {
            "phase": ["pre", "fault", "post"],
            "fault_score": [
                row["pre_mean_fault_score"],
                row["fault_mean_fault_score"],
                row["post_mean_fault_score"],
            ],
        }
    )
    metadata = {
        "episode_id": row["episode_id"],
        "detected": bool(row["detected"]),
        "recovered": bool(row["recovered"]),
        "detection_delay_steps": row["detection_delay_steps"],
        "recovery_steps": row["recovery_steps"],
        "episode_count": len(selected),
    }
    return timeline, metadata
