"""Artifact-validated inference helpers for the offline Streamlit replay demo."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset, collate_multimodal
from robustsense.models.fusion import build_phase3_model
from robustsense.training.phase3_trainer import _checkpoint_contract
from robustsense.training.trainer import _move_batch, _verify_checkpoint_contract
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json

DEMO_MODELS = ("gated", "robust-gated", "quality-aware")
DEMO_FAULTS = ("clean", "gaussian", "bias", "scale")


class DemoArtifactError(ValueError):
    """Raised when a demo run is absent or violates its saved artifact contract."""


@dataclass
class DemoBundle:
    root: Path
    run_path: Path
    run_id: str
    model_name: str
    fold: int
    seed: int
    device: torch.device
    model: torch.nn.Module
    dataset: MultiModalDataset
    labels: list[str]
    thresholds: np.ndarray
    modality_slices: dict[str, tuple[int, int]]
    corruption: CorruptionRegistry


def run_directory(
    project_root: str | Path,
    model_name: str,
    fold: int,
    seed: int = 13,
    profile: str = "credible",
) -> Path:
    if model_name not in DEMO_MODELS:
        raise DemoArtifactError(f"Unsupported demo model: {model_name}")
    return (
        Path(project_root).resolve()
        / "runs"
        / f"extrasensory-{model_name}-fold{int(fold)}-seed{int(seed)}-{profile}"
    )


def validate_demo_artifacts(run_path: str | Path) -> dict[str, Path]:
    run = Path(run_path).resolve()
    root = run.parents[1] if len(run.parents) > 1 else run.parent
    required = {
        "checkpoint": run / "best_checkpoint.pt",
        "resolved_config": run / "resolved_config.yaml",
        "thresholds": run / "thresholds.json",
        "evaluation_manifest": run / "evaluation_manifest.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise DemoArtifactError(
            "Demo requires a complete real run. Missing artifacts: " + ", ".join(missing)
        )
    resolved = load_config(required["resolved_config"])
    processed_dir = root / str(resolved.get("processed_dir", ""))
    required.update(
        {
            "processed_manifest": processed_dir / "processed_manifest.json",
            "preprocessor": processed_dir / "preprocessor.json",
            "preprocessor_arrays": processed_dir / "preprocessor.npz",
            "test_split": processed_dir / "test.npz",
        }
    )
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise DemoArtifactError(
            "Demo run is not replayable. Missing matching data artifacts: "
            + ", ".join(missing)
        )
    return required


def load_demo_bundle(
    project_root: str | Path,
    model_name: str,
    fold: int,
    *,
    seed: int = 13,
    profile: str = "credible",
    device_name: str = "auto",
) -> DemoBundle:
    root = Path(project_root).resolve()
    run_path = run_directory(root, model_name, fold, seed, profile)
    artifacts = validate_demo_artifacts(run_path)
    resolved = load_config(artifacts["resolved_config"])
    expected_identity = {
        "model_name": model_name,
        "fold": int(fold),
        "seed": int(seed),
        "profile": profile,
        "phase": 3,
    }
    mismatches = {
        name: (expected, resolved.get(name))
        for name, expected in expected_identity.items()
        if resolved.get(name) != expected
    }
    if mismatches:
        raise DemoArtifactError(f"Resolved run identity mismatch: {mismatches}")

    manifest = read_json(artifacts["processed_manifest"])
    labels = list(manifest["labels"])
    modality_slices = {
        name: tuple(manifest["modality_slices"][name]) for name in MODALITIES
    }
    modality_dims = {
        name: end - start for name, (start, end) in modality_slices.items()
    }
    model = build_phase3_model(
        model_name,
        modality_dims,
        len(labels),
        resolved["model_config"],
    )
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise DemoArtifactError("CUDA was selected for the demo but is unavailable")
    checkpoint = torch.load(
        artifacts["checkpoint"], map_location=device, weights_only=False
    )
    expected_contract = _checkpoint_contract(
        model_name,
        manifest,
        artifacts["processed_manifest"],
        artifacts["preprocessor"],
        labels,
        modality_dims,
    )
    try:
        _verify_checkpoint_contract(expected_contract, checkpoint["contract"])
        _verify_checkpoint_contract(resolved["checkpoint_contract"], checkpoint["contract"])
    except (AssertionError, KeyError, ValueError) as exc:
        raise DemoArtifactError(f"Checkpoint contract verification failed: {exc}") from exc
    model.load_state_dict(checkpoint["state_dict"])
    model = model.to(device)
    model.eval()

    threshold_artifact = read_json(artifacts["thresholds"])
    if (
        threshold_artifact.get("source_split") != "val"
        or threshold_artifact.get("labels") != labels
    ):
        raise DemoArtifactError(
            "Thresholds are not validation-derived or do not match checkpoint labels"
        )
    thresholds = np.asarray(threshold_artifact["values"], dtype=np.float32)
    if thresholds.shape != (len(labels),):
        raise DemoArtifactError("Threshold vector shape does not match the label schema")

    preprocessor = read_json(artifacts["preprocessor"])
    evaluation_corruption = load_config(root / "configs/corruption/eval.yaml")
    corruption = CorruptionRegistry(
        evaluation_corruption,
        modality_slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )
    dataset = MultiModalDataset(artifacts["test_split"], artifacts["processed_manifest"])
    return DemoBundle(
        root=root,
        run_path=run_path,
        run_id=str(resolved["run_id"]),
        model_name=model_name,
        fold=int(fold),
        seed=int(seed),
        device=device,
        model=model,
        dataset=dataset,
        labels=labels,
        thresholds=thresholds,
        modality_slices=modality_slices,
        corruption=corruption,
    )


def _pseudonymous_user(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:12]


def infer_sample(
    bundle: DemoBundle,
    sample_index: int,
    *,
    disabled_modalities: tuple[str, ...] = (),
    fault_type: str = "clean",
    target_modality: str | None = None,
    severity: float = 0.0,
) -> dict[str, Any]:
    if not 0 <= int(sample_index) < len(bundle.dataset):
        raise IndexError(
            f"Sample index {sample_index} is outside [0, {len(bundle.dataset) - 1}]"
        )
    unknown_disabled = sorted(set(disabled_modalities) - set(MODALITIES))
    if unknown_disabled:
        raise ValueError(f"Unknown disabled modalities: {unknown_disabled}")
    if len(disabled_modalities) >= len(MODALITIES):
        raise ValueError("The demo must keep at least one modality available")
    if fault_type not in DEMO_FAULTS:
        raise ValueError(f"Unsupported demo fault: {fault_type}")
    if fault_type != "clean" and target_modality not in MODALITIES:
        raise ValueError("A supported target modality is required for feature corruption")
    if target_modality in disabled_modalities:
        raise ValueError("The corrupted modality cannot also be disabled")

    row = bundle.dataset[int(sample_index)]
    batch = collate_multimodal([row])
    if disabled_modalities:
        batch, _ = bundle.corruption.apply_controlled(
            batch,
            "missing",
            seed=bundle.seed,
            view=f"demo-missing-{sample_index}",
            drop_modalities=tuple(disabled_modalities),
        )
    else:
        batch, _ = bundle.corruption.apply_controlled(
            batch,
            "clean",
            seed=bundle.seed,
            view=f"demo-clean-{sample_index}",
        )
    if fault_type != "clean":
        batch, fault_metadata = bundle.corruption.apply_controlled(
            batch,
            fault_type,
            seed=bundle.seed,
            view=f"demo-{sample_index}-{fault_type}",
            target_modality=target_modality,
            severity=float(severity),
        )
    else:
        fault_metadata = None

    batch = _move_batch(batch, bundle.device, bundle.modality_slices)
    with torch.inference_mode():
        output = bundle.model(batch)
        probabilities = torch.sigmoid(output["logits"])[0].detach().cpu().numpy()
        weights = output["fusion_weights"][0].detach().cpu().numpy()
        reliability_tensor = output["reliability"]
        reliability = (
            reliability_tensor[0].detach().cpu().numpy()
            if reliability_tensor is not None
            else np.full(len(MODALITIES), np.nan, dtype=np.float32)
        )
    availability = batch["availability"][0].detach().cpu().numpy()
    quality = batch["quality_features"][0].detach().cpu().numpy()
    targets = batch["targets"][0].detach().cpu().numpy()
    target_mask = batch["target_mask"][0].detach().cpu().numpy()
    if np.any(weights[~availability] != 0.0):
        raise AssertionError("Unavailable modalities must have exactly zero fusion weight")

    return {
        "run_id": bundle.run_id,
        "model_name": bundle.model_name,
        "fold": bundle.fold,
        "seed": bundle.seed,
        "device": str(bundle.device),
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
        "fusion_weights": weights,
        "reliability": reliability,
        "disabled_modalities": list(disabled_modalities),
        "fault_type": fault_type,
        "target_modality": target_modality,
        "severity": float(severity),
        "fault_metadata": fault_metadata,
    }
