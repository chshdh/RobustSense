"""V2-4 wrappers around the frozen V2-3 P2 training and evaluation semantics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

import robustsense.training.phase3_trainer as v1_phase3
import robustsense.training.v2_phase3_trainer as phase3
from robustsense.data.extrasensory import sha256_file
from robustsense.training.v2_phase3_trainer import state_dict_sha256
from robustsense.utils.config import dump_config, load_config
from robustsense.utils.io import read_json, write_json

PHASE4_VARIANTS = ("P2", "P2-A3", "P2-A1")
PHASE4_V1_MODELS = ("gated", "robust-gated", "quality-aware")


def v2_phase4_run_id(variant: str, fold: int, seed: int) -> str:
    if variant not in PHASE4_VARIANTS:
        raise ValueError(f"Unknown V2-4 variant: {variant}")
    return (
        f"extrasensory-{phase3.VARIANT_SLUGS[variant]}-fold{fold}-seed{seed}-"
        "v2_multiseed"
    )


def _load_plan(root: Path, plan_path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(plan_path)
    if not path.is_absolute():
        path = root / path
    return load_config(path), path


def _run_path(root: Path, plan: dict[str, Any], variant: str, fold: int, seed: int) -> Path:
    profile = load_config(root / plan["experiment_config"])
    return root / profile["run_root"] / v2_phase4_run_id(variant, fold, seed)


def _v1_run_path(root: Path, model_name: str, fold: int, seed: int) -> Path:
    return root / "runs" / f"extrasensory-{model_name}-fold{fold}-seed{seed}-full"


def train_v2_phase4_v1(
    project_root: str | Path,
    *,
    model_name: str,
    fold: int,
    seed: int,
    protocol_sha256: str,
    plan_path: str | Path = "configs/v2/phase_v2_4_plan.yaml",
) -> Path:
    """Run a frozen V1 model while preventing its legacy implicit test evaluation."""
    root = Path(project_root).resolve()
    plan, _ = _load_plan(root, plan_path)
    if model_name not in PHASE4_V1_MODELS:
        raise ValueError(f"Model is outside the V2-4 core plan: {model_name}")
    if fold not in plan["folds"] or seed not in plan["new_training_seeds"]:
        raise ValueError("V2-4 training only permits frozen folds and new seeds")
    run_path = _v1_run_path(root, model_name, fold, seed)
    run_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "test_metrics.json",
        "predictions.parquet",
        "evaluation_manifest.json",
        "modality_diagnostics.parquet",
    ):
        (run_path / name).unlink(missing_ok=True)

    def validation_only_stub(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {"status": "test_evaluation_deferred_by_v2_4"}

    original_evaluate = v1_phase3.evaluate_phase3
    v1_phase3.evaluate_phase3 = validation_only_stub
    try:
        trained = v1_phase3.train_phase3(
            root,
            root / plan["data_config"],
            model_name,
            fold,
            seed,
            "full",
        )
    finally:
        v1_phase3.evaluate_phase3 = original_evaluate
    if trained != run_path:
        raise AssertionError("V2-4 V1 wrapper returned an unexpected run directory")
    if (run_path / "test_metrics.json").exists():
        raise AssertionError("V2-4 training must not create test metrics")
    resolved = load_config(run_path / "resolved_config.yaml")
    resolved.update(
        {
            "v2_governance_phase": "V2-4",
            "v2_protocol_sha256": protocol_sha256,
            "test_split_opened_during_training": False,
            "source_training_implementation": "frozen_v1_phase3_semantics",
        }
    )
    dump_config(resolved, run_path / "resolved_config.yaml")
    write_json(
        run_path / "v2_run_manifest.json",
        {
            "phase": "V2-4",
            "run_id": resolved["run_id"],
            "model_name": model_name,
            "fold": fold,
            "seed": seed,
            "protocol_sha256": protocol_sha256,
            "status": "trained_validation_frozen_test_not_opened",
            "checkpoint_sha256": sha256_file(run_path / "best_checkpoint.pt"),
            "test_split_opened_during_training": False,
        },
    )
    return run_path


def evaluate_v2_phase4_v1(run_dir: str | Path) -> dict[str, Any]:
    """Open test once for a V2-4 B4/B5/P run using frozen V1 semantics."""
    run_path = Path(run_dir).resolve()
    resolved = load_config(run_path / "resolved_config.yaml")
    if resolved.get("v2_governance_phase") != "V2-4":
        raise ValueError("Not a V2-4 governed V1-model run")
    result = v1_phase3.evaluate_phase3(run_path, suite="v2_multiseed_core")
    threshold_artifact = read_json(run_path / "thresholds.json")
    evaluation = {
        "phase": "V2-4",
        "profile": "v2_multiseed_core",
        "run_id": resolved["run_id"],
        "model_name": resolved["model_name"],
        "fold": int(resolved["fold"]),
        "seed": int(resolved["seed"]),
        "protocol_sha256": resolved["v2_protocol_sha256"],
        "threshold_source_split": threshold_artifact.get("source_split"),
        "test_opened_by_separate_evaluation_process": True,
        "test_metrics_sha256": sha256_file(run_path / "test_metrics.json"),
        "predictions_sha256": sha256_file(run_path / "predictions.parquet"),
    }
    write_json(run_path / "evaluation_manifest.json", evaluation)
    manifest = read_json(run_path / "v2_run_manifest.json")
    manifest.update(
        {
            "status": "success_test_evaluated_once",
            "evaluation_manifest_sha256": sha256_file(
                run_path / "evaluation_manifest.json"
            ),
        }
    )
    write_json(run_path / "v2_run_manifest.json", manifest)
    return result


def train_v2_phase4(
    project_root: str | Path,
    *,
    variant: str,
    fold: int,
    seed: int,
    protocol_sha256: str,
    plan_path: str | Path = "configs/v2/phase_v2_4_plan.yaml",
) -> Path:
    """Train a new-seed P2-family unit without modifying the frozen V2-3 module."""
    root = Path(project_root).resolve()
    plan, plan_file = _load_plan(root, plan_path)
    if variant not in PHASE4_VARIANTS:
        raise ValueError(f"Variant is outside the frozen V2-4 plan: {variant}")
    if fold not in plan["folds"] or seed not in plan["new_training_seeds"]:
        raise ValueError("V2-4 training only permits frozen folds and new seeds")
    run_id = v2_phase4_run_id(variant, fold, seed)
    run_path = _run_path(root, plan, variant, fold, seed)
    run_path.mkdir(parents=True, exist_ok=True)
    for name in (
        "test_metrics.json",
        "predictions.parquet",
        "evaluation_manifest.json",
        "risk_coverage.json",
    ):
        (run_path / name).unlink(missing_ok=True)

    original_run_id = phase3.v2_phase3_run_id
    phase3.v2_phase3_run_id = lambda _variant, _fold, _seed: v2_phase4_run_id(
        _variant, _fold, _seed
    )
    try:
        trained = phase3.train_v2_phase3(
            root,
            variant=variant,
            fold=fold,
            seed=seed,
            protocol_sha256=protocol_sha256,
            plan_path=plan_file,
        )
    finally:
        phase3.v2_phase3_run_id = original_run_id
    if trained != run_path:
        raise AssertionError("V2-4 run identity wrapper returned an unexpected directory")

    checkpoint_path = run_path / "best_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    contract = {
        **checkpoint["contract"],
        "version": 4,
        "phase": "V2-4",
        "profile": "v2_multiseed_core",
        "protocol_sha256": protocol_sha256,
    }
    checkpoint["contract"] = contract
    torch.save(checkpoint, checkpoint_path)
    resolved = load_config(run_path / "resolved_config.yaml")
    resolved.update(
        {
            "phase": "V2-4",
            "profile": "v2_multiseed_core",
            "run_id": run_id,
            "run_dir": str(run_path.relative_to(root)),
            "plan": str(plan_file.relative_to(root)),
            "protocol_sha256": protocol_sha256,
            "checkpoint_contract": contract,
            "source_training_implementation": "frozen_v2_phase3_semantics",
        }
    )
    dump_config(resolved, run_path / "resolved_config.yaml")
    manifest = read_json(run_path / "run_manifest.json")
    manifest.update(
        {
            "phase": "V2-4",
            "run_id": run_id,
            "protocol_sha256": protocol_sha256,
            "status": "trained_validation_frozen_test_not_opened",
            "checkpoint_sha256": sha256_file(checkpoint_path),
            "state_dict_sha256": state_dict_sha256(checkpoint["state_dict"]),
        }
    )
    write_json(run_path / "run_manifest.json", manifest)
    (run_path / "run_summary.md").write_text(
        "# V2-4 多种子训练摘要\n\n"
        f"- Run：`{run_id}`\n"
        f"- 变体：`{variant}`\n"
        f"- Fold：{fold}\n"
        f"- Seed：{seed}\n"
        f"- 完成 epoch：{manifest['epochs_completed']}\n\n"
        "训练阶段未打开 test；分类阈值和拒绝阈值均来自 validation。\n",
        encoding="utf-8",
    )
    return run_path


def evaluate_v2_phase4(run_dir: str | Path) -> dict[str, Any]:
    """Evaluate once with frozen V2-3 semantics, then promote artifacts to V2-4."""
    run_path = Path(run_dir).resolve()
    resolved_path = run_path / "resolved_config.yaml"
    resolved = load_config(resolved_path)
    if resolved.get("phase") != "V2-4":
        raise ValueError("Not a V2-4 run")
    temporary = {**resolved, "phase": "V2-3"}
    dump_config(temporary, resolved_path)
    try:
        result = phase3.evaluate_v2_phase3(run_path)
    finally:
        dump_config(resolved, resolved_path)
    result["phase"] = "V2-4"
    write_json(run_path / "test_metrics.json", result)
    evaluation = read_json(run_path / "evaluation_manifest.json")
    evaluation.update(
        {
            "phase": "V2-4",
            "profile": "v2_multiseed_core",
            "protocol_sha256": resolved["protocol_sha256"],
        }
    )
    write_json(run_path / "evaluation_manifest.json", evaluation)
    manifest = read_json(run_path / "run_manifest.json")
    manifest.update(
        {
            "phase": "V2-4",
            "status": "success_test_evaluated_once",
            "evaluation_manifest_sha256": sha256_file(
                run_path / "evaluation_manifest.json"
            ),
        }
    )
    write_json(run_path / "run_manifest.json", manifest)
    return result
