"""Phase-0 prepare, train, and evaluate orchestration."""

from __future__ import annotations

import csv
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from robustsense.constants import LABELS
from robustsense.data.synthetic import prepare_synthetic
from robustsense.models.baselines import LinearBaseline
from robustsense.training.losses import masked_binary_cross_entropy, sigmoid
from robustsense.training.metrics import masked_f1_metrics
from robustsense.utils.config import dump_config, load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed


def _root_from_config(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "data" and path.parent.parent.name == "configs":
        return path.parent.parent.parent
    raise ValueError(f"Expected config under <project>/configs/data: {path}")


def _load_dataset(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Prepared dataset not found: {path}; run prepare first")
    with np.load(path) as loaded:
        return {name: loaded[name] for name in loaded.files}


def _design_matrix(dataset: dict[str, np.ndarray]) -> np.ndarray:
    features = np.nan_to_num(dataset["features"].astype(np.float64), nan=0.0)
    feature_mask = dataset["feature_mask"].astype(np.float64)
    availability = dataset["availability"].astype(np.float64)
    return np.concatenate([features, feature_mask, availability], axis=1)


def prepare(config_path: str | Path, fold: int = 0) -> dict[str, str]:
    project_root = _root_from_config(config_path)
    config = load_config(config_path)
    if config.get("dataset") == "synthetic":
        prepared = prepare_synthetic(config, project_root=project_root, fold=fold)
        return {
            "dataset_path": str(prepared.dataset_path),
            "manifest_path": str(prepared.manifest_path),
            "split_manifest_path": str(prepared.split_manifest_path),
        }
    if config.get("dataset") == "extrasensory":
        from robustsense.data.real import prepare_extrasensory

        return prepare_extrasensory(config, project_root=project_root, fold=fold)
    raise ValueError(f"Unsupported dataset: {config.get('dataset')!r}")


def audit(
    config_path: str | Path, fold: int = 0, probe_only: bool = False
) -> dict[str, object]:
    from robustsense.data.extrasensory import run_extrasensory_audit

    project_root = _root_from_config(config_path)
    config = load_config(config_path)
    if config.get("dataset") != "extrasensory":
        raise ValueError("The audit command requires an ExtraSensory data config")
    return run_extrasensory_audit(config, project_root, fold=fold, probe_only=probe_only)


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


def train(
    project_root: str | Path,
    model_name: str = "early",
    fold: int = 0,
    seed: int = 13,
    profile: str = "dev",
    data_config: str | Path | None = None,
) -> Path:
    root = Path(project_root).resolve()
    if data_config is not None:
        config_path = Path(data_config)
        if not config_path.is_absolute():
            config_path = root / config_path
        if load_config(config_path).get("dataset") == "extrasensory":
            if model_name in {
                "gated",
                "robust-gated",
                "quality-aware-cls",
                "quality-aware",
                "reliability-constrained",
                "ablation-a1-no-sensor-dropout",
                "ablation-a2-no-noise",
                "ablation-a3-no-quality",
                "ablation-a4-no-reliability",
                "ablation-a5-no-reliability-loss",
            }:
                from robustsense.training.phase3_trainer import train_phase3

                return train_phase3(root, config_path, model_name, fold, seed, profile)
            from robustsense.training.trainer import train_phase2

            return train_phase2(root, config_path, model_name, fold, seed, profile)
    if model_name != "early":
        raise ValueError("Phase 0 implements only the early NumPy smoke baseline")
    data_config = load_config(root / "configs/data/synthetic.yaml")
    experiment = load_config(root / f"configs/experiment/{profile}.yaml")
    dataset_path = root / data_config["output_path"]
    dataset = _load_dataset(dataset_path)
    set_global_seed(seed)

    matrix = _design_matrix(dataset)
    targets = dataset["targets"].astype(np.float64)
    target_mask = dataset["target_mask"].astype(bool)
    split = dataset["split"].astype(str)
    train_rows = split == "train"
    val_rows = split == "val"
    if not train_rows.any() or not val_rows.any():
        raise ValueError("Prepared data has an empty training or validation split")

    model = LinearBaseline.initialize(matrix.shape[1], len(LABELS), seed)
    learning_rate = float(experiment.get("learning_rate", 0.05))
    batch_size = int(experiment.get("batch_size", 32))
    epochs = int(experiment.get("epochs", 1))
    rng = np.random.default_rng(seed)
    train_indices = np.flatnonzero(train_rows)
    log_rows: list[dict[str, float | int]] = []

    for epoch in range(epochs):
        shuffled = rng.permutation(train_indices)
        for start in range(0, len(shuffled), batch_size):
            batch = shuffled[start : start + batch_size]
            logits = model.logits(matrix[batch])
            clean_targets = np.nan_to_num(targets[batch], nan=0.0)
            mask = target_mask[batch].astype(np.float64)
            denominator = float(mask.sum())
            if denominator == 0:
                continue
            grad_logits = (sigmoid(logits) - clean_targets) * mask / denominator
            model.weights -= learning_rate * (matrix[batch].T @ grad_logits)
            model.bias -= learning_rate * grad_logits.sum(axis=0)

        train_logits = model.logits(matrix[train_rows])
        train_loss = masked_binary_cross_entropy(
            train_logits, targets[train_rows], target_mask[train_rows]
        )
        val_probabilities = sigmoid(model.logits(matrix[val_rows]))
        val_metrics = masked_f1_metrics(val_probabilities, targets[val_rows], target_mask[val_rows])
        log_rows.append(
            {"epoch": epoch + 1, "train_loss": train_loss, "val_macro_f1": val_metrics["macro_f1"]}
        )

    run_id = f"synthetic-{model_name}-fold{fold}-seed{seed}"
    run_dir = root / experiment.get("run_root", "runs") / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_checkpoint.npz"
    model.save(checkpoint_path)
    thresholds = np.full(len(LABELS), float(experiment.get("threshold", 0.5)))
    write_json(
        run_dir / "thresholds.json", {"source": "phase_0_fixed", "values": thresholds.tolist()}
    )
    write_json(run_dir / "val_metrics.json", val_metrics)
    dump_config(
        {
            "phase": 0,
            "synthetic_only": True,
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "profile": profile,
            "data": data_config,
            "experiment": experiment,
        },
        run_dir / "resolved_config.yaml",
    )
    split_manifest = read_json(
        root / data_config["manifest_dir"] / f"synthetic_split_fold{fold}.json"
    )
    write_json(run_dir / "split_manifest.json", split_manifest)
    write_json(
        run_dir / "data_manifest.json",
        read_json(root / data_config["manifest_dir"] / "synthetic_data_manifest.json"),
    )
    write_json(
        run_dir / "environment.json",
        {
            "created_at": datetime.now(UTC).isoformat(),
            "python": sys.version,
            "platform": platform.platform(),
            "git_commit": _git_commit(root),
            "seed": seed,
            "wsl_native_validation": False,
        },
    )
    with (run_dir / "train_log.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_macro_f1"])
        writer.writeheader()
        writer.writerows(log_rows)
    return run_dir


def evaluate(run_dir: str | Path, suite: str = "smoke") -> dict[str, object]:
    run_path = Path(run_dir).resolve()
    resolved = load_config(run_path / "resolved_config.yaml")
    if suite == "full" and resolved.get("phase") in {2, 3}:
        from robustsense.evaluation.suite import evaluate_full_suite

        return evaluate_full_suite(run_path)
    if resolved.get("phase") == 2:
        from robustsense.training.trainer import evaluate_phase2

        return evaluate_phase2(run_path, suite)
    if resolved.get("phase") == 3:
        from robustsense.training.phase3_trainer import evaluate_phase3

        return evaluate_phase3(run_path, suite)
    root = run_path.parents[1]
    dataset_path = root / resolved["data"]["output_path"]
    dataset = _load_dataset(dataset_path)
    test_rows = dataset["split"].astype(str) == "test"
    if not test_rows.any():
        raise ValueError("Prepared data has an empty test split")

    model = LinearBaseline.load(run_path / "best_checkpoint.npz")
    threshold_values = np.asarray(read_json(run_path / "thresholds.json")["values"])
    probabilities = sigmoid(model.logits(_design_matrix(dataset)[test_rows]))
    targets = dataset["targets"][test_rows]
    target_mask = dataset["target_mask"][test_rows].astype(bool)
    metrics = masked_f1_metrics(probabilities, targets, target_mask, threshold_values)
    metrics.update({"suite": suite, "synthetic_only": True, "sample_count": int(test_rows.sum())})
    write_json(run_path / "test_metrics.json", metrics)

    test_users = dataset["user_id"][test_rows].astype(str)
    test_timestamps = dataset["timestamp"][test_rows]
    with (run_path / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "user_id",
                "timestamp",
                "scenario",
                "label",
                "target",
                "target_known",
                "probability",
                "prediction",
            ]
        )
        for row_index, (user, timestamp) in enumerate(
            zip(test_users, test_timestamps, strict=True)
        ):
            for label_index, label in enumerate(LABELS):
                known = bool(target_mask[row_index, label_index])
                writer.writerow(
                    [
                        user,
                        int(timestamp),
                        "synthetic_clean",
                        label,
                        "" if not known else int(targets[row_index, label_index]),
                        int(known),
                        float(probabilities[row_index, label_index]),
                        int(probabilities[row_index, label_index] >= threshold_values[label_index]),
                    ]
                )
    (run_path / "run_summary.md").write_text(
        "# Phase-0 synthetic smoke run\n\n"
        f"- Suite: `{suite}`\n"
        f"- Samples: {metrics['sample_count']}\n"
        f"- Macro-F1: {metrics['macro_f1']:.6f}\n"
        f"- Micro-F1: {metrics['micro_f1']:.6f}\n\n"
        "These metrics come from synthetic data and are not research results.\n",
        encoding="utf-8",
    )
    return metrics
