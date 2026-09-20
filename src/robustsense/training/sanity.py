"""Scientific validity checks for the Phase-2 data path."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW

from robustsense.data.dataset import MultiModalDataset
from robustsense.training.phase2_metrics import phase2_multilabel_metrics
from robustsense.training.torch_losses import masked_weighted_bce_with_logits, training_pos_weight
from robustsense.utils.io import read_json, write_json


def mean_label_prevalence(targets: np.ndarray, target_mask: np.ndarray) -> float:
    values: list[float] = []
    for index in range(targets.shape[1]):
        known = target_mask[:, index].astype(bool)
        if known.any():
            values.append(float((targets[known, index] == 1).mean()))
    if not values:
        raise ValueError("No known labels for prevalence calculation")
    return float(np.mean(values))


def _design_matrix(dataset: MultiModalDataset, indices: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            dataset.features[indices],
            dataset.feature_masks[indices].astype(np.float32),
            dataset.availability[indices].astype(np.float32),
        ],
        axis=1,
    )


def run_random_label_sanity(
    project_root: str | Path,
    fold: int = 0,
    seed: int = 13,
    *,
    epochs: int = 1,
    batch_size: int = 4096,
    tolerance: float = 0.15,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    processed = root / "data/processed/extrasensory" / f"fold{fold}"
    manifest_path = processed / "processed_manifest.json"
    manifest = read_json(manifest_path)
    train = MultiModalDataset(processed / "train.npz", manifest_path)
    validation = MultiModalDataset(processed / "val.npz", manifest_path)
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train.features.shape[1] * 2 + train.availability.shape[1]
    model = nn.Linear(input_dim, train.targets.shape[1]).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    pos_weight = training_pos_weight(
        torch.from_numpy(train.targets), torch.from_numpy(train.target_mask), cap=20.0
    ).to(device)
    shuffled_target_rows = rng.permutation(len(train))

    for _ in range(epochs):
        order = rng.permutation(len(train))
        model.train()
        for start in range(0, len(order), batch_size):
            indices = order[start : start + batch_size]
            shuffled_indices = shuffled_target_rows[indices]
            features = torch.from_numpy(_design_matrix(train, indices)).to(device)
            targets = torch.from_numpy(train.targets[shuffled_indices]).to(device)
            mask = torch.from_numpy(train.target_mask[shuffled_indices]).to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = masked_weighted_bce_with_logits(logits, targets, mask, pos_weight)
            loss.backward()
            optimizer.step()

    model.eval()
    probabilities: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(validation), batch_size):
            indices = np.arange(start, min(start + batch_size, len(validation)))
            features = torch.from_numpy(_design_matrix(validation, indices)).to(device)
            probabilities.append(torch.sigmoid(model(features)).cpu().numpy())
    probability_matrix = np.concatenate(probabilities)
    metrics = phase2_multilabel_metrics(
        probability_matrix,
        validation.targets,
        validation.target_mask,
        list(manifest["labels"]),
        0.5,
    )
    chance_map = mean_label_prevalence(validation.targets, validation.target_mask)
    observed_map = float(metrics["mean_average_precision"])
    gap = abs(observed_map - chance_map)
    result = {
        "status": "passed" if gap <= tolerance else "failed",
        "phase": 2,
        "fold": fold,
        "seed": seed,
        "training_labels": "deterministically_row_shuffled",
        "evaluation_split": "val",
        "test_fold_read_for_tuning": False,
        "epochs": epochs,
        "sample_count": len(validation),
        "chance_mean_prevalence": chance_map,
        "shuffled_label_mean_average_precision": observed_map,
        "absolute_gap": gap,
        "tolerance": tolerance,
    }
    output = root / "reports/sanity/phase2_random_label.json"
    write_json(output, result)
    if result["status"] != "passed":
        raise RuntimeError(f"Random-label sanity check failed: {result}")
    return {"output": str(output), **result}
