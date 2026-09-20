"""Profile V2-3 training choices on train data only; never open validation or test."""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.corruption.vectorized import VectorizedTrainingCorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.models.v2_phase3 import build_v2_phase3_model
from robustsense.training.phase3_losses import (
    consistency_mse,
    reliability_mse,
    reliability_ranking_loss,
)
from robustsense.training.torch_losses import (
    masked_weighted_bce_with_logits,
    training_pos_weight,
)
from robustsense.training.trainer import _loader, _move_batch
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed


def _measure_corruption(
    registry: CorruptionRegistry,
    batches: list[dict[str, Any]],
) -> float:
    started = time.perf_counter()
    for batch_index, batch in enumerate(batches):
        registry.apply_training(batch, seed=13, epoch=1, view=f"profile-{batch_index}")
    return time.perf_counter() - started


def _training_steps(
    dataset: MultiModalDataset,
    *,
    batch_size: int,
    step_count: int,
    device: torch.device,
    modality_slices: dict[str, tuple[int, int]],
    modality_dims: dict[str, int],
    model_config: dict[str, Any],
    corruption_config: dict[str, Any],
    outlier_threshold: float,
    clip_value: float,
) -> dict[str, Any]:
    set_global_seed(13)
    model = build_v2_phase3_model(
        "P2", modality_dims, len(dataset.metadata["labels"]), model_config
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    registry = VectorizedTrainingCorruptionRegistry(
        corruption_config,
        modality_slices,
        outlier_threshold=outlier_threshold,
        clip_value=clip_value,
    )
    sample_mask = dataset.availability.any(axis=1)
    pos_weight = training_pos_weight(
        torch.from_numpy(dataset.targets),
        torch.from_numpy(dataset.target_mask & sample_mask[:, None]),
    ).to(device)
    loader = _loader(dataset, batch_size, False, 13, device)
    timings = []
    samples = 0
    torch.cuda.reset_peak_memory_stats(device)
    for batch_index, batch in enumerate(loader):
        if batch_index >= step_count + 1:
            break
        started = time.perf_counter()
        corrupted, metadata = registry.apply_training(
            batch, seed=13, epoch=1, view=f"profile-{batch_index}"
        )
        corruption_finished = time.perf_counter()
        clean = _move_batch(batch, device, modality_slices)
        corrupted = _move_batch(corrupted, device, modality_slices)
        reliability_target = metadata["reliability_target"].to(device, non_blocking=True)
        artificial_fault = metadata["artificial_fault"].to(device, non_blocking=True)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        clean_output = model(clean)
        output = model(corrupted)
        effective_mask = corrupted["target_mask"] & model.valid_sample_mask(corrupted).unsqueeze(1)
        classification = masked_weighted_bce_with_logits(
            output["logits"], corrupted["targets"], effective_mask, pos_weight
        )
        reliability = reliability_mse(output["reliability"], reliability_target)
        consistency = consistency_mse(
            clean_output["logits"], output["logits"], effective_mask
        )
        ranking = reliability_ranking_loss(
            clean_output["reliability"],
            output["reliability"],
            artificial_fault,
            margin=0.1,
        )
        loss = classification + 0.1 * reliability + 0.1 * consistency + 0.05 * ranking
        loss.backward()
        clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        torch.cuda.synchronize(device)
        finished = time.perf_counter()
        if batch_index:
            timings.append(
                {
                    "corruption_seconds": corruption_finished - started,
                    "total_seconds": finished - started,
                }
            )
            samples += len(batch["user_id"])
    elapsed = sum(row["total_seconds"] for row in timings)
    return {
        "batch_size": batch_size,
        "measured_steps": len(timings),
        "measured_samples": samples,
        "samples_per_second": samples / elapsed,
        "mean_step_seconds": float(np.mean([row["total_seconds"] for row in timings])),
        "mean_corruption_seconds": float(
            np.mean([row["corruption_seconds"] for row in timings])
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / (1024**2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_3_performance_preflight.json")
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    processed = root / "data/processed/extrasensory/fold0"
    manifest_path = processed / "processed_manifest.json"
    manifest = read_json(manifest_path)
    dataset = MultiModalDataset(processed / "train.npz", manifest_path)
    modality_slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    modality_dims = {name: end - start for name, (start, end) in modality_slices.items()}
    model_config = load_config(root / "configs/v2/models/reliability_constrained.yaml")
    corruption_config = load_config(root / "configs/v2/corruption/train.yaml")
    preprocessor = read_json(processed / "preprocessor.json")
    sample_loader = _loader(dataset, 512, False, 13, torch.device("cpu"))
    batches = []
    for batch_index, batch in enumerate(sample_loader):
        if batch_index == arguments.steps:
            break
        batches.append(batch)
    registry_arguments = {
        "outlier_threshold": float(preprocessor["outlier_threshold"]),
        "clip_value": float(preprocessor["clip_value"]),
    }
    reference = CorruptionRegistry(
        corruption_config, modality_slices, **registry_arguments
    )
    vectorized = VectorizedTrainingCorruptionRegistry(
        corruption_config, modality_slices, **registry_arguments
    )
    reference_seconds = _measure_corruption(reference, batches)
    vectorized_seconds = _measure_corruption(vectorized, batches)
    device = torch.device("cuda")
    throughput = [
        _training_steps(
            dataset,
            batch_size=batch_size,
            step_count=arguments.steps,
            device=device,
            modality_slices=modality_slices,
            modality_dims=modality_dims,
            model_config=model_config,
            corruption_config=corruption_config,
            outlier_threshold=float(preprocessor["outlier_threshold"]),
            clip_value=float(preprocessor["clip_value"]),
        )
        for batch_size in (512, 1024, 2048, 4096, 8192, 16384)
    ]
    result = {
        "phase": "V2-3",
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "source_split": "train",
        "test_split_read": False,
        "fold": 0,
        "seed": 13,
        "corruption": {
            "batch_size": 512,
            "measured_batches": len(batches),
            "reference_seconds": reference_seconds,
            "vectorized_seconds": vectorized_seconds,
            "speedup": reference_seconds / vectorized_seconds,
            "exact_equivalence_test": "tests/unit/test_vectorized_corruption.py",
        },
        "throughput": throughput,
    }
    output = arguments.output if arguments.output.is_absolute() else root / arguments.output
    write_json(output, result)
    print(result)


if __name__ == "__main__":
    main()
