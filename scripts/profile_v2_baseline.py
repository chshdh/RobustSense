"""Profile the V1 quality-aware path without training a persistent model."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import torch
from torch.optim import AdamW

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import MultiModalDataset
from robustsense.models.fusion import build_phase3_model
from robustsense.training.phase3_losses import consistency_mse, reliability_mse
from robustsense.training.torch_losses import masked_weighted_bce_with_logits
from robustsense.training.trainer import _loader, _move_batch
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json
from robustsense.utils.reproducibility import set_global_seed

T = TypeVar("T")


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(device: torch.device, operation: Callable[[], T]) -> tuple[T, float]:
    _sync(device)
    started = time.perf_counter()
    result = operation()
    _sync(device)
    return result, (time.perf_counter() - started) * 1000.0


def _summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered))))
    return {
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
        "min_ms": min(values),
        "max_ms": max(values),
    }


def profile_baseline(
    project_root: Path,
    *,
    fold: int,
    batch_size: int,
    warmup_steps: int,
    measured_steps: int,
    seed: int,
    requested_device: str,
) -> dict[str, Any]:
    root = project_root.resolve()
    if requested_device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    processed = root / "data/processed/extrasensory" / f"fold{fold}"
    manifest_path = processed / "processed_manifest.json"
    manifest = read_json(manifest_path)
    preprocessor = read_json(processed / "preprocessor.json")
    slices = {name: tuple(manifest["modality_slices"][name]) for name in MODALITIES}
    dimensions = {name: end - start for name, (start, end) in slices.items()}
    model_config = load_config(root / "configs/model/quality_aware.yaml")
    corruption_config = load_config(root / "configs/corruption/train.yaml")
    registry = CorruptionRegistry(
        corruption_config,
        slices,
        outlier_threshold=float(preprocessor["outlier_threshold"]),
        clip_value=float(preprocessor["clip_value"]),
    )

    set_global_seed(seed)
    model = build_phase3_model(
        "quality-aware", dimensions, len(manifest["labels"]), model_config
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.0001)
    pos_weight = torch.ones(len(manifest["labels"]), device=device)
    train_dataset = MultiModalDataset(processed / "train.npz", manifest_path)
    val_dataset = MultiModalDataset(processed / "val.npz", manifest_path)
    train_loader = _loader(train_dataset, batch_size, True, seed, device)
    val_loader = _loader(val_dataset, batch_size, False, seed, device)
    train_iterator = iter(train_loader)
    val_iterator = iter(val_loader)

    timings: dict[str, list[float]] = {
        "dataloader_wait": [],
        "batch_transfer": [],
        "corruption": [],
        "forward": [],
        "backward_optimizer": [],
        "validation": [],
        "controlled_evaluation": [],
    }
    raw_rows: list[dict[str, float | int]] = []

    def next_train() -> dict[str, Any]:
        nonlocal train_iterator
        try:
            return next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            return next(train_iterator)

    def next_val() -> dict[str, Any]:
        nonlocal val_iterator
        try:
            return next(val_iterator)
        except StopIteration:
            val_iterator = iter(val_loader)
            return next(val_iterator)

    def one_step(step: int) -> dict[str, float]:
        batch, data_ms = _timed(device, next_train)
        (corrupted, fault_metadata), corruption_ms = _timed(
            device,
            lambda: registry.apply_training(batch, seed=seed, epoch=0, view="v2-phase0"),
        )

        def transfer() -> tuple[dict[str, Any], dict[str, Any], torch.Tensor]:
            return (
                _move_batch(corrupted, device, slices),
                _move_batch(batch, device, slices),
                fault_metadata["reliability_target"].to(device, non_blocking=True),
            )

        (train_batch, clean_batch, reliability_target), transfer_ms = _timed(device, transfer)

        def forward() -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
            model.eval()
            with torch.no_grad():
                clean_logits = model(clean_batch)["logits"]
            model.train()
            return model(train_batch), clean_logits

        (output, clean_logits), forward_ms = _timed(device, forward)

        def backward_optimizer() -> None:
            optimizer.zero_grad(set_to_none=True)
            valid = model.valid_sample_mask(train_batch)
            effective_mask = train_batch["target_mask"] & valid.unsqueeze(1)
            classification = masked_weighted_bce_with_logits(
                output["logits"], train_batch["targets"], effective_mask, pos_weight
            )
            reliability = reliability_mse(output["reliability"], reliability_target)
            consistency = consistency_mse(clean_logits, output["logits"], effective_mask)
            loss = classification + 0.1 * reliability + 0.1 * consistency
            loss.backward()
            optimizer.step()

        _, backward_ms = _timed(device, backward_optimizer)

        def validation() -> None:
            val_batch = _move_batch(next_val(), device, slices)
            model.eval()
            with torch.inference_mode():
                model(val_batch)

        _, validation_ms = _timed(device, validation)

        def controlled_evaluation() -> None:
            controlled, _ = registry.apply_controlled(
                next_val(),
                "gaussian",
                seed=seed,
                view="v2-phase0-controlled",
                target_modality="phone_acc",
                severity=0.5,
            )
            controlled = _move_batch(controlled, device, slices)
            model.eval()
            with torch.inference_mode():
                model(controlled)

        _, controlled_ms = _timed(device, controlled_evaluation)
        return {
            "dataloader_wait": data_ms,
            "batch_transfer": transfer_ms,
            "corruption": corruption_ms,
            "forward": forward_ms,
            "backward_optimizer": backward_ms,
            "validation": validation_ms,
            "controlled_evaluation": controlled_ms,
        }

    for step in range(warmup_steps):
        one_step(step)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    measured_started = time.perf_counter()
    for step in range(measured_steps):
        row = one_step(warmup_steps + step)
        raw_rows.append({"step": step, **row})
        for name, value in row.items():
            timings[name].append(value)
    _sync(device)
    elapsed_seconds = time.perf_counter() - measured_started

    io_dir = root / "work/v2/performance_io"
    io_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(raw_rows)
    _, json_write_ms = _timed(
        device, lambda: frame.to_json(io_dir / "timings.json", orient="records", indent=2)
    )
    _, parquet_write_ms = _timed(
        device, lambda: frame.to_parquet(io_dir / "timings.parquet", index=False)
    )
    timings["json_write"] = [json_write_ms]
    timings["parquet_write"] = [parquet_write_ms]

    phase_means = {name: statistics.fmean(values) for name, values in timings.items()}
    training_phases = (
        "dataloader_wait",
        "batch_transfer",
        "corruption",
        "forward",
        "backward_optimizer",
    )
    training_step_ms = sum(phase_means[name] for name in training_phases)
    bottleneck = max(training_phases, key=phase_means.__getitem__)
    result: dict[str, Any] = {
        "version": 1,
        "phase": "V2-0",
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "purpose": "V1 performance baseline before any P2 implementation",
        "guardrails": {
            "persistent_training_run_created": False,
            "test_split_read": False,
            "accuracy_metrics_computed": False,
            "source_splits": ["train", "val"],
        },
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "workload": {
            "model": "quality-aware (V1 P)",
            "fold": fold,
            "seed": seed,
            "batch_size": batch_size,
            "warmup_steps": warmup_steps,
            "measured_steps": measured_steps,
            "train_samples": len(train_dataset),
            "validation_samples": len(val_dataset),
        },
        "timings": {name: _summary(values) for name, values in timings.items()},
        "analysis": {
            "estimated_training_step_ms": training_step_ms,
            "estimated_training_samples_per_second": batch_size / (training_step_ms / 1000.0),
            "measured_all_operations_samples_per_second": (
                batch_size * measured_steps / elapsed_seconds
            ),
            "largest_training_phase": bottleneck,
            "largest_training_phase_share": phase_means[bottleneck] / training_step_ms,
            "cuda_peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
            ),
        },
        "raw_steps": raw_rows,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("reports/v2/performance_baseline.json"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measured-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.measured_steps < 1 or args.warmup_steps < 0:
        raise ValueError("measured-steps must be positive and warmup-steps cannot be negative")
    root = args.project_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    result = profile_baseline(
        root,
        fold=args.fold,
        batch_size=args.batch_size,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        seed=args.seed,
        requested_device=args.device,
    )
    write_json(output, result)
    print(json.dumps({"output": str(output), **result["analysis"]}, indent=2))


if __name__ == "__main__":
    main()
