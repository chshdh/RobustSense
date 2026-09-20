"""Batch-write optimization for sample-stable V2 training corruption."""

from __future__ import annotations

from copy import copy
from typing import Any

import torch

from robustsense.constants import MODALITIES
from robustsense.corruption.registry import CorruptionRegistry, _sample_seed
from robustsense.data.dataset import add_modality_views


class VectorizedTrainingCorruptionRegistry(CorruptionRegistry):
    """Preserve V1 random draws while grouping expensive feature writes."""

    def apply_training(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
        epoch: int,
        view: str = "train",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        corrupted = copy(batch)
        corrupted["features_flat"] = batch["features_flat"].clone()
        corrupted["feature_masks_flat"] = batch["feature_masks_flat"].clone()
        corrupted["availability"] = batch["availability"].clone()
        corrupted["quality_features"] = batch["quality_features"].clone()
        availability = corrupted["availability"]
        if availability.shape[1] != len(MODALITIES):
            raise ValueError("Availability tensor does not match frozen modalities")
        if (~availability.any(dim=1)).any():
            raise ValueError("Cannot corrupt a sample with no available modality")

        batch_size = len(batch["user_id"])
        dropped = torch.zeros_like(availability)
        artificial = torch.zeros_like(availability)
        severity = torch.zeros((batch_size, len(MODALITIES)), dtype=torch.float32)
        reliability_target = availability.to(torch.float32).clone()
        fault_type = [["clean" for _ in MODALITIES] for _ in range(batch_size)]
        dropped_rows: dict[int, list[int]] = {index: [] for index in range(len(MODALITIES))}
        noise_rows: dict[tuple[int, str], list[tuple[int, float, torch.Tensor | None]]] = {}

        for row in range(batch_size):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(
                _sample_seed(
                    seed,
                    epoch,
                    str(batch["user_id"][row]),
                    int(batch["timestamp"][row]),
                    view,
                )
            )
            original_available = torch.nonzero(availability[row]).flatten().cpu()
            apply_view = float(torch.rand((), generator=generator)) < float(
                self.config["view_probability"]
            )
            if not apply_view:
                continue

            missing_count = self._missing_count(generator, "mixed", None)
            missing_count = min(missing_count, max(len(original_available) - 1, 0))
            if missing_count:
                permutation = torch.randperm(len(original_available), generator=generator)
                selected_drops = original_available[permutation[:missing_count]].tolist()
                for modality_index in selected_drops:
                    availability[row, modality_index] = False
                    dropped[row, modality_index] = True
                    artificial[row, modality_index] = True
                    severity[row, modality_index] = 1.0
                    reliability_target[row, modality_index] = 0.0
                    fault_type[row][modality_index] = "missing"
                    dropped_rows[modality_index].append(row)

            add_noise = float(torch.rand((), generator=generator)) < float(
                self.config["noisy_modality_probability"]
            )
            if not add_noise:
                continue
            remaining = torch.nonzero(availability[row]).flatten().cpu()
            modality_index = int(
                remaining[int(torch.randint(len(remaining), (), generator=generator))]
            )
            noise_types = list(self.config["noise_types"])
            noise_type = noise_types[int(torch.randint(len(noise_types), (), generator=generator))]
            value, normalized = self._noise_value(noise_type, generator, None)
            start, end = self.modality_slices[MODALITIES[modality_index]]
            gaussian = None
            if noise_type == "gaussian":
                gaussian = torch.randn(
                    (end - start,), generator=generator, dtype=batch["features_flat"].dtype
                )
            noise_rows.setdefault((modality_index, noise_type), []).append(
                (row, value, gaussian)
            )
            artificial[row, modality_index] = True
            severity[row, modality_index] = normalized
            reliability_target[row, modality_index] = max(0.0, 1.0 - normalized)
            fault_type[row][modality_index] = noise_type

        for modality_index, rows in dropped_rows.items():
            if not rows:
                continue
            start, end = self.modality_slices[MODALITIES[modality_index]]
            corrupted["features_flat"][rows, start:end] = 0.0
            corrupted["feature_masks_flat"][rows, start:end] = False

        for (modality_index, noise_type), plans in noise_rows.items():
            start, end = self.modality_slices[MODALITIES[modality_index]]
            rows = [plan[0] for plan in plans]
            values = corrupted["features_flat"][rows, start:end]
            observed = corrupted["feature_masks_flat"][rows, start:end]
            amounts = torch.tensor(
                [plan[1] for plan in plans], dtype=values.dtype
            ).unsqueeze(1)
            if noise_type == "gaussian":
                noise = torch.stack([plan[2] for plan in plans])
                changed = values + noise * amounts
            elif noise_type == "bias":
                changed = values + amounts
            elif noise_type == "scale":
                changed = values * amounts
            else:
                raise ValueError(f"Unknown noise type: {noise_type}")
            changed = changed.clamp(-self.clip_value, self.clip_value)
            corrupted["features_flat"][rows, start:end] = torch.where(
                observed, changed, values
            )

        if (~availability.any(dim=1)).any():
            raise AssertionError("Corruption disabled every modality for a sample")
        self._refresh_quality(corrupted)
        add_modality_views(corrupted, self.modality_slices)
        return corrupted, {
            "fault_type": fault_type,
            "severity": severity,
            "reliability_target": reliability_target,
            "artificial_fault": artificial,
            "dropped": dropped,
        }
