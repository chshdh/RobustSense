"""Deterministic, label-preserving sensor corruption registry."""

from __future__ import annotations

import hashlib
from copy import copy
from typing import Any

import torch

from robustsense.constants import MODALITIES
from robustsense.data.dataset import add_modality_views


def _sample_seed(seed: int, epoch: int, user_id: str, timestamp: int, view: str) -> int:
    payload = f"{seed}|{epoch}|{user_id}|{timestamp}|{view}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _uniform(generator: torch.Generator, low: float, high: float) -> float:
    return low + (high - low) * float(torch.rand((), generator=generator))


class CorruptionRegistry:
    """Apply reproducible missing-modality and feature-noise transformations.

    Randomness is independently keyed by sample identity, epoch and view name, so
    results do not depend on DataLoader ordering or batch size.
    """

    def __init__(
        self,
        config: dict[str, Any],
        modality_slices: dict[str, tuple[int, int]],
        *,
        outlier_threshold: float = 4.0,
        clip_value: float = 20.0,
    ):
        self.config = config
        self.modality_slices = modality_slices
        self.outlier_threshold = float(outlier_threshold)
        self.clip_value = float(clip_value)
        if tuple(modality_slices) != MODALITIES:
            raise ValueError("Corruption registry requires the frozen modality order")

    def apply_training(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
        epoch: int,
        view: str = "train",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._apply(batch, seed=seed, epoch=epoch, view=view, scenario="mixed")

    def apply_scenario(
        self,
        batch: dict[str, Any],
        scenario: str,
        *,
        seed: int,
        severity: float = 0.5,
        epoch: int = 0,
        view: str = "controlled",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if scenario not in {"clean", "missing", "gaussian", "bias", "scale"}:
            raise ValueError(f"Unknown corruption scenario: {scenario}")
        return self._apply(
            batch,
            seed=seed,
            epoch=epoch,
            view=view,
            scenario=scenario,
            controlled_severity=float(severity),
        )

    def apply_controlled(
        self,
        batch: dict[str, Any],
        scenario: str,
        *,
        seed: int,
        view: str,
        target_modality: str | None = None,
        severity: float = 0.0,
        drop_modalities: tuple[str, ...] = (),
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Apply one fixed evaluation scenario to every sample in a batch."""
        if scenario not in {"clean", "missing", "gaussian", "bias", "scale"}:
            raise ValueError(f"Unknown controlled scenario: {scenario}")
        if target_modality is not None and target_modality not in MODALITIES:
            raise ValueError(f"Unknown target modality: {target_modality}")
        if any(name not in MODALITIES for name in drop_modalities):
            raise ValueError(f"Unknown dropped modality set: {drop_modalities}")
        corrupted = copy(batch)
        corrupted["features_flat"] = batch["features_flat"].clone()
        corrupted["feature_masks_flat"] = batch["feature_masks_flat"].clone()
        corrupted["availability"] = batch["availability"].clone()
        corrupted["quality_features"] = batch["quality_features"].clone()
        if (~corrupted["availability"].any(dim=1)).any():
            raise ValueError("Cannot corrupt a sample with no available modality")

        batch_size = corrupted["availability"].shape[0]
        device = corrupted["availability"].device
        dropped = torch.zeros_like(corrupted["availability"])
        artificial = torch.zeros_like(corrupted["availability"])
        severity_values = torch.zeros(
            (batch_size, len(MODALITIES)), dtype=torch.float32, device=device
        )
        reliability_target = corrupted["availability"].to(torch.float32).clone()
        fault_type = [["clean" for _ in MODALITIES] for _ in range(batch_size)]

        if scenario == "missing":
            if not drop_modalities:
                raise ValueError("Controlled missing scenario requires drop_modalities")
            if len(drop_modalities) >= len(MODALITIES):
                raise ValueError("Controlled corruption cannot drop every modality")
            for modality in drop_modalities:
                modality_index = MODALITIES.index(modality)
                start, end = self.modality_slices[modality]
                corrupted["features_flat"][:, start:end] = 0.0
                corrupted["feature_masks_flat"][:, start:end] = False
                corrupted["availability"][:, modality_index] = False
                dropped[:, modality_index] = True
                artificial[:, modality_index] = True
                severity_values[:, modality_index] = 1.0
                reliability_target[:, modality_index] = 0.0
                for row in fault_type:
                    row[modality_index] = "missing"
        elif scenario != "clean":
            if target_modality is None:
                raise ValueError(f"Controlled {scenario} requires target_modality")
            modality_index = MODALITIES.index(target_modality)
            start, end = self.modality_slices[target_modality]
            values = corrupted["features_flat"][:, start:end]
            observed = corrupted["feature_masks_flat"][:, start:end]
            if scenario == "gaussian":
                payload = f"{seed}|{view}|{scenario}|{target_modality}".encode()
                scenario_seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (
                    2**63 - 1
                )
                generator = torch.Generator(device="cpu").manual_seed(scenario_seed)
                noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
                noise = noise.to(values.device) * float(severity)
                values[observed] += noise[observed]
            elif scenario == "bias":
                values[observed] += float(severity)
            elif scenario == "scale":
                values[observed] *= float(severity)
            values.clamp_(-self.clip_value, self.clip_value)
            normalized = self._controlled_normalized_severity(scenario, float(severity))
            artificial[:, modality_index] = True
            severity_values[:, modality_index] = normalized
            reliability_target[:, modality_index] = max(0.0, 1.0 - normalized)
            for row in fault_type:
                row[modality_index] = scenario

        if (~corrupted["availability"].any(dim=1)).any():
            raise AssertionError("Controlled corruption disabled every modality")
        self._refresh_quality(corrupted)
        add_modality_views(corrupted, self.modality_slices)
        return corrupted, {
            "fault_type": fault_type,
            "severity": severity_values,
            "reliability_target": reliability_target,
            "artificial_fault": artificial,
            "dropped": dropped,
        }

    def _controlled_normalized_severity(self, noise_type: str, value: float) -> float:
        if noise_type == "gaussian":
            values = self.config.get("gaussian_sigma")
            denominator = max(map(abs, values)) if values is not None else float(
                self.config["gaussian_sigma_range"][1]
            )
            return min(abs(value) / max(denominator, 1e-8), 1.0)
        if noise_type == "bias":
            values = self.config.get("bias")
            if values is None:
                values = self.config["bias_range"]
            return min(abs(value) / max(max(map(abs, values)), 1e-8), 1.0)
        if noise_type == "scale":
            values = self.config.get("scale")
            if values is None:
                values = self.config["scale_range"]
            denominator = max(abs(float(item) - 1.0) for item in values)
            return min(abs(value - 1.0) / max(denominator, 1e-8), 1.0)
        raise ValueError(f"Unknown noise type: {noise_type}")

    def _apply(
        self,
        batch: dict[str, Any],
        *,
        seed: int,
        epoch: int,
        view: str,
        scenario: str,
        controlled_severity: float | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        corrupted = copy(batch)
        corrupted["features_flat"] = batch["features_flat"].clone()
        corrupted["feature_masks_flat"] = batch["feature_masks_flat"].clone()
        corrupted["availability"] = batch["availability"].clone()
        corrupted["quality_features"] = batch["quality_features"].clone()

        availability = corrupted["availability"]
        batch_size, modality_count = availability.shape
        if modality_count != len(MODALITIES):
            raise ValueError("Availability tensor does not match frozen modalities")
        if (~availability.any(dim=1)).any():
            raise ValueError("Cannot corrupt a sample with no available modality")

        device = availability.device
        dropped = torch.zeros_like(availability)
        artificial = torch.zeros_like(availability)
        severity = torch.zeros((batch_size, modality_count), device=device)
        reliability_target = availability.to(torch.float32).clone()
        fault_type = [["clean" for _ in MODALITIES] for _ in range(batch_size)]

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
            apply_view = scenario != "mixed" or (
                float(torch.rand((), generator=generator)) < float(self.config["view_probability"])
            )
            if not apply_view or scenario == "clean":
                continue

            if scenario in {"mixed", "missing"}:
                missing_count = self._missing_count(generator, scenario, controlled_severity)
                missing_count = min(missing_count, max(len(original_available) - 1, 0))
                if missing_count:
                    permutation = torch.randperm(len(original_available), generator=generator)
                    for modality_index in original_available[permutation[:missing_count]].tolist():
                        self._drop_modality(corrupted, row, modality_index)
                        dropped[row, modality_index] = True
                        artificial[row, modality_index] = True
                        severity[row, modality_index] = 1.0
                        reliability_target[row, modality_index] = 0.0
                        fault_type[row][modality_index] = "missing"

            add_noise = scenario in {"gaussian", "bias", "scale"}
            if scenario == "mixed":
                add_noise = float(torch.rand((), generator=generator)) < float(
                    self.config["noisy_modality_probability"]
                )
            if add_noise:
                remaining = torch.nonzero(corrupted["availability"][row]).flatten().cpu()
                selected = int(
                    remaining[int(torch.randint(len(remaining), (), generator=generator))]
                )
                noise_type = scenario
                if scenario == "mixed":
                    noise_types = list(self.config["noise_types"])
                    noise_type = noise_types[
                        int(torch.randint(len(noise_types), (), generator=generator))
                    ]
                value, normalized_severity = self._noise_value(
                    noise_type, generator, controlled_severity
                )
                self._noise_modality(corrupted, row, selected, noise_type, value, generator)
                artificial[row, selected] = True
                severity[row, selected] = normalized_severity
                reliability_target[row, selected] = max(0.0, 1.0 - normalized_severity)
                fault_type[row][selected] = noise_type

        if (~corrupted["availability"].any(dim=1)).any():
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

    def _missing_count(
        self,
        generator: torch.Generator,
        scenario: str,
        controlled_severity: float | None,
    ) -> int:
        if scenario == "missing":
            return max(1, int(round(controlled_severity or 1.0)))
        items = sorted(
            (int(key), float(value)) for key, value in self.config["missing_count_probs"].items()
        )
        counts = [item[0] for item in items]
        probabilities = torch.tensor([item[1] for item in items], dtype=torch.float64)
        index = int(torch.multinomial(probabilities, 1, generator=generator))
        return counts[index]

    def _noise_value(
        self,
        noise_type: str,
        generator: torch.Generator,
        controlled_severity: float | None,
    ) -> tuple[float, float]:
        if noise_type == "gaussian":
            low, high = map(float, self.config["gaussian_sigma_range"])
            value = (
                controlled_severity
                if controlled_severity is not None
                else _uniform(generator, low, high)
            )
            return value, min(abs(value) / max(abs(high), 1e-8), 1.0)
        if noise_type == "bias":
            low, high = map(float, self.config["bias_range"])
            value = (
                controlled_severity
                if controlled_severity is not None
                else _uniform(generator, low, high)
            )
            return value, min(abs(value) / max(abs(low), abs(high), 1e-8), 1.0)
        if noise_type == "scale":
            low, high = map(float, self.config["scale_range"])
            value = (
                controlled_severity
                if controlled_severity is not None
                else _uniform(generator, low, high)
            )
            denominator = max(abs(low - 1.0), abs(high - 1.0), 1e-8)
            return value, min(abs(value - 1.0) / denominator, 1.0)
        raise ValueError(f"Unknown noise type: {noise_type}")

    def _drop_modality(self, batch: dict[str, Any], row: int, modality_index: int) -> None:
        start, end = self.modality_slices[MODALITIES[modality_index]]
        batch["features_flat"][row, start:end] = 0.0
        batch["feature_masks_flat"][row, start:end] = False
        batch["availability"][row, modality_index] = False

    def _noise_modality(
        self,
        batch: dict[str, Any],
        row: int,
        modality_index: int,
        noise_type: str,
        value: float,
        generator: torch.Generator,
    ) -> None:
        start, end = self.modality_slices[MODALITIES[modality_index]]
        values = batch["features_flat"][row, start:end]
        observed = batch["feature_masks_flat"][row, start:end]
        if noise_type == "gaussian":
            noise = torch.randn(values.shape, generator=generator, dtype=values.dtype)
            noise = noise.to(values.device) * value
            values[observed] += noise[observed]
        elif noise_type == "bias":
            values[observed] += value
        elif noise_type == "scale":
            values[observed] *= value
        values.clamp_(-self.clip_value, self.clip_value)

    def _refresh_quality(self, batch: dict[str, Any]) -> None:
        features = batch["features_flat"]
        masks = batch["feature_masks_flat"]
        for modality_index, modality in enumerate(MODALITIES):
            start, end = self.modality_slices[modality]
            modality_mask = masks[:, start:end]
            observed_count = modality_mask.sum(dim=1)
            observed_fraction = observed_count.to(features.dtype) / (end - start)
            denominator = observed_count.clamp_min(1).to(features.dtype)
            abs_values = features[:, start:end].abs()
            outlier_fraction = ((abs_values > self.outlier_threshold) & modality_mask).sum(
                dim=1
            ).to(features.dtype) / denominator
            mean_abs = (abs_values * modality_mask).sum(dim=1) / denominator
            max_abs = torch.where(modality_mask, abs_values, 0.0).max(dim=1).values
            batch["quality_features"][:, modality_index] = torch.stack(
                [
                    batch["availability"][:, modality_index].to(features.dtype),
                    observed_fraction,
                    outlier_fraction,
                    mean_abs.clamp_max(self.clip_value),
                    max_abs.clamp_max(self.clip_value),
                ],
                dim=1,
            )
