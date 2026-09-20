"""Frozen Phase-4 controlled evaluation scenario matrix."""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from robustsense.constants import MODALITIES


@dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    family: str
    corruption: str
    severity: float
    target_modality: str | None = None
    drop_modalities: tuple[str, ...] = ()
    replicate: int | None = None

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["drop_modalities"] = list(self.drop_modalities)
        return value


def _fixed_drop_masks(k: int, count: int, seed: int) -> list[tuple[str, ...]]:
    combinations = list(itertools.combinations(MODALITIES, k))
    generator = np.random.default_rng(seed + k * 1009)
    order = generator.permutation(len(combinations))
    return [combinations[int(index)] for index in order[: min(count, len(combinations))]]


def build_scenarios(config: dict[str, Any]) -> list[ScenarioSpec]:
    scenarios = [
        ScenarioSpec(
            scenario_id="clean_complete",
            family="clean_complete",
            corruption="clean",
            severity=0.0,
        )
    ]
    for modality in MODALITIES:
        scenarios.append(
            ScenarioSpec(
                scenario_id=f"drop_each_one.{modality}",
                family="drop_each_one",
                corruption="missing",
                severity=1.0,
                target_modality=modality,
                drop_modalities=(modality,),
            )
        )
    mask_count = int(config.get("drop_random_masks_per_k", 5))
    scenario_seed = int(config["scenario_seed"])
    for k in config["drop_random_k"]:
        for replicate, mask in enumerate(_fixed_drop_masks(int(k), mask_count, scenario_seed)):
            mask_key = "-".join(mask)
            scenarios.append(
                ScenarioSpec(
                    scenario_id=f"drop_random_k.k{k}.mask{replicate}.{mask_key}",
                    family="drop_random_k",
                    corruption="missing",
                    severity=float(k),
                    drop_modalities=mask,
                    replicate=replicate,
                )
            )
    noise_families = (
        ("gaussian_noise", "gaussian", config["gaussian_sigma"]),
        ("bias_drift", "bias", config["bias"]),
        ("scale_error", "scale", config["scale"]),
    )
    for family, corruption, severities in noise_families:
        for modality in MODALITIES:
            for severity in severities:
                scenarios.append(
                    ScenarioSpec(
                        scenario_id=f"{family}.{modality}.{float(severity):g}",
                        family=family,
                        corruption=corruption,
                        severity=float(severity),
                        target_modality=modality,
                    )
                )
    identifiers = [scenario.scenario_id for scenario in scenarios]
    if len(identifiers) != len(set(identifiers)):
        raise AssertionError("Controlled scenario identifiers are not unique")
    return scenarios


def scenario_manifest_sha256(scenarios: list[ScenarioSpec]) -> str:
    payload = "\n".join(scenario.scenario_id for scenario in scenarios) + "\n"
    return hashlib.sha256(payload.encode()).hexdigest()
