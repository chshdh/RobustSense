"""Aggregation and cross-model sample-contract checks for V2-2."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite_summary(values: np.ndarray) -> tuple[float, float, int]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return np.nan, np.nan, 0
    return float(finite.mean()), float(finite.min()), int(len(finite))


def aggregate_mask_metrics(
    rows: list[dict[str, Any]], natural_mask_frequencies: dict[str, int]
) -> dict[str, Any]:
    """Aggregate one model/fold/seed's complete 63-mask result table."""
    mask_ids = [str(row["mask_id"]) for row in rows]
    if len(rows) != 63 or len(set(mask_ids)) != 63:
        raise ValueError("Mask aggregation requires exactly 63 unique rows")
    values = np.asarray([row["macro_f1"] for row in rows], dtype=np.float64)
    macro_mean, worst, valid_count = _finite_summary(values)
    frequencies = np.asarray(
        [max(0, int(natural_mask_frequencies.get(mask_id, 0))) for mask_id in mask_ids],
        dtype=np.float64,
    )
    usable = np.isfinite(values) & (frequencies > 0)
    weighted = (
        float(np.average(values[usable], weights=frequencies[usable]))
        if usable.any()
        else np.nan
    )
    by_available_count = []
    for count in range(1, 7):
        selected = np.asarray(
            [int(row["available_count"]) == count for row in rows], dtype=bool
        )
        mean, minimum, group_valid_count = _finite_summary(values[selected])
        by_available_count.append(
            {
                "available_count": count,
                "mask_count": int(selected.sum()),
                "valid_metric_count": group_valid_count,
                "mean_macro_f1": mean,
                "worst_macro_f1": minimum,
            }
        )
    return {
        "macro_mean_macro_f1": macro_mean,
        "natural_frequency_weighted_macro_f1": weighted,
        "worst_macro_f1": worst,
        "valid_metric_count": valid_count,
        "by_available_count": by_available_count,
    }


def validate_shared_scenario_samples(
    rows: list[dict[str, Any]], *, expected_models: set[str]
) -> dict[str, str]:
    """Require every model in a scenario to use the same ordered sample set."""
    by_scenario: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
    if not by_scenario:
        raise ValueError("At least one scenario row is required")
    hashes: dict[str, str] = {}
    for scenario_id, scenario_rows in sorted(by_scenario.items()):
        models = {str(row["model_id"]) for row in scenario_rows}
        if models != expected_models:
            raise ValueError(
                f"Scenario {scenario_id} has models {sorted(models)}, "
                f"expected {sorted(expected_models)}"
            )
        sample_hashes = {str(row["scenario_sample_sha256"]) for row in scenario_rows}
        if len(sample_hashes) != 1:
            raise ValueError(f"Scenario {scenario_id} does not share one sample hash")
        hashes[scenario_id] = sample_hashes.pop()
    return hashes
