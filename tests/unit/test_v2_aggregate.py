import numpy as np
import pytest

from robustsense.evaluation.v2_aggregate import (
    aggregate_mask_metrics,
    validate_shared_scenario_samples,
)
from robustsense.evaluation.v2_scenarios import build_exhaustive_masks


def test_mask_aggregation_reports_required_views_and_preserves_empty_group_nan():
    masks = build_exhaustive_masks()
    rows = [
        {
            "mask_id": mask.mask_id,
            "available_count": mask.available_count,
            "macro_f1": np.nan if mask.available_count == 1 else mask.available_count / 6,
        }
        for mask in masks
    ]
    frequencies = {mask.mask_id: mask.available_count for mask in masks}
    result = aggregate_mask_metrics(rows, frequencies)
    assert result["valid_metric_count"] == 57
    assert np.isnan(result["by_available_count"][0]["mean_macro_f1"])
    assert result["by_available_count"][-1]["worst_macro_f1"] == pytest.approx(1.0)
    assert np.isfinite(result["natural_frequency_weighted_macro_f1"])


def test_shared_scenario_sample_contract_rejects_model_hash_drift():
    rows = [
        {"scenario_id": "mask-a", "model_id": "P", "scenario_sample_sha256": "same"},
        {"scenario_id": "mask-a", "model_id": "P2", "scenario_sample_sha256": "same"},
    ]
    assert validate_shared_scenario_samples(rows, expected_models={"P", "P2"}) == {
        "mask-a": "same"
    }
    rows[1]["scenario_sample_sha256"] = "different"
    with pytest.raises(ValueError, match="sample hash"):
        validate_shared_scenario_samples(rows, expected_models={"P", "P2"})
