import numpy as np
import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import add_modality_views
from robustsense.evaluation.v2_scenarios import (
    apply_mixed_failure,
    availability_mask_manifest_bytes,
    availability_mask_manifest_sha256,
    build_exhaustive_masks,
    build_mixed_failures,
    build_persistent_episodes,
    mixed_failure_manifest_bytes,
    persistent_episode_manifest_bytes,
    scenario_sample_sha256,
)


def _batch() -> tuple[dict, dict[str, tuple[int, int]]]:
    slices = {name: (index * 2, index * 2 + 2) for index, name in enumerate(MODALITIES)}
    batch = {
        "features_flat": torch.arange(24, dtype=torch.float32).reshape(2, 12) / 10,
        "feature_masks_flat": torch.ones((2, 12), dtype=torch.bool),
        "availability": torch.ones((2, 6), dtype=torch.bool),
        "quality_features": torch.ones((2, 6, 5), dtype=torch.float32),
        "targets": torch.zeros((2, 2)),
        "target_mask": torch.ones((2, 2), dtype=torch.bool),
        "user_id": ["a", "b"],
        "timestamp": torch.tensor([1, 2]),
    }
    return add_modality_views(batch, slices), slices


def _corruption_config() -> dict:
    return {
        "view_probability": 1.0,
        "missing_count_probs": {"0": 0.0, "1": 1.0},
        "noisy_modality_probability": 1.0,
        "noise_types": ["gaussian"],
        "gaussian_sigma_range": [0.1, 2.0],
        "bias_range": [-1.0, 1.0],
        "scale_range": [0.5, 1.5],
    }


def test_exhaustive_mask_contract_is_complete_unique_and_byte_stable():
    masks = build_exhaustive_masks()
    assert len(masks) == 63
    assert {mask.available_count for mask in masks} == {1, 2, 3, 4, 5, 6}
    assert {mask.bits for mask in masks} == {
        tuple(bool(code & (1 << index)) for index in range(6))
        for code in range(1, 64)
    }
    assert availability_mask_manifest_bytes(masks) == availability_mask_manifest_bytes(
        build_exhaustive_masks()
    )
    assert availability_mask_manifest_sha256(masks) == availability_mask_manifest_sha256(
        build_exhaustive_masks()
    )


def test_mixed_failure_matrix_and_application_are_deterministic_and_disjoint():
    scenarios = build_mixed_failures()
    assert len(scenarios) == 60
    assert all(item.drop_modality != item.noisy_modality for item in scenarios)
    assert mixed_failure_manifest_bytes(scenarios) == mixed_failure_manifest_bytes(
        build_mixed_failures()
    )
    scenario = scenarios[0]
    batch, slices = _batch()
    registry = CorruptionRegistry(_corruption_config(), slices)
    first, first_meta = apply_mixed_failure(
        registry, batch, scenario, seed=13, view="test"
    )
    second, second_meta = apply_mixed_failure(
        registry, batch, scenario, seed=13, view="test"
    )
    drop_index = MODALITIES.index(scenario.drop_modality)
    noisy_index = MODALITIES.index(scenario.noisy_modality)
    drop_start, drop_end = slices[scenario.drop_modality]
    noisy_start, noisy_end = slices[scenario.noisy_modality]
    assert torch.equal(first["features_flat"][:, drop_start:drop_end], torch.zeros((2, 2)))
    assert not torch.equal(
        first["features_flat"][:, noisy_start:noisy_end],
        batch["features_flat"][:, noisy_start:noisy_end],
    )
    assert first_meta["artificial_fault"][:, [drop_index, noisy_index]].all()
    assert torch.equal(first_meta["artificial_fault"], second_meta["artificial_fault"])
    torch.testing.assert_close(first["features_flat"], second["features_flat"], rtol=0, atol=0)


def test_persistent_episodes_respect_users_gaps_and_report_exclusions():
    users = np.array(["a"] * 12 + ["b"] * 4 + ["c"] * 6)
    timestamps = np.array(list(range(12)) + list(range(4)) + [0, 1, 2, 200, 201, 202])
    episodes, summary = build_persistent_episodes(
        users, timestamps, episode_lengths=(2,), max_gap_seconds=90
    )
    assert len(episodes) == 2
    assert all(episode.user_id == "a" for episode in episodes)
    assert all(len(episode.fault_indices) == 2 for episode in episodes)
    assert summary == [
        {
            "fault_length": 2,
            "episode_count": 2,
            "excluded_short_segment_count": 3,
            "unused_row_count": 10,
        }
    ]
    assert persistent_episode_manifest_bytes(episodes) == persistent_episode_manifest_bytes(
        build_persistent_episodes(
            users, timestamps, episode_lengths=(2,), max_gap_seconds=90
        )[0]
    )


def test_scenario_sample_hash_is_data_based_not_model_based():
    users = np.array(["a", "b", "a"])
    timestamps = np.array([1, 2, 3])
    first = scenario_sample_sha256(users, timestamps)
    second = scenario_sample_sha256(users.copy(), timestamps.copy())
    assert first == second
    assert first != scenario_sample_sha256(users, np.array([1, 2, 4]))
