import pytest
import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.corruption.vectorized import VectorizedTrainingCorruptionRegistry
from robustsense.data.dataset import add_modality_views


def _config(noise_type: str) -> dict:
    return {
        "view_probability": 0.8,
        "missing_count_probs": {"0": 0.4, "1": 0.4, "2": 0.2},
        "noisy_modality_probability": 0.8,
        "noise_types": [noise_type],
        "gaussian_sigma_range": [0.1, 1.0],
        "bias_range": [-0.75, 0.75],
        "scale_range": [0.7, 1.3],
    }


def _batch() -> tuple[dict, dict[str, tuple[int, int]]]:
    slices = {name: (index * 3, index * 3 + 3) for index, name in enumerate(MODALITIES)}
    size = 24
    generator = torch.Generator().manual_seed(7)
    feature_masks = torch.rand((size, 18), generator=generator) > 0.15
    availability = torch.ones((size, 6), dtype=torch.bool)
    batch = {
        "features_flat": torch.randn((size, 18), generator=generator),
        "feature_masks_flat": feature_masks,
        "availability": availability,
        "quality_features": torch.ones((size, 6, 5), dtype=torch.float32),
        "targets": torch.zeros((size, 2)),
        "target_mask": torch.ones((size, 2), dtype=torch.bool),
        "user_id": [f"user-{index % 5}" for index in range(size)],
        "timestamp": torch.arange(size),
    }
    return add_modality_views(batch, slices), slices


@pytest.mark.parametrize("noise_type", ["gaussian", "bias", "scale"])
def test_vectorized_training_corruption_is_exactly_equal(noise_type: str):
    batch, slices = _batch()
    expected_batch, expected_meta = CorruptionRegistry(
        _config(noise_type), slices
    ).apply_training(batch, seed=13, epoch=2, view="train")
    actual_batch, actual_meta = VectorizedTrainingCorruptionRegistry(
        _config(noise_type), slices
    ).apply_training(batch, seed=13, epoch=2, view="train")

    for key in ("features_flat", "feature_masks_flat", "availability", "quality_features"):
        torch.testing.assert_close(actual_batch[key], expected_batch[key], rtol=0, atol=0)
    assert actual_meta["fault_type"] == expected_meta["fault_type"]
    for key in ("severity", "reliability_target", "artificial_fault", "dropped"):
        torch.testing.assert_close(actual_meta[key], expected_meta[key], rtol=0, atol=0)
