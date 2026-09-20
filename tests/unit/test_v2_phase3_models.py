import pytest
import torch

from robustsense.constants import MODALITIES
from robustsense.data.dataset import add_modality_views
from robustsense.models.fusion import ReliabilityConstrainedFusionModel
from robustsense.models.v2_phase3 import (
    build_v2_phase3_model,
    copy_p2_weights_for_equivalence,
)


def _batch(size: int = 4) -> dict:
    slices = {name: (index * 2, index * 2 + 2) for index, name in enumerate(MODALITIES)}
    generator = torch.Generator().manual_seed(17)
    batch = {
        "features_flat": torch.randn((size, 12), generator=generator),
        "feature_masks_flat": torch.ones((size, 12), dtype=torch.bool),
        "availability": torch.ones((size, 6), dtype=torch.bool),
        "quality_features": torch.randn((size, 6, 5), generator=generator),
        "targets": torch.zeros((size, 2)),
        "target_mask": torch.ones((size, 2), dtype=torch.bool),
        "user_id": [f"u-{index}" for index in range(size)],
        "timestamp": torch.arange(size),
    }
    return add_modality_views(batch, slices)


def _config() -> dict:
    return {
        "hidden_dim": 8,
        "embedding_dim": 4,
        "quality_dim": 5,
        "reliability_hidden_dim": 4,
        "utility_hidden_dim": 4,
        "classifier_hidden_dim": 8,
        "dropout": 0.0,
        "beta_init": 1.0,
        "epsilon": 1.0e-6,
    }


def test_phase_v2_3_complete_variant_is_exactly_v2_1_p2():
    base = ReliabilityConstrainedFusionModel(
        {name: 2 for name in MODALITIES},
        2,
        hidden_dim=8,
        latent_dim=4,
        quality_dim=5,
        reliability_hidden_dim=4,
        utility_hidden_dim=4,
        classifier_hidden_dim=8,
        dropout=0.0,
    ).eval()
    variant = build_v2_phase3_model(
        "P2", {name: 2 for name in MODALITIES}, 2, _config()
    ).eval()
    copy_p2_weights_for_equivalence(variant, base)
    expected = base(_batch())
    actual = variant(_batch())
    for key in (
        "logits",
        "fusion_weights",
        "reliability",
        "utility_scores",
        "system_reliability",
    ):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_a1_weights_are_utility_only_even_when_quality_changes():
    model = build_v2_phase3_model(
        "P2-A1", {name: 2 for name in MODALITIES}, 2, _config()
    ).eval()
    original = _batch()
    changed = _batch()
    changed["quality_features"] = changed["quality_features"] + 10.0
    first = model(original)
    second = model(changed)
    torch.testing.assert_close(first["fusion_weights"], second["fusion_weights"], rtol=0, atol=0)
    assert not torch.equal(first["reliability"], second["reliability"])


def test_a3_reliability_cannot_read_quality_and_a4_rejects_abstention():
    a3 = build_v2_phase3_model(
        "P2-A3", {name: 2 for name in MODALITIES}, 2, _config()
    ).eval()
    original = _batch()
    changed = _batch()
    changed["quality_features"] = changed["quality_features"] + 10.0
    first = a3(original)
    second = a3(changed)
    torch.testing.assert_close(first["reliability"], second["reliability"], rtol=0, atol=0)
    torch.testing.assert_close(first["fusion_weights"], second["fusion_weights"], rtol=0, atol=0)

    a4 = build_v2_phase3_model(
        "P2-A4", {name: 2 for name in MODALITIES}, 2, _config()
    )
    with pytest.raises(ValueError, match="removes selective"):
        a4.set_abstention_threshold(0.5)
