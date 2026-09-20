from pathlib import Path

import numpy as np
import pytest
import torch

from robustsense.constants import MODALITIES
from robustsense.data.dataset import add_modality_views
from robustsense.models.fusion import (
    ReliabilityConstrainedFusionModel,
    reliability_constrained_weights,
    system_reliability_score,
)
from robustsense.training.phase3_losses import reliability_ranking_loss
from robustsense.training.selective import (
    apply_abstention_threshold,
    tune_abstention_threshold,
)


def _batch(batch_size: int = 3) -> dict:
    slices = {name: (index * 2, index * 2 + 2) for index, name in enumerate(MODALITIES)}
    values = torch.arange(batch_size * 12, dtype=torch.float32).reshape(batch_size, 12) / 10
    batch = {
        "features_flat": values,
        "feature_masks_flat": torch.ones((batch_size, 12), dtype=torch.bool),
        "availability": torch.ones((batch_size, 6), dtype=torch.bool),
        "quality_features": torch.ones((batch_size, 6, 5), dtype=torch.float32),
        "targets": torch.zeros((batch_size, 2)),
        "target_mask": torch.ones((batch_size, 2), dtype=torch.bool),
        "user_id": [f"user-{index}" for index in range(batch_size)],
        "timestamp": torch.arange(batch_size),
    }
    return add_modality_views(batch, slices)


def _model() -> ReliabilityConstrainedFusionModel:
    return ReliabilityConstrainedFusionModel(
        {name: 2 for name in MODALITIES},
        2,
        hidden_dim=8,
        latent_dim=4,
        reliability_hidden_dim=4,
        utility_hidden_dim=4,
        classifier_hidden_dim=8,
        dropout=0.0,
        beta_init=1.0,
    )


def test_p2_output_contract_and_inference_needs_no_fault_metadata():
    model = _model().eval()
    batch = _batch()
    batch["availability"][0, 0] = False
    output = model(batch)
    assert tuple(model.utility_nets) == MODALITIES
    assert output["logits"].shape == (3, 2)
    assert output["utility_scores"].shape == (3, 6)
    assert output["reliability"].shape == (3, 6)
    assert output["system_reliability"].shape == (3,)
    assert output["abstain"] is None
    assert torch.all((output["reliability"] >= 0) & (output["reliability"] <= 1))
    assert torch.equal(
        output["fusion_weights"][~batch["availability"]],
        torch.zeros_like(output["fusion_weights"][~batch["availability"]]),
    )
    torch.testing.assert_close(output["fusion_weights"].sum(dim=1), torch.ones(3))
    assert float(model.beta.detach()) > 0


def test_reliability_constraint_is_monotonic_for_fixed_utility():
    utility = torch.zeros((1, 6))
    availability = torch.ones((1, 6), dtype=torch.bool)
    original = torch.full((1, 6), 0.8)
    reduced = original.clone()
    reduced[0, 2] = 0.2
    original_weight = reliability_constrained_weights(
        utility, original, availability, beta=1.0, epsilon=1.0e-6
    )
    reduced_weight = reliability_constrained_weights(
        utility, reduced, availability, beta=1.0, epsilon=1.0e-6
    )
    assert reduced_weight[0, 2] < original_weight[0, 2]


def test_ranking_loss_and_system_reliability_match_hand_calculation():
    cleaner = torch.tensor([[0.9, 0.4]], requires_grad=True)
    severe = torch.tensor([[0.6, 0.5]], requires_grad=True)
    mask = torch.tensor([[True, True]])
    loss = reliability_ranking_loss(cleaner, severe, mask, margin=0.1)
    assert float(loss.detach()) == pytest.approx(0.1)
    weights = torch.tensor([[0.25, 0.75]])
    reliability = torch.tensor([[0.8, 0.4]])
    torch.testing.assert_close(
        system_reliability_score(weights, reliability), torch.tensor([0.5])
    )


def test_abstention_threshold_is_validation_only_and_meets_coverage():
    reliability = np.linspace(0.1, 1.0, 10)
    artifact = tune_abstention_threshold(
        reliability, target_coverage=0.9, source_split="val"
    )
    assert artifact["threshold"] == pytest.approx(0.2)
    assert artifact["validation_coverage"] == pytest.approx(0.9)
    assert int(apply_abstention_threshold(reliability, artifact["threshold"]).sum()) == 1
    with pytest.raises(ValueError, match="validation"):
        tune_abstention_threshold(reliability, source_split="test")


def test_p2_abstention_and_checkpoint_reload_are_deterministic(tmp_path: Path):
    batch = _batch()
    model = _model().eval()
    model.set_abstention_threshold(0.6)
    first = model(batch)
    assert first["abstain"].dtype == torch.bool
    checkpoint = tmp_path / "p2.pt"
    torch.save(model.state_dict(), checkpoint)
    reloaded = _model().eval()
    reloaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    reloaded.set_abstention_threshold(0.6)
    second = reloaded(batch)
    for key in (
        "logits",
        "fusion_weights",
        "reliability",
        "utility_scores",
        "system_reliability",
    ):
        torch.testing.assert_close(first[key], second[key], rtol=0, atol=0)
    assert torch.equal(first["abstain"], second["abstain"])


def test_p2_rejects_all_unavailable_modalities():
    batch = _batch(1)
    batch["availability"][:] = False
    with pytest.raises(ValueError, match="at least one available"):
        _model()(batch)

