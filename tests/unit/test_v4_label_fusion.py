import numpy as np
import pytest
import torch

from robustsense.constants import MODALITIES
from robustsense.models.v2_phase3 import build_v2_phase3_model
from robustsense.models.v4_label_fusion import (
    build_label_conditioned_model,
    initialize_label_conditioned_from_p2,
    label_conditioned_weights,
)


def _config():
    return {
        "hidden_dim": 8,
        "embedding_dim": 4,
        "quality_dim": 5,
        "reliability_hidden_dim": 3,
        "utility_hidden_dim": 3,
        "classifier_hidden_dim": 6,
        "dropout": 0.0,
        "beta_init": 1.0,
        "epsilon": 1.0e-6,
    }


def _batch(batch_size=3):
    dims = {name: 2 for name in MODALITIES}
    availability = torch.ones(batch_size, len(MODALITIES), dtype=torch.bool)
    availability[0, 0] = False
    return dims, {
        "features": {
            name: torch.randn(batch_size, dims[name]) for name in MODALITIES
        },
        "feature_masks": {
            name: torch.ones(batch_size, dims[name], dtype=torch.bool)
            for name in MODALITIES
        },
        "quality_features": torch.rand(batch_size, len(MODALITIES), 5),
        "availability": availability,
    }


def test_label_conditioned_weights_mask_and_normalize_each_label():
    utilities = torch.randn(2, 3, 4)
    reliability = torch.rand(2, 3)
    availability = torch.tensor([[True, False, True], [False, True, True]])
    weights = label_conditioned_weights(
        utilities, reliability, availability, beta=1.0, epsilon=1.0e-6
    )
    assert weights.shape == (2, 4, 3)
    torch.testing.assert_close(weights.sum(dim=2), torch.ones(2, 4))
    assert torch.count_nonzero(weights[:, :, 1][0]) == 0
    assert torch.count_nonzero(weights[:, :, 0][1]) == 0


def test_p2_initialization_is_logit_and_weight_equivalent():
    torch.manual_seed(7)
    dims, batch = _batch()
    p2 = build_v2_phase3_model("P2", dims, 5, _config()).eval()
    candidate = build_label_conditioned_model(dims, 5, _config()).eval()
    initialize_label_conditioned_from_p2(candidate, p2.state_dict())
    with torch.inference_mode():
        source = p2(batch)
        target = candidate(batch)
    torch.testing.assert_close(target["logits"], source["logits"], atol=1e-6, rtol=0)
    for label in range(5):
        torch.testing.assert_close(
            target["label_fusion_weights"][:, label],
            source["fusion_weights"],
            atol=1e-6,
            rtol=0,
        )
    torch.testing.assert_close(
        target["fusion_weights"], source["fusion_weights"], atol=1e-6, rtol=0
    )


def test_adapter_freeze_leaves_only_utility_parameters_trainable():
    dims, _ = _batch()
    model = build_label_conditioned_model(dims, 5, _config())
    model.freeze_p2_backbone_for_adapter_training()
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    assert trainable
    assert all(name.startswith("utility_nets.") for name in trainable)


def test_bad_reliability_shape_is_rejected():
    with pytest.raises(ValueError, match="match utility"):
        label_conditioned_weights(
            torch.randn(2, 3, 4),
            torch.rand(2, 4),
            torch.ones(2, 3, dtype=torch.bool),
            beta=1.0,
            epsilon=1.0e-6,
        )


def test_model_emits_label_specific_diagnostics():
    torch.manual_seed(11)
    dims, batch = _batch(batch_size=2)
    model = build_label_conditioned_model(dims, 5, _config()).eval()
    output = model(batch)
    assert output["logits"].shape == (2, 5)
    assert output["label_fusion_weights"].shape == (2, 5, len(MODALITIES))
    assert output["fusion_weights"].shape == (2, len(MODALITIES))
    assert output["label_system_reliability"].shape == (2, 5)
    assert np.isfinite(output["logits"].detach().numpy()).all()
