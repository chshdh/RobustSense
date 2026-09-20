import torch

from robustsense.constants import MODALITIES
from robustsense.models.torch_baselines import build_phase2_model


def sample_batch() -> dict:
    modality_dims = {name: 2 for name in MODALITIES}
    availability = torch.tensor(
        [[True, False, True, True, False, True], [False, True, False, False, True, True]]
    )
    return {
        "features": {name: torch.zeros(2, 2) for name in MODALITIES},
        "feature_masks": {name: torch.ones(2, 2, dtype=torch.bool) for name in MODALITIES},
        "features_flat": torch.zeros(2, 12),
        "feature_masks_flat": torch.ones(2, 12, dtype=torch.bool),
        "availability": availability,
        "quality_features": torch.zeros(2, 6, 5),
        "targets": torch.zeros(2, 3),
        "target_mask": torch.ones(2, 3, dtype=torch.bool),
        "modality_dims": modality_dims,
    }


def test_late_fusion_masks_unavailable_modalities():
    batch = sample_batch()
    model = build_phase2_model("late", batch["modality_dims"], 3, {})

    output = model(batch)

    weights = output["fusion_weights"]
    assert weights is not None
    assert torch.all(weights[~batch["availability"]] == 0)
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))


def test_single_sensor_exposes_valid_sample_mask():
    batch = sample_batch()
    model = build_phase2_model("single-phone_acc", batch["modality_dims"], 3, {})

    torch.testing.assert_close(model.valid_sample_mask(batch), torch.tensor([True, False]))


def test_eval_inference_is_repeatable_for_same_batch():
    batch = sample_batch()
    model = build_phase2_model("mlp", batch["modality_dims"], 3, {})
    model.eval()

    with torch.inference_mode():
        first = model(batch)["logits"]
        second = model(batch)["logits"]

    assert first is not None and second is not None
    torch.testing.assert_close(first, second)
