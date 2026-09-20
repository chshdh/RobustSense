import unittest

import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry
from robustsense.data.dataset import add_modality_views
from robustsense.evaluation.scenarios import build_scenarios, scenario_manifest_sha256
from robustsense.models.fusion import (
    GatedFusionModel,
    QualityAwareFusionModel,
    masked_softmax,
)
from robustsense.training.phase3_losses import consistency_mse, reliability_mse


def _batch() -> tuple[dict, dict[str, tuple[int, int]], dict[str, int]]:
    slices = {name: (index * 2, index * 2 + 2) for index, name in enumerate(MODALITIES)}
    dims = {name: 2 for name in MODALITIES}
    batch = {
        "features_flat": torch.arange(36, dtype=torch.float32).reshape(3, 12) / 10,
        "feature_masks_flat": torch.ones((3, 12), dtype=torch.bool),
        "availability": torch.ones((3, 6), dtype=torch.bool),
        "quality_features": torch.ones((3, 6, 5), dtype=torch.float32),
        "targets": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, float("nan")]]),
        "target_mask": torch.tensor([[True, True], [True, True], [True, False]]),
        "user_id": ["a", "b", "c"],
        "timestamp": torch.tensor([1, 2, 3]),
    }
    return add_modality_views(batch, slices), slices, dims


def _config() -> dict:
    return {
        "view_probability": 1.0,
        "missing_count_probs": {"0": 0.0, "1": 1.0, "2": 0.0},
        "noisy_modality_probability": 1.0,
        "noise_types": ["gaussian", "bias", "scale"],
        "gaussian_sigma_range": [0.1, 1.0],
        "bias_range": [-0.75, 0.75],
        "scale_range": [0.7, 1.3],
        "keep_at_least_one": True,
    }


class Phase3FusionTest(unittest.TestCase):
    def test_masked_softmax_exact_contract(self):
        scores = torch.tensor([[1.0, 2.0, 9.0], [1.0, -1.0, 3.0]])
        available = torch.tensor([[True, True, False], [False, True, True]])
        weights = masked_softmax(scores, available)
        self.assertTrue(torch.equal(weights[~available], torch.zeros_like(weights[~available])))
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))
        with self.assertRaises(ValueError):
            masked_softmax(torch.zeros((1, 3)), torch.zeros((1, 3), dtype=torch.bool))

    def test_models_return_weights_and_optional_reliability(self):
        batch, _, dims = _batch()
        batch["availability"][0, 0] = False
        batch["feature_masks_flat"][0, :2] = False
        batch["features_flat"][0, :2] = 0
        gated = GatedFusionModel(dims, 2, hidden_dim=8, latent_dim=4)
        output = gated(batch)
        self.assertEqual(tuple(output["logits"].shape), (3, 2))
        self.assertEqual(float(output["fusion_weights"][0, 0].detach()), 0.0)
        torch.testing.assert_close(output["fusion_weights"].sum(dim=1), torch.ones(3))

        quality = QualityAwareFusionModel(
            dims,
            2,
            hidden_dim=8,
            latent_dim=4,
            reliability_hidden_dim=4,
            gate_hidden_dim=4,
            classifier_hidden_dim=8,
        )
        # Inference deliberately has no artificial-fault severity fields.
        quality_output = quality(batch)
        self.assertEqual(tuple(quality_output["reliability"].shape), (3, 6))
        self.assertEqual(float(quality_output["fusion_weights"][0, 0].detach()), 0.0)


class CorruptionRegistryTest(unittest.TestCase):
    def test_seed_reproduces_masks_noise_and_metadata(self):
        batch, slices, _ = _batch()
        registry = CorruptionRegistry(_config(), slices)
        first, first_meta = registry.apply_training(batch, seed=13, epoch=2)
        second, second_meta = registry.apply_training(batch, seed=13, epoch=2)
        self.assertTrue(torch.equal(first["availability"], second["availability"]))
        self.assertTrue(torch.equal(first["feature_masks_flat"], second["feature_masks_flat"]))
        torch.testing.assert_close(first["features_flat"], second["features_flat"], rtol=0, atol=0)
        self.assertEqual(first_meta["fault_type"], second_meta["fault_type"])
        torch.testing.assert_close(first_meta["severity"], second_meta["severity"])
        self.assertTrue(first["availability"].any(dim=1).all())
        torch.testing.assert_close(batch["targets"], first["targets"], equal_nan=True)
        self.assertTrue(torch.equal(batch["target_mask"], first["target_mask"]))

    def test_clean_missing_and_noise_scenarios(self):
        batch, slices, _ = _batch()
        registry = CorruptionRegistry(_config(), slices)
        clean, clean_meta = registry.apply_scenario(batch, "clean", seed=7)
        torch.testing.assert_close(clean["features_flat"], batch["features_flat"])
        self.assertFalse(clean_meta["artificial_fault"].any())

        missing, missing_meta = registry.apply_scenario(batch, "missing", seed=7, severity=2)
        self.assertTrue((missing_meta["dropped"].sum(dim=1) == 2).all())
        self.assertTrue(missing["availability"].any(dim=1).all())

        noisy, noisy_meta = registry.apply_scenario(batch, "gaussian", seed=7, severity=0.5)
        self.assertTrue((noisy_meta["artificial_fault"].sum(dim=1) == 1).all())
        self.assertFalse(torch.equal(noisy["features_flat"], batch["features_flat"]))
        self.assertEqual(noisy["quality_features"].shape[-1], 5)

    def test_auxiliary_losses(self):
        reliability = torch.tensor([[0.2, 0.8]], requires_grad=True)
        target = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(float(reliability_mse(reliability, target).detach()), 0.04, places=6)
        clean = torch.tensor([[0.0, 1.0]])
        corrupt = torch.tensor([[0.0, 0.0]], requires_grad=True)
        mask = torch.tensor([[True, False]])
        self.assertEqual(float(consistency_mse(clean, corrupt, mask).detach()), 0.0)

    def test_controlled_corruption_targets_fixed_modalities(self):
        batch, slices, _ = _batch()
        registry = CorruptionRegistry(_config(), slices)
        first, first_meta = registry.apply_controlled(
            batch,
            "gaussian",
            seed=13,
            view="scenario|batch0",
            target_modality="audio",
            severity=0.5,
        )
        second, second_meta = registry.apply_controlled(
            batch,
            "gaussian",
            seed=13,
            view="scenario|batch0",
            target_modality="audio",
            severity=0.5,
        )
        torch.testing.assert_close(first["features_flat"], second["features_flat"], rtol=0, atol=0)
        self.assertTrue(
            torch.equal(first_meta["artificial_fault"], second_meta["artificial_fault"])
        )
        audio_index = MODALITIES.index("audio")
        self.assertTrue(first_meta["artificial_fault"][:, audio_index].all())
        self.assertEqual(int(first_meta["artificial_fault"].sum()), len(batch["user_id"]))

        missing, _ = registry.apply_controlled(
            batch,
            "missing",
            seed=13,
            view="drop",
            drop_modalities=("phone_acc", "watch_acc", "audio"),
        )
        self.assertTrue((missing["availability"].sum(dim=1) == 3).all())

    def test_phase4_scenario_matrix_is_frozen_and_unique(self):
        config = {
            "drop_random_k": [1, 2, 3],
            "gaussian_sigma": [0.25, 0.5, 1.0, 2.0],
            "bias": [0.25, 0.5, 1.0],
            "scale": [0.5, 0.75, 1.25, 1.5],
            "scenario_seed": 13,
            "drop_random_masks_per_k": 5,
        }
        scenarios = build_scenarios(config)
        self.assertEqual(len(scenarios), 88)
        self.assertEqual(scenarios[0].scenario_id, "clean_complete")
        self.assertEqual(len({scenario.scenario_id for scenario in scenarios}), 88)
        self.assertEqual(
            scenario_manifest_sha256(scenarios), scenario_manifest_sha256(build_scenarios(config))
        )


if __name__ == "__main__":
    unittest.main()
