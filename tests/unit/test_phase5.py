from pathlib import Path

import pytest
import torch

from robustsense.constants import MODALITIES
from robustsense.data.dataset import add_modality_views
from robustsense.experiments.registry import (
    ProtocolMutationError,
    RunRegistry,
    build_run_plan,
)
from robustsense.models.fusion import build_phase3_model
from robustsense.training.phase3_trainer import (
    _resolved_corruption_config,
    _training_policy,
)


def _quality_config() -> dict:
    return {
        "hidden_dim": 8,
        "embedding_dim": 4,
        "quality_dim": 5,
        "reliability_hidden_dim": 4,
        "gate_hidden_dim": 4,
        "classifier_hidden_dim": 8,
        "dropout": 0.0,
        "reliability_loss_weight": 0.1,
        "consistency_loss_weight": 0.1,
    }


def _batch() -> dict:
    slices = {name: (index * 2, index * 2 + 2) for index, name in enumerate(MODALITIES)}
    batch = {
        "features_flat": torch.zeros((2, 12)),
        "feature_masks_flat": torch.ones((2, 12), dtype=torch.bool),
        "availability": torch.ones((2, 6), dtype=torch.bool),
        "quality_features": torch.ones((2, 6, 5)),
    }
    return add_modality_views(batch, slices)


def test_ablation_architectures_and_training_policies_are_isolated():
    dims = {name: 2 for name in MODALITIES}
    no_quality = build_phase3_model(
        "ablation-a3-no-quality", dims, 3, _quality_config()
    )
    no_reliability = build_phase3_model(
        "ablation-a4-no-reliability", dims, 3, _quality_config()
    )
    assert no_quality(_batch())["reliability"].shape == (2, 6)
    assert no_reliability(_batch())["reliability"] is None

    a1 = _training_policy("ablation-a1-no-sensor-dropout", _quality_config())
    a2 = _training_policy("ablation-a2-no-noise", _quality_config())
    a5 = _training_policy("ablation-a5-no-reliability-loss", _quality_config())
    base = {
        "missing_count_probs": {"0": 0.4, "1": 0.4, "2": 0.2},
        "noisy_modality_probability": 0.3,
    }
    assert _resolved_corruption_config(base, a1)["missing_count_probs"] == {
        "0": 1.0,
        "1": 0.0,
        "2": 0.0,
    }
    assert _resolved_corruption_config(base, a2)["noisy_modality_probability"] == 0.0
    assert a5["reliability_loss_weight"] == 0.0
    assert a5["consistency_loss_weight"] == 0.1


def test_checked_in_phase5_matrix_is_internally_consistent():
    root = Path(__file__).resolve().parents[2]
    plan = root / "configs/experiment/phase5_plan.yaml"
    assert len(build_run_plan(root, plan, "ablation_dev")) == 6
    assert len(build_run_plan(root, plan, "credible")) == 60
    assert len(build_run_plan(root, plan, "full")) == 45


def test_registry_recovers_interruption_and_rejects_protocol_change(tmp_path: Path):
    registry = RunRegistry(tmp_path)
    planned = [
        {
            "run_key": "credible|mlp|fold0|seed13",
            "group": "credible",
            "profile": "credible",
            "model_name": "mlp",
            "fold": "0",
            "seed": "13",
            "status": "pending",
            "attempt_count": "0",
            "created_at": "2026-09-10T00:00:00+00:00",
            "updated_at": "2026-09-10T00:00:00+00:00",
            "run_dir": "runs/example",
            "reason": "registered_before_execution",
        }
    ]
    registry.register(planned, "hash-one")
    registry.update(planned[0]["run_key"], status="running", attempt_count=1)
    assert registry.recover_interrupted() == 1
    row = registry.rows()[0]
    assert row["status"] == "failed"
    assert "--retry-failed" in row["reason"]
    with pytest.raises(ProtocolMutationError):
        registry.register(planned, "hash-two")
