import json
from pathlib import Path

import torch

from robustsense.data.extrasensory import run_extrasensory_audit
from robustsense.data.real import prepare_extrasensory
from robustsense.training.v2_phase3_trainer import (
    derive_p2_a4_run,
    evaluate_v2_phase3,
    state_dict_sha256,
    train_v2_phase3,
)
from robustsense.utils.io import read_json
from tests.integration.test_phase1_audit import (
    create_official_shape_archives,
    fixture_config,
)


def _write_configs(root: Path, data_config: dict) -> None:
    values = {
        "configs/data/extrasensory.yaml": data_config,
        "configs/v2/models/reliability_constrained.yaml": {
            "hidden_dim": 8,
            "embedding_dim": 4,
            "quality_dim": 5,
            "reliability_hidden_dim": 4,
            "utility_hidden_dim": 4,
            "classifier_hidden_dim": 8,
            "dropout": 0.0,
            "reliability_loss_weight": 0.1,
            "consistency_loss_weight": 0.1,
            "ranking_loss_weight": 0.05,
            "ranking_margin": 0.1,
            "beta_init": 1.0,
            "epsilon": 1.0e-6,
            "target_coverage": 0.9,
        },
        "configs/v2/corruption/train.yaml": {
            "view_probability": 1.0,
            "missing_count_probs": {"0": 0.0, "1": 1.0, "2": 0.0},
            "noisy_modality_probability": 1.0,
            "noise_types": ["gaussian", "bias", "scale"],
            "gaussian_sigma_range": [0.1, 1.0],
            "bias_range": [-0.75, 0.75],
            "scale_range": [0.7, 1.3],
        },
        "configs/v2/evaluation/seed13_credible.yaml": {
            "profile": "v2_seed13",
            "folds": [0],
            "seeds": [13],
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "pos_weight_cap": 20.0,
            "early_stopping_patience": 1,
            "threshold_grid": [0.25, 0.5, 0.75],
            "risk_coverage_grid": [0.0, 0.5, 1.0],
            "device": "cpu",
            "run_root": "runs/v2/phase_v2_3",
        },
        "configs/v2/ablations/phase_v2_3.yaml": {},
        "configs/v2/phase_v2_3_plan.yaml": {
            "data_config": "configs/data/extrasensory.yaml",
            "model_config": "configs/v2/models/reliability_constrained.yaml",
            "corruption_config": "configs/v2/corruption/train.yaml",
            "experiment_config": "configs/v2/evaluation/seed13_credible.yaml",
            "ablation_config": "configs/v2/ablations/phase_v2_3.yaml",
            "folds": [0],
            "seeds": [13],
        },
    }
    for relative, value in values.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")


def test_v2_phase3_train_evaluate_and_a4_derivation(tmp_path: Path):
    create_official_shape_archives(tmp_path)
    config = fixture_config()
    run_extrasensory_audit(config, tmp_path, fold=0)
    prepare_extrasensory(config, tmp_path, fold=0)
    _write_configs(tmp_path, config)

    p2_run = train_v2_phase3(
        tmp_path,
        variant="P2",
        fold=0,
        seed=13,
        protocol_sha256="fixture-protocol",
    )
    assert not (p2_run / "test_metrics.json").exists()
    assert read_json(p2_run / "thresholds.json")["source_split"] == "val"
    assert read_json(p2_run / "abstention_threshold.json")["source_split"] == "val"
    p2_metrics = evaluate_v2_phase3(p2_run)
    assert p2_metrics["selective"]["threshold_source_split"] == "val"
    assert (p2_run / "risk_coverage.json").is_file()

    variant_runs = {}
    for variant in ("P2-A1", "P2-A2", "P2-A3"):
        variant_run = train_v2_phase3(
            tmp_path,
            variant=variant,
            fold=0,
            seed=13,
            protocol_sha256="fixture-protocol",
        )
        assert not (variant_run / "test_metrics.json").exists()
        variant_metrics = evaluate_v2_phase3(variant_run)
        assert variant_metrics["variant"] == variant
        variant_runs[variant] = variant_run
    a1_resolved = read_json(variant_runs["P2-A1"] / "resolved_config.yaml")
    assert a1_resolved["learned_beta"] is None
    a2_log = (variant_runs["P2-A2"] / "train_log.csv").read_text(encoding="utf-8")
    assert a2_log.splitlines()[1].split(",")[5] == "0.0"

    a4_run = derive_p2_a4_run(
        tmp_path,
        fold=0,
        seed=13,
        protocol_sha256="fixture-protocol",
    )
    assert not (a4_run / "abstention_threshold.json").exists()
    assert not (a4_run / "test_metrics.json").exists()
    a4_metrics = evaluate_v2_phase3(a4_run)
    assert a4_metrics["selective"] is None
    assert not (a4_run / "risk_coverage.json").exists()
    assert a4_metrics["forced"]["macro_f1"] == p2_metrics["forced"]["macro_f1"]

    p2_state = torch.load(p2_run / "best_checkpoint.pt", weights_only=False)["state_dict"]
    a4_state = torch.load(a4_run / "best_checkpoint.pt", weights_only=False)["state_dict"]
    assert state_dict_sha256(p2_state) == state_dict_sha256(a4_state)
