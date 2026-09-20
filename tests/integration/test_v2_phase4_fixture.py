import json
from pathlib import Path

from robustsense.data.extrasensory import run_extrasensory_audit
from robustsense.data.real import prepare_extrasensory
from robustsense.evaluation.v2_phase4 import evaluate_v2_phase4_extended
from robustsense.training.v2_phase4_trainer import (
    evaluate_v2_phase4,
    evaluate_v2_phase4_v1,
    train_v2_phase4,
    train_v2_phase4_v1,
)
from robustsense.utils.io import read_json
from tests.integration.test_phase1_audit import (
    create_official_shape_archives,
    fixture_config,
)
from tests.integration.test_v2_phase3_fixture import _write_configs


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v2_phase4_new_seed_uses_isolated_identity_and_validation_thresholds(
    tmp_path: Path,
):
    create_official_shape_archives(tmp_path)
    config = fixture_config()
    run_extrasensory_audit(config, tmp_path, fold=0)
    prepare_extrasensory(config, tmp_path, fold=0)
    _write_configs(tmp_path, config)
    _write(
        tmp_path / "configs/v2/evaluation/multiseed_core.yaml",
        {
            "profile": "v2_multiseed_core",
            "folds": [0],
            "seeds": [13, 29, 47],
            "new_training_seeds": [29, 47],
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
            "run_root": "runs/v2/phase_v2_4",
        },
    )
    _write(
        tmp_path / "configs/v2/phase_v2_4_plan.yaml",
        {
            "data_config": "configs/data/extrasensory.yaml",
            "model_config": "configs/v2/models/reliability_constrained.yaml",
            "corruption_config": "configs/v2/corruption/train.yaml",
            "experiment_config": "configs/v2/evaluation/multiseed_core.yaml",
            "folds": [0],
            "seeds": [13, 29, 47],
            "new_training_seeds": [29, 47],
        },
    )
    run = train_v2_phase4(
        tmp_path,
        variant="P2",
        fold=0,
        seed=29,
        protocol_sha256="fixture-v2-4",
    )
    assert run.name == "extrasensory-p2-fold0-seed29-v2_multiseed"
    resolved = read_json(run / "resolved_config.yaml")
    assert resolved["phase"] == "V2-4"
    assert resolved["test_split_opened_during_training"] is False
    assert not (run / "test_metrics.json").exists()
    result = evaluate_v2_phase4(run)
    assert result["phase"] == "V2-4"
    assert result["threshold_source_split"] == "val"
    assert read_json(run / "run_manifest.json")["status"] == "success_test_evaluated_once"

    _write(
        tmp_path / "configs/v2/evaluation/extended_core.yaml",
        {
            "profile": "v2_multiseed_extended",
            "models": ["P2"],
            "folds": [0],
            "seeds": [29],
            "scenario_seed": 2404,
            "scenario_group_size": 16,
            "persistent_episode_lengths": [1],
            "persistent_faults": ["drop", "gaussian_sigma_2"],
            "persistent_target_modalities": [
                "phone_acc",
                "phone_gyro",
                "watch_acc",
                "location",
                "audio",
                "phone_state",
            ],
            "max_gap_seconds": 90,
            "calibration_bins": 4,
            "fault_detection_threshold": 0.5,
            "recovery_tolerance": 0.05,
        },
    )
    extended = evaluate_v2_phase4_extended(run, model_id="P2")
    assert extended["mask_scenario_count"] == 63
    assert extended["mixed_scenario_count"] == 60
    assert extended["persistent_scenario_count"] == 12
    assert extended["scenario_group_size"] == 16
    assert extended["per_sample_predictions_stored"] is False
    assert (run / "extended_v2_4/mask_metrics.csv").is_file()
    assert (run / "extended_v2_4/persistent_response.csv").is_file()

    _write(
        tmp_path / "configs/model/gated.yaml",
        {
            "name": "gated",
            "hidden_dim": 8,
            "embedding_dim": 4,
            "gate_hidden_dim": 4,
            "classifier_hidden_dim": 8,
            "dropout": 0.0,
        },
    )
    _write(
        tmp_path / "configs/corruption/train.yaml",
        {
            "view_probability": 1.0,
            "missing_count_probs": {"0": 0.0, "1": 1.0, "2": 0.0},
            "noisy_modality_probability": 1.0,
            "noise_types": ["gaussian", "bias", "scale"],
            "gaussian_sigma_range": [0.1, 1.0],
            "bias_range": [-0.75, 0.75],
            "scale_range": [0.7, 1.3],
        },
    )
    _write(
        tmp_path / "configs/experiment/full.yaml",
        {
            "profile": "full",
            "folds": [0],
            "seeds": [13, 29, 47],
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "gradient_clip_norm": 5.0,
            "pos_weight_cap": 20.0,
            "early_stopping_patience": 1,
            "threshold_grid": [0.25, 0.5, 0.75],
            "device": "cpu",
            "run_root": "runs",
        },
    )
    v1_run = train_v2_phase4_v1(
        tmp_path,
        model_name="gated",
        fold=0,
        seed=29,
        protocol_sha256="fixture-v2-4",
    )
    assert not (v1_run / "test_metrics.json").exists()
    assert read_json(v1_run / "resolved_config.yaml")[
        "test_split_opened_during_training"
    ] is False
    v1_result = evaluate_v2_phase4_v1(v1_run)
    assert v1_result["suite"] == "v2_multiseed_core"
    assert read_json(v1_run / "evaluation_manifest.json")[
        "test_opened_by_separate_evaluation_process"
    ] is True
