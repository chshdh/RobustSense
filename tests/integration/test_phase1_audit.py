import csv
import gzip
import hashlib
import io
import json
import zipfile
from pathlib import Path

from robustsense.constants import LABELS
from robustsense.data.extrasensory import run_extrasensory_audit
from robustsense.data.real import prepare_extrasensory
from robustsense.evaluation.report import generate_report
from robustsense.evaluation.suite import evaluate_full_suite
from robustsense.training.phase3_trainer import train_phase3
from robustsense.training.trainer import train_phase2
from robustsense.utils.io import read_json

USERS = [f"10000000-0000-0000-0000-{index:012d}" for index in range(10)]
FEATURES = (
    "raw_acc:mean",
    "proc_gyro:mean",
    "watch_acceleration:mean",
    "location:variance",
    "location_quick_features:speed",
    "audio_naive:mfcc0:mean",
    "discrete:screen_on",
    "raw_magnet:mean",
    "watch_heading:mean",
    "lf_measurements:battery_level",
)


def fixture_config() -> dict:
    return {
        "raw_dir": "data/raw",
        "extracted_dir": "data/extracted",
        "interim_dir": "data/interim",
        "manifest_dir": "data/manifests",
        "report_dir": "reports/data_audit",
        "feature_archive": "features.zip",
        "fold_archive": "folds.zip",
        "timestamp_column": "timestamp",
        "label_prefix": "label:",
        "metadata_columns": ["label_source"],
        "labels": list(LABELS),
        "modalities": {
            "phone_acc": ["^raw_acc:"],
            "phone_gyro": ["^proc_gyro:"],
            "watch_acc": ["^watch_acceleration:"],
            "location": ["^location:", "^location_quick_features:"],
            "audio": ["^audio_naive:"],
            "phone_state": ["^discrete:"],
        },
        "ignored_feature_patterns": [
            "^raw_magnet:",
            "^watch_heading:",
            "^lf_measurements:",
        ],
        "expected_user_count": 10,
        "expected_sample_count_range": [20, 20],
        "audit_chunk_size": 2,
    }


def compressed_user_csv(user_index: int) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["timestamp", *FEATURES, *(f"label:{name}" for name in LABELS), "label_source"])
    for row in range(2):
        labels = [
            "nan" if (row + label_index) % 7 == 0 else (row + label_index + user_index) % 2
            for label_index in range(len(LABELS))
        ]
        writer.writerow([1_700_000_000 + row, *([0.25] * len(FEATURES)), *labels, 0])
    return gzip.compress(stream.getvalue().encode("utf-8"))


def create_official_shape_archives(root: Path) -> None:
    raw = root / "data/raw"
    raw.mkdir(parents=True)
    with zipfile.ZipFile(raw / "features.zip", "w") as bundle:
        for index, user in enumerate(USERS):
            bundle.writestr(f"{user}.features_labels.csv.gz", compressed_user_csv(index))

    android = USERS[::2]
    iphone = USERS[1::2]
    with zipfile.ZipFile(raw / "folds.zip", "w") as bundle:
        for fold in range(5):
            for platform, platform_users in (("android", android), ("iphone", iphone)):
                test_user = platform_users[fold]
                train_users = [user for user in platform_users if user != test_user]
                bundle.writestr(f"cv_5_folds/fold_{fold}_test_{platform}_uuids.txt", test_user)
                bundle.writestr(
                    f"cv_5_folds/fold_{fold}_train_{platform}_uuids.txt",
                    "\n".join(train_users),
                )


def test_phase1_fixture_rebuilds_complete_audit(tmp_path: Path):
    create_official_shape_archives(tmp_path)

    result = run_extrasensory_audit(fixture_config(), tmp_path, fold=0)

    assert result["status"] == "passed"
    assert result["user_count"] == 10
    assert result["sample_count"] == 20
    report = tmp_path / "reports/data_audit"
    expected = {
        "dataset_summary.json",
        "feature_schema.csv",
        "modality_missingness.csv",
        "label_prevalence.csv",
        "label_by_fold.csv",
        "user_sample_counts.csv",
        "modality_coavailability.png",
        "label_cooccurrence.png",
        "audit_report.md",
    }
    assert expected <= {path.name for path in report.iterdir()}
    assert read_json(report / "dataset_summary.json")["label_value_contract"].startswith("0/1/NaN")
    split = read_json(tmp_path / "data/manifests/extrasensory_split_fold0.json")
    assert {len(split[name]) for name in ("train", "val", "test")} == {2, 6}


def test_small_real_shape_fixture_prepares_and_trains_b2_b4_p_and_p2(tmp_path: Path):
    create_official_shape_archives(tmp_path)
    config = fixture_config()
    run_extrasensory_audit(config, tmp_path, fold=0)
    prepared = prepare_extrasensory(config, tmp_path, fold=0)
    assert prepared["status"] == "passed"
    assert prepared["split_counts"]["train"]["kept_rows"] == 12

    data_config = tmp_path / "configs/data/extrasensory.yaml"
    data_config.parent.mkdir(parents=True)
    data_config.write_text(json.dumps(config), encoding="utf-8")
    experiment = tmp_path / "configs/experiment/dev.yaml"
    experiment.parent.mkdir(parents=True)
    experiment.write_text(
        json.dumps(
            {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "device": "cpu",
                "run_root": "runs",
                "threshold_grid": [0.25, 0.5, 0.75],
            }
        ),
        encoding="utf-8",
    )
    model_config = tmp_path / "configs/model/mlp_early.yaml"
    model_config.parent.mkdir(parents=True)
    model_config.write_text(
        json.dumps({"hidden_dim": 8, "latent_dim": 4, "dropout": 0.0}),
        encoding="utf-8",
    )
    gated_config = tmp_path / "configs/model/gated.yaml"
    gated_config.write_text(
        json.dumps(
            {
                "hidden_dim": 8,
                "embedding_dim": 4,
                "gate_hidden_dim": 4,
                "classifier_hidden_dim": 8,
                "dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )
    quality_config = tmp_path / "configs/model/quality_aware.yaml"
    quality_config.write_text(
        json.dumps(
            {
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
        ),
        encoding="utf-8",
    )
    p2_config = tmp_path / "configs/v2/models/reliability_constrained.yaml"
    p2_config.parent.mkdir(parents=True)
    p2_config.write_text(
        json.dumps(
            {
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
            }
        ),
        encoding="utf-8",
    )
    v2_experiment = tmp_path / "configs/v2/evaluation/dev.yaml"
    v2_experiment.parent.mkdir(parents=True)
    v2_experiment.write_text(
        json.dumps(
            {
                "epochs": 1,
                "batch_size": 4,
                "learning_rate": 0.001,
                "weight_decay": 0.0001,
                "gradient_clip_norm": 5.0,
                "device": "cpu",
                "run_root": "runs/v2",
                "threshold_grid": [0.25, 0.5, 0.75],
            }
        ),
        encoding="utf-8",
    )
    corruption_config = tmp_path / "configs/corruption/train.yaml"
    corruption_config.parent.mkdir(parents=True)
    corruption_config.write_text(
        json.dumps(
            {
                "view_probability": 1.0,
                "missing_count_probs": {"0": 0.0, "1": 1.0, "2": 0.0},
                "noisy_modality_probability": 1.0,
                "noise_types": ["gaussian", "bias", "scale"],
                "gaussian_sigma_range": [0.1, 1.0],
                "bias_range": [-0.75, 0.75],
                "scale_range": [0.7, 1.3],
                "keep_at_least_one": True,
            }
        ),
        encoding="utf-8",
    )
    v2_corruption = tmp_path / "configs/v2/corruption/train.yaml"
    v2_corruption.parent.mkdir(parents=True)
    v2_corruption.write_text(corruption_config.read_text(encoding="utf-8"), encoding="utf-8")
    evaluation_corruption = tmp_path / "configs/corruption/eval.yaml"
    evaluation_corruption.write_text(
        json.dumps(
            {
                "drop_random_k": [1, 2, 3],
                "gaussian_sigma": [0.25, 0.5, 1.0, 2.0],
                "bias": [0.25, 0.5, 1.0],
                "scale": [0.5, 0.75, 1.25, 1.5],
                "scenario_seed": 13,
                "drop_random_masks_per_k": 5,
                "dev_complete_sample_limit": 4,
            }
        ),
        encoding="utf-8",
    )
    evaluation_plan = tmp_path / "configs/evaluation/dev.yaml"
    evaluation_plan.parent.mkdir(parents=True)
    evaluation_plan.write_text(
        json.dumps(
            {
                "name": "fixture",
                "profile": "dev",
                "folds": [0],
                "seeds": [13],
                "models": ["mlp", "gated", "quality-aware"],
                "controlled_models": ["mlp", "gated", "quality-aware"],
                "suite_version": 1,
            }
        ),
        encoding="utf-8",
    )

    run_dir = train_phase2(tmp_path, data_config, "mlp", 0, 13, "dev")

    assert (run_dir / "best_checkpoint.pt").is_file()
    assert (run_dir / "thresholds.json").is_file()
    assert (run_dir / "test_metrics.json").is_file()
    assert (run_dir / "predictions.parquet").is_file()
    assert read_json(run_dir / "thresholds.json")["source_split"] == "val"

    phase3_runs = []
    for model_name in ("gated", "quality-aware"):
        phase3_run = train_phase3(tmp_path, data_config, model_name, 0, 13, "dev")
        phase3_runs.append(phase3_run)
        assert (phase3_run / "best_checkpoint.pt").is_file()
        assert (phase3_run / "modality_diagnostics.parquet").is_file()
        assert read_json(phase3_run / "test_metrics.json")["model_name"] == model_name

    p2_run = train_phase3(
        tmp_path, data_config, "reliability-constrained", 0, 13, "dev"
    )
    assert p2_run.parent == tmp_path / "runs/v2"
    assert (p2_run / "best_checkpoint.pt").is_file()
    assert not (p2_run / "test_metrics.json").exists()
    abstention = read_json(p2_run / "abstention_threshold.json")
    assert abstention["source_split"] == "val"
    assert abstention["validation_coverage"] >= abstention["target_coverage"]
    p2_resolved = read_json(p2_run / "resolved_config.yaml")
    assert p2_resolved["checkpoint_contract"]["version"] == 2
    assert p2_resolved["learned_beta"] > 0
    assert "train_ranking_loss" in (p2_run / "train_log.csv").read_text(encoding="utf-8")

    for evaluated_run in (run_dir, *phase3_runs):
        evaluation = evaluate_full_suite(evaluated_run)
        assert evaluation["scenario_count"] == 88
        assert evaluation["controlled_sample_count"] == 4
        assert (evaluated_run / "robustness_metrics.parquet").is_file()

    first_report = generate_report(tmp_path)
    assert first_report["validated_run_count"] == 3
    tracked = sorted(
        path
        for path in (tmp_path / "reports").rglob("*")
        if path.is_file() and "data_audit" not in path.parts
    )
    first_hashes = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
    }
    second_report = generate_report(tmp_path)
    second_hashes = {
        str(path.relative_to(tmp_path)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tracked
    }
    assert second_report == first_report
    assert second_hashes == first_hashes
