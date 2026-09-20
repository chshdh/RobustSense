import numpy as np
import pytest

from robustsense.evaluation.v2_metrics import (
    fault_detection_metrics,
    paired_user_bootstrap,
    persistent_classification_metrics,
    persistent_episode_response,
    reliability_calibration,
    risk_coverage_curve,
)


def test_calibration_preserves_empty_bins_as_nan_and_run_lineage():
    result = reliability_calibration(
        np.array([[0.05, 0.95], [0.15, 0.85]]),
        np.array([[0.0, 1.0], [0.2, 0.8]]),
        np.ones((2, 2), dtype=bool),
        modality_names=["a", "b"],
        n_bins=5,
        run_id="run-a",
    )
    empty = [row for row in result["bins"] if row["count"] == 0]
    assert empty
    assert all(np.isnan(row["mean_prediction"]) for row in empty)
    assert all(np.isnan(row["mean_target"]) for row in empty)
    assert {row["run_id"] for row in result["bins"]} == {"run-a"}


def test_fault_detection_returns_nan_when_auroc_is_undefined():
    undefined = fault_detection_metrics(
        np.array([[0.8, 0.9]]),
        np.array([[False, False]]),
        np.ones((1, 2), dtype=bool),
        run_id="run-a",
    )
    assert np.isnan(undefined["fault_auroc"])
    assert np.isnan(undefined["fault_auprc"])
    defined = fault_detection_metrics(
        np.array([[0.9, 0.1]]),
        np.array([[False, True]]),
        np.ones((1, 2), dtype=bool),
        run_id="run-a",
    )
    assert defined["fault_auroc"] == pytest.approx(1.0)


def test_risk_coverage_uses_stable_reliability_order_and_preserves_unknowns():
    probabilities = np.array([[0.9], [0.1], [0.8]])
    targets = np.array([[1.0], [0.0], [np.nan]])
    known = np.array([[True], [True], [False]])
    curve = risk_coverage_curve(
        probabilities,
        targets,
        known,
        np.array([0.9, 0.1, 0.8]),
        labels=["label"],
        thresholds=0.5,
        coverage_grid=np.array([0.0, 1 / 3, 2 / 3, 1.0]),
        run_id="run-a",
    )
    assert np.isnan(curve[0]["risk_masked_bce"])
    assert curve[1]["accepted_count"] == 1
    assert curve[2]["known_target_count"] == 1
    assert curve[-1]["known_target_count"] == 2
    assert {row["run_id"] for row in curve} == {"run-a"}


def test_persistent_response_reports_detection_and_missing_recovery_as_nan():
    result = persistent_episode_response(
        np.array([0.1, 0.2, 0.4, 0.8, 0.7, 0.6]),
        pre_indices=(0, 1),
        fault_indices=(2, 3),
        post_indices=(4, 5),
        detection_threshold=0.5,
        episode_id="episode-a",
        run_id="run-a",
    )
    assert result["detection_delay_steps"] == 1
    assert np.isnan(result["recovery_steps"])
    assert result["detected"] is True
    assert result["recovered"] is False


def test_persistent_classification_keeps_an_empty_phase_as_nan():
    rows = persistent_classification_metrics(
        np.array([[0.9], [0.1]]),
        np.array([[1.0], [0.0]]),
        np.ones((2, 1), dtype=bool),
        phases={"pre": (0,), "fault": (1,), "post": ()},
        labels=["label"],
        thresholds=0.5,
        episode_id="episode-a",
        run_id="run-a",
    )
    assert [row["phase"] for row in rows] == ["pre", "fault", "post"]
    assert np.isnan(rows[-1]["macro_f1"])
    assert np.isnan(rows[-1]["masked_bce"])


def test_paired_bootstrap_is_user_level_paired_and_reproducible():
    probabilities_a = np.array([[0.9], [0.8], [0.2], [0.1]])
    probabilities_b = np.array([[0.6], [0.4], [0.6], [0.4]])
    targets = np.array([[1.0], [1.0], [0.0], [0.0]])
    known = np.ones_like(targets, dtype=bool)
    users = np.array(["a", "b", "a", "b"])
    arguments = {
        "labels": ["label"],
        "thresholds": 0.5,
        "run_id_a": "run-a",
        "run_id_b": "run-b",
        "repeats": 50,
        "seed": 13,
    }
    first = paired_user_bootstrap(
        probabilities_a, probabilities_b, targets, known, users, **arguments
    )
    second = paired_user_bootstrap(
        probabilities_a, probabilities_b, targets, known, users, **arguments
    )
    assert first["summary"] == second["summary"]
    np.testing.assert_allclose(
        [row["delta_a_minus_b"] for row in first["draws"]],
        [row["delta_a_minus_b"] for row in second["draws"]],
        equal_nan=True,
    )
    assert first["summary"]["sampling_unit"] == "user"
    assert first["summary"]["valid_repeat_count"] == 50
    assert len(first["summary"]["sample_index_sha256"]) == 64
