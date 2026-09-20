import numpy as np
import pytest

from robustsense.evaluation.v3_dual_policy import (
    DualPolicyDecisionLayer,
    fixed_threshold_policy_metrics,
    tune_coverage_threshold,
)


def test_coverage_threshold_is_validation_only_and_hits_distinct_score_target():
    artifact = tune_coverage_threshold(
        np.array([0.9, 0.8, 0.7, 0.6, 0.5]),
        target_coverage=0.8,
        source_split="val",
    )
    assert artifact["validation_achieved_coverage"] == pytest.approx(0.8)
    assert 0.5 < artifact["threshold"] < 0.6
    with pytest.raises(ValueError, match="validation"):
        tune_coverage_threshold(
            np.array([0.9, 0.8]), target_coverage=0.5, source_split="test"
        )


def test_coverage_threshold_records_unavoidable_tie_overshoot():
    artifact = tune_coverage_threshold(
        np.array([0.9, 0.8, 0.8, 0.1]),
        target_coverage=0.5,
        source_split="val",
    )
    assert artifact["validation_achieved_coverage"] == pytest.approx(0.75)
    assert artifact["validation_coverage_absolute_error"] == pytest.approx(0.25)


def test_dual_policy_requires_explicit_known_mode_and_frozen_coverage():
    layer = DualPolicyDecisionLayer(
        thresholds={
            "risk_control": {"0.80": 0.4},
            "error_alert": {"0.80": 0.6},
        }
    )
    decisions = layer.decide(
        "risk_control", np.array([0.3, 0.5]), target_coverage=0.8
    )
    assert decisions.tolist() == [False, True]
    with pytest.raises(ValueError, match="Unknown"):
        layer.decide("automatic", np.array([0.5]), target_coverage=0.8)
    with pytest.raises(ValueError, match="not frozen"):
        layer.decide("risk_control", np.array([0.5]), target_coverage=0.9)


def test_dual_policy_artifact_round_trip_preserves_thresholds():
    layer = DualPolicyDecisionLayer(
        thresholds={
            "risk_control": {"0.80": 0.4},
            "error_alert": {"0.80": 0.6},
        }
    )
    restored = DualPolicyDecisionLayer.from_artifact(layer.to_artifact())
    assert restored.thresholds == layer.thresholds


def test_fixed_threshold_metrics_report_risk_and_rejected_error_capture():
    result = fixed_threshold_policy_metrics(
        np.array([[0.9], [0.8], [0.2], [0.1]]),
        np.array([[1.0], [0.0], [0.0], [1.0]]),
        np.ones((4, 1), dtype=bool),
        np.array([0.9, 0.2, 0.8, 0.1]),
        np.array([0.0, 1.0, 0.0, 1.0]),
        labels=["label"],
        classification_thresholds=np.array([0.5]),
        acceptance_threshold=0.5,
        target_coverage=0.5,
    )
    assert result["test_coverage"] == pytest.approx(0.5)
    assert result["rejected_error_recall"] == pytest.approx(1.0)
    assert result["rejected_error_precision"] == pytest.approx(1.0)
