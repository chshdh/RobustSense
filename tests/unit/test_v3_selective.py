import numpy as np
import pytest

from robustsense.evaluation.v3_selective import (
    error_detection_metrics,
    mean_binary_certainty,
    mean_normalized_threshold_margin,
    normalized_aurc,
    sample_error_indicator,
    selective_curve,
    summarize_selector,
)


def test_binary_certainty_is_low_at_half_and_high_at_probability_limits():
    scores = mean_binary_certainty(np.array([[0.5, 0.5], [0.0, 1.0]]))
    assert scores[0] == pytest.approx(0.0)
    assert scores[1] == pytest.approx(1.0, abs=1.0e-9)


def test_normalized_margin_uses_each_validation_threshold():
    scores = mean_normalized_threshold_margin(
        np.array([[0.2, 0.8], [0.0, 1.0]]), np.array([0.2, 0.8])
    )
    assert scores.tolist() == pytest.approx([0.0, 1.0])


def test_sample_error_indicator_ignores_unknown_labels_and_marks_no_known_nan():
    errors = sample_error_indicator(
        np.array([[0.9, 0.9], [0.1, 0.9], [0.1, 0.9]]),
        np.array([[1.0, np.nan], [1.0, 1.0], [np.nan, np.nan]]),
        np.array([[True, False], [True, True], [False, False]]),
        0.5,
    )
    assert errors[0] == 0.0
    assert errors[1] == 1.0
    assert np.isnan(errors[2])


def test_error_detection_treats_low_confidence_as_positive_error_score():
    result = error_detection_metrics(
        np.array([0.9, 0.8, 0.2, 0.1]),
        np.array([0.0, 0.0, 1.0, 1.0]),
        run_id="run-a",
        score_method="certainty",
    )
    assert result["error_detection_auroc"] == pytest.approx(1.0)
    assert result["error_detection_auprc"] == pytest.approx(1.0)


def test_selective_curve_keeps_the_most_confident_sample_first():
    probabilities = np.array([[0.9], [0.6], [0.1], [0.4]])
    targets = np.array([[1.0], [0.0], [0.0], [1.0]])
    known = np.ones_like(targets, dtype=bool)
    curve = selective_curve(
        probabilities,
        targets,
        known,
        np.array([0.9, 0.2, 0.8, 0.1]),
        labels=["label"],
        thresholds=0.5,
        coverage_grid=np.array([0.25, 0.5, 0.75, 1.0]),
        run_id="run-a",
        score_method="certainty",
    )
    assert curve[0]["accepted_count"] == 1
    assert curve[0]["confidence_cutoff"] == pytest.approx(0.9)
    assert curve[0]["risk_masked_bce"] < curve[-1]["risk_masked_bce"]


def test_normalized_aurc_uses_the_observed_coverage_interval():
    curve = [
        {"coverage": 0.25, "risk_masked_bce": 0.1},
        {"coverage": 0.50, "risk_masked_bce": 0.2},
        {"coverage": 1.00, "risk_masked_bce": 0.4},
    ]
    assert normalized_aurc(curve) == pytest.approx(0.25)


def test_selector_summary_requires_all_frozen_fixed_coverages():
    curve = [
        {
            "requested_coverage": coverage,
            "coverage": coverage,
            "risk_masked_bce": coverage,
            "macro_f1": 1.0 - coverage,
            "micro_f1": 1.0 - coverage,
        }
        for coverage in (0.8, 0.9, 0.95, 1.0)
    ]
    result = summarize_selector(
        curve,
        np.array([0.9, 0.1]),
        np.array([0.0, 1.0]),
        run_id="run-a",
        score_method="certainty",
        fixed_coverages=(0.8, 0.9, 0.95, 1.0),
    )
    assert result["risk_masked_bce_0_95"] == pytest.approx(0.95)
