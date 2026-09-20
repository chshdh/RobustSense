import numpy as np
import pytest

from robustsense.evaluation.v3_task_confidence import (
    TaskConfidenceSelector,
    build_task_confidence_features,
    fit_task_confidence_selector,
    per_label_binary_certainty,
    per_label_normalized_threshold_margin,
)


def _collected() -> dict[str, np.ndarray]:
    return {
        "probabilities": np.array([[0.9, 0.4], [0.2, 0.8], [0.6, 0.3]]),
        "availability": np.array([[1, 1], [1, 0], [0, 1]]),
        "fusion_weights": np.array([[0.6, 0.4], [1.0, 0.0], [0.0, 1.0]]),
        "reliability": np.array([[0.9, 0.7], [0.8, 0.3], [0.2, 0.6]]),
        "system_reliability": np.array([0.82, 0.80, 0.60]),
    }


def test_per_label_features_preserve_label_width_and_probability_limits():
    probabilities = np.array([[0.5, 0.0], [1.0, 0.8]])
    certainty = per_label_binary_certainty(probabilities)
    margins = per_label_normalized_threshold_margin(
        probabilities, np.array([0.5, 0.8])
    )
    assert certainty.shape == probabilities.shape
    assert certainty[0, 0] == pytest.approx(0.0)
    assert certainty[0, 1] == pytest.approx(1.0, abs=1.0e-9)
    assert margins[0, 0] == pytest.approx(0.0)
    assert margins[1, 1] == pytest.approx(0.0)


def test_output_only_features_do_not_require_sensor_outputs():
    features, names = build_task_confidence_features(
        {"probabilities": _collected()["probabilities"]},
        labels=["a", "b"],
        modality_names=["m1", "m2"],
        thresholds=np.array([0.5, 0.5]),
        include_sensor_features=False,
    )
    assert features.shape == (3, 4)
    assert names == [
        "label_certainty::a",
        "label_certainty::b",
        "label_threshold_margin::a",
        "label_threshold_margin::b",
    ]


def test_full_features_mask_unavailable_modality_reliability():
    features, names = build_task_confidence_features(
        _collected(),
        labels=["a", "b"],
        modality_names=["m1", "m2"],
        thresholds=np.array([0.5, 0.5]),
        include_sensor_features=True,
    )
    assert features.shape == (3, 11)
    m2_index = names.index("masked_reliability::m2")
    assert features[1, m2_index] == pytest.approx(0.0)
    assert names[-1] == "system_reliability"


def test_fitted_selector_learns_lower_confidence_for_error_linked_feature():
    x = np.arange(20, dtype=np.float64)[:, None]
    errors = (x[:, 0] >= 10).astype(np.float64)
    selector = fit_task_confidence_selector(
        x,
        ["risk_feature"],
        errors,
        regularization_c=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    confidence = selector.predict_confidence(np.array([[2.0], [18.0]]))
    assert confidence[0] > confidence[1]
    assert selector.iterations < selector.max_iter


def test_selector_json_artifact_round_trip_preserves_predictions():
    x = np.column_stack([np.arange(20), np.arange(20) % 3]).astype(np.float64)
    errors = (x[:, 0] >= 10).astype(np.float64)
    selector = fit_task_confidence_selector(
        x,
        ["a", "b"],
        errors,
        regularization_c=1.0,
        solver="lbfgs",
        max_iter=1000,
    )
    restored = TaskConfidenceSelector.from_artifact(selector.to_artifact())
    np.testing.assert_allclose(
        selector.predict_confidence(x), restored.predict_confidence(x), atol=0.0
    )


def test_selector_rejects_single_class_validation_target():
    with pytest.raises(ValueError, match="Both correct and error"):
        fit_task_confidence_selector(
            np.ones((4, 2)),
            ["a", "b"],
            np.ones(4),
            regularization_c=1.0,
            solver="lbfgs",
            max_iter=1000,
        )
