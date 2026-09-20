import numpy as np
import pytest

from robustsense.evaluation.v3_grouped_risk import (
    RidgeRiskSelector,
    build_compact_risk_features,
    fit_ridge_risk_selector,
    normalized_risk_aurc_from_losses,
    select_grouped_ridge_risk_selector,
)


def _collected() -> dict[str, np.ndarray]:
    return {
        "probabilities": np.array([[0.9, 0.4], [0.2, 0.8], [0.6, 0.3], [0.1, 0.7]]),
        "availability": np.array([[1, 1], [1, 0], [0, 1], [1, 1]]),
        "fusion_weights": np.array(
            [[0.6, 0.4], [1.0, 0.0], [0.0, 1.0], [0.3, 0.7]]
        ),
        "reliability": np.array(
            [[0.9, 0.7], [0.8, 0.3], [0.2, 0.6], [0.5, 0.9]]
        ),
        "system_reliability": np.array([0.82, 0.80, 0.60, 0.78]),
    }


def test_compact_output_and_full_feature_contracts():
    output, output_names = build_compact_risk_features(
        _collected(),
        modality_names=["m1", "m2"],
        thresholds=np.array([0.5, 0.5]),
        include_sensor_features=False,
    )
    full, full_names = build_compact_risk_features(
        _collected(),
        modality_names=["m1", "m2"],
        thresholds=np.array([0.5, 0.5]),
        include_sensor_features=True,
    )
    assert output.shape == (4, 6)
    assert full.shape == (4, 11)
    assert full_names[:6] == output_names
    assert full_names[-1] == "fusion_weight_concentration"


def test_ridge_selector_round_trip_and_risk_order():
    features = np.arange(20, dtype=np.float64)[:, None]
    losses = features[:, 0] / 20.0
    selector = fit_ridge_risk_selector(
        features, ["risk_signal"], losses, alpha=1.0, solver="svd"
    )
    restored = RidgeRiskSelector.from_artifact(selector.to_artifact())
    np.testing.assert_allclose(
        selector.predict_confidence(features), restored.predict_confidence(features)
    )
    assert selector.predict_confidence(np.array([[1.0]]))[0] > selector.predict_confidence(
        np.array([[18.0]])
    )[0]


def test_direct_loss_aurc_rewards_correct_risk_ordering():
    losses = np.array([0.1, 0.2, 0.8, 1.0])
    good = normalized_risk_aurc_from_losses(
        losses, np.array([0.9, 0.8, 0.2, 0.1]), np.array([0.25, 0.5, 0.75, 1.0])
    )
    bad = normalized_risk_aurc_from_losses(
        losses, np.array([0.1, 0.2, 0.8, 0.9]), np.array([0.25, 0.5, 0.75, 1.0])
    )
    assert good < bad


def test_grouped_selection_has_no_group_overlap_and_refits_selected_alpha():
    rng = np.random.default_rng(13)
    groups = np.repeat(["u1", "u2", "u3", "u4"], 10)
    features = rng.normal(size=(40, 2))
    losses = 0.5 + 0.3 * features[:, 0] + rng.normal(scale=0.05, size=40)
    selector, selection = select_grouped_ridge_risk_selector(
        features,
        ["a", "b"],
        losses,
        groups,
        alphas=[0.1, 1.0, 10.0],
        n_splits=4,
        solver="svd",
        coverage_grid=np.array([0.25, 0.5, 0.75, 1.0]),
    )
    assert selector.alpha == selection["selected_alpha"]
    assert len(selection["candidates"]) == 3
    assert all(row["group_overlap_count"] == 0 for row in selection["fold_contract"])


def test_grouped_selection_rejects_too_few_users():
    with pytest.raises(ValueError, match="Not enough unique groups"):
        select_grouped_ridge_risk_selector(
            np.ones((6, 2)),
            ["a", "b"],
            np.ones(6),
            np.array(["u1"] * 3 + ["u2"] * 3),
            alphas=[1.0],
            n_splits=3,
            solver="svd",
            coverage_grid=np.array([0.5, 1.0]),
        )
