import numpy as np
import pytest

from robustsense.training.sanity import mean_label_prevalence
from robustsense.training.thresholds import tune_per_label_thresholds


def test_threshold_tuning_is_validation_only_and_masks_unknowns():
    probabilities = np.asarray([[0.9], [0.4], [0.8]], dtype=np.float32)
    targets = np.asarray([[1.0], [0.0], [np.nan]], dtype=np.float32)
    mask = np.isfinite(targets)

    thresholds = tune_per_label_thresholds(
        probabilities, targets, mask, [0.3, 0.5, 0.7], source_split="val"
    )

    np.testing.assert_allclose(thresholds, [0.5])
    with pytest.raises(ValueError, match="validation"):
        tune_per_label_thresholds(
            probabilities, targets, mask, [0.5], source_split="test"
        )


def test_mean_label_prevalence_ignores_unknown_targets():
    targets = np.asarray([[1.0, np.nan], [0.0, 1.0], [np.nan, 0.0]])
    mask = np.isfinite(targets)

    assert mean_label_prevalence(targets, mask) == 0.5
