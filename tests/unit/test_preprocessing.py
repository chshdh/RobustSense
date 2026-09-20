from pathlib import Path

import numpy as np

from robustsense.constants import MODALITIES
from robustsense.data.preprocessing import RobustPreprocessor


def test_preprocessor_uses_train_statistics_and_handles_constant_columns(tmp_path: Path):
    train = np.asarray(
        [
            [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            [2.0, 1.0, 4.0, 5.0, 6.0, 7.0],
            [np.nan, 1.0, 6.0, 7.0, 8.0, 9.0],
        ],
        dtype=np.float32,
    )
    names = [f"feature_{index}" for index in range(6)]
    processor = RobustPreprocessor.fit(train, names, ["train-user"])
    medians_before = processor.medians.copy()
    slices = {name: (index, index + 1) for index, name in enumerate(MODALITIES)}

    transformed = processor.transform(
        np.asarray([[10_000.0, np.nan, 2.0, 3.0, 4.0, 5.0]], dtype=np.float32),
        slices,
    )

    np.testing.assert_array_equal(processor.medians, medians_before)
    assert processor.iqrs[1] == 1.0
    assert transformed["features"][0, 0] == processor.clip_value
    assert transformed["features"][0, 1] == 0.0
    assert not transformed["feature_masks"][0, 1]
    processor.save(tmp_path, ["train-user"])
    loaded = RobustPreprocessor.load(tmp_path)
    np.testing.assert_allclose(loaded.medians, processor.medians)
    np.testing.assert_allclose(loaded.iqrs, processor.iqrs)
