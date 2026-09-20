from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from robustsense.evaluation.report import (
    IncompleteRunMatrixError,
    aggregate_mean_std,
    validate_run_matrix,
)


def test_aggregate_preserves_missing_metrics_instead_of_zero():
    frame = pd.DataFrame(
        {
            "model_name": ["a", "a", "b", "b"],
            "run_id": ["a0", "a1", "b0", "b1"],
            "macro_f1": [0.5, np.nan, np.nan, np.nan],
        }
    )
    result = aggregate_mean_std(frame, ["model_name"], ["macro_f1"]).set_index("model_name")
    assert result.loc["a", "macro_f1_mean"] == 0.5
    assert result.loc["a", "macro_f1_count"] == 1
    assert np.isnan(result.loc["b", "macro_f1_mean"])
    assert result.loc["b", "macro_f1_count"] == 0


def test_completeness_check_rejects_missing_run(tmp_path: Path):
    plan = {
        "profile": "dev",
        "folds": [0],
        "seeds": [13],
        "models": ["mlp"],
        "controlled_models": ["mlp"],
        "suite_version": 1,
    }
    with pytest.raises(IncompleteRunMatrixError, match="missing run directory"):
        validate_run_matrix(tmp_path, tmp_path / "runs", plan)
