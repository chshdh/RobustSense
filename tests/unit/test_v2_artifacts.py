from pathlib import Path

import numpy as np
import pytest

from robustsense.evaluation.v2_artifacts import (
    stable_csv_bytes,
    write_artifact_lineage,
    write_stable_csv,
)


def test_stable_csv_is_byte_reproducible_and_serializes_nan_explicitly(tmp_path: Path):
    rows = [{"run_id": "run-a", "value": np.nan}, {"run_id": "run-b", "value": 0.5}]
    columns = ["run_id", "value"]
    assert stable_csv_bytes(rows, columns) == stable_csv_bytes(rows, columns)
    first = write_stable_csv(tmp_path / "first.csv", rows, columns)
    second = write_stable_csv(tmp_path / "second.csv", rows, columns)
    assert first["sha256"] == second["sha256"]
    assert (tmp_path / "first.csv").read_text(encoding="utf-8").splitlines()[1].endswith(
        "NaN"
    )


def test_artifact_lineage_requires_and_sorts_source_run_ids(tmp_path: Path):
    result = write_artifact_lineage(
        tmp_path / "lineage.json",
        [
            {
                "artifact_path": "figures/risk.png",
                "artifact_type": "figure",
                "source_run_ids": ["run-b", "run-a", "run-a"],
            }
        ],
        protocol_id="V2-2",
    )
    assert result["artifact_count"] == 1
    content = (tmp_path / "lineage.json").read_text(encoding="utf-8")
    assert content.index("run-a") < content.index("run-b")
    with pytest.raises(ValueError, match="source run ID"):
        write_artifact_lineage(
            tmp_path / "invalid.json",
            [{"artifact_path": "figures/invalid.png", "source_run_ids": []}],
            protocol_id="V2-2",
        )
