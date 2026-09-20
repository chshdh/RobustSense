from pathlib import Path

import pytest

from robustsense.demo import DemoArtifactError, run_directory, validate_demo_artifacts


def test_demo_run_directory_is_registered_and_model_limited(tmp_path: Path):
    path = run_directory(tmp_path, "quality-aware", 3)
    assert path == (
        tmp_path
        / "runs"
        / "extrasensory-quality-aware-fold3-seed13-credible"
    )
    with pytest.raises(DemoArtifactError, match="Unsupported demo model"):
        run_directory(tmp_path, "made-up", 0)


def test_demo_artifact_validation_fails_closed(tmp_path: Path):
    run_path = tmp_path / "runs" / "missing-run"
    with pytest.raises(DemoArtifactError, match="Missing artifacts"):
        validate_demo_artifacts(run_path)
