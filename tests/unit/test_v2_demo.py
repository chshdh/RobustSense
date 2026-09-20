from pathlib import Path

import pytest

from robustsense.v2_demo import (
    V2DemoArtifactError,
    resolve_demo_run,
    validate_demo_artifacts,
    validate_v2_demo_context,
)


def test_v2_demo_context_fails_closed_without_frozen_plan(tmp_path: Path):
    with pytest.raises(V2DemoArtifactError, match="缺少 V2-5 冻结计划"):
        validate_v2_demo_context(tmp_path)


def test_v2_demo_rejects_unknown_model_before_loading_registry(tmp_path: Path):
    with pytest.raises(V2DemoArtifactError, match="不支持的 Demo 模型"):
        resolve_demo_run(tmp_path, "UNKNOWN", 0, 13)


def test_v2_demo_requires_real_run_artifacts(tmp_path: Path):
    with pytest.raises(V2DemoArtifactError, match="真实 run 产物不完整"):
        validate_demo_artifacts(tmp_path / "missing", "P2")
