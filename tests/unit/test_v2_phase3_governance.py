import json
from pathlib import Path

from robustsense.experiments.v2_phase3 import (
    V2Phase3Registry,
    build_v2_phase3_plan,
    select_v2_4_ablations,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_v2_phase3_plan_has_25_isolated_units_and_registry(tmp_path: Path):
    _write(
        tmp_path / "configs/v2/evaluation/seed13_credible.yaml",
        {
            "profile": "v2_seed13",
            "folds": list(range(5)),
            "seeds": [13],
            "run_root": "runs/v2/phase_v2_3",
        },
    )
    variants = ["P2", "P2-A1", "P2-A2", "P2-A3", "P2-A4"]
    _write(
        tmp_path / "configs/v2/ablations/phase_v2_3.yaml",
        {variant: {} for variant in variants},
    )
    _write(
        tmp_path / "configs/v2/phase_v2_3_plan.yaml",
        {
            "experiment_config": "configs/v2/evaluation/seed13_credible.yaml",
            "ablation_config": "configs/v2/ablations/phase_v2_3.yaml",
            "folds": list(range(5)),
            "seeds": [13],
            "variants": variants,
            "expected_unit_count": 25,
        },
    )
    planned = build_v2_phase3_plan(tmp_path)
    assert len(planned) == 25
    assert len({row["run_key"] for row in planned}) == 25
    registry = V2Phase3Registry(tmp_path)
    registry.register(planned, "protocol")
    assert registry.path == tmp_path / "reports/v2/phase_v2_3_run_registry.csv"
    assert not (tmp_path / "reports/run_registry.csv").exists()


def test_v2_4_selection_reads_validation_only_and_uses_frozen_ranking(tmp_path: Path):
    rows = []
    variants = ["P2", "P2-A1", "P2-A2", "P2-A3", "P2-A4"]
    deltas = {"P2": 0.0, "P2-A1": 0.01, "P2-A2": -0.04, "P2-A3": 0.02, "P2-A4": 0.0}
    for fold in range(5):
        for variant in variants:
            run_dir = Path("runs") / f"{variant}-{fold}"
            _write(
                tmp_path / run_dir / "val_metrics.json",
                {"macro_f1": 0.5 + deltas[variant]},
            )
            rows.append(
                {
                    "run_key": f"{variant}-{fold}",
                    "run_dir": run_dir.as_posix(),
                    "model_name": variant,
                    "fold": str(fold),
                    "seed": "13",
                    "status": "success",
                }
            )
    selection = select_v2_4_ablations(tmp_path, rows)
    assert selection["test_metrics_read"] is False
    assert selection["selected_variants"] == ["P2-A2", "P2-A3"]
