import json
from pathlib import Path

import pandas as pd

from robustsense.constants import LABELS
from robustsense.experiments.v2_phase4 import (
    V2Phase4Registry,
    aggregate_v2_phase4_core,
    bootstrap_v2_phase4_natural,
    build_v2_phase4_core_plan,
    build_v2_phase4_optional_plan,
)


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _fixture_plan(root: Path) -> None:
    entries = []
    for fold in range(5):
        for model_id in ("B4", "B5", "P", "P2"):
            entries.append(
                {
                    "run_key": f"V2-4|{model_id}|fold{fold}|seed13",
                    "source_run_id": f"source-{model_id}-{fold}",
                    "source_run_dir": f"runs/reused-{model_id}-{fold}",
                }
            )
    _write(
        root / "reports/v2/phase_v2_4_reuse_manifest.json",
        {
            "status": "verified_before_first_v2_4_test_run",
            "entry_count": 20,
            "entries": entries,
        },
    )
    _write(
        root / "configs/v2/phase_v2_4_plan.yaml",
        {
            "reuse_manifest": "reports/v2/phase_v2_4_reuse_manifest.json",
            "folds": list(range(5)),
            "seeds": [13, 29, 47],
            "reuse_seeds": [13],
            "new_training_seeds": [29, 47],
            "expected_core_unit_count": 60,
            "expected_reused_unit_count": 20,
            "run_root": "runs/v2/phase_v2_4",
        },
    )


def test_v2_phase4_plan_has_60_units_and_exactly_20_reused(tmp_path: Path):
    _fixture_plan(tmp_path)
    rows = build_v2_phase4_core_plan(tmp_path)
    assert len(rows) == 60
    assert len({row["run_key"] for row in rows}) == 60
    assert sum(row["status"] == "success" for row in rows) == 20
    assert sum(row["status"] == "pending" for row in rows) == 40
    registry = V2Phase4Registry(tmp_path)
    registry.register(rows, "protocol")
    assert registry.path == tmp_path / "reports/v2/phase_v2_4_run_registry.csv"
    assert not (tmp_path / "reports/run_registry.csv").exists()

    plan = json.loads(
        (tmp_path / "configs/v2/phase_v2_4_plan.yaml").read_text(encoding="utf-8")
    )
    plan.update(
        {
            "optional_ablation_variants": ["P2-A3", "P2-A1"],
            "optional_ablation_seeds": [29, 47],
            "expected_optional_ablation_unit_count": 20,
        }
    )
    _write(tmp_path / "configs/v2/phase_v2_4_plan.yaml", plan)
    optional = build_v2_phase4_optional_plan(tmp_path)
    assert len(optional) == 20
    assert {row["model_name"] for row in optional} == {"P2-A3", "P2-A1"}


def test_v2_phase4_aggregation_averages_seeds_inside_each_fold(tmp_path: Path):
    _fixture_plan(tmp_path)
    rows = build_v2_phase4_core_plan(tmp_path)
    for row in rows:
        row["status"] = "success"
        run = tmp_path / row["run_dir"]
        if row["model_name"] == "P2":
            _write(
                run / "test_metrics.json",
                {
                    "forced": {
                        "macro_f1": 0.5 + int(row["fold"]) / 100,
                        "micro_f1": 0.6,
                        "mean_average_precision": 0.55,
                        "brier_score": 0.1,
                    }
                },
            )
        else:
            _write(
                run / "test_metrics.json",
                {
                    "macro_f1": 0.4 + int(row["fold"]) / 100,
                    "micro_f1": 0.5,
                    "mean_average_precision": 0.45,
                    "brier_score": 0.2,
                },
            )
    registry = V2Phase4Registry(tmp_path)
    registry.register(rows, "protocol")
    report = aggregate_v2_phase4_core(tmp_path, registry)
    assert report["successful_core_unit_count"] == 60
    summary = (tmp_path / "reports/v2/phase_v2_4_core_summary.csv").read_text(
        encoding="utf-8"
    )
    assert "seeds_within_fold_then_folds" in json.dumps(report)
    assert "0.52" in summary


def test_v2_phase4_bootstrap_pairs_users_and_preserves_aggregation_order(tmp_path: Path):
    _fixture_plan(tmp_path)
    rows = build_v2_phase4_core_plan(tmp_path)
    for row in rows:
        row["status"] = "success"
        if row["model_name"] not in {"P", "P2"}:
            continue
        records = []
        for user_index, user in enumerate(("user-a", "user-b")):
            for label_index, label in enumerate(LABELS):
                target = bool((user_index + label_index) % 2)
                p2_prediction = target
                p_prediction = target if label_index % 3 else not target
                records.append(
                    {
                        "user_id": user,
                        "timestamp": 100 + user_index,
                        "scenario": "natural_missingness",
                        "label": label,
                        "target": float(target),
                        "target_known": True,
                        "prediction": (
                            p2_prediction if row["model_name"] == "P2" else p_prediction
                        ),
                    }
                )
        output = tmp_path / row["run_dir"] / "predictions.parquet"
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_parquet(output, index=False)
    _write(
        tmp_path / "data/processed/extrasensory/fold0/processed_manifest.json",
        {"labels": list(LABELS)},
    )
    _write(
        tmp_path / "configs/v2/evaluation/extended_core.yaml",
        {
            "folds": list(range(5)),
            "seeds": [13, 29, 47],
            "bootstrap_repeats": 50,
            "bootstrap_seed": 2404,
        },
    )
    registry = V2Phase4Registry(tmp_path)
    registry.register(rows, "protocol")
    result = bootstrap_v2_phase4_natural(tmp_path, registry)
    assert result["valid_repeat_count"] == 50
    assert result["mean_delta_p2_minus_p"] > 0
    assert result["aggregation_order"] == "seeds_within_fold_then_folds"
    assert len(result["source_run_keys"]) == 30
