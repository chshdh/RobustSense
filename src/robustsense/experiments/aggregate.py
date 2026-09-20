"""Small Phase-5 aggregates that only read registered successful runs."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "robustsense-matplotlib"))

import matplotlib
import pandas as pd

from robustsense.experiments.registry import RunRegistry
from robustsense.utils.config import load_config
from robustsense.utils.io import read_json, write_json

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


def aggregate_ablation_development_screen(
    project_root: Path, plan_path: Path
) -> dict[str, Any]:
    root = project_root.resolve()
    plan_file = plan_path if plan_path.is_absolute() else root / plan_path
    plan = load_config(plan_file)
    experiment = load_config(
        root / plan["profiles"]["ablation_dev"]["experiment_config"]
    )
    definitions = {item["model"]: item for item in plan["ablations"]}
    registry_rows = [
        row
        for row in RunRegistry(root).rows()
        if row["profile"] == "ablation_dev" and row["status"] == "success"
    ]
    expected_models = plan["profiles"]["ablation_dev"]["models"]
    if {row["model_name"] for row in registry_rows} != set(expected_models):
        raise ValueError("Ablation aggregate requires every registered development-screen run")

    rows = []
    for registry_row in registry_rows:
        run_path = root / registry_row["run_dir"]
        natural = read_json(run_path / "test_metrics.json")
        robustness = pd.read_parquet(run_path / "robustness_metrics.parquet")
        clean = robustness.loc[robustness["family"] == "clean_complete"]
        if len(clean) != 1:
            raise ValueError(f"Expected one clean row in {run_path}")
        definition = definitions[registry_row["model_name"]]
        rows.append(
            {
                "ablation_id": definition["id"],
                "model_name": registry_row["model_name"],
                "change": definition["change"],
                "run_id": run_path.name,
                "fold": int(registry_row["fold"]),
                "seed": int(registry_row["seed"]),
                "epochs_budget": int(experiment["epochs"]),
                "natural_macro_f1": natural["macro_f1"],
                "clean_complete_macro_f1": float(clean.iloc[0]["macro_f1"]),
                "development_only": True,
            }
        )
    frame = pd.DataFrame(rows).sort_values("ablation_id")
    table_path = root / "reports/tables/phase5_ablation_dev.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(table_path, index=False)

    figure_path = root / "reports/figures/ablation.png"
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(frame["ablation_id"], frame["clean_complete_macro_f1"], color="#3478bf")
    axis.set_title("Phase-5 development ablation screen (fold 0, seed 13)")
    axis.set_xlabel("Frozen ablation")
    axis.set_ylabel("Controlled-complete Macro-F1")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        figure_path,
        dpi=150,
        bbox_inches="tight",
        metadata={"Software": "RobustSense Phase 5"},
    )
    plt.close(figure)

    summary = {
        "status": "complete_development_screen",
        "research_conclusion": False,
        "profile": "ablation_dev",
        "folds": [0],
        "seeds": [13],
        "source_run_ids": frame["run_id"].tolist(),
        "table": str(table_path.relative_to(root)),
        "figure": str(figure_path.relative_to(root)),
    }
    write_json(root / "reports/phase5_ablation_dev_summary.json", summary)
    return summary
