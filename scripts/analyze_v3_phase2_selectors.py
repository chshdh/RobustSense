"""汇总 V3-2 selector 的 validation-test 泛化差距与系数稳定性。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    summary = pd.read_csv(root / "reports/v3/phase_v3_2_fold_seed_summary.csv")
    selector_paths = sorted(
        (root / "reports/v3/phase_v3_2/units").glob("fold*_seed*/selectors.json")
    )
    if len(selector_paths) != 15:
        raise RuntimeError(f"V3-2 selector 单元不是 15/15：{len(selector_paths)}")

    generalization_rows: list[dict[str, Any]] = []
    coefficient_rows: list[dict[str, Any]] = []
    for path in selector_paths:
        artifact = read_json(path)
        for selector_name, selector in artifact["selectors"].items():
            matches = summary.loc[
                (summary["fold"] == artifact["fold"])
                & (summary["seed"] == artifact["seed"])
                & (summary["score_method"] == selector_name)
            ]
            if len(matches) != 1:
                raise RuntimeError(f"无法唯一匹配 selector test 摘要：{path}")
            test = matches.iloc[0]
            validation_auroc = selector["fit_in_sample_diagnostics"][
                "error_detection_auroc"
            ]
            generalization_rows.append(
                {
                    "fold": artifact["fold"],
                    "seed": artifact["seed"],
                    "selector": selector_name,
                    "feature_count": len(selector["feature_names"]),
                    "iterations": selector["iterations"],
                    "val_in_sample_error_detection_auroc": validation_auroc,
                    "test_error_detection_auroc": test["error_detection_auroc"],
                    "val_minus_test_auroc": (
                        validation_auroc - test["error_detection_auroc"]
                    ),
                }
            )
            for feature, coefficient in zip(
                selector["feature_names"], selector["coefficients"], strict=True
            ):
                coefficient_rows.append(
                    {
                        "fold": artifact["fold"],
                        "seed": artifact["seed"],
                        "selector": selector_name,
                        "feature": feature,
                        "coefficient": coefficient,
                    }
                )

    generalization = pd.DataFrame(generalization_rows)
    coefficients = pd.DataFrame(coefficient_rows)
    stability_rows = []
    for (selector_name, feature), group in coefficients.groupby(
        ["selector", "feature"], sort=True
    ):
        values = group["coefficient"].to_numpy(dtype=np.float64)
        positive_fraction = float(np.mean(values > 0))
        negative_fraction = float(np.mean(values < 0))
        stability_rows.append(
            {
                "selector": selector_name,
                "feature": feature,
                "unit_count": len(values),
                "coefficient_mean": float(values.mean()),
                "coefficient_std": float(values.std(ddof=1)),
                "coefficient_abs_mean": float(np.abs(values).mean()),
                "positive_fraction": positive_fraction,
                "negative_fraction": negative_fraction,
                "sign_agreement": max(positive_fraction, negative_fraction),
            }
        )
    stability = pd.DataFrame(stability_rows)
    generalization_path = root / "reports/v3/phase_v3_2_fit_generalization.csv"
    stability_path = root / "reports/v3/phase_v3_2_coefficient_stability.csv"
    generalization.to_csv(generalization_path, index=False)
    stability.to_csv(stability_path, index=False)

    selector_summary = []
    for selector_name, group in generalization.groupby("selector", sort=True):
        selector_stability = stability.loc[stability["selector"] == selector_name]
        selector_summary.append(
            {
                "selector": selector_name,
                "unit_count": len(group),
                "feature_count": int(group["feature_count"].iloc[0]),
                "mean_iterations": float(group["iterations"].mean()),
                "mean_val_in_sample_error_detection_auroc": float(
                    group["val_in_sample_error_detection_auroc"].mean()
                ),
                "mean_test_error_detection_auroc": float(
                    group["test_error_detection_auroc"].mean()
                ),
                "mean_val_minus_test_auroc": float(
                    group["val_minus_test_auroc"].mean()
                ),
                "median_coefficient_sign_agreement": float(
                    selector_stability["sign_agreement"].median()
                ),
                "feature_count_with_sign_agreement_at_least_0_8": int(
                    (selector_stability["sign_agreement"] >= 0.8).sum()
                ),
            }
        )
    diagnostics = {
        "version": 1,
        "phase": "V3-2",
        "status": "complete_posthoc_diagnostic_not_primary_model_selection",
        "selector_summary": selector_summary,
        "interpretation": (
            "Validation in-sample gains do not transfer consistently to held-out test users; "
            "the full feature set has the larger generalization gap and lower coefficient-sign "
            "stability."
        ),
    }
    diagnostics_path = root / "reports/v3/phase_v3_2_diagnostics.json"
    write_json(diagnostics_path, diagnostics)
    print(generalization.groupby("selector")["val_minus_test_auroc"].mean())
    print(diagnostics_path)


if __name__ == "__main__":
    main()
