"""汇总 V3-3 validation 分组选择与 test 表现。"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from robustsense.utils.io import read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    test_summary = pd.read_csv(root / "reports/v3/phase_v3_3_fold_seed_summary.csv")
    paths = sorted(
        (root / "reports/v3/phase_v3_3/units").glob("fold*_seed*/selectors.json")
    )
    if len(paths) != 15:
        raise RuntimeError(f"V3-3 selector 单元不是 15/15：{len(paths)}")
    rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    alpha_counts: dict[str, Counter[str]] = {}
    max_overlap = 0
    for path in paths:
        artifact = read_json(path)
        family_counts[artifact["selected_family"]] += 1
        for selector_name, selector in artifact["selectors"].items():
            selection = selector["selection"]
            alpha_counts.setdefault(selector_name, Counter())[str(selection["selected_alpha"])] += 1
            overlap = max(row["group_overlap_count"] for row in selection["fold_contract"])
            max_overlap = max(max_overlap, overlap)
            match = test_summary.loc[
                (test_summary["fold"] == artifact["fold"])
                & (test_summary["seed"] == artifact["seed"])
                & (test_summary["score_method"] == selector_name)
            ]
            if len(match) != 1:
                raise RuntimeError(f"无法唯一匹配 V3-3 test 摘要：{path}")
            rows.append(
                {
                    "fold": artifact["fold"],
                    "seed": artifact["seed"],
                    "selector": selector_name,
                    "feature_count": selector["feature_count"],
                    "selected_alpha": selection["selected_alpha"],
                    "validation_oof_normalized_aurc": selection[
                        "selected_oof_normalized_aurc"
                    ],
                    "test_normalized_aurc": float(match.iloc[0]["normalized_aurc"]),
                    "selected_family_for_unit": artifact["selected_family"],
                    "max_group_overlap_count": overlap,
                }
            )
    frame = pd.DataFrame(rows)
    csv_path = root / "reports/v3/phase_v3_3_selection_diagnostics.csv"
    frame.to_csv(csv_path, index=False)
    report = read_json(root / "reports/v3/phase_v3_3_report.json")
    result = {
        "version": 1,
        "phase": "V3-3",
        "status": "complete_posthoc_diagnostic_not_primary_model_selection",
        "unit_count": len(paths),
        "selected_family_counts": dict(family_counts),
        "selected_alpha_counts": {
            selector: dict(counts) for selector, counts in alpha_counts.items()
        },
        "max_group_overlap_count": max_overlap,
        "primary_comparisons": report["primary_comparisons"],
        "interpretation": (
            "Grouped continuous-risk selection improves AURC and fixed-coverage BCE but does "
            "not improve the discrete any-label-error AUROC."
        ),
    }
    output = root / "reports/v3/phase_v3_3_diagnostics.json"
    write_json(output, result)
    print(output)


if __name__ == "__main__":
    main()
