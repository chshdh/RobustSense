"""生成 V3-4 的后验精度与覆盖率差异诊断，不改变首要假设。"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robustsense.utils.io import read_json, write_json


def _paired_diagnostics(fold: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for coverage in sorted(fold["target_coverage"].unique()):
        risk = fold[
            (fold["policy_mode"] == "risk_control")
            & np.isclose(fold["target_coverage"], coverage)
        ].set_index("fold")
        alert = fold[
            (fold["policy_mode"] == "error_alert")
            & np.isclose(fold["target_coverage"], coverage)
        ].set_index("fold")
        if set(risk.index) != set(alert.index):
            raise RuntimeError("V3-4 后验诊断无法按折配对")
        precision_delta = (
            alert["rejected_error_precision"] - risk["rejected_error_precision"]
        )
        recall_delta = alert["rejected_error_recall"] - risk["rejected_error_recall"]
        coverage_delta = alert["test_coverage"] - risk["test_coverage"]
        rows.append(
            {
                "target_coverage": float(coverage),
                "alert_minus_risk_test_coverage_mean": float(coverage_delta.mean()),
                "alert_minus_risk_rejected_error_precision_mean": float(
                    precision_delta.mean()
                ),
                "alert_precision_win_fold_count": int((precision_delta > 0).sum()),
                "alert_minus_risk_rejected_error_recall_mean": float(
                    recall_delta.mean()
                ),
                "alert_recall_win_fold_count": int((recall_delta > 0).sum()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    report = read_json(root / "reports/v3/phase_v3_4_report.json")
    if report.get("status") != "complete" or report.get("result_row_count") != 90:
        raise RuntimeError("V3-4 正式报告尚未完整生成")
    fold = pd.read_csv(root / "reports/v3/phase_v3_4_fold_summary.csv")
    rows = _paired_diagnostics(fold)
    result = {
        "version": 1,
        "phase": "V3-4",
        "status": "complete_posthoc_diagnostic_not_primary_hypothesis",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": report["protocol_sha256"],
        "interpretation": (
            "error_alert 的拒绝错误精度较高，但实际覆盖率也较高、拒绝样本更少；"
            "因此其预声明拒绝错误召回假设失败。后验精度不能替代首要召回判据。"
        ),
        "paired_diagnostics": rows,
    }
    output = root / "reports/v3/phase_v3_4_diagnostics.json"
    write_json(output, result)

    document_path = root / "docs/v3/PHASE_V3_4_REPORT.md"
    text = document_path.read_text(encoding="utf-8")
    marker = "## 6. 后验诊断（非首要假设）"
    text = text.split(marker, maxsplit=1)[0].rstrip()
    lines = [
        text,
        "",
        marker,
        "",
        (
            "`error_alert` 的首要召回假设仍判为失败。后验查看表明，它在三个"
            "覆盖率上的拒绝错误精度均高于 `risk_control`，但实际覆盖率也均更高，"
            "即拒绝样本更少；精度提升不能替代预声明的召回要求。"
        ),
        "",
        "| 目标覆盖率 | test 覆盖率差（alert-risk） | 拒绝精度差 | 精度胜出折数 | 拒绝召回差 |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {coverage:.2f} | {coverage_delta:+.6f} | {precision_delta:+.6f} | "
            "{wins}/5 | {recall_delta:+.6f} |".format(
                coverage=row["target_coverage"],
                coverage_delta=row["alert_minus_risk_test_coverage_mean"],
                precision_delta=row[
                    "alert_minus_risk_rejected_error_precision_mean"
                ],
                wins=row["alert_precision_win_fold_count"],
                recall_delta=row["alert_minus_risk_rejected_error_recall_mean"],
            )
        )
    document_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
