"""Validate and aggregate Phase-4 run artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from robustsense.cli._common import run_cli
from robustsense.evaluation.report import generate_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Aggregate RobustSense run artifacts")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--plan", default="configs/evaluation/dev.yaml")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cli(
        lambda: generate_report(Path(args.project_root), args.runs_dir, args.output_dir, args.plan)
    )


if __name__ == "__main__":
    main()
