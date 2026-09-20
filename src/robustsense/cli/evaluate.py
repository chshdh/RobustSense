"""Evaluate run CLI."""

from __future__ import annotations

import argparse

from robustsense.cli._common import run_cli
from robustsense.pipeline import evaluate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a RobustSense run")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--suite", default="smoke")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cli(lambda: evaluate(args.run_dir, args.suite))


if __name__ == "__main__":
    main()
