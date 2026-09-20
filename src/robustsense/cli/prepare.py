"""Prepare data CLI."""

from __future__ import annotations

import argparse

from robustsense.cli._common import run_cli
from robustsense.pipeline import prepare


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare a RobustSense dataset")
    parser.add_argument("--config", required=True, help="Path to a data config")
    parser.add_argument("--fold", type=int, default=0, help="Outer test fold")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cli(lambda: prepare(args.config, args.fold))


if __name__ == "__main__":
    main()
