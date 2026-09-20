"""Train model CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

from robustsense.cli._common import run_cli
from robustsense.pipeline import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a RobustSense model")
    parser.add_argument("--model", default="early")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--profile", default="dev")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--data-config",
        default=None,
        help="Data config; set to configs/data/extrasensory.yaml for Phase 2",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cli(
        lambda: {
            "run_dir": str(
                train(
                    Path(args.project_root),
                    model_name=args.model,
                    fold=args.fold,
                    seed=args.seed,
                    profile=args.profile,
                    data_config=args.data_config,
                )
            )
        }
    )


if __name__ == "__main__":
    main()
