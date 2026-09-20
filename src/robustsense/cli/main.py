"""Console-script multiplexer."""

from __future__ import annotations

import argparse
from pathlib import Path

from robustsense.cli import audit as audit_cli
from robustsense.cli import evaluate as evaluate_cli
from robustsense.cli import prepare as prepare_cli
from robustsense.cli import report as report_cli
from robustsense.cli import sweep as sweep_cli
from robustsense.cli import train as train_cli
from robustsense.pipeline import audit, evaluate, prepare, train


def main() -> None:
    parser = argparse.ArgumentParser(prog="robustsense")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("prepare", parents=[prepare_cli.build_parser()], add_help=False)
    subparsers.add_parser("audit", parents=[audit_cli.build_parser()], add_help=False)
    subparsers.add_parser("train", parents=[train_cli.build_parser()], add_help=False)
    subparsers.add_parser("evaluate", parents=[evaluate_cli.build_parser()], add_help=False)
    subparsers.add_parser("report", parents=[report_cli.build_parser()], add_help=False)
    subparsers.add_parser("sweep", parents=[sweep_cli.build_parser()], add_help=False)
    args = parser.parse_args()

    if args.command == "audit":
        audit_cli.run_cli(lambda: audit(args.config, args.fold, args.probe_only))
    elif args.command == "prepare":
        prepare_cli.run_cli(lambda: prepare(args.config, args.fold))
    elif args.command == "train":
        train_cli.run_cli(
            lambda: {
                "run_dir": str(
                    train(
                        args.project_root,
                        model_name=args.model,
                        fold=args.fold,
                        seed=args.seed,
                        profile=args.profile,
                        data_config=args.data_config,
                    )
                )
            }
        )
    elif args.command == "evaluate":
        evaluate_cli.run_cli(lambda: evaluate(args.run_dir, args.suite))
    elif args.command == "report":
        report_cli.run_cli(
            lambda: report_cli.generate_report(
                Path(args.project_root), args.runs_dir, args.output_dir, args.plan
            )
        )
    else:
        sweep_cli.run_cli(lambda: sweep_cli.execute_sweep(args))


if __name__ == "__main__":
    main()
