"""ExtraSensory Phase-1 audit CLI."""

from __future__ import annotations

import argparse

from robustsense.cli._common import run_cli
from robustsense.pipeline import audit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit the official ExtraSensory archives")
    parser.add_argument("--config", required=True, help="Path to the ExtraSensory data config")
    parser.add_argument("--fold", type=int, default=0, help="Outer test fold for split manifest")
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Validate archives, headers, mapping, and folds without scanning all rows",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_cli(lambda: audit(args.config, args.fold, args.probe_only))


if __name__ == "__main__":
    main()
