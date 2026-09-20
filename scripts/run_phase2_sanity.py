"""Run the deterministic Phase-2 shuffled-label validity check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.training.sanity import run_random_label_sanity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    result = run_random_label_sanity(Path(args.project_root), args.fold, args.seed)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
