"""Run one frozen V2-4 metrics-only extended evaluation unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.evaluation.v2_phase4 import evaluate_v2_phase4_extended


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-id", choices=("B4", "B5", "P", "P2"), required=True)
    parser.add_argument(
        "--config", default="configs/v2/evaluation/extended_core.yaml"
    )
    arguments = parser.parse_args()
    result = evaluate_v2_phase4_extended(
        arguments.run_dir,
        model_id=arguments.model_id,
        config_path=arguments.config,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
