"""Run one train, derivation, or evaluation operation for a registered V2-3 unit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.training.v2_phase3_trainer import (
    derive_p2_a4_run,
    evaluate_v2_phase3,
    train_v2_phase3,
    v2_phase3_run_id,
)
from robustsense.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "derive-a4", "evaluate"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--variant", required=True)
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--plan", default="configs/v2/phase_v2_3_plan.yaml")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    if arguments.mode == "train":
        run_dir = train_v2_phase3(
            root,
            variant=arguments.variant,
            fold=arguments.fold,
            seed=arguments.seed,
            protocol_sha256=arguments.protocol_sha256,
            plan_path=arguments.plan,
        )
        result = {"operation": "train", "run_dir": str(run_dir)}
    elif arguments.mode == "derive-a4":
        if arguments.variant != "P2-A4":
            raise ValueError("derive-a4 mode requires P2-A4")
        run_dir = derive_p2_a4_run(
            root,
            fold=arguments.fold,
            seed=arguments.seed,
            protocol_sha256=arguments.protocol_sha256,
            plan_path=arguments.plan,
        )
        result = {"operation": "derive-a4", "run_dir": str(run_dir)}
    else:
        plan = load_config(root / arguments.plan)
        profile = load_config(root / plan["experiment_config"])
        run_dir = (
            root
            / profile["run_root"]
            / v2_phase3_run_id(arguments.variant, arguments.fold, arguments.seed)
        )
        result = evaluate_v2_phase3(run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
