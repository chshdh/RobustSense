"""Run one P2-family V2-4 train or evaluation operation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.training.v2_phase4_trainer import (
    evaluate_v2_phase4,
    evaluate_v2_phase4_v1,
    train_v2_phase4,
    train_v2_phase4_v1,
    v2_phase4_run_id,
)
from robustsense.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--variant")
    identity.add_argument("--v1-model")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--plan", default="configs/v2/phase_v2_4_plan.yaml")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    if arguments.mode == "train":
        if arguments.v1_model:
            run_dir = train_v2_phase4_v1(
                root,
                model_name=arguments.v1_model,
                fold=arguments.fold,
                seed=arguments.seed,
                protocol_sha256=arguments.protocol_sha256,
                plan_path=arguments.plan,
            )
        else:
            run_dir = train_v2_phase4(
                root,
                variant=arguments.variant,
                fold=arguments.fold,
                seed=arguments.seed,
                protocol_sha256=arguments.protocol_sha256,
                plan_path=arguments.plan,
            )
        result = {"operation": "train", "run_dir": str(run_dir)}
    else:
        plan = load_config(root / arguments.plan)
        if arguments.v1_model:
            run_dir = (
                root
                / "runs"
                / (
                    f"extrasensory-{arguments.v1_model}-fold{arguments.fold}-"
                    f"seed{arguments.seed}-full"
                )
            )
            result = evaluate_v2_phase4_v1(run_dir)
        else:
            profile = load_config(root / plan["experiment_config"])
            run_dir = (
                root
                / profile["run_root"]
                / v2_phase4_run_id(arguments.variant, arguments.fold, arguments.seed)
            )
            result = evaluate_v2_phase4(run_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
