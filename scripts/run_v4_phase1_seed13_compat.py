"""Run V4-1 seed-13 units whose P2 sources are verified V2-3 reuses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import run_v4_phase1 as v4
import torch

from robustsense.utils.config import load_config


def _verified_reuse_source(
    root: Path,
    plan: dict[str, Any],
    registry: dict[tuple[int, int], dict[str, str]],
    fold: int,
    seed: int,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    row = registry[(fold, seed)]
    source_dir = root / row["run_dir"]
    resolved = load_config(source_dir / "resolved_config.yaml")
    if not (
        seed == 13
        and row["status"] == "success"
        and row["reason"].startswith("reused_contract_verified:")
        and resolved.get("phase") == "V2-3"
        and resolved.get("variant") == "P2"
        and int(resolved.get("fold", -1)) == fold
        and int(resolved.get("seed", -1)) == seed
    ):
        raise RuntimeError("来源不符合冻结的 V2-3 → V2-4 seed-13 复用合同")
    checkpoint = torch.load(
        source_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False
    )
    return source_dir, resolved, checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("train", "evaluate"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, required=True)
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan, lock = v4._load_governance(root)
    if arguments.fold not in plan["folds"]:
        parser.error("fold 不在冻结矩阵")
    registry = v4._p2_registry(root, plan)
    v4._source_context = _verified_reuse_source
    if arguments.mode == "train":
        v4.train_unit(
            root,
            plan,
            lock,
            registry,
            fold=arguments.fold,
            seed=13,
            force=False,
        )
    else:
        v4.evaluate_unit(
            root,
            plan,
            lock,
            registry,
            fold=arguments.fold,
            seed=13,
            force=False,
        )


if __name__ == "__main__":
    main()
