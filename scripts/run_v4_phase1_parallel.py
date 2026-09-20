"""以有限并发运行 V4-1 的 15 个训练与独立测试评估单元。"""

from __future__ import annotations

import argparse
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from robustsense.utils.config import load_config


def _unit(
    root: Path, fold: int, seed: int, mode: str, log_dir: Path
) -> tuple[int, int, int, str]:
    log_path = log_dir / f"fold{fold}_seed{seed}.log"
    operations = ("train", "evaluate") if mode == "both" else (mode,)
    for operation in operations:
        command = [
            sys.executable,
            str(root / "scripts/run_v4_phase1.py"),
            operation,
            "--project-root",
            str(root),
            "--fold",
            str(fold),
            "--seed",
            str(seed),
        ]
        with log_path.open("a", encoding="utf-8") as stream:
            stream.write(
                f"\n[{datetime.now(UTC).isoformat()}] {' '.join(command)}\n"
            )
            stream.flush()
            completed = subprocess.run(
                command,
                cwd=root,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if completed.returncode:
            return fold, seed, completed.returncode, str(log_path)
    return fold, seed, 0, str(log_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--mode", choices=("train", "evaluate", "both"), default="both")
    arguments = parser.parse_args()
    if arguments.workers < 1 or arguments.workers > 3:
        parser.error("workers 必须在 1 到 3 之间")
    root = arguments.project_root.resolve()
    plan = load_config(root / "configs/v4/phase_v4_1_plan.yaml")
    units = [(int(fold), int(seed)) for fold in plan["folds"] for seed in plan["seeds"]]
    log_dir = root / "runs/v4/_phase_v4_1_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    with ThreadPoolExecutor(max_workers=arguments.workers) as executor:
        futures = {
            executor.submit(
                _unit, root, fold, seed, arguments.mode, log_dir
            ): (fold, seed)
            for fold, seed in units
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            fold, seed, return_code, log_path = future.result()
            status = "完成" if return_code == 0 else f"失败({return_code})"
            print(
                f"[{completed_count}/{len(units)}] fold={fold} seed={seed} {status}",
                flush=True,
            )
            if return_code:
                failures.append((fold, seed, log_path))
    if failures:
        raise RuntimeError(f"V4-1 有 {len(failures)} 个失败单元：{failures}")


if __name__ == "__main__":
    main()
