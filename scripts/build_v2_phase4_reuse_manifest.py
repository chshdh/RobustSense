"""Build the metric-blind V2-4 seed-13 reuse manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.data.extrasensory import sha256_file
from robustsense.experiments.v2_phase4 import build_v2_phase4_reuse_manifest
from robustsense.utils.config import load_config
from robustsense.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--plan", default="configs/v2/phase_v2_4_plan.yaml")
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    plan = load_config(root / arguments.plan)
    output = root / plan["reuse_manifest"]
    manifest = build_v2_phase4_reuse_manifest(root, arguments.plan)
    write_json(output, manifest)
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256_file(output),
                "entry_count": manifest["entry_count"],
                "metric_values_read": manifest["metric_values_read"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
