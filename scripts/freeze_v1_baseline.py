"""Create or verify the immutable V1 baseline manifest for RobustSense V2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from robustsense.experiments.v2_phase0 import freeze_v1_manifest, verify_v1_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "verify"))
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--manifest", type=Path, default=Path("reports/v2/v1_baseline_manifest.json")
    )
    args = parser.parse_args()
    root = args.project_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if args.mode == "freeze":
        result = freeze_v1_manifest(root, manifest_path)
        summary = {
            "status": "frozen",
            "file_count": result["file_count"],
            "content_sha256": result["content_sha256"],
            "manifest": str(manifest_path),
        }
    else:
        result = verify_v1_manifest(root, manifest_path)
        summary = {**result, "manifest": str(manifest_path)}
        if result["status"] != "passed":
            raise SystemExit(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

