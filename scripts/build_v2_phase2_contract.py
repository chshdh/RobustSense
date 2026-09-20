"""Generate byte-stable V2-2 scenario protocol artifacts (not model results)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from robustsense.evaluation.v2_scenarios import (
    availability_mask_manifest_bytes,
    build_exhaustive_masks,
    build_mixed_failures,
    mixed_failure_manifest_bytes,
)


def _write(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def build_contract(output_dir: Path) -> dict[str, object]:
    masks = build_exhaustive_masks()
    mixed = build_mixed_failures()
    artifacts = {
        "availability_masks": _write(
            output_dir / "availability_masks.jsonl",
            availability_mask_manifest_bytes(masks),
        ),
        "mixed_failures": _write(
            output_dir / "mixed_failures.jsonl",
            mixed_failure_manifest_bytes(mixed),
        ),
    }
    contract = {
        "protocol_id": "V2-2",
        "artifact_kind": "scenario_contract_not_model_results",
        "availability_mask_count": len(masks),
        "mixed_failure_count": len(mixed),
        "persistent_faults": ["drop", "gaussian_sigma_2", "bias_1"],
        "persistent_episode_lengths": [5, 15, 30],
        "maximum_adjacent_gap_seconds": 90,
        "artifacts": artifacts,
    }
    payload = (json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    _write(output_dir / "scenario_contract.json", payload)
    return contract


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("reports/v2/scenarios"))
    arguments = parser.parse_args()
    contract = build_contract(arguments.output_dir)
    print(json.dumps(contract, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
