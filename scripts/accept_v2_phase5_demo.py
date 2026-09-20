"""Run the fail-closed V2-5 Demo 2.0 acceptance replay."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from robustsense.utils.io import write_json
from robustsense.v2_demo import (
    DEMO_MODELS,
    infer_sample,
    load_demo_matrix,
    mask_background,
    mixed_background,
    persistent_episode_timeline,
    persistent_phase_timeline,
    validate_v2_demo_context,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2/phase_v2_5_acceptance.json")
    )
    arguments = parser.parse_args()
    root = arguments.project_root.resolve()
    context = validate_v2_demo_context(root)
    bundles = load_demo_matrix(
        root, arguments.fold, arguments.seed, device_name=arguments.device
    )
    availability = bundles["P2"].dataset.availability
    complete = np.flatnonzero(availability.all(axis=1))
    if len(complete) == 0:
        raise RuntimeError("Demo acceptance requires at least one complete test sample")
    sample_index = int(complete[0])
    results = {
        model_id: infer_sample(
            bundle,
            sample_index,
            disabled_modalities=("phone_acc",),
            fault_type="gaussian",
            target_modality="phone_gyro",
            severity=2.0,
        )
        for model_id, bundle in bundles.items()
    }
    if len({result["timestamp"] for result in results.values()}) != 1:
        raise AssertionError("Four-model replay did not use the same sample")
    for result in results.values():
        if not np.isclose(float(result["fusion_weights"].sum()), 1.0, atol=1.0e-6):
            raise AssertionError("Fusion weights do not sum to one")
        phone_acc = result["modalities"].index("phone_acc")
        if result["fusion_weights"][phone_acc] != 0.0:
            raise AssertionError("Disabled modality retained non-zero weight")
    p2 = results["P2"]
    if (
        not np.isfinite(p2["utility_scores"]).all()
        or not np.isfinite(p2["reliability"]).all()
        or p2["system_reliability"] is None
        or p2["abstain"] is None
    ):
        raise AssertionError("P2 did not emit the required real diagnostics")

    masks = mask_background(root)
    mixed = mixed_background(root)
    persistent = persistent_phase_timeline(
        root, fault="drop", target_modality="phone_acc", fault_length=15
    )
    episode, episode_metadata = persistent_episode_timeline(
        bundles["P2"],
        fault="drop",
        target_modality="phone_acc",
        fault_length=15,
    )
    report = {
        "version": 1,
        "phase": "V2-5",
        "status": "passed",
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "protocol_sha256": context["protocol_sha256"],
        "fold": arguments.fold,
        "seed": arguments.seed,
        "sample_index": sample_index,
        "models": list(DEMO_MODELS),
        "run_ids": {model_id: bundle.run_id for model_id, bundle in bundles.items()},
        "same_test_split_sha256": next(
            iter({bundle.test_split_sha256 for bundle in bundles.values()})
        ),
        "mixed_replay": {
            "disabled_modalities": ["phone_acc"],
            "fault": "gaussian",
            "target_modality": "phone_gyro",
            "severity": 2.0,
        },
        "p2": {
            "system_reliability": p2["system_reliability"],
            "abstention_threshold": p2["abstention_threshold"],
            "acceptance": p2["acceptance"],
            "utility_finite": bool(np.isfinite(p2["utility_scores"]).all()),
            "reliability_finite": bool(np.isfinite(p2["reliability"]).all()),
        },
        "report_contracts": {
            "mask_rows": len(masks),
            "mixed_rows": len(mixed),
            "persistent_phase_rows": len(persistent),
            "persistent_episode_points": len(episode),
            "persistent_episode_id": episode_metadata["episode_id"],
        },
        "random_prediction_fallback": False,
        "contract_mismatch_fallback": False,
    }
    output = arguments.output
    if not output.is_absolute():
        output = root / output
    write_json(output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
