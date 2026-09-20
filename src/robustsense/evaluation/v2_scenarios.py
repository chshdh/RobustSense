"""Deterministic Phase V2-2 masks, mixed faults, and persistent episodes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robustsense.constants import MODALITIES
from robustsense.corruption import CorruptionRegistry


def _canonical_bytes(values: list[dict[str, Any]]) -> bytes:
    lines = [
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for value in values
    ]
    return ("\n".join(lines) + "\n").encode("utf-8")


@dataclass(frozen=True)
class AvailabilityMaskSpec:
    mask_id: str
    bits: tuple[bool, ...]
    available_modalities: tuple[str, ...]
    drop_modalities: tuple[str, ...]

    @property
    def available_count(self) -> int:
        return sum(self.bits)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mask_id": self.mask_id,
            "bits": [int(value) for value in self.bits],
            "available_modalities": list(self.available_modalities),
            "drop_modalities": list(self.drop_modalities),
            "available_count": self.available_count,
        }


def build_exhaustive_masks() -> list[AvailabilityMaskSpec]:
    masks = []
    for code in range(1, 1 << len(MODALITIES)):
        bits = tuple(bool(code & (1 << index)) for index in range(len(MODALITIES)))
        available = tuple(name for name, enabled in zip(MODALITIES, bits, strict=True) if enabled)
        dropped = tuple(name for name, enabled in zip(MODALITIES, bits, strict=True) if not enabled)
        bit_string = "".join("1" if enabled else "0" for enabled in bits)
        masks.append(
            AvailabilityMaskSpec(
                mask_id=f"mask.{bit_string}",
                bits=bits,
                available_modalities=available,
                drop_modalities=dropped,
            )
        )
    if len(masks) != 63 or len({item.mask_id for item in masks}) != 63:
        raise AssertionError("Six modalities must produce 63 unique non-empty masks")
    return masks


def availability_mask_manifest_bytes(masks: list[AvailabilityMaskSpec]) -> bytes:
    return _canonical_bytes([item.as_dict() for item in masks])


def availability_mask_manifest_sha256(masks: list[AvailabilityMaskSpec]) -> str:
    return hashlib.sha256(availability_mask_manifest_bytes(masks)).hexdigest()


@dataclass(frozen=True)
class MixedFailureSpec:
    scenario_id: str
    drop_modality: str
    noisy_modality: str
    gaussian_sigma: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_mixed_failures(sigmas: tuple[float, ...] = (1.0, 2.0)) -> list[MixedFailureSpec]:
    scenarios = []
    for dropped in MODALITIES:
        for noisy in MODALITIES:
            if noisy == dropped:
                continue
            for sigma in sigmas:
                scenarios.append(
                    MixedFailureSpec(
                        scenario_id=f"mixed.drop-{dropped}.gaussian-{noisy}.sigma-{sigma:g}",
                        drop_modality=dropped,
                        noisy_modality=noisy,
                        gaussian_sigma=float(sigma),
                    )
                )
    if len(scenarios) != 60 or len({item.scenario_id for item in scenarios}) != 60:
        raise AssertionError("Mixed failure protocol must contain 6 x 5 x 2 scenarios")
    return scenarios


def mixed_failure_manifest_bytes(scenarios: list[MixedFailureSpec]) -> bytes:
    return _canonical_bytes([item.as_dict() for item in scenarios])


def mixed_failure_manifest_sha256(scenarios: list[MixedFailureSpec]) -> str:
    return hashlib.sha256(mixed_failure_manifest_bytes(scenarios)).hexdigest()


def apply_mixed_failure(
    registry: CorruptionRegistry,
    batch: dict[str, Any],
    scenario: MixedFailureSpec,
    *,
    seed: int,
    view: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if scenario.drop_modality == scenario.noisy_modality:
        raise ValueError("Mixed failure cannot corrupt the modality it already dropped")
    missing, missing_meta = registry.apply_controlled(
        batch,
        "missing",
        seed=seed,
        view=f"{view}|missing",
        drop_modalities=(scenario.drop_modality,),
    )
    mixed, noise_meta = registry.apply_controlled(
        missing,
        "gaussian",
        seed=seed,
        view=f"{view}|gaussian",
        target_modality=scenario.noisy_modality,
        severity=scenario.gaussian_sigma,
    )
    fault_type = [row.copy() for row in missing_meta["fault_type"]]
    noisy_index = MODALITIES.index(scenario.noisy_modality)
    for row in fault_type:
        row[noisy_index] = "gaussian"
    metadata = {
        "fault_type": fault_type,
        "severity": torch.maximum(missing_meta["severity"], noise_meta["severity"]),
        "reliability_target": torch.minimum(
            missing_meta["reliability_target"], noise_meta["reliability_target"]
        ),
        "artificial_fault": (
            missing_meta["artificial_fault"] | noise_meta["artificial_fault"]
        ),
        "dropped": missing_meta["dropped"],
    }
    return mixed, metadata


@dataclass(frozen=True)
class PersistentEpisode:
    episode_id: str
    user_id: str
    fault_length: int
    pre_indices: tuple[int, ...]
    fault_indices: tuple[int, ...]
    post_indices: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "user_id": self.user_id,
            "fault_length": self.fault_length,
            "pre_indices": list(self.pre_indices),
            "fault_indices": list(self.fault_indices),
            "post_indices": list(self.post_indices),
        }


def build_persistent_episodes(
    user_ids: np.ndarray,
    timestamps: np.ndarray,
    episode_lengths: tuple[int, ...] = (5, 15, 30),
    *,
    max_gap_seconds: int = 90,
) -> tuple[list[PersistentEpisode], list[dict[str, int]]]:
    users = np.asarray(user_ids).astype(str)
    times = np.asarray(timestamps, dtype=np.int64)
    if users.ndim != 1 or times.ndim != 1 or len(users) != len(times):
        raise ValueError("Users and timestamps must be aligned one-dimensional arrays")
    if any(length <= 0 for length in episode_lengths):
        raise ValueError("Episode lengths must be positive")
    episodes: list[PersistentEpisode] = []
    summaries: list[dict[str, int]] = []
    segments: list[tuple[str, list[int]]] = []
    for user in sorted(set(users.tolist())):
        ordered = sorted(
            np.flatnonzero(users == user).tolist(),
            key=lambda index: (times[index], index),
        )
        current: list[int] = []
        for index in ordered:
            if current and int(times[index] - times[current[-1]]) > max_gap_seconds:
                segments.append((user, current))
                current = []
            current.append(index)
        if current:
            segments.append((user, current))

    for length in episode_lengths:
        window = 3 * int(length)
        short_segments = 0
        unused_rows = 0
        count_before = len(episodes)
        for user, segment in segments:
            if len(segment) < window:
                short_segments += 1
                unused_rows += len(segment)
                continue
            window_count = len(segment) // window
            unused_rows += len(segment) - window_count * window
            for window_index in range(window_count):
                block = segment[window_index * window : (window_index + 1) * window]
                pre = tuple(block[:length])
                fault = tuple(block[length : 2 * length])
                post = tuple(block[2 * length :])
                payload = f"{user}|{length}|{times[pre[0]]}|{window_index}".encode()
                identifier = hashlib.sha256(payload).hexdigest()[:16]
                episodes.append(
                    PersistentEpisode(
                        episode_id=f"episode.{identifier}",
                        user_id=user,
                        fault_length=int(length),
                        pre_indices=pre,
                        fault_indices=fault,
                        post_indices=post,
                    )
                )
        summaries.append(
            {
                "fault_length": int(length),
                "episode_count": len(episodes) - count_before,
                "excluded_short_segment_count": short_segments,
                "unused_row_count": unused_rows,
            }
        )
    return episodes, summaries


def persistent_episode_manifest_bytes(episodes: list[PersistentEpisode]) -> bytes:
    return _canonical_bytes([item.as_dict() for item in episodes])


def write_scenario_manifest(path: str | Path, values: bytes) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(values)
    return hashlib.sha256(values).hexdigest()


def scenario_sample_sha256(user_ids: np.ndarray, timestamps: np.ndarray) -> str:
    users = np.asarray(user_ids).astype(str)
    times = np.asarray(timestamps, dtype=np.int64)
    if len(users) != len(times):
        raise ValueError("Users and timestamps must align")
    payload = "\n".join(
        f"{user}|{int(timestamp)}" for user, timestamp in zip(users, times, strict=True)
    )
    return hashlib.sha256((payload + "\n").encode("utf-8")).hexdigest()
