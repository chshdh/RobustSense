"""Processed multimodal ExtraSensory dataset and batch collation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from robustsense.constants import MODALITIES
from robustsense.utils.io import read_json


class MultiModalDataset(Dataset):
    def __init__(self, split_path: str | Path, metadata_path: str | Path):
        self.split_path = Path(split_path)
        self.metadata = read_json(metadata_path)
        with np.load(self.split_path) as arrays:
            self.features = arrays["features"].astype(np.float32)
            self.feature_masks = arrays["feature_masks"].astype(bool)
            self.availability = arrays["availability"].astype(bool)
            self.quality_features = arrays["quality_features"].astype(np.float32)
            self.targets = arrays["targets"].astype(np.float32)
            self.target_mask = arrays["target_mask"].astype(bool)
            self.user_id = arrays["user_id"].astype(str)
            self.timestamp = arrays["timestamp"].astype(np.int64)
        self.modality_slices = {
            name: tuple(self.metadata["modality_slices"][name]) for name in MODALITIES
        }

    def __len__(self) -> int:
        return len(self.timestamp)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {
            "features_flat": self.features[index],
            "feature_masks_flat": self.feature_masks[index],
            "availability": self.availability[index],
            "quality_features": self.quality_features[index],
            "targets": self.targets[index],
            "target_mask": self.target_mask[index],
            "user_id": self.user_id[index],
            "timestamp": self.timestamp[index],
        }


def collate_multimodal(rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat_features = torch.from_numpy(np.stack([row["features_flat"] for row in rows]))
    flat_masks = torch.from_numpy(np.stack([row["feature_masks_flat"] for row in rows]))
    availability = torch.from_numpy(np.stack([row["availability"] for row in rows]))
    quality = torch.from_numpy(np.stack([row["quality_features"] for row in rows]))
    targets = torch.from_numpy(np.stack([row["targets"] for row in rows]))
    target_mask = torch.from_numpy(np.stack([row["target_mask"] for row in rows]))
    slices = rows[0].get("modality_slices")
    return {
        "features_flat": flat_features,
        "feature_masks_flat": flat_masks,
        "availability": availability,
        "quality_features": quality,
        "targets": targets,
        "target_mask": target_mask,
        "user_id": [str(row["user_id"]) for row in rows],
        "timestamp": torch.tensor([int(row["timestamp"]) for row in rows]),
        "modality_slices": slices,
    }


def add_modality_views(
    batch: dict[str, Any], modality_slices: dict[str, tuple[int, int]]
) -> dict[str, Any]:
    batch["features"] = {
        name: batch["features_flat"][:, start:end]
        for name, (start, end) in modality_slices.items()
    }
    batch["feature_masks"] = {
        name: batch["feature_masks_flat"][:, start:end]
        for name, (start, end) in modality_slices.items()
    }
    return batch
