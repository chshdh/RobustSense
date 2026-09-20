"""PyTorch B0-B3 baselines with a shared multimodal batch contract."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from robustsense.constants import MODALITIES


class ModalityEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([features, mask.to(features.dtype)], dim=1))


class Phase2Model(nn.Module):
    model_id: str

    def valid_sample_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        return batch["availability"].any(dim=1)


class SingleSensorModel(Phase2Model):
    model_id = "B0"

    def __init__(
        self,
        modality: str,
        input_dim: int,
        n_labels: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.modality = modality
        self.modality_index = MODALITIES.index(modality)
        self.encoder = ModalityEncoder(input_dim, hidden_dim, latent_dim, dropout)
        self.classifier = nn.Linear(latent_dim, n_labels)

    def valid_sample_mask(self, batch: dict[str, Any]) -> torch.Tensor:
        return batch["availability"][:, self.modality_index]

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        encoded = self.encoder(
            batch["features"][self.modality], batch["feature_masks"][self.modality]
        )
        return {
            "logits": self.classifier(encoded),
            "fusion_weights": None,
            "reliability": None,
        }


class LinearEarlyFusion(Phase2Model):
    model_id = "B1"

    def __init__(self, total_features: int, n_labels: int):
        super().__init__()
        self.classifier = nn.Linear(total_features * 2 + len(MODALITIES), n_labels)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        inputs = torch.cat(
            [
                batch["features_flat"],
                batch["feature_masks_flat"].to(batch["features_flat"].dtype),
                batch["availability"].to(batch["features_flat"].dtype),
            ],
            dim=1,
        )
        return {
            "logits": self.classifier(inputs),
            "fusion_weights": None,
            "reliability": None,
        }


class MLPEarlyFusion(Phase2Model):
    model_id = "B2"

    def __init__(
        self,
        total_features: int,
        n_labels: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()
        input_dim = total_features * 2 + len(MODALITIES)
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, n_labels),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        inputs = torch.cat(
            [
                batch["features_flat"],
                batch["feature_masks_flat"].to(batch["features_flat"].dtype),
                batch["availability"].to(batch["features_flat"].dtype),
            ],
            dim=1,
        )
        return {
            "logits": self.network(inputs),
            "fusion_weights": None,
            "reliability": None,
        }


class LateFusion(Phase2Model):
    model_id = "B3"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        hidden_dim: int,
        latent_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(modality_dims[name], hidden_dim, latent_dim, dropout)
                for name in MODALITIES
            }
        )
        self.classifiers = nn.ModuleDict(
            {name: nn.Linear(latent_dim, n_labels) for name in MODALITIES}
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        modality_logits = []
        for name in MODALITIES:
            encoded = self.encoders[name](batch["features"][name], batch["feature_masks"][name])
            modality_logits.append(self.classifiers[name](encoded))
        stacked = torch.stack(modality_logits, dim=1)
        availability = batch["availability"].to(stacked.dtype)
        weights = availability / availability.sum(dim=1, keepdim=True).clamp_min(1.0)
        logits = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return {"logits": logits, "fusion_weights": weights, "reliability": None}


def build_phase2_model(
    model_name: str,
    modality_dims: dict[str, int],
    n_labels: int,
    config: dict[str, Any],
) -> Phase2Model:
    hidden_dim = int(config.get("hidden_dim", 128))
    latent_dim = int(config.get("latent_dim", 64))
    dropout = float(config.get("dropout", 0.2))
    total_features = sum(modality_dims.values())
    if model_name.startswith("single-"):
        modality = model_name.removeprefix("single-")
        if modality not in MODALITIES:
            raise ValueError(f"Unknown single-sensor modality: {modality}")
        return SingleSensorModel(
            modality,
            modality_dims[modality],
            n_labels,
            hidden_dim,
            latent_dim,
            dropout,
        )
    if model_name == "linear":
        return LinearEarlyFusion(total_features, n_labels)
    if model_name == "mlp":
        return MLPEarlyFusion(total_features, n_labels, hidden_dim, latent_dim, dropout)
    if model_name == "late":
        return LateFusion(modality_dims, n_labels, hidden_dim, latent_dim, dropout)
    raise ValueError(f"Phase 2 model must be single-<modality>, linear, mlp, or late: {model_name}")


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
