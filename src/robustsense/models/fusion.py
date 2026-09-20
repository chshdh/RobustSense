"""Phase 3 gated and quality-aware multimodal fusion models."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from robustsense.constants import MODALITIES
from robustsense.models.torch_baselines import ModalityEncoder, Phase2Model


def masked_softmax(scores: torch.Tensor, availability: torch.Tensor) -> torch.Tensor:
    """Softmax over available modalities with exact zeros elsewhere."""
    if scores.ndim == 3 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.shape != availability.shape:
        raise ValueError("Scores and availability must have identical [batch, modality] shape")
    if (~availability.any(dim=1)).any():
        raise ValueError("Masked softmax requires at least one available modality")
    masked_scores = scores.masked_fill(~availability, -torch.inf)
    weights = torch.softmax(masked_scores, dim=1)
    weights = torch.where(availability, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=1, keepdim=True)


def reliability_constrained_weights(
    utility_scores: torch.Tensor,
    reliability: torch.Tensor,
    availability: torch.Tensor,
    beta: torch.Tensor | float,
    epsilon: float,
) -> torch.Tensor:
    """Apply the P2 gate while preserving exact availability invariants."""
    if utility_scores.shape != reliability.shape or reliability.shape != availability.shape:
        raise ValueError("Utility, reliability, and availability must share [batch, modality]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if torch.any((reliability < 0) | (reliability > 1)):
        raise ValueError("Reliability values must be in [0, 1]")
    beta_tensor = torch.as_tensor(beta, dtype=utility_scores.dtype, device=utility_scores.device)
    if beta_tensor.numel() != 1 or not bool((beta_tensor > 0).item()):
        raise ValueError("beta must be a positive scalar")
    scores = utility_scores + beta_tensor * torch.log(reliability + epsilon)
    return masked_softmax(scores, availability)


def system_reliability_score(
    fusion_weights: torch.Tensor, reliability: torch.Tensor
) -> torch.Tensor:
    if fusion_weights.shape != reliability.shape or fusion_weights.ndim != 2:
        raise ValueError("Fusion weights and reliability must share [batch, modality]")
    return (fusion_weights * reliability).sum(dim=1)


class GatedFusionModel(Phase2Model):
    """B4 architecture; B5 uses the same network with corrupted training views."""

    model_id = "B4/B5"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        gate_hidden_dim: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(modality_dims[name], hidden_dim, latent_dim, dropout)
                for name in MODALITIES
            }
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim, gate_hidden_dim),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for name in MODALITIES
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, n_labels),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        embeddings = []
        gate_scores = []
        for name in MODALITIES:
            encoded = self.encoders[name](batch["features"][name], batch["feature_masks"][name])
            embeddings.append(encoded)
            gate_scores.append(self.gates[name](encoded))
        stacked = torch.stack(embeddings, dim=1)
        weights = masked_softmax(torch.cat(gate_scores, dim=1), batch["availability"])
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": self.classifier(fused),
            "fusion_weights": weights,
            "reliability": None,
        }


class QualityAwareFusionModel(Phase2Model):
    """P: content gates augmented by explicit quality and learned reliability."""

    model_id = "P"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        quality_dim: int = 5,
        reliability_hidden_dim: int = 32,
        gate_hidden_dim: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        use_quality: bool = True,
        use_reliability: bool = True,
    ):
        super().__init__()
        self.quality_dim = quality_dim
        self.use_quality = bool(use_quality)
        self.use_reliability = bool(use_reliability)
        effective_quality_dim = quality_dim if self.use_quality else 0
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(modality_dims[name], hidden_dim, latent_dim, dropout)
                for name in MODALITIES
            }
        )
        self.reliability_nets = (
            nn.ModuleDict(
                {
                    name: nn.Sequential(
                        nn.Linear(
                            latent_dim + effective_quality_dim,
                            reliability_hidden_dim,
                        ),
                        nn.GELU(),
                        nn.Linear(reliability_hidden_dim, 1),
                        nn.Sigmoid(),
                    )
                    for name in MODALITIES
                }
            )
            if self.use_reliability
            else nn.ModuleDict()
        )
        self.gates = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(
                        latent_dim
                        + effective_quality_dim
                        + (1 if self.use_reliability else 0),
                        gate_hidden_dim,
                    ),
                    nn.GELU(),
                    nn.Linear(gate_hidden_dim, 1),
                )
                for name in MODALITIES
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, n_labels),
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        quality = batch["quality_features"]
        if quality.shape[-1] != self.quality_dim:
            raise ValueError(
                f"Expected {self.quality_dim} quality features, got {quality.shape[-1]}"
            )
        embeddings = []
        reliability_values = []
        gate_scores = []
        for modality_index, name in enumerate(MODALITIES):
            encoded = self.encoders[name](batch["features"][name], batch["feature_masks"][name])
            modality_quality = quality[:, modality_index]
            quality_input = modality_quality if self.use_quality else encoded[:, :0]
            if self.use_reliability:
                reliability = self.reliability_nets[name](
                    torch.cat([encoded, quality_input], dim=1)
                )
                reliability_values.append(reliability)
                gate_input = torch.cat([encoded, quality_input, reliability], dim=1)
            else:
                gate_input = torch.cat([encoded, quality_input], dim=1)
            embeddings.append(encoded)
            gate_scores.append(self.gates[name](gate_input))
        stacked = torch.stack(embeddings, dim=1)
        reliability_tensor = (
            torch.cat(reliability_values, dim=1) if self.use_reliability else None
        )
        weights = masked_softmax(torch.cat(gate_scores, dim=1), batch["availability"])
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        return {
            "logits": self.classifier(fused),
            "fusion_weights": weights,
            "reliability": reliability_tensor,
        }


class ReliabilityConstrainedFusionModel(Phase2Model):
    """P2: task utility and reliability are separate, then constrained at fusion."""

    model_id = "P2"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        quality_dim: int = 5,
        reliability_hidden_dim: int = 32,
        utility_hidden_dim: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        beta_init: float = 1.0,
        epsilon: float = 1.0e-6,
        abstention_threshold: float | None = None,
    ):
        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if beta_init <= epsilon:
            raise ValueError("beta_init must be greater than epsilon")
        self.quality_dim = int(quality_dim)
        self.epsilon = float(epsilon)
        inverse_softplus = math.log(math.expm1(float(beta_init) - self.epsilon))
        self.raw_beta = nn.Parameter(torch.tensor(inverse_softplus, dtype=torch.float32))
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(modality_dims[name], hidden_dim, latent_dim, dropout)
                for name in MODALITIES
            }
        )
        self.reliability_nets = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim + self.quality_dim, reliability_hidden_dim),
                    nn.GELU(),
                    nn.Linear(reliability_hidden_dim, 1),
                    nn.Sigmoid(),
                )
                for name in MODALITIES
            }
        )
        self.utility_nets = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(latent_dim, utility_hidden_dim),
                    nn.GELU(),
                    nn.Linear(utility_hidden_dim, 1),
                )
                for name in MODALITIES
            }
        )
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_dim, n_labels),
        )
        self.abstention_threshold: float | None = None
        self.set_abstention_threshold(abstention_threshold)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) + self.epsilon

    def set_abstention_threshold(self, threshold: float | None) -> None:
        if threshold is not None and not 0 <= float(threshold) <= 1:
            raise ValueError("Abstention threshold must be in [0, 1]")
        self.abstention_threshold = None if threshold is None else float(threshold)

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor | None]:
        quality = batch["quality_features"]
        if quality.shape[-1] != self.quality_dim:
            raise ValueError(
                f"Expected {self.quality_dim} quality features, got {quality.shape[-1]}"
            )
        embeddings = []
        reliability_values = []
        utility_values = []
        for modality_index, name in enumerate(MODALITIES):
            encoded = self.encoders[name](batch["features"][name], batch["feature_masks"][name])
            embeddings.append(encoded)
            reliability_values.append(
                self.reliability_nets[name](
                    torch.cat([encoded, quality[:, modality_index]], dim=1)
                )
            )
            utility_values.append(self.utility_nets[name](encoded))
        stacked = torch.stack(embeddings, dim=1)
        reliability = torch.cat(reliability_values, dim=1)
        utility_scores = torch.cat(utility_values, dim=1)
        weights = reliability_constrained_weights(
            utility_scores,
            reliability,
            batch["availability"],
            self.beta,
            self.epsilon,
        )
        fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)
        system_reliability = system_reliability_score(weights, reliability)
        abstain = (
            system_reliability < self.abstention_threshold
            if self.abstention_threshold is not None
            else None
        )
        return {
            "logits": self.classifier(fused),
            "fusion_weights": weights,
            "reliability": reliability,
            "utility_scores": utility_scores,
            "system_reliability": system_reliability,
            "abstain": abstain,
        }


def build_phase3_model(
    model_name: str,
    modality_dims: dict[str, int],
    n_labels: int,
    config: dict[str, Any],
) -> Phase2Model:
    common = {
        "modality_dims": modality_dims,
        "n_labels": n_labels,
        "hidden_dim": int(config.get("hidden_dim", 128)),
        "latent_dim": int(config.get("latent_dim", config.get("embedding_dim", 64))),
        "gate_hidden_dim": int(config.get("gate_hidden_dim", 32)),
        "classifier_hidden_dim": int(config.get("classifier_hidden_dim", 128)),
        "dropout": float(config.get("dropout", 0.2)),
    }
    if model_name in {"gated", "robust-gated"}:
        return GatedFusionModel(**common)
    if model_name in {
        "quality-aware-cls",
        "quality-aware",
        "ablation-a1-no-sensor-dropout",
        "ablation-a2-no-noise",
        "ablation-a3-no-quality",
        "ablation-a4-no-reliability",
        "ablation-a5-no-reliability-loss",
    }:
        return QualityAwareFusionModel(
            **common,
            quality_dim=int(config.get("quality_dim", 5)),
            reliability_hidden_dim=int(config.get("reliability_hidden_dim", 32)),
            use_quality=model_name not in {
                "ablation-a3-no-quality",
                "ablation-a4-no-reliability",
            },
            use_reliability=model_name != "ablation-a4-no-reliability",
        )
    if model_name == "reliability-constrained":
        return ReliabilityConstrainedFusionModel(
            modality_dims=modality_dims,
            n_labels=n_labels,
            hidden_dim=common["hidden_dim"],
            latent_dim=common["latent_dim"],
            quality_dim=int(config.get("quality_dim", 5)),
            reliability_hidden_dim=int(config.get("reliability_hidden_dim", 32)),
            utility_hidden_dim=int(config.get("utility_hidden_dim", 32)),
            classifier_hidden_dim=common["classifier_hidden_dim"],
            dropout=common["dropout"],
            beta_init=float(config.get("beta_init", 1.0)),
            epsilon=float(config.get("epsilon", 1.0e-6)),
            abstention_threshold=config.get("abstention_threshold"),
        )
    raise ValueError(
        "Fusion model must be gated, robust-gated, quality-aware, or "
        f"reliability-constrained: {model_name}"
    )
