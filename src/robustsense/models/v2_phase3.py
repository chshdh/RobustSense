"""Frozen P2 ablation variants for Phase V2-3."""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from robustsense.constants import MODALITIES
from robustsense.models.fusion import (
    ModalityEncoder,
    ReliabilityConstrainedFusionModel,
    masked_softmax,
    reliability_constrained_weights,
    system_reliability_score,
)
from robustsense.models.torch_baselines import Phase2Model

V2_PHASE3_VARIANTS = (
    "P2",
    "P2-A1",
    "P2-A2",
    "P2-A3",
    "P2-A4",
)


class P2AblationModel(Phase2Model):
    """P2 with exactly one predeclared component removed."""

    model_id = "P2"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        *,
        variant: str,
        hidden_dim: int = 128,
        latent_dim: int = 64,
        quality_dim: int = 5,
        reliability_hidden_dim: int = 32,
        utility_hidden_dim: int = 32,
        classifier_hidden_dim: int = 128,
        dropout: float = 0.2,
        beta_init: float = 1.0,
        epsilon: float = 1.0e-6,
    ):
        super().__init__()
        if variant not in V2_PHASE3_VARIANTS:
            raise ValueError(f"Unknown V2 Phase-3 variant: {variant}")
        if epsilon <= 0 or beta_init <= epsilon:
            raise ValueError("beta_init must be greater than positive epsilon")
        self.variant = variant
        self.quality_dim = int(quality_dim)
        self.epsilon = float(epsilon)
        self.use_reliability_constraint = variant != "P2-A1"
        self.reliability_uses_quality = variant != "P2-A3"
        self.selective_prediction = variant != "P2-A4"
        inverse_softplus = math.log(math.expm1(float(beta_init) - self.epsilon))
        self.raw_beta = nn.Parameter(
            torch.tensor(inverse_softplus, dtype=torch.float32),
            requires_grad=self.use_reliability_constraint,
        )
        self.encoders = nn.ModuleDict(
            {
                name: ModalityEncoder(modality_dims[name], hidden_dim, latent_dim, dropout)
                for name in MODALITIES
            }
        )
        reliability_input_dim = latent_dim + (
            self.quality_dim if self.reliability_uses_quality else 0
        )
        self.reliability_nets = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Linear(reliability_input_dim, reliability_hidden_dim),
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

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) + self.epsilon

    def set_abstention_threshold(self, threshold: float | None) -> None:
        if threshold is not None and not self.selective_prediction:
            raise ValueError("P2-A4 removes selective prediction")
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
            encoded = self.encoders[name](
                batch["features"][name], batch["feature_masks"][name]
            )
            reliability_input = encoded
            if self.reliability_uses_quality:
                reliability_input = torch.cat(
                    [encoded, quality[:, modality_index]], dim=1
                )
            embeddings.append(encoded)
            reliability_values.append(self.reliability_nets[name](reliability_input))
            utility_values.append(self.utility_nets[name](encoded))
        stacked = torch.stack(embeddings, dim=1)
        reliability = torch.cat(reliability_values, dim=1)
        utility_scores = torch.cat(utility_values, dim=1)
        if self.use_reliability_constraint:
            weights = reliability_constrained_weights(
                utility_scores,
                reliability,
                batch["availability"],
                self.beta,
                self.epsilon,
            )
        else:
            weights = masked_softmax(utility_scores, batch["availability"])
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


def build_v2_phase3_model(
    variant: str,
    modality_dims: dict[str, int],
    n_labels: int,
    config: dict[str, Any],
) -> P2AblationModel:
    return P2AblationModel(
        modality_dims,
        n_labels,
        variant=variant,
        hidden_dim=int(config.get("hidden_dim", 128)),
        latent_dim=int(config.get("latent_dim", config.get("embedding_dim", 64))),
        quality_dim=int(config.get("quality_dim", 5)),
        reliability_hidden_dim=int(config.get("reliability_hidden_dim", 32)),
        utility_hidden_dim=int(config.get("utility_hidden_dim", 32)),
        classifier_hidden_dim=int(config.get("classifier_hidden_dim", 128)),
        dropout=float(config.get("dropout", 0.2)),
        beta_init=float(config.get("beta_init", 1.0)),
        epsilon=float(config.get("epsilon", 1.0e-6)),
    )


def copy_p2_weights_for_equivalence(
    target: P2AblationModel, source: ReliabilityConstrainedFusionModel
) -> None:
    """Test helper: P2 and its generalized implementation must be state-compatible."""
    if target.variant != "P2":
        raise ValueError("Only the complete P2 variant is equivalent to the V2-1 model")
    target.load_state_dict(source.state_dict(), strict=True)
