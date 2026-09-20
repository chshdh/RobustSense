"""V4-1 label-conditioned modality routing with exact P2 initialization."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from robustsense.constants import MODALITIES
from robustsense.models.torch_baselines import ModalityEncoder, Phase2Model


def label_conditioned_weights(
    utility_scores: torch.Tensor,
    reliability: torch.Tensor,
    availability: torch.Tensor,
    beta: torch.Tensor | float,
    epsilon: float,
) -> torch.Tensor:
    """Return [batch, label, modality] weights with exact availability masking."""
    if utility_scores.ndim != 3:
        raise ValueError("Utility scores must have [batch, modality, label] shape")
    batch, modality_count, _ = utility_scores.shape
    expected = (batch, modality_count)
    if reliability.shape != expected or availability.shape != expected:
        raise ValueError("Reliability and availability must match utility modalities")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if torch.any((reliability < 0) | (reliability > 1)):
        raise ValueError("Reliability values must be in [0, 1]")
    if (~availability.any(dim=1)).any():
        raise ValueError("Every sample needs at least one available modality")
    beta_tensor = torch.as_tensor(
        beta, dtype=utility_scores.dtype, device=utility_scores.device
    )
    if beta_tensor.numel() != 1 or not bool((beta_tensor > 0).item()):
        raise ValueError("beta must be a positive scalar")
    scores = utility_scores.permute(0, 2, 1)
    scores = scores + beta_tensor * torch.log(reliability[:, None, :] + epsilon)
    expanded_availability = availability[:, None, :].expand_as(scores)
    weights = torch.softmax(scores.masked_fill(~expanded_availability, -torch.inf), dim=2)
    weights = torch.where(expanded_availability, weights, torch.zeros_like(weights))
    return weights / weights.sum(dim=2, keepdim=True)


class LabelConditionedFusionModel(Phase2Model):
    """P2 adapter whose modality routing is different for every output label."""

    model_id = "P2-LC"

    def __init__(
        self,
        modality_dims: dict[str, int],
        n_labels: int,
        *,
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
        if n_labels <= 0 or epsilon <= 0 or beta_init <= epsilon:
            raise ValueError("Model dimensions and reliability constants are invalid")
        self.n_labels = int(n_labels)
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
                    nn.Linear(utility_hidden_dim, self.n_labels),
                )
                for name in MODALITIES
            }
        )
        self.classifier_hidden = nn.Sequential(
            nn.Linear(latent_dim, classifier_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.label_classifier_weight = nn.Parameter(
            torch.empty(self.n_labels, classifier_hidden_dim)
        )
        self.label_classifier_bias = nn.Parameter(torch.empty(self.n_labels))
        nn.init.kaiming_uniform_(self.label_classifier_weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(classifier_hidden_dim)
        nn.init.uniform_(self.label_classifier_bias, -bound, bound)

    @property
    def beta(self) -> torch.Tensor:
        return F.softplus(self.raw_beta) + self.epsilon

    def freeze_p2_backbone_for_adapter_training(self) -> None:
        """Freeze inherited P2 components and train only label-conditioned utilities."""
        self.raw_beta.requires_grad_(False)
        for module in (self.encoders, self.reliability_nets, self.classifier_hidden):
            module.requires_grad_(False)
        self.label_classifier_weight.requires_grad_(False)
        self.label_classifier_bias.requires_grad_(False)
        self.utility_nets.requires_grad_(True)

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
            embeddings.append(encoded)
            reliability_values.append(
                self.reliability_nets[name](
                    torch.cat([encoded, quality[:, modality_index]], dim=1)
                )
            )
            utility_values.append(self.utility_nets[name](encoded))
        stacked = torch.stack(embeddings, dim=1)
        reliability = torch.cat(reliability_values, dim=1)
        utility_scores = torch.stack(utility_values, dim=1)
        label_weights = label_conditioned_weights(
            utility_scores,
            reliability,
            batch["availability"],
            self.beta,
            self.epsilon,
        )
        # Keep P2's multiply-then-sum reduction order.  Besides making the
        # warm start numerically auditable, this avoids a backend-dependent
        # einsum contraction changing otherwise equivalent logits.
        label_fused = (
            stacked[:, None, :, :] * label_weights[:, :, :, None]
        ).sum(dim=2)
        hidden = self.classifier_hidden(label_fused)
        logits = torch.einsum("blh,lh->bl", hidden, self.label_classifier_weight)
        logits = logits + self.label_classifier_bias
        mean_weights = label_weights.mean(dim=1)
        label_system_reliability = (
            label_weights * reliability[:, None, :]
        ).sum(dim=2)
        return {
            "logits": logits,
            "fusion_weights": mean_weights,
            "label_fusion_weights": label_weights,
            "reliability": reliability,
            "utility_scores": utility_scores,
            "system_reliability": label_system_reliability.mean(dim=1),
            "label_system_reliability": label_system_reliability,
            "abstain": None,
        }


def build_label_conditioned_model(
    modality_dims: dict[str, int], n_labels: int, config: dict[str, Any]
) -> LabelConditionedFusionModel:
    return LabelConditionedFusionModel(
        modality_dims,
        n_labels,
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


def initialize_label_conditioned_from_p2(
    target: LabelConditionedFusionModel,
    source_state: Mapping[str, torch.Tensor],
) -> None:
    """Map a P2 state into an exactly equivalent label-conditioned initialization."""
    target_state = target.state_dict()
    copied = set()
    direct_prefixes = ("encoders.", "reliability_nets.")
    for name, value in source_state.items():
        if name == "raw_beta" or name.startswith(direct_prefixes):
            if name not in target_state or target_state[name].shape != value.shape:
                raise ValueError(f"P2 initialization mismatch: {name}")
            target_state[name].copy_(value)
            copied.add(name)
    for modality in MODALITIES:
        for suffix in ("0.weight", "0.bias"):
            source_name = f"utility_nets.{modality}.{suffix}"
            target_state[source_name].copy_(source_state[source_name])
            copied.add(source_name)
        for suffix in ("weight", "bias"):
            source_name = f"utility_nets.{modality}.2.{suffix}"
            value = source_state[source_name]
            expanded = value.expand(target_state[source_name].shape).clone()
            target_state[source_name].copy_(expanded)
            copied.add(source_name)
    for suffix in ("weight", "bias"):
        source_name = f"classifier.0.{suffix}"
        target_name = f"classifier_hidden.0.{suffix}"
        target_state[target_name].copy_(source_state[source_name])
        copied.add(source_name)
    target_state["label_classifier_weight"].copy_(source_state["classifier.3.weight"])
    target_state["label_classifier_bias"].copy_(source_state["classifier.3.bias"])
    copied.update({"classifier.3.weight", "classifier.3.bias"})
    expected_source = set(source_state)
    if copied != expected_source:
        missing = sorted(expected_source - copied)
        raise ValueError(f"P2 state contains unmapped parameters: {missing}")
    target.load_state_dict(target_state, strict=True)
