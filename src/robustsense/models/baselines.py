"""Dependency-light Phase-0 baseline model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class LinearBaseline:
    weights: np.ndarray
    bias: np.ndarray

    @classmethod
    def initialize(cls, input_dim: int, output_dim: int, seed: int) -> LinearBaseline:
        rng = np.random.default_rng(seed)
        weights = rng.normal(scale=0.02, size=(input_dim, output_dim)).astype(np.float64)
        return cls(weights=weights, bias=np.zeros(output_dim, dtype=np.float64))

    def logits(self, features: np.ndarray) -> np.ndarray:
        return features @ self.weights + self.bias

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, weights=self.weights, bias=self.bias)

    @classmethod
    def load(cls, path: str | Path) -> LinearBaseline:
        with np.load(path) as checkpoint:
            return cls(weights=checkpoint["weights"], bias=checkpoint["bias"])
