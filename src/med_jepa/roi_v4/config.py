"""Strict configuration for the first development-only Med-JEPA v4 run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4-training-config-v1"


@dataclass(frozen=True)
class V4TrainingConfig:
    seed: int
    hidden_channels: int
    blocks: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    masked_fraction: float
    latent_weight: float
    semigroup_weight: float
    variance_weight: float
    variance_margin: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    bootstrap_replicates: int
    bootstrap_seed: int
    minimum_effective_rank_ratio: float

    @classmethod
    def load(cls, path: str | Path) -> "V4TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4 training config")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != allowed:
            raise ValueError("v4 training config fields do not match the frozen schema")
        config = cls(**{name: value[name] for name in cls.__dataclass_fields__})
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026:
            raise ValueError("the first v4 development run is frozen to seed 2026")
        for name in ("hidden_channels", "blocks", "epochs", "batch_size", "early_stopping_patience"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate", "weight_decay", "latent_weight", "semigroup_weight",
            "variance_weight", "variance_margin", "early_stopping_min_delta",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if not 0.1 <= self.masked_fraction <= 0.9:
            raise ValueError("masked_fraction must be in [0.1,0.9]")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("v4 gate requires 10,000 bootstrap replicates")
        if not 0 < self.minimum_effective_rank_ratio <= 1:
            raise ValueError("minimum_effective_rank_ratio must be in (0,1]")


__all__ = ["SCHEMA", "V4TrainingConfig"]
