"""Immutable train409-only configuration for Med-JEPA v4.3.1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4p3p1-training-config-v1"


@dataclass(frozen=True)
class V431TrainingConfig:
    seed: int
    hidden_channels: int
    conditioned_blocks: int
    maximum_cv_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mask_fractions: tuple[float, ...]
    grouped_cv_folds: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    evaluation_mask_draws_per_fraction: int
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: str | Path) -> "V431TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4.3.1 training config")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != allowed:
            raise ValueError("v4.3.1 config fields do not match the frozen schema")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        payload["mask_fractions"] = tuple(payload["mask_fractions"])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026:
            raise ValueError("v4.3.1 development is frozen to seed 2026")
        if self.hidden_channels != 128 or self.conditioned_blocks != 4:
            raise ValueError("v4.3.1 capacity changed")
        if self.maximum_cv_epochs != 100 or self.batch_size != 4:
            raise ValueError("v4.3.1 epoch or batch contract changed")
        if self.mask_fractions != (0.5, 0.625, 0.75, 0.875, 1.0):
            raise ValueError("v4.3.1 uniform mask schedule changed")
        if self.grouped_cv_folds != 5 or self.bootstrap_replicates != 10_000:
            raise ValueError("v4.3.1 grouped-CV/bootstrap contract changed")
        for name in (
            "maximum_cv_epochs",
            "batch_size",
            "grouped_cv_folds",
            "early_stopping_patience",
            "evaluation_mask_draws_per_fraction",
            "bootstrap_replicates",
            "bootstrap_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "early_stopping_min_delta",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")


__all__ = ["SCHEMA", "V431TrainingConfig"]

