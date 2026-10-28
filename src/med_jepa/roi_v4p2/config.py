"""Immutable configuration for the first Med-JEPA v4.2 development run."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4p2-training-config-v1"


@dataclass(frozen=True)
class V42TrainingConfig:
    seed: int
    hidden_channels: int
    blocks: int
    maximum_cv_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mask_fractions: tuple[float, ...]
    latent_weight: float
    semigroup_weight: float
    occupancy_lambda_candidates: tuple[float, ...]
    grouped_cv_folds: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    evaluation_mask_draws_per_fraction: int
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: str | Path) -> "V42TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4.2 training config")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != allowed:
            raise ValueError("v4.2 config fields do not match the frozen schema")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        payload["mask_fractions"] = tuple(
            float(item) for item in payload["mask_fractions"]
        )
        payload["occupancy_lambda_candidates"] = tuple(
            float(item) for item in payload["occupancy_lambda_candidates"]
        )
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026:
            raise ValueError("v4.2 development is frozen to seed 2026")
        for name in (
            "hidden_channels",
            "blocks",
            "maximum_cv_epochs",
            "batch_size",
            "grouped_cv_folds",
            "early_stopping_patience",
            "evaluation_mask_draws_per_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "latent_weight",
            "semigroup_weight",
            "early_stopping_min_delta",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.hidden_channels != 128 or self.blocks != 4:
            raise ValueError("v4.2 reuses the frozen v4.1 architecture")
        if self.maximum_cv_epochs != 100:
            raise ValueError("v4.2 reuses the v4.1 100-epoch ceiling")
        if self.mask_fractions != (0.5, 0.625, 0.75, 0.875, 1.0):
            raise ValueError("v4.2 uniform mask schedule is frozen")
        if self.occupancy_lambda_candidates != (0.01, 0.05, 0.1, 0.25):
            raise ValueError("v4.2 occupancy lambda candidates are frozen")
        if self.grouped_cv_folds != 5:
            raise ValueError("v4.2 requires five train-only grouped folds")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("v4.2 requires 10,000 paired bootstrap replicates")


__all__ = ["SCHEMA", "V42TrainingConfig"]
