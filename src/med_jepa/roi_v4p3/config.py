"""Immutable train409-only development configuration for Med-JEPA v4.3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4p3-training-config-v1"


@dataclass(frozen=True)
class V43TrainingConfig:
    seed: int
    hidden_channels: int
    conditioned_blocks: int
    global_coarse_grid: tuple[int, int, int]
    maximum_cv_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mask_fractions: tuple[float, ...]
    ring_gate_beta_candidates: tuple[float, ...]
    outside_energy_weight_candidates: tuple[float, ...]
    global_background_weight: float
    disease_weight: float
    full_grid_safety_weight: float
    decorrelation_weight: float
    semigroup_weight: float
    grouped_cv_folds: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    evaluation_mask_draws_per_fraction: int
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: str | Path) -> "V43TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4.3 training config")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != allowed:
            raise ValueError("v4.3 config fields do not match the frozen schema")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        for name in (
            "global_coarse_grid",
            "mask_fractions",
            "ring_gate_beta_candidates",
            "outside_energy_weight_candidates",
        ):
            payload[name] = tuple(payload[name])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026:
            raise ValueError("v4.3 development is frozen to seed 2026")
        if self.hidden_channels != 128 or self.conditioned_blocks != 4:
            raise ValueError("v4.3 capacity is frozen before training")
        if self.global_coarse_grid != (2, 2, 2):
            raise ValueError("v4.3 global branch must use a fixed 2-cube bottleneck")
        if self.maximum_cv_epochs != 100 or self.batch_size != 4:
            raise ValueError("v4.3 epoch and batch contracts changed")
        if self.mask_fractions != (0.5, 0.625, 0.75, 0.875, 1.0):
            raise ValueError("v4.3 uniform mask schedule changed")
        if self.ring_gate_beta_candidates != (0.25, 0.5, 0.75):
            raise ValueError("v4.3 ring-gate candidates changed")
        if self.outside_energy_weight_candidates != (0.01, 0.05):
            raise ValueError("v4.3 outside-energy candidates changed")
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
        if self.grouped_cv_folds != 5 or self.bootstrap_replicates != 10_000:
            raise ValueError("v4.3 grouped-CV/bootstrap contract changed")
        for name in (
            "learning_rate",
            "weight_decay",
            "global_background_weight",
            "disease_weight",
            "full_grid_safety_weight",
            "decorrelation_weight",
            "semigroup_weight",
            "early_stopping_min_delta",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if (
            self.global_background_weight != 1.0
            or self.disease_weight != 1.0
            or self.full_grid_safety_weight != 0.25
            or self.decorrelation_weight != 0.01
            or self.semigroup_weight != 0.25
        ):
            raise ValueError("v4.3 fixed loss weights changed")


__all__ = ["SCHEMA", "V43TrainingConfig"]
