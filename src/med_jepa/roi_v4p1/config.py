"""Immutable configuration contract for the single-seed v4.1 experiment."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4p1-training-config-v1"


@dataclass(frozen=True)
class V41TrainingConfig:
    seed: int
    hidden_channels: int
    blocks: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    mask_fractions: tuple[float, ...]
    latent_weight: float
    semigroup_weight: float
    variance_weight: float
    variance_margin: float
    early_stopping_patience: int
    early_stopping_min_delta: float
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: str | Path) -> "V41TrainingConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4.1 training config")
        allowed = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != allowed:
            raise ValueError("v4.1 config fields do not match the frozen schema")
        payload = {name: value[name] for name in cls.__dataclass_fields__}
        payload["mask_fractions"] = tuple(float(x) for x in payload["mask_fractions"])
        config = cls(**payload)
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026:
            raise ValueError("v4.1 development is frozen to seed 2026")
        for name in (
            "hidden_channels",
            "blocks",
            "epochs",
            "batch_size",
            "early_stopping_patience",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "latent_weight",
            "semigroup_weight",
            "variance_weight",
            "variance_margin",
            "early_stopping_min_delta",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.mask_fractions != (0.0, 0.5, 0.75, 0.9, 1.0):
            raise ValueError("v4.1 mask schedule is frozen to 0,.5,.75,.9,1")
        if self.variance_weight != 0.0:
            raise ValueError("v4.1 forbids variance/rank matching pressure")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("v4.1 requires 10,000 paired bootstrap replicates")


__all__ = ["SCHEMA", "V41TrainingConfig"]
