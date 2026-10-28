"""Immutable Med-JEPA v4.3.2 recovery configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SCHEMA = "med-jepa-roi-v4p3p2-recovery-config-v1"


@dataclass(frozen=True)
class V432RecoveryConfig:
    seed: int
    final_epoch_count: int
    background_numerical_noninferiority_margin: float
    bootstrap_replicates: int
    bootstrap_seed: int

    @classmethod
    def load(cls, path: str | Path) -> "V432RecoveryConfig":
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA:
            raise ValueError("unsupported v4.3.2 recovery config")
        expected = {"schema_version", *cls.__dataclass_fields__}
        if set(value) != expected:
            raise ValueError("v4.3.2 recovery config fields changed")
        config = cls(
            **{name: value[name] for name in cls.__dataclass_fields__}
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.seed != 2026 or self.final_epoch_count != 18:
            raise ValueError("v4.3.2 seed/final epoch contract changed")
        if self.background_numerical_noninferiority_margin != 1e-6:
            raise ValueError("v4.3.2 numerical margin changed")
        if self.bootstrap_replicates != 10_000:
            raise ValueError("v4.3.2 bootstrap contract changed")
        if (
            isinstance(self.bootstrap_seed, bool)
            or not isinstance(self.bootstrap_seed, int)
            or self.bootstrap_seed < 1
        ):
            raise ValueError("v4.3.2 bootstrap seed is invalid")


__all__ = ["SCHEMA", "V432RecoveryConfig"]

