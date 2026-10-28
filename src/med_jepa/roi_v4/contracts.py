"""Strict train/validation contracts for token-level Med-JEPA v4.

FTV and outcome labels are deliberately absent from the model-facing batches.
They can therefore be used only by detached probes and downstream evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import torch
from torch import Tensor


VISIT_ORDER = ("T0", "T1", "T2", "T3")
CURRENT_RELEASE_VISITS = ("T0", "T1", "T3")
FEATURE_SHAPE = (1024, 7, 7, 7)
ACTION_DIM = 2


def ordered_pairs_and_triplets(
    visits: Sequence[str],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]]:
    normalized = tuple(str(value).strip().upper() for value in visits)
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("visits must be non-empty and unique")
    try:
        ranks = tuple(VISIT_ORDER.index(value) for value in normalized)
    except ValueError as error:
        raise ValueError("unsupported visit label") from error
    if tuple(sorted(ranks)) != ranks:
        raise ValueError("visits must be chronologically ordered")
    return tuple(combinations(range(len(ranks)), 2)), tuple(
        combinations(range(len(ranks)), 3)
    )


def _finite(name: str, value: Tensor) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")


@dataclass(frozen=True)
class TransitionBatch:
    source: Tensor
    target: Tensor
    source_time_days: Tensor
    target_time_days: Tensor
    action: Tensor
    prediction_mask: Tensor

    def validate(self) -> None:
        if self.source.ndim != 5:
            raise ValueError("source must be [B,1024,7,7,7]")
        batch = self.source.shape[0]
        expected = (batch, *FEATURE_SHAPE)
        if tuple(self.source.shape) != expected or tuple(self.target.shape) != expected:
            raise ValueError("transition tensors violate the frozen 3DINO grid")
        for name in ("source", "target", "source_time_days", "target_time_days", "action"):
            _finite(name, getattr(self, name))
        if self.source_time_days.shape != (batch,) or self.target_time_days.shape != (batch,):
            raise ValueError("transition times must be [B]")
        if torch.any(self.target_time_days <= self.source_time_days):
            raise ValueError("every transition must move forward in time")
        if self.action.shape != (batch, ACTION_DIM):
            raise ValueError("action must be [B,2]")
        if self.prediction_mask.dtype != torch.bool or self.prediction_mask.shape != (
            batch, 1, 7, 7, 7
        ):
            raise ValueError("prediction_mask must be boolean [B,1,7,7,7]")
        counts = self.prediction_mask.flatten(1).sum(1)
        if torch.any(counts == 0) or torch.any(counts == 343):
            raise ValueError("each sample needs visible and masked tokens")


@dataclass(frozen=True)
class CompositionBatch:
    a: Tensor
    c: Tensor
    times_days: Tensor
    action_ab: Tensor
    action_bc: Tensor
    action_ac: Tensor
    consistency_mask: Tensor

    def validate(self) -> None:
        if self.a.ndim != 5:
            raise ValueError("composition states must be [B,1024,7,7,7]")
        batch = self.a.shape[0]
        expected = (batch, *FEATURE_SHAPE)
        if self.a.shape != expected or self.c.shape != expected:
            raise ValueError("composition tensors violate the frozen grid")
        if self.times_days.shape != (batch, 3):
            raise ValueError("times_days must be [B,3]")
        if torch.any(self.times_days[:, 0] >= self.times_days[:, 1]) or torch.any(
            self.times_days[:, 1] >= self.times_days[:, 2]
        ):
            raise ValueError("composition times must satisfy a<b<c")
        for value in (self.action_ab, self.action_bc, self.action_ac):
            if value.shape != (batch, ACTION_DIM):
                raise ValueError("composition actions must be [B,2]")
        if self.consistency_mask.dtype != torch.bool or self.consistency_mask.shape != (
            batch, 1, 7, 7, 7
        ):
            raise ValueError("consistency_mask must be boolean [B,1,7,7,7]")


__all__ = [
    "ACTION_DIM",
    "CURRENT_RELEASE_VISITS",
    "CompositionBatch",
    "FEATURE_SHAPE",
    "TransitionBatch",
    "VISIT_ORDER",
    "ordered_pairs_and_triplets",
]
