"""Non-pixel token dynamics for Med-JEPA v4.

The model predicts immutable 3DINO feature tokens.  It contains no image
decoder and cannot emit a future MRI volume.  Fixed coordinates condition the
velocity network but are never added to the state.  With zero velocity the
prediction is exactly the source state; no terminal normalization changes it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from med_jepa.roi_v4.contracts import CompositionBatch, TransitionBatch


def _coordinates() -> Tensor:
    axis = torch.linspace(-1.0, 1.0, 7)
    return torch.stack(torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0)[None]


class _ConditionedBlock(nn.Module):
    def __init__(self, channels: int, condition_dim: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv3d(channels, channels, 3, padding=1, groups=channels)
        self.norm = nn.GroupNorm(1, channels)
        self.condition = nn.Linear(condition_dim, channels * 2)
        self.pointwise = nn.Sequential(
            nn.Conv3d(channels, channels * 2, 1), nn.GELU(), nn.Conv3d(channels * 2, channels, 1)
        )

    def forward(self, values: Tensor, condition: Tensor) -> Tensor:
        residual = values
        values = self.norm(self.depthwise(values))
        scale, shift = self.condition(condition).chunk(2, dim=1)
        values = values * (1.0 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        return residual + self.pointwise(values)


@dataclass(frozen=True)
class TransitionOutput:
    predicted: Tensor
    target: Tensor
    source_persistence: Tensor
    mask: Tensor


class TokenDynamicsJEPA(nn.Module):
    """Masked residual predictor in the frozen [1024,7,7,7] token space."""

    def __init__(self, hidden_channels: int = 256, blocks: int = 4) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1024, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("fixed_coordinates", _coordinates(), persistent=True)
        self.input = nn.Conv3d(1027, hidden_channels, 1, bias=False)
        self.blocks = nn.ModuleList(_ConditionedBlock(hidden_channels, 6) for _ in range(blocks))
        self.velocity = nn.Conv3d(hidden_channels, 1024, 1)

    @staticmethod
    def _condition(source_time: Tensor, target_time: Tensor, action: Tensor) -> Tensor:
        delta = target_time - source_time
        if torch.any(delta <= 0):
            raise ValueError("target time must be after source time")
        scaled = torch.log1p(delta / 30.0)
        return torch.cat((scaled[:, None], scaled.sin()[:, None], scaled.cos()[:, None], action, torch.log1p(source_time.clamp_min(0) / 30)[:, None]), dim=1)

    def step(
        self,
        source: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        mask: Tensor | None = None,
    ) -> Tensor:
        if source.shape[1:] != (1024, 7, 7, 7):
            raise ValueError("source violates the frozen-token contract")
        base = source if mask is None else torch.where(mask, self.mask_token.to(source.dtype), source)
        coordinates = self.fixed_coordinates.to(source.dtype).expand(source.shape[0], -1, -1, -1, -1)
        hidden = self.input(torch.cat((base, coordinates), dim=1))
        condition = self._condition(source_time, target_time, action)
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale = (torch.log1p((target_time - source_time) / 30.0) / math.log1p(6.0)).clamp(max=3.0)
        return base + scale[:, None, None, None, None] * self.velocity(hidden)

    def forward(self, batch: TransitionBatch) -> TransitionOutput:
        batch.validate()
        return TransitionOutput(
            predicted=self.step(batch.source, batch.source_time_days, batch.target_time_days, batch.action, batch.prediction_mask),
            target=batch.target.detach(),
            source_persistence=batch.source.detach(),
            mask=batch.prediction_mask,
        )

    def compose(self, batch: CompositionBatch) -> tuple[Tensor, Tensor]:
        batch.validate()
        t0, t1, t2 = batch.times_days.unbind(1)
        direct = self.step(batch.a, t0, t2, batch.action_ac)
        middle = self.step(batch.a, t0, t1, batch.action_ab)
        composed = self.step(middle, t1, t2, batch.action_bc)
        return direct, composed


class DetachedFTVProbe(nn.Module):
    """Evaluation-only probe; detach is enforced inside the module."""

    def __init__(self) -> None:
        super().__init__()
        self.readout = nn.Linear(1024 * 27, 1)

    def forward(self, features: Tensor) -> Tensor:
        pooled = F.adaptive_avg_pool3d(features.detach().float(), (3, 3, 3)).flatten(1)
        return self.readout(pooled).squeeze(1)


__all__ = ["DetachedFTVProbe", "TokenDynamicsJEPA", "TransitionOutput"]
