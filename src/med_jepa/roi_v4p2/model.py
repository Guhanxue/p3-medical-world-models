"""Shrinkage-anchored residual token dynamics for Med-JEPA v4.2."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from med_jepa.roi_v4.model import _ConditionedBlock


def _coordinates(grid_shape: tuple[int, int, int], device: torch.device) -> Tensor:
    if len(grid_shape) != 3 or any(value < 2 for value in grid_shape):
        raise ValueError("token grid must have three dimensions of size at least two")
    axes = [
        torch.linspace(-1.0, 1.0, steps=value, device=device)
        for value in grid_shape
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)[None]


class ShrinkageAnchoredTokenDynamicsJEPA(nn.Module):
    """Predict only the residual beyond a train-fold shrinkage anchor."""

    def __init__(self, hidden_channels: int = 128, blocks: int = 4) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1024, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.input = nn.Conv3d(1027, hidden_channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            _ConditionedBlock(hidden_channels, 6) for _ in range(blocks)
        )
        self.residual_velocity = nn.Conv3d(hidden_channels, 1024, 1)
        # Begin at the exact prespecified shrinkage baseline.  The residual can
        # move away from zero only when training evidence supports doing so.
        nn.init.zeros_(self.residual_velocity.weight)
        nn.init.zeros_(self.residual_velocity.bias)

    @staticmethod
    def _condition(
        source_time: Tensor, target_time: Tensor, action: Tensor
    ) -> Tensor:
        delta = target_time - source_time
        if torch.any(delta <= 0):
            raise ValueError("target time must be after source time")
        scaled = torch.log1p(delta / 30.0)
        return torch.cat(
            (
                scaled[:, None],
                scaled.sin()[:, None],
                scaled.cos()[:, None],
                action,
                torch.log1p(source_time.clamp_min(0) / 30.0)[:, None],
            ),
            dim=1,
        )

    @staticmethod
    def shrinkage_anchor(
        source: Tensor,
        population_target_mean: Tensor,
        alpha: float | Tensor,
    ) -> Tensor:
        if source.ndim != 5 or source.shape[1] != 1024:
            raise ValueError("source violates the frozen-token contract")
        if population_target_mean.ndim != 5 or population_target_mean.shape[1:] != source.shape[1:]:
            raise ValueError("population target mean does not match the source grid")
        if population_target_mean.shape[0] not in (1, source.shape[0]):
            raise ValueError("population target mean batch dimension is invalid")
        alpha_tensor = torch.as_tensor(alpha, dtype=source.dtype, device=source.device)
        if alpha_tensor.numel() != 1 or not bool((alpha_tensor >= 0) & (alpha_tensor <= 1)):
            raise ValueError("shrinkage alpha must be a scalar in [0,1]")
        return alpha_tensor * source + (1.0 - alpha_tensor) * population_target_mean

    def step(
        self,
        source: Tensor,
        population_target_mean: Tensor,
        alpha: float | Tensor,
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        anchor = self.shrinkage_anchor(source, population_target_mean, alpha)
        grid_shape = tuple(int(value) for value in source.shape[-3:])
        if context_mask is not None:
            if context_mask.dtype != torch.bool or context_mask.shape != (
                source.shape[0], 1, *grid_shape
            ):
                raise ValueError("context mask violates the token-grid contract")
            context = torch.where(
                context_mask, self.mask_token.to(source.dtype), source
            )
        else:
            context = source
        coordinates = _coordinates(grid_shape, source.device).to(source.dtype)
        coordinates = coordinates.expand(source.shape[0], -1, -1, -1, -1)
        hidden = self.input(torch.cat((context, coordinates), dim=1))
        condition = self._condition(source_time, target_time, action)
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale = (
            torch.log1p((target_time - source_time) / 30.0) / math.log1p(6.0)
        ).clamp(max=3.0)
        residual = scale[:, None, None, None, None] * self.residual_velocity(hidden)
        return anchor + residual

    def compose(
        self,
        source: Tensor,
        population_mean_b: Tensor,
        population_mean_c: Tensor,
        alpha_ab: float | Tensor,
        alpha_bc: float | Tensor,
        alpha_ac: float | Tensor,
        times_days: Tensor,
        action_ab: Tensor,
        action_bc: Tensor,
        action_ac: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if times_days.ndim != 2 or times_days.shape[1] != 3:
            raise ValueError("composition times must have shape [B,3]")
        t0, t1, t2 = times_days.unbind(1)
        direct = self.step(
            source, population_mean_c, alpha_ac, t0, t2, action_ac
        )
        middle = self.step(
            source, population_mean_b, alpha_ab, t0, t1, action_ab
        )
        composed = self.step(
            middle, population_mean_c, alpha_bc, t1, t2, action_bc
        )
        return direct, composed


__all__ = ["ShrinkageAnchoredTokenDynamicsJEPA"]
