"""Source-anchored masked token dynamics for Med-JEPA v4.1.

The masking operation is confined to the velocity-network context.  The state
update is always residual around the observed source token grid, including at
masked locations.  Consequently zero velocity is exact latent persistence for
every mask, not only for an unmasked forward pass.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from med_jepa.roi_v4.model import _ConditionedBlock, _coordinates


class SourceAnchoredTokenDynamicsJEPA(nn.Module):
    """Predict future frozen 3DINO tokens with an exact source residual base."""

    def __init__(self, hidden_channels: int = 256, blocks: int = 4) -> None:
        super().__init__()
        self.mask_token = nn.Parameter(torch.zeros(1, 1024, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("fixed_coordinates", _coordinates(), persistent=True)
        self.input = nn.Conv3d(1027, hidden_channels, 1, bias=False)
        self.blocks = nn.ModuleList(
            _ConditionedBlock(hidden_channels, 6) for _ in range(blocks)
        )
        self.velocity = nn.Conv3d(hidden_channels, 1024, 1)

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

    def step(
        self,
        source: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        context_mask: Tensor | None = None,
    ) -> Tensor:
        if source.shape[1:] != (1024, 7, 7, 7):
            raise ValueError("source violates the frozen-token contract")
        if context_mask is not None:
            if context_mask.dtype != torch.bool or context_mask.shape != (
                source.shape[0],
                1,
                7,
                7,
                7,
            ):
                raise ValueError("context mask violates the token-grid contract")
            context = torch.where(
                context_mask, self.mask_token.to(source.dtype), source
            )
        else:
            context = source
        coordinates = self.fixed_coordinates.to(source.dtype).expand(
            source.shape[0], -1, -1, -1, -1
        )
        hidden = self.input(torch.cat((context, coordinates), dim=1))
        condition = self._condition(source_time, target_time, action)
        for block in self.blocks:
            hidden = block(hidden, condition)
        scale = (
            torch.log1p((target_time - source_time) / 30.0) / math.log1p(6.0)
        ).clamp(max=3.0)
        return source + scale[:, None, None, None, None] * self.velocity(hidden)

    def compose(
        self,
        a: Tensor,
        times_days: Tensor,
        action_ab: Tensor,
        action_bc: Tensor,
        action_ac: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if times_days.ndim != 2 or times_days.shape[1] != 3:
            raise ValueError("composition times must have shape [B,3]")
        t0, t1, t2 = times_days.unbind(1)
        direct = self.step(a, t0, t2, action_ac)
        middle = self.step(a, t0, t1, action_ab)
        composed = self.step(middle, t1, t2, action_bc)
        return direct, composed


__all__ = ["SourceAnchoredTokenDynamicsJEPA"]

