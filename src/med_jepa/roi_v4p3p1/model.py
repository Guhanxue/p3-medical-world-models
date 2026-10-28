"""Coarse-RRR-anchored core-only residual predictor for Med-JEPA v4.3.1.

The externally fitted coarse RRR prediction is an exact, immutable anchor.
The neural branch can change that prediction only at tokens with non-zero
fractional T0 core occupancy.  Ring occupancy remains a context covariate but
never opens the residual gate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from med_jepa.roi_v4.model import _ConditionedBlock


def _coordinates(device: torch.device) -> Tensor:
    axis = torch.linspace(-1.0, 1.0, steps=7, device=device)
    return torch.stack(
        torch.meshgrid(axis, axis, axis, indexing="ij"), dim=0
    )[None]


@dataclass(frozen=True)
class CoreResidualOutput:
    predicted: Tensor
    coarse_rrr_anchor: Tensor
    raw_core_residual: Tensor
    gated_core_residual: Tensor
    core_gate: Tensor


class CoarseRRRAnchoredCoreResidualJEPA(nn.Module):
    """Predict only the nonlinear residual left inside the baseline T0 core."""

    def __init__(self, hidden_channels: int = 128, blocks: int = 4) -> None:
        super().__init__()
        if hidden_channels < 1 or blocks < 1:
            raise ValueError("unsupported v4.3.1 architecture")
        self.mask_token = nn.Parameter(torch.zeros(1, 1024, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)
        # Current state + previous change + core/ring occupancy + xyz.
        self.input_projection = nn.Conv3d(
            1024 * 2 + 2 + 3, hidden_channels, 1, bias=False
        )
        self.blocks = nn.ModuleList(
            _ConditionedBlock(hidden_channels, 7) for _ in range(blocks)
        )
        self.head = nn.Conv3d(hidden_channels, 1024, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _condition(
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        history_available: Tensor,
    ) -> Tensor:
        delta = target_time - source_time
        if (
            source_time.ndim != 1
            or target_time.shape != source_time.shape
            or action.shape != (source_time.shape[0], 2)
            or history_available.shape != source_time.shape
            or torch.any(delta <= 0)
            or torch.any((history_available < 0) | (history_available > 1))
        ):
            raise ValueError("temporal condition violates the v4.3.1 contract")
        scaled = torch.log1p(delta / 30.0)
        return torch.cat(
            (
                scaled[:, None],
                scaled.sin()[:, None],
                scaled.cos()[:, None],
                action,
                torch.log1p(source_time.clamp_min(0) / 30.0)[:, None],
                history_available[:, None],
            ),
            dim=1,
        )

    @staticmethod
    def _occupancy(core: Tensor, ring: Tensor) -> tuple[Tensor, Tensor]:
        if core.shape != ring.shape or core.ndim != 5:
            raise ValueError("core/ring occupancy tensors do not align")
        if core.shape[1:] != (1, 7, 7, 7):
            raise ValueError("occupancy must use the local64 7-cube grid")
        core = core.float()
        ring = ring.float()
        if (
            not torch.isfinite(core).all()
            or not torch.isfinite(ring).all()
            or torch.any((core < 0) | (core > 1))
            or torch.any((ring < 0) | (ring > 1))
            or torch.any(core + ring > 1.00001)
        ):
            raise ValueError("T0 fractional occupancy is invalid")
        return core, ring

    def step(
        self,
        source: Tensor,
        coarse_rrr_anchor: Tensor,
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        core_occupancy: Tensor,
        ring_occupancy: Tensor,
        *,
        history_delta: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> CoreResidualOutput:
        if (
            source.ndim != 5
            or source.shape[1:] != (1024, 7, 7, 7)
            or coarse_rrr_anchor.shape != source.shape
            or not torch.isfinite(source).all()
            or not torch.isfinite(coarse_rrr_anchor).all()
        ):
            raise ValueError("source/anchor violates the local64 token contract")
        batch = source.shape[0]
        if history_delta is None:
            history = torch.zeros_like(source)
            history_available = source_time.new_zeros(batch)
        else:
            if history_delta.shape != source.shape or not torch.isfinite(history_delta).all():
                raise ValueError("history delta does not align")
            history = history_delta
            history_available = source_time.new_ones(batch)
        if context_mask is None:
            mask = torch.zeros(
                batch, 1, 7, 7, 7, dtype=torch.bool, device=source.device
            )
        else:
            if context_mask.dtype != torch.bool or context_mask.shape != (
                batch, 1, 7, 7, 7
            ):
                raise ValueError("context mask violates the local64 grid")
            mask = context_mask
        core, ring = self._occupancy(core_occupancy, ring_occupancy)
        core = core.to(device=source.device, dtype=source.dtype)
        ring = ring.to(device=source.device, dtype=source.dtype)
        context = torch.where(mask, self.mask_token.to(source.dtype), source)
        history_context = torch.where(mask, torch.zeros_like(history), history)
        coordinates = _coordinates(source.device).to(source.dtype).expand(
            batch, -1, -1, -1, -1
        )
        condition = self._condition(
            source_time, target_time, action, history_available
        ).to(source.dtype)
        hidden = self.input_projection(
            torch.cat(
                (context, history_context, core, ring, coordinates), dim=1
            )
        )
        for block in self.blocks:
            hidden = block(hidden, condition)
        raw = self.head(hidden)
        scale = (
            torch.log1p((target_time - source_time) / 30.0) / math.log1p(6.0)
        ).clamp(max=3.0)[:, None, None, None, None]
        raw = scale * raw
        gated = raw * core
        return CoreResidualOutput(
            predicted=coarse_rrr_anchor + gated,
            coarse_rrr_anchor=coarse_rrr_anchor,
            raw_core_residual=raw,
            gated_core_residual=gated,
            core_gate=core,
        )


__all__ = ["CoreResidualOutput", "CoarseRRRAnchoredCoreResidualJEPA"]

