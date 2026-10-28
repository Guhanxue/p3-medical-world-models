"""Dual-path shrinkage-anchored token dynamics for Med-JEPA v4.3.

The global branch can emit only a trilinearly upsampled 2x2x2 residual.  The
disease branch can emit a local 7-cube residual, but its contribution is
multiplied by a T0-only continuous core/ring gate and is therefore exactly zero
outside the gate.  Both terminal heads start at zero, so the initial model is
bitwise equal to the train-fold shrinkage anchor.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from med_jepa.roi_v4.model import _ConditionedBlock


def _coordinates(grid_shape: tuple[int, int, int], device: torch.device) -> Tensor:
    if len(grid_shape) != 3 or any(value < 2 for value in grid_shape):
        raise ValueError("token grid must contain three non-trivial dimensions")
    axes = [
        torch.linspace(-1.0, 1.0, steps=value, device=device)
        for value in grid_shape
    ]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=0)[None]


@dataclass(frozen=True)
class DualPathOutput:
    predicted: Tensor
    shrinkage_anchor: Tensor
    global_coarse_residual: Tensor
    global_residual: Tensor
    raw_disease_residual: Tensor
    gated_disease_residual: Tensor
    disease_gate: Tensor


class DualPathShrinkageTokenDynamicsJEPA(nn.Module):
    """Separate low-frequency global nuisance and gated disease dynamics."""

    def __init__(
        self,
        hidden_channels: int = 128,
        blocks: int = 4,
        global_coarse_grid: tuple[int, int, int] = (2, 2, 2),
    ) -> None:
        super().__init__()
        if hidden_channels < 1 or blocks < 1 or global_coarse_grid != (2, 2, 2):
            raise ValueError("unsupported v4.3 architecture")
        self.hidden_channels = hidden_channels
        self.global_coarse_grid = global_coarse_grid
        self.mask_token = nn.Parameter(torch.zeros(1, 1024, 1, 1, 1))
        nn.init.normal_(self.mask_token, std=0.02)

        # Background-pooled current state + background-pooled prior delta +
        # seven temporal/treatment/history condition scalars.
        self.global_encoder = nn.Sequential(
            nn.Linear(1024 * 2 + 7, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
        )
        self.global_head = nn.Linear(
            hidden_channels,
            1024 * math.prod(global_coarse_grid),
        )

        # Current context + prior delta + core/ring fractional occupancy + xyz.
        self.disease_input = nn.Conv3d(1024 * 2 + 2 + 3, hidden_channels, 1, bias=False)
        self.disease_blocks = nn.ModuleList(
            _ConditionedBlock(hidden_channels, 7) for _ in range(blocks)
        )
        self.disease_head = nn.Conv3d(hidden_channels, 1024, 1)

        nn.init.zeros_(self.global_head.weight)
        nn.init.zeros_(self.global_head.bias)
        nn.init.zeros_(self.disease_head.weight)
        nn.init.zeros_(self.disease_head.bias)

    @staticmethod
    def shrinkage_anchor(
        source: Tensor,
        population_target_mean: Tensor,
        alpha: float | Tensor,
    ) -> Tensor:
        if source.ndim != 5 or source.shape[1:] != (1024, 7, 7, 7):
            raise ValueError("source violates the local64 frozen-token contract")
        if (
            population_target_mean.ndim != 5
            or population_target_mean.shape[1:] != source.shape[1:]
            or population_target_mean.shape[0] not in (1, source.shape[0])
        ):
            raise ValueError("population target mean does not align")
        value = torch.as_tensor(alpha, device=source.device, dtype=source.dtype)
        if value.numel() != 1 or not bool((value >= 0) & (value <= 1)):
            raise ValueError("shrinkage alpha must be a scalar in [0,1]")
        return value * source + (1.0 - value) * population_target_mean

    @staticmethod
    def continuous_disease_gate(
        core_occupancy: Tensor,
        ring_occupancy: Tensor,
        ring_gate_beta: float | Tensor,
    ) -> Tensor:
        if core_occupancy.shape != ring_occupancy.shape or core_occupancy.ndim != 5:
            raise ValueError("core/ring occupancy tensors do not align")
        if core_occupancy.shape[1:] != (1, 7, 7, 7):
            raise ValueError("occupancy must use the local64 7-cube grid")
        core = core_occupancy.float()
        ring = ring_occupancy.float()
        if (
            not torch.isfinite(core).all()
            or not torch.isfinite(ring).all()
            or torch.any((core < 0) | (core > 1))
            or torch.any((ring < 0) | (ring > 1))
            or torch.any(core + ring > 1.00001)
        ):
            raise ValueError("T0 core/ring fractional occupancy is invalid")
        beta = torch.as_tensor(ring_gate_beta, dtype=core.dtype, device=core.device)
        if beta.numel() != 1 or not bool((beta >= 0) & (beta <= 1)):
            raise ValueError("ring gate beta must be a scalar in [0,1]")
        return (core + beta * ring).clamp(0, 1)

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
            raise ValueError("temporal condition violates the v4.3 contract")
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
    def _weighted_pool(values: Tensor, weights: Tensor) -> Tensor:
        denominator = weights.flatten(1).sum(1).clamp_min(1e-6)
        return (
            (values.float() * weights.float()).flatten(2).sum(2)
            / denominator[:, None]
        )

    def step(
        self,
        source: Tensor,
        population_target_mean: Tensor,
        alpha: float | Tensor,
        source_time: Tensor,
        target_time: Tensor,
        action: Tensor,
        core_occupancy: Tensor,
        ring_occupancy: Tensor,
        *,
        ring_gate_beta: float | Tensor,
        valid_occupancy: Tensor | None = None,
        history_delta: Tensor | None = None,
        context_mask: Tensor | None = None,
    ) -> DualPathOutput:
        anchor = self.shrinkage_anchor(source, population_target_mean, alpha)
        batch = source.shape[0]
        if history_delta is None:
            history = torch.zeros_like(source)
            history_available = source_time.new_zeros(batch)
        else:
            if history_delta.shape != source.shape or not torch.isfinite(history_delta).all():
                raise ValueError("history delta does not align with source")
            history = history_delta
            history_available = source_time.new_ones(batch)
        if context_mask is None:
            mask = torch.zeros(
                (batch, 1, 7, 7, 7), dtype=torch.bool, device=source.device
            )
        else:
            if context_mask.dtype != torch.bool or context_mask.shape != (
                batch, 1, 7, 7, 7
            ):
                raise ValueError("context mask violates the local64 token grid")
            mask = context_mask
        context = torch.where(mask, self.mask_token.to(source.dtype), source)
        history_context = torch.where(mask, torch.zeros_like(history), history)
        core = core_occupancy.to(device=source.device, dtype=source.dtype)
        ring = ring_occupancy.to(device=source.device, dtype=source.dtype)
        gate = self.continuous_disease_gate(core, ring, ring_gate_beta).to(source.dtype)
        disease_bed = (core + ring).clamp(0, 1)
        if valid_occupancy is None:
            valid = torch.ones_like(disease_bed)
        else:
            if (
                valid_occupancy.shape != disease_bed.shape
                or not torch.isfinite(valid_occupancy).all()
                or torch.any((valid_occupancy < 0) | (valid_occupancy > 1))
                or torch.any(disease_bed > valid_occupancy + 1e-5)
            ):
                raise ValueError("T0 valid occupancy does not align")
            valid = valid_occupancy.to(device=source.device, dtype=source.dtype)
        background = (valid - disease_bed).clamp(0, 1)
        condition = self._condition(
            source_time, target_time, action, history_available
        ).to(source.dtype)

        global_state = torch.cat(
            (
                self._weighted_pool(context, background),
                self._weighted_pool(history_context, background),
                condition,
            ),
            dim=1,
        )
        global_hidden = self.global_encoder(global_state)
        coarse = self.global_head(global_hidden).reshape(
            batch, 1024, *self.global_coarse_grid
        )
        global_residual = F.interpolate(
            coarse.float(), size=(7, 7, 7), mode="trilinear", align_corners=False
        ).to(source.dtype)

        coordinates = _coordinates((7, 7, 7), source.device).to(source.dtype)
        coordinates = coordinates.expand(batch, -1, -1, -1, -1)
        disease_hidden = self.disease_input(
            torch.cat((context, history_context, core, ring, coordinates), dim=1)
        )
        for block in self.disease_blocks:
            disease_hidden = block(disease_hidden, condition)
        raw_disease = self.disease_head(disease_hidden)
        gated_disease = raw_disease * gate

        scale = (
            torch.log1p((target_time - source_time) / 30.0) / math.log1p(6.0)
        ).clamp(max=3.0)
        scale = scale[:, None, None, None, None]
        global_residual = scale * global_residual
        raw_disease = scale * raw_disease
        gated_disease = scale * gated_disease
        predicted = anchor + global_residual + gated_disease
        return DualPathOutput(
            predicted=predicted,
            shrinkage_anchor=anchor,
            global_coarse_residual=scale * coarse,
            global_residual=global_residual,
            raw_disease_residual=raw_disease,
            gated_disease_residual=gated_disease,
            disease_gate=gate,
        )


__all__ = ["DualPathOutput", "DualPathShrinkageTokenDynamicsJEPA"]
