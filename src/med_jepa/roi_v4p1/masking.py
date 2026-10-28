"""Deterministic input-mask schedule for source-anchored v4.1 dynamics."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def block_mask(
    batch_size: int,
    *,
    seed: int,
    masked_fraction: float,
) -> Tensor:
    """Draw a block mask, explicitly supporting both 0% and 100% masking."""

    if batch_size < 1 or not 0.0 <= masked_fraction <= 1.0:
        raise ValueError("invalid v4.1 block-mask request")
    if masked_fraction == 0.0:
        return torch.zeros(batch_size, 1, 7, 7, 7, dtype=torch.bool)
    if masked_fraction == 1.0:
        return torch.ones(batch_size, 1, 7, 7, 7, dtype=torch.bool)
    generator = torch.Generator().manual_seed(int(seed))
    mask = torch.zeros(batch_size, 1, 7, 7, 7, dtype=torch.bool)
    target = max(1, min(342, round(343 * masked_fraction)))
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            size = int(torch.randint(2, 5, (1,), generator=generator))
            starts = [
                int(torch.randint(0, 8 - size, (1,), generator=generator))
                for _ in range(3)
            ]
            z, y, x = starts
            mask[row, :, z : z + size, y : y + size, x : x + size] = True
        if int(mask[row].sum()) == 343:
            mask[row, :, 0, 0, 0] = False
    return mask


def scheduled_masks(
    batch_size: int,
    *,
    seed: int,
    fractions: Sequence[float],
) -> tuple[Tensor, Tensor, float]:
    """Return an input mask and the corresponding supervised-token mask.

    A zero input-mask fraction represents the deployment condition (the full
    source is visible), so all target positions are supervised.  For non-zero
    fractions, supervision remains confined to hidden context positions.
    """

    if not fractions:
        raise ValueError("v4.1 mask schedule cannot be empty")
    normalized = tuple(float(value) for value in fractions)
    if any(not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("v4.1 mask fractions must be in [0,1]")
    generator = torch.Generator().manual_seed(int(seed))
    choice = int(torch.randint(0, len(normalized), (1,), generator=generator))
    fraction = normalized[choice]
    context = block_mask(batch_size, seed=seed + 17, masked_fraction=fraction)
    supervised = torch.ones_like(context) if fraction == 0.0 else context.clone()
    return context, supervised, fraction


__all__ = ["block_mask", "scheduled_masks"]

