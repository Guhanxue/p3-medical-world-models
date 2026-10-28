"""Grid-agnostic uniform block masking for Med-JEPA v4.2."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor


def uniform_block_mask(
    batch_size: int,
    grid_shape: tuple[int, int, int],
    *,
    seed: int,
    masked_fraction: float,
) -> Tensor:
    if (
        batch_size < 1
        or len(grid_shape) != 3
        or any(value < 2 for value in grid_shape)
        or not 0.0 <= masked_fraction <= 1.0
    ):
        raise ValueError("invalid uniform block-mask request")
    mask = torch.zeros(batch_size, 1, *grid_shape, dtype=torch.bool)
    if masked_fraction == 0.0:
        return mask
    if masked_fraction == 1.0:
        return torch.ones_like(mask)
    total = grid_shape[0] * grid_shape[1] * grid_shape[2]
    target = max(1, min(total - 1, round(total * masked_fraction)))
    generator = torch.Generator().manual_seed(int(seed))
    maximum_block = max(2, min(grid_shape) // 2)
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            sizes = [
                int(torch.randint(2, maximum_block + 1, (1,), generator=generator))
                for _ in grid_shape
            ]
            starts = [
                int(torch.randint(0, extent - size + 1, (1,), generator=generator))
                for extent, size in zip(grid_shape, sizes, strict=True)
            ]
            z, y, x = starts
            dz, dy, dx = sizes
            mask[row, :, z : z + dz, y : y + dy, x : x + dx] = True
        if int(mask[row].sum()) == total:
            mask[row, :, 0, 0, 0] = False
    return mask


def scheduled_uniform_masks(
    batch_size: int,
    grid_shape: tuple[int, int, int],
    *,
    seed: int,
    fractions: Sequence[float],
) -> tuple[Tensor, Tensor, float]:
    normalized = tuple(float(value) for value in fractions)
    if not normalized or any(not 0.0 <= value <= 1.0 for value in normalized):
        raise ValueError("mask fractions must be a non-empty sequence in [0,1]")
    generator = torch.Generator().manual_seed(int(seed))
    fraction = normalized[
        int(torch.randint(0, len(normalized), (1,), generator=generator))
    ]
    context = uniform_block_mask(
        batch_size,
        grid_shape,
        seed=seed + 17,
        masked_fraction=fraction,
    )
    supervised = torch.ones_like(context) if fraction == 0.0 else context.clone()
    return context, supervised, fraction


__all__ = ["scheduled_uniform_masks", "uniform_block_mask"]

