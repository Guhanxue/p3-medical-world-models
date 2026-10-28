"""Deterministic block masking for a 7x7x7 frozen-token grid."""

from __future__ import annotations

import torch
from torch import Tensor


def block_mask(batch_size: int, *, seed: int, masked_fraction: float = 0.5) -> Tensor:
    if batch_size < 1 or not 0.1 <= masked_fraction <= 0.9:
        raise ValueError("invalid block-mask request")
    generator = torch.Generator().manual_seed(int(seed))
    mask = torch.zeros(batch_size, 1, 7, 7, 7, dtype=torch.bool)
    target = max(1, min(342, round(343 * masked_fraction)))
    for row in range(batch_size):
        while int(mask[row].sum()) < target:
            size = int(torch.randint(2, 5, (1,), generator=generator))
            starts = [int(torch.randint(0, 8 - size, (1,), generator=generator)) for _ in range(3)]
            z, y, x = starts
            mask[row, :, z : z + size, y : y + size, x : x + size] = True
        if int(mask[row].sum()) == 343:
            mask[row, :, 0, 0, 0] = False
    return mask


__all__ = ["block_mask"]
