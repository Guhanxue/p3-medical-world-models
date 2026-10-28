"""Masked latent and semigroup objectives; no pixel or FTV training loss."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def _masked(values: Tensor, mask: Tensor) -> Tensor:
    return values.masked_select(mask.expand_as(values))


def masked_latent_loss(predicted: Tensor, target: Tensor, mask: Tensor) -> Tensor:
    return F.smooth_l1_loss(_masked(predicted, mask), _masked(target.detach(), mask))


def semigroup_loss(direct: Tensor, composed: Tensor, mask: Tensor) -> Tensor:
    return F.smooth_l1_loss(_masked(direct, mask), _masked(composed, mask))


def token_variance_loss(values: Tensor, margin: float = 1.0) -> Tensor:
    tokens = values.permute(0, 2, 3, 4, 1).reshape(-1, values.shape[1])
    return torch.relu(tokens.new_tensor(margin) - torch.sqrt(tokens.var(0, unbiased=False) + 1e-4)).mean()


__all__ = ["masked_latent_loss", "semigroup_loss", "token_variance_loss"]
