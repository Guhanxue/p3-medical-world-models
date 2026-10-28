"""Occupancy-aligned losses for Med-JEPA v4.2."""

from __future__ import annotations

import torch
from torch import Tensor
import torch.nn.functional as F


def occupancy_weighted_masked_latent_loss(
    prediction: Tensor,
    target: Tensor,
    supervised_mask: Tensor,
    t0_disease_occupancy: Tensor,
    *,
    uniform_floor_lambda: float,
) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share [B,C,D,H,W]")
    expected = (prediction.shape[0], 1, *prediction.shape[-3:])
    if supervised_mask.dtype != torch.bool or supervised_mask.shape != expected:
        raise ValueError("supervised mask does not match the token grid")
    if t0_disease_occupancy.shape != expected:
        raise ValueError("T0 disease occupancy does not match the token grid")
    if not 0.0 < float(uniform_floor_lambda) <= 1.0:
        raise ValueError("uniform occupancy floor must be in (0,1]")
    occupancy = t0_disease_occupancy.float()
    if torch.any((occupancy < 0) | (occupancy > 1)):
        raise ValueError("fractional disease occupancy must lie in [0,1]")
    token_error = F.smooth_l1_loss(
        prediction.float(), target.float(), reduction="none"
    ).mean(1, keepdim=True)
    weights = (occupancy + float(uniform_floor_lambda)) * supervised_mask.float()
    denominator = weights.flatten(1).sum(1)
    if torch.any(denominator <= 0):
        raise ValueError("every patient must have supervised weighted tokens")
    patient_loss = (token_error * weights).flatten(1).sum(1) / denominator
    return patient_loss.mean()


__all__ = ["occupancy_weighted_masked_latent_loss"]
