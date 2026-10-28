"""Masked fractional-core objective for Med-JEPA v4.3.1."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from med_jepa.roi_v4p3p1.model import CoreResidualOutput


def masked_fractional_core_objective(
    output: CoreResidualOutput,
    target: Tensor,
    supervised_mask: Tensor,
    core_occupancy: Tensor,
) -> Mapping[str, Tensor]:
    if (
        output.predicted.shape != target.shape
        or output.predicted.ndim != 5
        or supervised_mask.dtype != torch.bool
        or supervised_mask.shape
        != (target.shape[0], 1, *target.shape[-3:])
        or core_occupancy.shape != supervised_mask.shape
        or not torch.isfinite(target).all()
        or not torch.isfinite(core_occupancy).all()
        or torch.any((core_occupancy < 0) | (core_occupancy > 1))
    ):
        raise ValueError("v4.3.1 core objective inputs do not align")
    weights = supervised_mask.float() * core_occupancy.float()
    denominator = weights.flatten(1).sum(1)
    eligible = denominator > 0
    if not bool(eligible.any()):
        # Uniform masks are not changed to chase the tumour.  A rare batch with
        # no masked core token contributes a differentiable zero instead.
        zero = output.predicted.float().sum() * 0.0
        return {"total": zero, "masked_core": zero, "eligible_patients": zero}
    error = F.smooth_l1_loss(
        output.predicted.float(), target.float(), reduction="none"
    ).mean(1, keepdim=True)
    patient = (
        (error[eligible] * weights[eligible]).flatten(1).sum(1)
        / denominator[eligible]
    )
    core = patient.mean()
    return {
        "total": core,
        "masked_core": core,
        "eligible_patients": core.new_tensor(int(eligible.sum())),
    }


__all__ = ["masked_fractional_core_objective"]
