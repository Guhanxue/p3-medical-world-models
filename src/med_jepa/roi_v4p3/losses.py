"""Role-separated train409 losses for Med-JEPA v4.3."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import Tensor
import torch.nn.functional as F

from med_jepa.roi_v4p3.model import DualPathOutput


def _validate_spatial_inputs(
    output: DualPathOutput,
    target: Tensor,
    supervised_mask: Tensor,
    core_occupancy: Tensor,
    ring_occupancy: Tensor,
) -> None:
    expected = output.predicted.shape
    spatial = (expected[0], 1, *expected[-3:])
    if expected != target.shape or target.ndim != 5:
        raise ValueError("target does not align with the dual-path prediction")
    if supervised_mask.dtype != torch.bool or supervised_mask.shape != spatial:
        raise ValueError("supervised mask does not align")
    if core_occupancy.shape != spatial or ring_occupancy.shape != spatial:
        raise ValueError("T0 occupancy does not align")
    for value in (target, core_occupancy, ring_occupancy):
        if not torch.isfinite(value).all():
            raise ValueError("dual-path loss input is non-finite")
    if (
        torch.any((core_occupancy < 0) | (core_occupancy > 1))
        or torch.any((ring_occupancy < 0) | (ring_occupancy > 1))
        or torch.any(core_occupancy + ring_occupancy > 1.00001)
    ):
        raise ValueError("core/ring fractional occupancy is invalid")


def _patient_weighted_mean(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.flatten(1).sum(1)
    eligible = denominator > 0
    if int(eligible.sum()) == 0:
        raise ValueError("loss term has no eligible patients")
    patient = (
        (values[eligible] * weights[eligible]).flatten(1).sum(1)
        / denominator[eligible]
    )
    return patient.mean()


def branch_decorrelation_loss(global_residual: Tensor, disease_residual: Tensor) -> Tensor:
    if global_residual.shape != disease_residual.shape or global_residual.ndim != 5:
        raise ValueError("branch residuals do not align")
    left = global_residual.float().flatten(1)
    right = disease_residual.float().flatten(1)
    numerator = (left * right).sum(1)
    denominator = left.norm(dim=1) * right.norm(dim=1)
    cosine = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1e-12),
        torch.zeros_like(numerator),
    )
    return cosine.square().mean()


def semigroup_consistency_loss(
    direct_prediction: Tensor,
    composed_prediction: Tensor,
    supervised_mask: Tensor,
) -> Tensor:
    """Match T0->T3 with the composed T0->T1->T3 transition.

    This term uses no T3 pixel reconstruction and no clinical target.  Both
    arguments are frozen-token state predictions made with the same fold-bound
    anchors and treatment/time information.
    """

    if (
        direct_prediction.shape != composed_prediction.shape
        or direct_prediction.ndim != 5
        or supervised_mask.dtype != torch.bool
        or supervised_mask.shape
        != (direct_prediction.shape[0], 1, *direct_prediction.shape[-3:])
        or not torch.isfinite(direct_prediction).all()
        or not torch.isfinite(composed_prediction).all()
    ):
        raise ValueError("semigroup consistency inputs do not align")
    error = F.smooth_l1_loss(
        direct_prediction.float(), composed_prediction.float(), reduction="none"
    ).mean(1, keepdim=True)
    return _patient_weighted_mean(error, supervised_mask.float())


def dual_path_objective(
    output: DualPathOutput,
    target: Tensor,
    supervised_mask: Tensor,
    core_occupancy: Tensor,
    ring_occupancy: Tensor,
    *,
    ring_gate_beta: float,
    outside_energy_weight: float,
    global_background_weight: float = 1.0,
    disease_weight: float = 1.0,
    full_grid_safety_weight: float = 0.25,
    decorrelation_weight: float = 0.01,
) -> Mapping[str, Tensor]:
    """Compute losses without letting background dominate the disease branch."""

    _validate_spatial_inputs(
        output, target, supervised_mask, core_occupancy, ring_occupancy
    )
    for name, value in (
        ("ring_gate_beta", ring_gate_beta),
        ("outside_energy_weight", outside_energy_weight),
        ("global_background_weight", global_background_weight),
        ("disease_weight", disease_weight),
        ("full_grid_safety_weight", full_grid_safety_weight),
        ("decorrelation_weight", decorrelation_weight),
    ):
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must lie in [0,1]")
    core = core_occupancy.float()
    ring = ring_occupancy.float()
    disease_bed = (core + ring).clamp(0, 1)
    disease_weight_map = (core + float(ring_gate_beta) * ring).clamp(0, 1)
    background = (1.0 - disease_bed).clamp(0, 1)
    mask = supervised_mask.float()

    global_prediction = output.shrinkage_anchor + output.global_residual
    global_error = F.smooth_l1_loss(
        global_prediction.float(), target.float(), reduction="none"
    ).mean(1, keepdim=True)
    full_error = F.smooth_l1_loss(
        output.predicted.float(), target.float(), reduction="none"
    ).mean(1, keepdim=True)
    global_background = _patient_weighted_mean(
        global_error, background * mask
    )
    disease = _patient_weighted_mean(
        full_error, disease_weight_map * mask
    )
    full_grid = _patient_weighted_mean(full_error, mask)
    outside_energy = _patient_weighted_mean(
        F.smooth_l1_loss(
            output.raw_disease_residual.float(),
            torch.zeros_like(output.raw_disease_residual, dtype=torch.float32),
            reduction="none",
        ).mean(1, keepdim=True),
        background * mask,
    )
    decorrelation = branch_decorrelation_loss(
        output.global_residual, output.gated_disease_residual
    )
    total = (
        float(global_background_weight) * global_background
        + float(disease_weight) * disease
        + float(full_grid_safety_weight) * full_grid
        + float(outside_energy_weight) * outside_energy
        + float(decorrelation_weight) * decorrelation
    )
    return {
        "total": total,
        "global_background": global_background,
        "disease": disease,
        "full_grid_safety": full_grid,
        "raw_disease_outside_energy": outside_energy,
        "global_disease_decorrelation": decorrelation,
    }


__all__ = [
    "branch_decorrelation_loss",
    "dual_path_objective",
    "semigroup_consistency_loss",
]
