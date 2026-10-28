"""Fold-local information-matched baselines for Med-JEPA v4.3.

The outer held fold is never used to select a reference family.  Reference
selection happens on a deterministic calibration subset of the outer training
fold, after which the selected family is refit on the complete outer training
fold and evaluated once on the held patients.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
import torch.nn.functional as F

from med_jepa.frozen_audit.run import _fit_rrr_temporal
from med_jepa.roi_v4.audit_mechanistic import REGIONS, patient_region_mae
from med_jepa.roi_v4.data import PatientStates


BASELINE_FAMILIES = (
    "train_fold_shrinkage_anchor",
    "occupancy_conditioned_global_RRR",
    "occupancy_conditioned_coarse_RRR",
    "dual_path_linear_global_plus_gated_local_residual",
)


@dataclass(frozen=True)
class FoldBaselineResult:
    selected_family: str
    prediction: Tensor
    selection_disease_mae: Mapping[str, float]
    selected_fit_report: Mapping[str, Any]
    outside_gated_local_maximum_absolute: float


def _weighted_pool(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.flatten(1).sum(1).clamp_min(1e-6)
    return (
        (values.float() * weights[:, None].float()).flatten(2).sum(2)
        / denominator[:, None]
    )


def _condition(records: Sequence[PatientStates]) -> Tensor:
    times = torch.stack([item.times_days for item in records]).float()
    actions = torch.stack([item.action_13 for item in records]).float()
    dt = torch.log1p((times[:, 2] - times[:, 1]).clamp_min(0) / 30.0)
    source = torch.log1p(times[:, 1].clamp_min(0) / 30.0)
    return torch.cat((dt[:, None], source[:, None], actions), dim=1)


def _adaptive_view(values: Tensor, size: int) -> Tensor:
    if values.ndim != 5:
        raise ValueError("adaptive baseline view expects [N,C,D,H,W]")
    return F.adaptive_avg_pool3d(values.float(), size).flatten(1)


def _occupancy_view(input_occupancy: Tensor, indices: Tensor, size: int) -> Tensor:
    core = input_occupancy[indices, 0, None].float()
    ring = input_occupancy[indices, 1, None].float()
    valid = input_occupancy[indices, 2, None].float()
    return torch.cat(
        (
            F.adaptive_avg_pool3d(core, size).flatten(1),
            F.adaptive_avg_pool3d(ring, size).flatten(1),
            F.adaptive_avg_pool3d(valid, size).flatten(1),
        ),
        dim=1,
    )


def _design(
    records: Sequence[PatientStates],
    features: Tensor,
    input_occupancy: Tensor,
    indices: Tensor,
    *,
    spatial_size: int,
) -> Tensor:
    visits = []
    for visit in (0, 1):
        visits.append(_adaptive_view(features[indices, visit], spatial_size))
    history = visits[1] - visits[0]
    selected_records = [records[int(index)] for index in indices.tolist()]
    return torch.cat(
        (
            visits[0],
            visits[1],
            history,
            _condition(selected_records),
            _occupancy_view(input_occupancy, indices, spatial_size),
        ),
        dim=1,
    ).float()


def _reconstruct(values: Tensor, size: int) -> Tensor:
    if values.ndim != 2 or values.shape[1] != 1024 * size**3:
        raise ValueError("RRR output does not match its frozen spatial view")
    coarse = values.reshape(-1, 1024, size, size, size)
    if size == 7:
        return coarse.contiguous()
    return F.interpolate(
        coarse,
        size=(7, 7, 7),
        mode="trilinear",
        align_corners=False,
    ).contiguous()


def _gate(input_occupancy: Tensor, indices: Tensor, beta: float) -> Tensor:
    core = input_occupancy[indices, 0, None].float()
    ring = input_occupancy[indices, 1, None].float()
    return (core + float(beta) * ring).clamp(0, 1)


def _rrr_prediction(
    family: str,
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    fit_indices: Tensor,
    predict_indices: Tensor,
    anchor_fit: Tensor,
    anchor_predict: Tensor,
    *,
    beta: float,
    seed: int,
) -> tuple[Tensor, dict[str, Any], float]:
    fit_target = features[fit_indices, 2].float()
    predict_target = features[predict_indices, 2].float()
    fit_ids = [records[int(index)].patient_id for index in fit_indices.tolist()]

    if family == "occupancy_conditioned_global_RRR":
        size = 1
        fit_x = _design(records, features, input_occupancy, fit_indices, spatial_size=2)
        predict_x = _design(
            records, features, input_occupancy, predict_indices, spatial_size=2
        )
        background_fit = evaluation_weights[
            fit_indices, REGIONS.index("valid_non_disease_bed")
        ]
        background_predict = evaluation_weights[
            predict_indices, REGIONS.index("valid_non_disease_bed")
        ]
        fit_y = _weighted_pool(fit_target - anchor_fit, background_fit)
        predict_y = _weighted_pool(
            predict_target - anchor_predict, background_predict
        )
        report, residual = _fit_rrr_temporal(
            fit_x, fit_y, predict_x, predict_y, fit_ids, seed=seed
        )
        prediction = anchor_predict + residual[:, :, None, None, None]
        return prediction, report, 0.0

    if family == "occupancy_conditioned_coarse_RRR":
        size = 3
        fit_x = _design(records, features, input_occupancy, fit_indices, spatial_size=size)
        predict_x = _design(
            records, features, input_occupancy, predict_indices, spatial_size=size
        )
        fit_y = _adaptive_view(fit_target - anchor_fit, size)
        predict_y = _adaptive_view(predict_target - anchor_predict, size)
        report, residual = _fit_rrr_temporal(
            fit_x, fit_y, predict_x, predict_y, fit_ids, seed=seed
        )
        return anchor_predict + _reconstruct(residual, size), report, 0.0

    if family != "dual_path_linear_global_plus_gated_local_residual":
        raise ValueError("unknown information-matched baseline family")

    # Low-frequency global path, fitted in exactly the same 2-cube output
    # space as the neural global branch.
    fit_x_global = _design(records, features, input_occupancy, fit_indices, spatial_size=2)
    predict_x_global = _design(
        records, features, input_occupancy, predict_indices, spatial_size=2
    )
    fit_y_global = _adaptive_view(fit_target - anchor_fit, 2)
    predict_y_global = _adaptive_view(predict_target - anchor_predict, 2)
    global_report, global_predict = _fit_rrr_temporal(
        fit_x_global,
        fit_y_global,
        predict_x_global,
        predict_y_global,
        fit_ids,
        seed=seed + 17,
    )
    # The training-side global residual is independently generated from the
    # same fold fit.  It is used only to define the local residual target.
    _, global_fit = _fit_rrr_temporal(
        fit_x_global,
        fit_y_global,
        fit_x_global,
        fit_y_global,
        fit_ids,
        seed=seed + 19,
    )
    global_fit_grid = _reconstruct(global_fit, 2)
    global_predict_grid = _reconstruct(global_predict, 2)

    fit_gate = _gate(input_occupancy, fit_indices, beta)
    predict_gate = _gate(input_occupancy, predict_indices, beta)
    fit_x_local = _design(records, features, input_occupancy, fit_indices, spatial_size=3)
    predict_x_local = _design(
        records, features, input_occupancy, predict_indices, spatial_size=3
    )
    local_target_fit = (fit_target - anchor_fit - global_fit_grid) * fit_gate
    local_target_predict = (
        predict_target - anchor_predict - global_predict_grid
    ) * predict_gate
    local_report, local_predict = _fit_rrr_temporal(
        fit_x_local,
        _adaptive_view(local_target_fit, 3),
        predict_x_local,
        _adaptive_view(local_target_predict, 3),
        fit_ids,
        seed=seed + 23,
    )
    raw_local = _reconstruct(local_predict, 3)
    gated_local = raw_local * predict_gate
    outside = predict_gate == 0
    outside_max = (
        float(gated_local[outside.expand_as(gated_local)].abs().max())
        if bool(outside.any())
        else 0.0
    )
    report = {
        "global_RRR": global_report,
        "gated_local_RRR": local_report,
        "gate": "T0_core_plus_beta_ring_fractional_occupancy",
        "ring_gate_beta": float(beta),
    }
    return (
        anchor_predict + global_predict_grid + gated_local,
        report,
        outside_max,
    )


def _calibration_split(indices: Tensor, *, seed: int) -> tuple[Tensor, Tensor]:
    if indices.ndim != 1 or len(indices) < 20:
        raise ValueError("outer training fold is too small")
    generator = torch.Generator().manual_seed(int(seed))
    shuffled = indices[torch.randperm(len(indices), generator=generator)]
    calibration_count = max(16, len(indices) // 5)
    return shuffled[calibration_count:].sort().values, shuffled[:calibration_count].sort().values


def select_and_predict_strongest_baseline(
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    outer_train_indices: Tensor,
    held_indices: Tensor,
    *,
    alpha: float,
    population_target_mean: Tensor,
    ring_gate_beta: float,
    seed: int,
) -> FoldBaselineResult:
    """Select a reference inside the outer fold, then score held patients once."""

    fit_indices, calibration_indices = _calibration_split(
        outer_train_indices, seed=seed
    )
    anchor = lambda idx: (  # noqa: E731 - compact fold-bound anchor definition
        float(alpha) * features[idx, 1].float()
        + (1.0 - float(alpha)) * population_target_mean.float()
    )
    calibration_target = features[calibration_indices, 2].float()
    calibration_anchor = anchor(calibration_indices)
    disease_weights = evaluation_weights[
        calibration_indices, REGIONS.index("t0_disease_bed")
    ]
    selection: dict[str, float] = {
        "train_fold_shrinkage_anchor": float(
            torch.nanmean(
                patient_region_mae(
                    calibration_anchor, calibration_target, disease_weights
                )
            )
        )
    }
    selection_reports: dict[str, Any] = {}
    for counter, family in enumerate(BASELINE_FAMILIES[1:]):
        prediction, report, _ = _rrr_prediction(
            family,
            records,
            features,
            evaluation_weights,
            input_occupancy,
            fit_indices,
            calibration_indices,
            anchor(fit_indices),
            calibration_anchor,
            beta=ring_gate_beta,
            seed=seed + (counter + 1) * 1_003,
        )
        selection[family] = float(
            torch.nanmean(
                patient_region_mae(prediction, calibration_target, disease_weights)
            )
        )
        selection_reports[family] = report
    selected = min(BASELINE_FAMILIES, key=lambda name: (selection[name], name))
    held_anchor = anchor(held_indices)
    if selected == "train_fold_shrinkage_anchor":
        prediction = held_anchor
        report: Mapping[str, Any] = {
            "model": "pair_specific_train_fold_shrinkage_anchor"
        }
        outside_max = 0.0
    else:
        prediction, report, outside_max = _rrr_prediction(
            selected,
            records,
            features,
            evaluation_weights,
            input_occupancy,
            outer_train_indices,
            held_indices,
            anchor(outer_train_indices),
            held_anchor,
            beta=ring_gate_beta,
            seed=seed + 90_001,
        )
    return FoldBaselineResult(
        selected_family=selected,
        prediction=prediction.float().cpu(),
        selection_disease_mae=selection,
        selected_fit_report={
            "selected": dict(report),
            "calibration_reports": selection_reports,
        },
        outside_gated_local_maximum_absolute=outside_max,
    )


__all__ = [
    "BASELINE_FAMILIES",
    "FoldBaselineResult",
    "select_and_predict_strongest_baseline",
]
