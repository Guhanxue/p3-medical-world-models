"""Leakage-safe fold-local coarse-RRR anchor construction for v4.3.1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.data import PatientStates
from med_jepa.roi_v4p2.train import (
    _mean_visit,
    _select_pair_alpha,
)
from med_jepa.roi_v4p3.baselines import _rrr_prediction


@dataclass(frozen=True)
class CoarseAnchorFit:
    predictions: Tensor
    shrinkage_alpha: float
    shrinkage_leave_one_out_scores: Mapping[str, float]
    coarse_rrr_report: Mapping[str, Any]
    fit_patient_count: int
    prediction_patient_count: int


def fit_predict_coarse_rrr_anchor(
    records: Sequence[PatientStates],
    features: Tensor,
    evaluation_weights: Tensor,
    input_occupancy: Tensor,
    fit_indices: Tensor,
    predict_indices: Tensor,
    *,
    seed: int,
    device: torch.device,
) -> CoarseAnchorFit:
    """Fit only on ``fit_indices`` and predict the requested patients once.

    The reduced-rank/regularization choice is selected by grouped CV inside
    ``fit_indices``.  ``predict_indices`` outcomes are passed only to the
    baseline routine's descriptive metrics and never influence fitting.
    """

    if (
        fit_indices.ndim != 1
        or predict_indices.ndim != 1
        or len(fit_indices) < 20
        or len(predict_indices) < 1
        or len(set(fit_indices.tolist())) != len(fit_indices)
        or len(set(predict_indices.tolist())) != len(predict_indices)
    ):
        raise ValueError("coarse RRR anchor indices are invalid")
    alpha, alpha_scores = _select_pair_alpha(
        features,
        evaluation_weights,
        fit_indices,
        (1, 2),
        device=device,
    )
    target_mean = _mean_visit(features, fit_indices, 2)
    fit_shrinkage = (
        float(alpha) * features[fit_indices, 1].float()
        + (1.0 - float(alpha)) * target_mean.float()
    )
    predict_shrinkage = (
        float(alpha) * features[predict_indices, 1].float()
        + (1.0 - float(alpha)) * target_mean.float()
    )
    prediction, report, outside = _rrr_prediction(
        "occupancy_conditioned_coarse_RRR",
        records,
        features,
        evaluation_weights,
        input_occupancy,
        fit_indices,
        predict_indices,
        fit_shrinkage,
        predict_shrinkage,
        beta=0.0,
        seed=int(seed),
    )
    if outside != 0.0 or prediction.shape != features[predict_indices, 2].shape:
        raise RuntimeError("coarse RRR anchor violated its spatial contract")
    return CoarseAnchorFit(
        predictions=prediction.float().cpu(),
        shrinkage_alpha=float(alpha),
        shrinkage_leave_one_out_scores=dict(alpha_scores),
        coarse_rrr_report=dict(report),
        fit_patient_count=len(fit_indices),
        prediction_patient_count=len(predict_indices),
    )


def anchor_bank(
    total_patients: int,
    indices: Tensor,
    predictions: Tensor,
) -> Tensor:
    if (
        total_patients < 1
        or indices.ndim != 1
        or predictions.ndim != 5
        or predictions.shape[0] != len(indices)
        or predictions.shape[1:] != (1024, 7, 7, 7)
    ):
        raise ValueError("anchor bank inputs do not align")
    bank = torch.full(
        (total_patients, 1024, 7, 7, 7),
        float("nan"),
        dtype=torch.float32,
    )
    bank[indices] = predictions.float()
    return bank


__all__ = ["CoarseAnchorFit", "anchor_bank", "fit_predict_coarse_rrr_anchor"]

