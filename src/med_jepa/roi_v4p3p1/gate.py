"""Frozen train409 OOF gate for Med-JEPA v4.3.1."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.frozen_audit.statistics import paired_patient_bootstrap
from med_jepa.roi_v4.audit_mechanistic import REGIONS


SCHEMA = "med-jepa-roi-v4p3p1-train409-oof-gate-v1"


def _comparison(
    candidate: Tensor,
    reference: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    eligible = torch.isfinite(candidate) & torch.isfinite(reference)
    ids = [
        patient_id
        for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
        if keep
    ]
    if len(ids) < 2:
        raise ValueError("v4.3.1 comparison has fewer than two patients")
    return asdict(
        paired_patient_bootstrap(
            candidate[eligible],
            reference[eligible],
            ids,
            higher_is_better=False,
            n_bootstrap=replicates,
            seed=seed,
        )
    )


def evaluate_v431_train_oof_gate(
    candidate_errors: Tensor,
    coarse_rrr_errors: Tensor,
    occupancy_shuffled_errors: Tensor,
    patient_ids: Sequence[str],
    *,
    maximum_outside_core_support_absolute: float,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Require a real core increment without buying it by harming other regions.

    Fractional core/ring/background weights can coexist in one 7-cube token,
    so exact equality of ring/background metrics is mathematically impossible.
    Exact zero is therefore checked only where T0 core occupancy is zero;
    ring/background/full-grid are frozen mean-safety checks and their complete
    paired intervals remain in the report.
    """

    expected = (len(patient_ids), len(REGIONS))
    if (
        len(patient_ids) != 409
        or len(set(patient_ids)) != 409
        or int(replicates) != 10_000
        or candidate_errors.shape != expected
        or coarse_rrr_errors.shape != expected
        or occupancy_shuffled_errors.shape != expected
        or not torch.isfinite(candidate_errors).all()
        or not torch.isfinite(coarse_rrr_errors).all()
        or not torch.isfinite(occupancy_shuffled_errors).all()
        or not torch.isfinite(
            torch.tensor(float(maximum_outside_core_support_absolute))
        )
        or maximum_outside_core_support_absolute < 0
    ):
        raise ValueError("v4.3.1 gate inputs violate the frozen contract")
    comparisons = {
        region: _comparison(
            candidate_errors[:, index],
            coarse_rrr_errors[:, index],
            patient_ids,
            seed=int(seed) + index,
            replicates=replicates,
        )
        for index, region in enumerate(REGIONS)
    }
    core_index = REGIONS.index("t0_core")
    occupancy_alignment = _comparison(
        candidate_errors[:, core_index],
        occupancy_shuffled_errors[:, core_index],
        patient_ids,
        seed=int(seed) + 101,
        replicates=replicates,
    )
    core = comparisons["t0_core"]
    disease = comparisons["t0_disease_bed"]
    ring = comparisons["t0_20mm_ring"]
    background = comparisons["valid_non_disease_bed"]
    full = comparisons["full_shared_valid_grid"]
    criteria = {
        "core_vs_coarse_RRR_CI_lower_positive": float(core["ci_low"]) > 0,
        "core_patient_win_fraction_gt_half": (
            float(core["patient_win_fraction"]) > 0.5
        ),
        "disease_bed_mean_not_worse_than_coarse_RRR": (
            float(disease["improvement"]) >= 0
        ),
        "ring_mean_not_worse_than_coarse_RRR": float(ring["improvement"]) >= 0,
        "background_mean_not_worse_than_coarse_RRR": (
            float(background["improvement"]) >= 0
        ),
        "full_grid_mean_not_worse_than_coarse_RRR": (
            float(full["improvement"]) >= 0
        ),
        "T0_occupancy_alignment_core_CI_lower_positive": (
            float(occupancy_alignment["ci_low"]) > 0
        ),
        "core_residual_exactly_zero_where_T0_core_occupancy_is_zero": (
            float(maximum_outside_core_support_absolute) == 0.0
        ),
    }
    passed = all(criteria.values())
    return {
        "schema_version": SCHEMA,
        "status": "passed" if passed else "failed",
        "development_gate_passed": passed,
        "scope": "train409_patient_grouped_out_of_fold_only",
        "primary_estimand": "T0_core_fractional_occupancy_weighted_token_MAE",
        "fixed_reference": "occupancy_conditioned_coarse_RRR",
        "candidate_vs_coarse_RRR": comparisons,
        "T0_occupancy_alignment_candidate_vs_patient_shuffle": (
            occupancy_alignment
        ),
        "maximum_outside_core_support_absolute": float(
            maximum_outside_core_support_absolute
        ),
        "fractional_region_overlap_precludes_exact_ring_background_equality": True,
        "criterion_results": criteria,
        "bootstrap_replicates": replicates,
        "best_seed_selection_permitted": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "positive_values_favor_med_jepa": True,
    }


__all__ = ["SCHEMA", "evaluate_v431_train_oof_gate"]
