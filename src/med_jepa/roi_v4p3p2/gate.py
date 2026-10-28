"""Adaptive train409 OOF recovery gate for Med-JEPA v4.3.2."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p3p1.gate import _comparison


SCHEMA = "med-jepa-roi-v4p3p2-numerical-recovery-gate-v1"


def recovery_comparisons(
    candidate_errors: Tensor,
    coarse_rrr_errors: Tensor,
    occupancy_shuffled_errors: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> tuple[dict[str, Mapping[str, Any]], Mapping[str, Any]]:
    expected = (len(patient_ids), len(REGIONS))
    if (
        len(patient_ids) not in {88, 409}
        or len(set(patient_ids)) != len(patient_ids)
        or candidate_errors.shape != expected
        or coarse_rrr_errors.shape != expected
        or occupancy_shuffled_errors.shape != expected
        # Fractional T0 regions can be empty for an individual patient after
        # local64 resampling.  patient_region_mae deliberately represents that
        # undefined patient-region estimand as NaN, and _comparison performs
        # the frozen paired complete-case filtering independently per region.
        # Infinities remain invalid and every region must retain at least two
        # paired patients (enforced by _comparison).
        or torch.isinf(candidate_errors).any()
        or torch.isinf(coarse_rrr_errors).any()
        or torch.isinf(occupancy_shuffled_errors).any()
        or int(replicates) != 10_000
    ):
        raise ValueError("v4.3.2 comparison inputs violate the frozen contract")
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
    alignment = _comparison(
        candidate_errors[:, core_index],
        occupancy_shuffled_errors[:, core_index],
        patient_ids,
        seed=int(seed) + 101,
        replicates=replicates,
    )
    return comparisons, alignment


def evaluate_v432_recovery_gate(
    candidate_errors: Tensor,
    coarse_rrr_errors: Tensor,
    occupancy_shuffled_errors: Tensor,
    patient_ids: Sequence[str],
    *,
    maximum_outside_core_support_absolute: float,
    numerical_noninferiority_margin: float,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Re-audit immutable v4.3.1 OOF measurements without refitting them."""

    if (
        len(patient_ids) != 409
        or float(numerical_noninferiority_margin) != 1e-6
        or not torch.isfinite(
            torch.tensor(float(maximum_outside_core_support_absolute))
        )
        or maximum_outside_core_support_absolute < 0
    ):
        raise ValueError("v4.3.2 recovery constants changed")
    comparisons, alignment = recovery_comparisons(
        candidate_errors,
        coarse_rrr_errors,
        occupancy_shuffled_errors,
        patient_ids,
        seed=seed,
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
        "ring_mean_not_worse_than_coarse_RRR": (
            float(ring["improvement"]) >= 0
        ),
        "full_grid_mean_not_worse_than_coarse_RRR": (
            float(full["improvement"]) >= 0
        ),
        "background_CI_lower_above_fixed_numerical_noninferiority_margin": (
            float(background["ci_low"])
            > -float(numerical_noninferiority_margin)
        ),
        "T0_occupancy_alignment_core_CI_lower_positive": (
            float(alignment["ci_low"]) > 0
        ),
        "core_residual_exactly_zero_where_T0_core_occupancy_is_zero": (
            float(maximum_outside_core_support_absolute) == 0.0
        ),
    }
    passed = all(criteria.values())
    return {
        "schema_version": SCHEMA,
        "status": "passed" if passed else "failed",
        "recovery_gate_passed": passed,
        "scope": "adaptive_reaudit_of_immutable_train409_OOF_measurements",
        "candidate_vs_coarse_RRR": comparisons,
        "T0_occupancy_alignment_candidate_vs_patient_shuffle": alignment,
        "criterion_results": criteria,
        "background_numerical_noninferiority": {
            "estimand": "paired_patient_bootstrap_improvement_CI_lower",
            "fixed_absolute_margin": float(numerical_noninferiority_margin),
            "pass_rule": "CI_lower_gt_negative_fixed_margin",
            "observed_CI_lower": float(background["ci_low"]),
        },
        "maximum_outside_core_support_absolute": float(
            maximum_outside_core_support_absolute
        ),
        "bootstrap_replicates": int(replicates),
        "original_v4p3p1_failure_overwritten": False,
        "OOF_models_refit_by_recovery": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "confirmatory_claim_permitted": False,
    }


__all__ = [
    "SCHEMA",
    "evaluate_v432_recovery_gate",
    "recovery_comparisons",
]
