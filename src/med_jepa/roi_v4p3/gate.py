"""Patient-grouped train409 OOF gate for Med-JEPA v4.3."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.frozen_audit.statistics import paired_patient_bootstrap
from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p1.region_audit import disease_specificity_bootstrap


SCHEMA = "med-jepa-roi-v4p3-train409-oof-gate-v1"


def _comparison(
    candidate: Tensor,
    reference: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> Mapping[str, Any]:
    eligible = torch.isfinite(candidate) & torch.isfinite(reference)
    eligible_ids = [
        patient_id
        for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
        if keep
    ]
    if len(eligible_ids) < 2:
        raise ValueError("v4.3 comparison has fewer than two eligible patients")
    return asdict(
        paired_patient_bootstrap(
            candidate[eligible],
            reference[eligible],
            eligible_ids,
            higher_is_better=False,
            n_bootstrap=replicates,
            seed=seed,
        )
    )


def evaluate_v4p3_train_oof_gate(
    candidate_errors: Tensor,
    strongest_baseline_errors: Tensor,
    global_only_errors: Tensor,
    anchor_errors: Tensor,
    occupancy_shuffled_errors: Tensor,
    patient_ids: Sequence[str],
    *,
    maximum_outside_gated_disease_absolute: float,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Evaluate a frozen, single-seed, patient-grouped train409 OOF release.

    Every error matrix has one row per patient and the five columns in
    :data:`med_jepa.roi_v4.audit_mechanistic.REGIONS`.  The strongest baseline
    is selected inside each training fold before its held-out patients are
    scored; this function never selects a baseline from OOF outcomes.
    """

    expected = (len(patient_ids), len(REGIONS))
    values = {
        "candidate": candidate_errors,
        "strongest_information_matched_baseline": strongest_baseline_errors,
        "global_only_ablation": global_only_errors,
        "shrinkage_anchor": anchor_errors,
        "T0_occupancy_shuffled_ablation": occupancy_shuffled_errors,
    }
    if (
        len(patient_ids) != 409
        or len(set(patient_ids)) != len(patient_ids)
        or int(replicates) != 10_000
        or any(value.shape != expected for value in values.values())
        or not torch.isfinite(torch.tensor(float(maximum_outside_gated_disease_absolute)))
        or float(maximum_outside_gated_disease_absolute) < 0
    ):
        raise ValueError("v4.3 gate inputs violate the frozen train409 contract")

    candidate_vs_baseline: dict[str, Mapping[str, Any]] = {}
    for index, region in enumerate(REGIONS):
        candidate_vs_baseline[region] = _comparison(
            candidate_errors[:, index],
            strongest_baseline_errors[:, index],
            patient_ids,
            seed=int(seed) + index,
            replicates=replicates,
        )

    core = REGIONS.index("t0_core")
    disease = REGIONS.index("t0_disease_bed")
    background = REGIONS.index("valid_non_disease_bed")
    disease_branch_increment = _comparison(
        candidate_errors[:, disease],
        global_only_errors[:, disease],
        patient_ids,
        seed=int(seed) + 100,
        replicates=replicates,
    )
    global_branch_increment = _comparison(
        global_only_errors[:, background],
        anchor_errors[:, background],
        patient_ids,
        seed=int(seed) + 101,
        replicates=replicates,
    )
    occupancy_alignment = _comparison(
        candidate_errors[:, disease],
        occupancy_shuffled_errors[:, disease],
        patient_ids,
        seed=int(seed) + 102,
        replicates=replicates,
    )
    specificity = disease_specificity_bootstrap(
        candidate_errors[:, disease],
        strongest_baseline_errors[:, disease],
        candidate_errors[:, background],
        strongest_baseline_errors[:, background],
        seed=int(seed) + 10_000,
        replicates=replicates,
    )

    criteria = {
        "disease_bed_vs_strongest_baseline_CI_lower_positive": (
            float(candidate_vs_baseline["t0_disease_bed"]["ci_low"]) > 0
        ),
        "core_vs_strongest_baseline_CI_lower_positive": (
            float(candidate_vs_baseline["t0_core"]["ci_low"]) > 0
        ),
        "disease_minus_background_relative_gain_CI_lower_positive": (
            float(specificity["paired_bootstrap_95ci"][0]) > 0
        ),
        "disease_branch_increment_CI_lower_positive": (
            float(disease_branch_increment["ci_low"]) > 0
        ),
        "global_branch_background_increment_CI_lower_positive": (
            float(global_branch_increment["ci_low"]) > 0
        ),
        "T0_occupancy_alignment_CI_lower_positive": (
            float(occupancy_alignment["ci_low"]) > 0
        ),
        "gated_disease_branch_exactly_zero_outside_gate": (
            float(maximum_outside_gated_disease_absolute) == 0.0
        ),
    }
    passed = all(criteria.values())
    return {
        "schema_version": SCHEMA,
        "status": "passed" if passed else "failed",
        "development_gate_passed": passed,
        "scope": "train409_patient_grouped_out_of_fold_only",
        "regions": list(REGIONS),
        "strongest_baseline_selection": (
            "inside_each_training_fold_before_heldout_predictions"
        ),
        "candidate_vs_strongest_information_matched_baseline": (
            candidate_vs_baseline
        ),
        "disease_specificity": specificity,
        "disease_branch_increment_over_global_only": disease_branch_increment,
        "global_branch_increment_over_anchor_in_background": (
            global_branch_increment
        ),
        "T0_occupancy_alignment_candidate_vs_shuffled": occupancy_alignment,
        "maximum_outside_gated_disease_absolute": float(
            maximum_outside_gated_disease_absolute
        ),
        "criterion_results": criteria,
        "bootstrap_replicates": replicates,
        "best_seed_selection_permitted": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "positive_values_favor_med_jepa": True,
    }


__all__ = ["SCHEMA", "evaluate_v4p3_train_oof_gate"]
