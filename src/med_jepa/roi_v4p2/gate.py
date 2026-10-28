"""Prespecified development gate for a future Med-JEPA v4.2 training release."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.frozen_audit.statistics import paired_patient_bootstrap
from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p1.region_audit import disease_specificity_bootstrap


SCHEMA = "med-jepa-roi-v4p2-development-gate-v1"


def evaluate_v4p2_gate(
    candidate_errors: Tensor,
    shrinkage_errors: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    """Evaluate the locked five-region gate without selecting a best seed."""

    expected = (len(patient_ids), len(REGIONS))
    if (
        candidate_errors.shape != expected
        or shrinkage_errors.shape != expected
        or len(set(patient_ids)) != len(patient_ids)
        or len(patient_ids) < 2
        or int(replicates) != 10_000
    ):
        raise ValueError("v4.2 gate inputs violate the frozen patient/region contract")
    comparisons: dict[str, Mapping[str, Any]] = {}
    for index, region in enumerate(REGIONS):
        eligible = (
            torch.isfinite(candidate_errors[:, index])
            & torch.isfinite(shrinkage_errors[:, index])
        )
        eligible_ids = [
            patient_id
            for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
            if keep
        ]
        if len(eligible_ids) < 2:
            raise ValueError(f"v4.2 gate region is not assessable: {region}")
        comparisons[region] = asdict(
            paired_patient_bootstrap(
                candidate_errors[eligible, index],
                shrinkage_errors[eligible, index],
                eligible_ids,
                higher_is_better=False,
                n_bootstrap=replicates,
                seed=int(seed) + index,
            )
        )
    disease_index = REGIONS.index("t0_disease_bed")
    background_index = REGIONS.index("valid_non_disease_bed")
    specificity = disease_specificity_bootstrap(
        candidate_errors[:, disease_index],
        shrinkage_errors[:, disease_index],
        candidate_errors[:, background_index],
        shrinkage_errors[:, background_index],
        seed=int(seed) + 10_000,
        replicates=replicates,
    )
    disease_pass = float(comparisons["t0_disease_bed"]["ci_low"]) > 0.0
    specificity_pass = float(specificity["paired_bootstrap_95ci"][0]) > 0.0
    passed = disease_pass and specificity_pass
    return {
        "schema_version": SCHEMA,
        "status": "passed" if passed else "failed",
        "development_gate_passed": passed,
        "primary_reference": "train_only_shrinkage_to_mean_t3",
        "regions": list(REGIONS),
        "patient_region_comparisons": comparisons,
        "disease_specificity": specificity,
        "criterion_results": {
            "disease_bed_candidate_vs_shrinkage_CI_lower_positive": disease_pass,
            "disease_minus_background_relative_improvement_CI_lower_positive": specificity_pass,
        },
        "best_seed_selection_permitted": False,
        "bootstrap_replicates": replicates,
        "positive_values_favor_med_jepa": True,
    }


__all__ = ["SCHEMA", "evaluate_v4p2_gate"]

