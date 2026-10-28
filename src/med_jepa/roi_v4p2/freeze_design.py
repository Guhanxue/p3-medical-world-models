"""Freeze Med-JEPA v4.2 before feature triage or model training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "med-jepa-roi-v4p2-frozen-design-v2"
REGION_AUDIT_SCHEMA = "med-jepa-roi-v4p1-regional-specificity-audit-v1"
CANDIDATE = "source_anchored_med_jepa_v4p1"
SHRINKAGE = "train_only_shrinkage_to_mean_t3"
REQUIRED_REGIONS = (
    "t0_core",
    "t0_20mm_ring",
    "t0_disease_bed",
    "valid_non_disease_bed",
    "full_shared_valid_grid",
)


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _validated_v4p1_failure(report: Mapping[str, Any]) -> float:
    if (
        report.get("schema_version") != REGION_AUDIT_SCHEMA
        or report.get("status")
        != "completed_read_only_development_regional_specificity_audit"
        or report.get("test_split_loaded") is not False
        or report.get("retraining_performed") is not False
        or report.get("patient_counts") != {"train": 409, "val": 88, "test": 0}
        or tuple(report.get("regions", ())) != REQUIRED_REGIONS
    ):
        raise ValueError("v4.1 regional audit is not an approved immutable release")
    shrinkage = report.get("train_only_shrinkage_baseline", {})
    alpha = float(shrinkage.get("selected_alpha", -1.0))
    if (
        shrinkage.get("selection")
        != "train_only_leave_one_patient_out_t0_disease_bed_MAE"
        or not 0.0 <= alpha <= 1.0
    ):
        raise ValueError("v4.1 train-only shrinkage evidence is malformed")
    comparisons = report.get("paired_patient_bootstrap", {})
    for region in REQUIRED_REGIONS:
        value = comparisons.get(region, {}).get(
            "v4p1_vs_train_only_shrinkage_to_mean_t3", {}
        )
        if (
            int(value.get("n_patients", 0)) < 87
            or int(value.get("n_bootstrap", 0)) != 10_000
            or float(value.get("ci_high", 1.0)) >= 0.0
        ):
            raise ValueError("v4.1 failure against shrinkage is not fully established")
    specificity = (
        report.get("disease_specificity", {})
        .get("comparisons", {})
        .get("v4p1_vs_train_only_shrinkage_to_mean_t3", {})
    )
    interval = specificity.get("paired_bootstrap_95ci", [1.0, 1.0])
    if len(interval) != 2 or float(interval[1]) >= 0.0:
        raise ValueError("v4.1 disease specificity failure is not established")
    return alpha


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.v4p1_region_audit).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("immutable v4.2 design report already exists")
    source = _json(source_path)
    alpha = _validated_v4p1_failure(source)
    report = {
        "schema_version": SCHEMA,
        "status": "frozen_feature_triage_required_training_prohibited",
        "repository_namespace": "med_jepa.roi_v4p2",
        "scientific_target": "patient_specific_nonlinear_future_frozen_token_residual_beyond_train_only_shrinkage",
        "v4p1_decision": {
            "v4p1_beats_persistence_and_full_token_ridge": True,
            "v4p1_loses_to_train_only_shrinkage_in_every_reported_region": True,
            "v4p1_disease_specificity_relative_to_shrinkage_is_negative": True,
            "three_seed_v4p1_rerun_permitted": False,
        },
        "shrinkage_anchor_contract": {
            "formula": "alpha_times_source_plus_one_minus_alpha_times_train_mean_target",
            "alpha_from_v4p1_train_only_evidence": alpha,
            "alpha_must_be_reselected_inside_each_training_fold": True,
            "population_target_mean_must_be_estimated_inside_each_training_fold": True,
            "prediction": "shrinkage_anchor_plus_learned_residual",
            "zero_residual_exactly_equals_shrinkage_baseline": True,
            "terminal_prediction_normalization": False,
        },
        "training_objective_contract": {
            "t0_occupancy_weighted": True,
            "token_weight_formula": "t0_disease_bed_fractional_occupancy_plus_lambda_uniform_floor",
            "lambda_candidates_train_only": [0.01, 0.05, 0.1, 0.25],
            "lambda_selection": "five_fold_patient_grouped_CV_within_409_training_patients_only",
            "lambda_selection_primary_metric": "t0_disease_bed_MAE_relative_to_fold_local_shrinkage",
            "lambda_selection_tiebreakers": [
                "disease_minus_background_relative_improvement",
                "lower_full_shared_valid_grid_MAE",
                "larger_lambda",
            ],
            "mask_sampling": "uniform_spatial_block_sampling_independent_of_occupancy",
            "mask_fraction_schedule": [0.5, 0.625, 0.75, 0.875, 1.0],
            "mask_sampling_and_loss_weighting_must_not_both_oversample_disease": True,
            "ftv_encoder_gradient": False,
            "ftv_role": "detached_external_probe_only",
            "semigroup_consistency": True,
            "variance_or_rank_matching_pressure": False,
            "future_pixel_reconstruction": False,
        },
        "development_gate": {
            "primary_region": "t0_disease_bed",
            "primary_reference": SHRINKAGE,
            "candidate": "shrinkage_anchored_med_jepa_v4p2",
            "patient_level_paired_bootstrap_replicates": 10_000,
            "candidate_minus_reference_MAE_improvement_ci_lower_bound_must_be_positive": True,
            "disease_minus_background_relative_improvement_ci_lower_bound_must_be_positive": True,
            "all_regions_required_in_report": list(REQUIRED_REGIONS),
            "full_grid_regression_must_be_reported": True,
            "best_seed_selection_permitted": False,
            "enforcement_implementation": "med_jepa.roi_v4p2.gate.evaluate_v4p2_gate",
        },
        "feature_triage_contract": {
            "scope": "409_training_patients_only_nested_grouped_cross_validation",
            "validation_88_may_be_loaded_during_triage": False,
            "test_may_be_loaded": False,
            "required_variants": [
                "canonical_cached_7cube",
                "symmetric_shifted_window_14cube",
                "t0_anchored_local_magnification_80_to_112",
                "t0_anchored_local_magnification_64_to_112",
                "patient_visit_reference_aligned_SER_7cube",
            ],
            "shifted_window_interpretation": "denser_token_sampling_not_smaller_patch_receptive_field",
            "local_magnification_interpretation": "native_2p1mm_information_rescaled_to_smaller_FOV_not_true_1mm_reconstruction",
            "true_high_resolution_release": "requires_new_raw_SER_conversion_and_is_not_fabricated_from_2p1mm_ROI",
            "reference_normalization": "patient_level_scale_hierarchy_IQR_then_P90_minus_P10_then_median_only_aligned_to_T0",
            "reference_normalization_scale_estimator_must_be_shared_across_visits": True,
            "reference_normalization_fallback_counts_must_be_reported": True,
            "selection_metrics": [
                "nested_CV_shrinkage_relative_MAE_reduction_in_t0_disease_bed",
                "nested_CV_disease_minus_background_relative_improvement",
                "nested_CV_T0_FTV_linear_probe_R2",
                "nested_CV_T0_core_volume_linear_probe_R2",
            ],
            "variant_selection_uses_validation_88": False,
        },
        "split_contract": {
            "hyperparameter_development_patients": 409,
            "nested_grouped_outer_folds": 5,
            "linear_probe_nested_grouped_inner_folds": 4,
            "shrinkage_alpha_selection": "leave_one_out_within_each_outer_training_fold",
            "validation_patients_reserved_until_design_and_hyperparameters_locked": 88,
            "validation_reuse_limitation": "previous_v4_development_results_make_future_claims_exploratory",
            "test_patients_loaded": 0,
        },
        "model_selection_contract": {
            "feature_variant_selected_once_from_train_only_triage": True,
            "architecture": {
                "hidden_channels": 128,
                "conditioned_blocks": 4,
                "architecture_reused_from_v4p1_without_validation_retuning": True,
            },
            "selectable_hyperparameters": [
                "feature_variant",
                "shrinkage_alpha",
                "occupancy_uniform_floor_lambda",
            ],
            "all_selectable_hyperparameters_use_only_409_training_patients": True,
            "validation_88_role": "one_fixed_post_lock_development_evaluation_only",
        },
        "required_before_training": [
            "completed_train_only_feature_triage_release",
            "locked_feature_variant_and_content_hashes",
            "locked_train_only_alpha_and_occupancy_lambda_search_protocol",
            "explicit_confirmation_validation_88_not_used_by_triage",
        ],
        "training_started": False,
        "validation_88_loaded_by_freeze": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "v4p1_region_audit_sha256": _sha(source_path),
            "design_source_sha256": _sha(Path(__file__)),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(output, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "training_started": False,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v4p1-region-audit", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REQUIRED_REGIONS", "SCHEMA", "run"]
