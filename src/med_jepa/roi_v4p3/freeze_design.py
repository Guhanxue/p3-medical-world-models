"""Freeze the v4.3 train409-only dual-path development protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from med_jepa.roi_v4p2.artifact_audit import SCHEMA as TOKEN_AUDIT_SCHEMA
from med_jepa.roi_v4p2.locality_v2 import SCHEMA as LOCALITY_SCHEMA
from med_jepa.roi_v4p2.lock import SELECTED_VARIANT
from med_jepa.roi_v4p2.recovery import SCHEMA as RECOVERY_SCHEMA
from med_jepa.roi_v4p3.config import V43TrainingConfig


SCHEMA = "med-jepa-roi-v4p3-frozen-design-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v4.3 evidence must be a JSON object")
    return value


def _validate_inputs(
    *,
    config_path: Path,
    recovery_path: Path,
    locality_path: Path,
    token_audit_path: Path,
) -> tuple[V43TrainingConfig, dict[str, Any]]:
    config = V43TrainingConfig.load(config_path)
    recovery = _json(recovery_path)
    locality = _json(locality_path)
    token_audit = _json(token_audit_path)
    recovery_sha = _sha(recovery_path)

    recovery_evidence = recovery.get("evidence", {})
    locality_evidence = locality.get("evidence", {})
    token_evidence = token_audit.get("evidence", {})
    diagnostic = locality.get("results", {}).get("diagnostic_contract", {})
    token_decision = token_audit.get("decision", {})
    if (
        recovery.get("schema_version") != RECOVERY_SCHEMA
        or recovery.get("status")
        not in {
            "completed_posthoc_selection_recovery_gate_pass",
            "completed_posthoc_selection_recovery_gate_fail",
        }
        or recovery.get("selected_feature_variant") != SELECTED_VARIANT
        or recovery.get("test_split_loaded") is not False
        or not isinstance(recovery_evidence, Mapping)
        or locality.get("schema_version") != LOCALITY_SCHEMA
        or locality.get("status") != "completed_train409_only_locality_v2"
        or locality.get("patients")
        != {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0}
        or locality.get("selected_feature_variant") != SELECTED_VARIANT
        or locality.get("validation_88_loaded") is not False
        or locality.get("test_split_loaded") is not False
        or diagnostic.get("core_local_information_sufficient_for_v4p3")
        is not True
        or diagnostic.get("recommendation")
        != "proceed_to_dual_path_v4p3_train409_design"
        or locality_evidence.get("recovery_report_sha256") != recovery_sha
        or token_audit.get("schema_version") != TOKEN_AUDIT_SCHEMA
        or token_audit.get("status")
        != "completed_posthoc_train_locked_token_locality_audit"
        or token_audit.get("selected_feature_variant") != SELECTED_VARIANT
        or token_audit.get("test_split_loaded") is not False
        or token_decision.get("final_norm_sparse_heavy_tail_signature") is not False
        or token_decision.get("retraining_authorized_by_this_audit") is not False
        or token_evidence.get("recovery_report_sha256") != recovery_sha
        or locality_evidence.get("manifest_sha256")
        != recovery_evidence.get("manifest_sha256")
        or locality_evidence.get("feature_triage_private_index_sha256")
        != recovery_evidence.get("feature_triage_private_index_sha256")
    ):
        raise ValueError("v4.3 prerequisite evidence chain is invalid")
    return config, {
        "recovery": recovery,
        "locality": locality,
        "token_audit": token_audit,
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "config": args.config,
            "recovery": args.recovery_report,
            "locality": args.locality_v2_report,
            "token_audit": args.token_audit_report,
            "output": args.output,
        }.items()
    }
    for name in ("config", "recovery", "locality", "token_audit"):
        if not paths[name].is_file():
            raise FileNotFoundError(f"v4.3 prerequisite is unavailable: {name}")
    if paths["output"].exists():
        raise FileExistsError("immutable v4.3 frozen design already exists")
    config, evidence = _validate_inputs(
        config_path=paths["config"],
        recovery_path=paths["recovery"],
        locality_path=paths["locality"],
        token_audit_path=paths["token_audit"],
    )
    locality_diagnostic = evidence["locality"]["results"]["diagnostic_contract"]
    report = {
        "schema_version": SCHEMA,
        "status": "frozen_train409_oof_required_training_not_started",
        "repository_namespace": "med_jepa.roi_v4p3",
        "scientific_target": (
            "disease_specific_future_frozen_local64_token_residual_beyond_"
            "information_matched_train_fold_baselines"
        ),
        "feature_contract": {
            "variant": SELECTED_VARIANT,
            "shape": [1024, 7, 7, 7],
            "source": "frozen_3DINO_local64_T0_anchored_magnification",
            "future_pixel_reconstruction": False,
            "core_information_train409_sufficient": locality_diagnostic[
                "core_local_information_sufficient_for_v4p3"
            ],
        },
        "architecture_contract": {
            "prediction_equation": (
                "train_fold_shrinkage_anchor + low_frequency_global_residual + "
                "T0_occupancy_gated_disease_residual"
            ),
            "shrinkage_anchor_exactly_expressible": True,
            "terminal_prediction_normalization": False,
            "global_branch": {
                "inputs": [
                    "background_pooled_current_context",
                    "background_pooled_available_history_delta",
                    "delta_t",
                    "source_time",
                    "treatment_action",
                    "history_available_flag",
                ],
                "coarse_grid": list(config.global_coarse_grid),
                "upsampling": "trilinear_to_7_cube",
                "arbitrary_local_output_prohibited": True,
            },
            "disease_branch": {
                "inputs": [
                    "masked_current_local64_tokens",
                    "masked_available_history_delta",
                    "T0_core_fractional_occupancy",
                    "T0_ring_fractional_occupancy",
                    "token_coordinates",
                    "delta_t",
                    "source_time",
                    "treatment_action",
                ],
                "gate": "T0_core + beta * T0_20mm_ring",
                "gate_candidates_train409_only": list(
                    config.ring_gate_beta_candidates
                ),
                "contribution_exactly_zero_where_gate_is_zero": True,
            },
            "history_contract": {
                "T1_to_T3": "T1_minus_T0_token_delta_available",
                "earlier_pairs": "zero_history_plus_explicit_unavailable_flag",
                "T2_fabricated": False,
            },
            "mask_contract": {
                "uniform_spatial_sampling": True,
                "fractions": list(config.mask_fractions),
                "applies_to_residual_context_not_unmasked_anchor": True,
                "target_token_leakage_prohibited": True,
            },
            "zero_initialized_terminal_heads_make_initial_prediction_exact_anchor": True,
        },
        "loss_contract": {
            "global_branch_supervision": "valid_non_disease_bed",
            "disease_branch_supervision": "T0_core_plus_beta_ring_fractional_weight",
            "full_grid_safety_weight": config.full_grid_safety_weight,
            "raw_disease_outside_gate_energy_weight_candidates": list(
                config.outside_energy_weight_candidates
            ),
            "branch_decorrelation_weight": config.decorrelation_weight,
            "semigroup_weight": config.semigroup_weight,
            "FTV_encoder_gradient": False,
            "FTV_role": "detached_external_probe_only",
            "mask_sampling_remains_uniform": True,
            "effective_rank_target_or_gate": False,
        },
        "information_matched_baselines": {
            "required": [
                "train_fold_shrinkage_anchor",
                "occupancy_conditioned_global_and_coarse_RRR",
                "dual_path_linear_global_plus_gated_local_residual",
            ],
            "same_inputs_required": [
                "T0_and_T1_when_available",
                "delta_t",
                "source_time",
                "treatment_action",
                "T0_core_and_ring_fractional_occupancy",
            ],
            "strongest_baseline_selected_inside_each_training_fold": True,
            "OOF_outcomes_must_not_select_the_reference": True,
        },
        "train409_selection_contract": {
            "patients": 409,
            "ordered_pairs_per_patient": ["T0_to_T1", "T0_to_T3", "T1_to_T3"],
            "composition_triplet": "T0_to_T1_to_T3",
            "T2_fabricated": False,
            "patient_grouped_folds": config.grouped_cv_folds,
            "seed": config.seed,
            "candidate_count": (
                len(config.ring_gate_beta_candidates)
                * len(config.outside_energy_weight_candidates)
            ),
            "ring_gate_beta_candidates": list(config.ring_gate_beta_candidates),
            "outside_energy_weight_candidates": list(
                config.outside_energy_weight_candidates
            ),
            "candidate_selection_metric": (
                "mean_train409_OOF_T0_disease_bed_MAE_subject_to_all_frozen_gates"
            ),
            "validation_88_must_not_select_architecture_or_hyperparameters": True,
            "best_seed_selection_permitted": False,
        },
        "train409_oof_gate": {
            "regions_reported": [
                "t0_core",
                "t0_20mm_ring",
                "t0_disease_bed",
                "valid_non_disease_bed",
                "full_shared_valid_grid",
            ],
            "all_required": [
                "disease_bed_vs_strongest_baseline_CI_lower_positive",
                "core_vs_strongest_baseline_CI_lower_positive",
                "disease_minus_background_relative_gain_CI_lower_positive",
                "disease_branch_increment_over_global_only_CI_lower_positive",
                "global_branch_increment_over_anchor_in_background_CI_lower_positive",
                "T0_occupancy_alignment_vs_patient_shuffle_CI_lower_positive",
                "gated_disease_branch_exactly_zero_outside_gate",
            ],
            "paired_patient_bootstrap_replicates": config.bootstrap_replicates,
            "positive_CI_definition": "two_sided_95_percent_interval_lower_bound_gt_0",
        },
        "protocol_boundary": {
            "validation_88_historically_reused_by_predecessor": True,
            "validation_88_loaded_by_freeze_process": False,
            "validation_88_authorized_for_v4p3_selection": False,
            "future_validation_88_result_confirmatory": False,
            "held_out_test90_authorized": False,
            "test_split_loaded": False,
            "training_started": False,
        },
        "evidence": {
            "config_sha256": _sha(paths["config"]),
            "v4p2r_recovery_report_sha256": _sha(paths["recovery"]),
            "train409_locality_v2_report_sha256": _sha(paths["locality"]),
            "posthoc_token_audit_report_sha256": _sha(paths["token_audit"]),
            "manifest_sha256": evidence["recovery"]["evidence"][
                "manifest_sha256"
            ],
            "train_feature_index_sha256": evidence["recovery"]["evidence"][
                "feature_triage_private_index_sha256"
            ],
            "freeze_source_sha256": _sha(Path(__file__)),
        },
        "training_started": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "paths_emitted": False,
    }
    paths["output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = paths["output"].with_suffix(paths["output"].suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, paths["output"])
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--locality-v2-report", required=True)
    parser.add_argument("--token-audit-report", required=True)
    parser.add_argument("--output", required=True)
    report = freeze(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
                "namespace": report["repository_namespace"],
                "training_started": report["training_started"],
                "validation_88_loaded": report["validation_88_loaded"],
                "test_split_loaded": report["test_split_loaded"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SCHEMA", "freeze"]
