"""Freeze the adaptive-but-train409-only Med-JEPA v4.3.1 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from med_jepa.roi_v4p3.train import SCHEMA as V43_TRAINING_SCHEMA
from med_jepa.roi_v4p3p1.config import V431TrainingConfig


SCHEMA = "med-jepa-roi-v4p3p1-frozen-design-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("v4.3.1 evidence must be a JSON object")
    return value


def _validate_v43_result(report: dict[str, Any]) -> dict[str, Any]:
    cv = report.get("train409_cross_validation")
    if (
        report.get("schema_version") != V43_TRAINING_SCHEMA
        or report.get("status")
        != "completed_train409_oof_gate_fail_no_final_model"
        or report.get("patient_counts")
        != {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0}
        or report.get("selected_candidate") is not None
        or report.get("final_checkpoint_created") is not False
        or report.get("validation_88_loaded") is not False
        or report.get("test_split_loaded") is not False
        or not isinstance(cv, dict)
        or cv.get("passing_candidates") != []
        or cv.get("outer_held_fold_used_for_early_stopping") is not False
        or cv.get("baseline_selected_inside_outer_training_fold") is not True
    ):
        raise ValueError("v4.3 negative-result prerequisite is invalid")
    gates = cv.get("candidate_oof_gates")
    summaries = cv.get("candidate_summaries")
    if not isinstance(gates, dict) or len(gates) != 6 or not isinstance(summaries, dict):
        raise ValueError("v4.3 candidate evidence is incomplete")
    core_positive = 0
    selected_families: set[str] = set()
    for name, gate in gates.items():
        if not isinstance(gate, dict) or name not in summaries:
            raise ValueError("v4.3 gate/summary keys do not align")
        core = gate.get("candidate_vs_strongest_information_matched_baseline", {}).get(
            "t0_core", {}
        )
        if float(core.get("ci_low", float("-inf"))) > 0:
            core_positive += 1
        fold_evidence = summaries[name].get("fold_evidence")
        if not isinstance(fold_evidence, list) or len(fold_evidence) != 5:
            raise ValueError("v4.3 outer-fold evidence is incomplete")
        selected_families.update(
            item.get("strongest_baseline", {}).get("selected_family", "")
            for item in fold_evidence
        )
    if core_positive < 4 or selected_families != {"occupancy_conditioned_coarse_RRR"}:
        raise ValueError("v4.3 does not support the frozen v4.3.1 hypothesis")
    return {
        "v4p3_candidates_with_positive_core_CI": core_positive,
        "v4p3_selected_baseline_families": sorted(selected_families),
    }


def freeze(args: argparse.Namespace) -> dict[str, Any]:
    config_path = Path(args.config).expanduser().resolve()
    parent_path = Path(args.v4p3_report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not config_path.is_file() or not parent_path.is_file():
        raise FileNotFoundError("v4.3.1 freeze prerequisite is unavailable")
    if output.exists():
        raise FileExistsError("immutable v4.3.1 design already exists")
    config = V431TrainingConfig.load(config_path)
    parent = _json(parent_path)
    parent_summary = _validate_v43_result(parent)
    report = {
        "schema_version": SCHEMA,
        "status": "frozen_train409_core_residual_oof_required_training_not_started",
        "repository_namespace": "med_jepa.roi_v4p3p1",
        "scientific_target": (
            "nonlinear_T0_core_residual_beyond_fold_local_occupancy_conditioned_"
            "coarse_RRR"
        ),
        "adaptive_development_disclosure": {
            "derived_after_reviewing_v4p3_train409_OOF": True,
            "independent_confirmatory_claim_permitted": False,
            **parent_summary,
        },
        "feature_contract": {
            "variant": parent["selected_feature_variant"],
            "shape": [1024, 7, 7, 7],
            "source": "frozen_3DINO_local64_T0_anchored_magnification",
            "future_pixel_reconstruction": False,
        },
        "prediction_contract": {
            "equation": (
                "fold_local_occupancy_conditioned_coarse_RRR + "
                "T0_fractional_core_occupancy * nonlinear_residual"
            ),
            "fixed_anchor": "occupancy_conditioned_coarse_RRR",
            "anchor_rank_and_ridge_selected_inside_each_fit_fold": True,
            "neural_global_branch": False,
            "neural_ring_residual_branch": False,
            "ring_occupancy_role": "context_covariate_only",
            "core_gate": "T0_fractional_core_occupancy",
            "residual_exactly_zero_where_core_occupancy_is_zero": True,
            "terminal_head_zero_initialized": True,
            "terminal_prediction_normalization": False,
        },
        "training_contract": {
            "single_transition": "T1_to_T3_with_T0_history",
            "loss": "masked_fractional_T0_core_SmoothL1",
            "mask_sampling": "uniform_spatial_blocks",
            "mask_fractions": list(config.mask_fractions),
            "FTV_encoder_gradient": False,
            "semigroup_loss": False,
            "candidate_count": 1,
            "seed": config.seed,
            "patient_grouped_outer_folds": config.grouped_cv_folds,
            "outer_held_patients_used_for_epoch_selection": False,
            "inner_calibration_used_for_epoch_selection": True,
        },
        "fractional_token_boundary": {
            "core_ring_background_can_coexist_within_one_7cube_token": True,
            "exact_ring_or_background_metric_equality_claimed": False,
            "ring_background_full_grid_reported_as_safety_estimands": True,
        },
        "train409_oof_gate": {
            "primary": [
                "core_vs_coarse_RRR_CI_lower_positive",
                "core_patient_win_fraction_gt_half",
            ],
            "safety": [
                "disease_bed_mean_not_worse_than_coarse_RRR",
                "ring_mean_not_worse_than_coarse_RRR",
                "background_mean_not_worse_than_coarse_RRR",
                "full_grid_mean_not_worse_than_coarse_RRR",
            ],
            "mechanistic": [
                "T0_occupancy_alignment_core_CI_lower_positive",
                "core_residual_exactly_zero_where_T0_core_occupancy_is_zero",
            ],
            "regions_reported": [
                "t0_core",
                "t0_20mm_ring",
                "t0_disease_bed",
                "valid_non_disease_bed",
                "full_shared_valid_grid",
            ],
            "paired_patient_bootstrap_replicates": config.bootstrap_replicates,
        },
        "protocol_boundary": {
            "train_patients": 409,
            "validation_88_loaded_by_freeze_process": False,
            "validation_88_authorized_for_selection": False,
            "held_out_test90_authorized": False,
            "training_started": False,
            "test_split_loaded": False,
        },
        "evidence": {
            "config_sha256": _sha(config_path),
            "v4p3_train409_report_sha256": _sha(parent_path),
            "manifest_sha256": parent["evidence"]["manifest_sha256"],
            "train_feature_index_sha256": parent["evidence"]["feature_index_sha256"],
            "freeze_source_sha256": _sha(Path(__file__)),
        },
        "training_started": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "paths_emitted": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.chmod(temporary, 0o600)
    os.replace(temporary, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--v4p3-report", required=True)
    parser.add_argument("--output", required=True)
    report = freeze(parser.parse_args())
    print(
        json.dumps(
            {
                "status": report["status"],
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
