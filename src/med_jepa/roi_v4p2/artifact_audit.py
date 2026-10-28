"""Post-hoc validation audit of v4.2r token locality and norm artifacts.

All thresholds and the tumour-locality probe are frozen from train409 before
this command opens the already-used validation release.  The primary removal
mask is defined by T1 source features only.  No model is trained, no image is
reconstructed, and the held-out test split is never loaded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary
from med_jepa.roi_v4p2.artifact_lock import (
    PRIVATE_SCHEMA as LOCK_PRIVATE_SCHEMA,
    SCHEMA as LOCK_SCHEMA,
)
from med_jepa.roi_v4p2.data import load_existing_selected_validation_release
from med_jepa.roi_v4p2.gate import evaluate_v4p2_gate
from med_jepa.roi_v4p2.lock import SELECTED_VARIANT
from med_jepa.roi_v4p2.recovery import (
    PRIVATE_SCHEMA as RECOVERY_PRIVATE_SCHEMA,
    SCHEMA as RECOVERY_SCHEMA,
)
from med_jepa.roi_v4p2.token_artifact import (
    apply_locality_probe,
    apply_source_norm_rule,
    binary_ranking_metrics,
    contribution_decomposition,
    region_group_errors,
    token_l2_norm,
)
from med_jepa.roi_v4p2.train import _record_features


SCHEMA = "med-jepa-roi-v4p2r-token-locality-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2r-private-token-locality-audit-v1"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _norm_summary(value: Tensor) -> dict[str, float | int]:
    flat = value.float().flatten()
    quantiles = torch.quantile(
        flat, torch.tensor([0.5, 0.9, 0.95, 0.99, 0.995, 0.999])
    )
    return {
        "count": int(flat.numel()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "q50": float(quantiles[0]),
        "q90": float(quantiles[1]),
        "q95": float(quantiles[2]),
        "q99": float(quantiles[3]),
        "q99_5": float(quantiles[4]),
        "q99_9": float(quantiles[5]),
        "maximum": float(flat.max()),
        "q99_to_median_ratio": float(quantiles[3] / quantiles[0]),
        "maximum_to_median_ratio": float(flat.max() / quantiles[0]),
    }


def _region_prevalence(selector: Tensor, weights: Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, region in enumerate(REGIONS):
        weight = weights[:, index].float()
        denominator = weight.flatten(1).sum(1)
        eligible = denominator > 0
        patient = (
            (weight[eligible] * selector[eligible].float()).flatten(1).sum(1)
            / denominator[eligible]
        )
        result[region] = {
            "aggregate_weight_fraction": float(
                (weight * selector.float()).sum() / weight.sum()
            ),
            "patient_fraction_summary": _summary(patient),
        }
    return result


def _error_summaries(errors: Mapping[str, Tensor]) -> dict[str, Any]:
    return {
        method: {
            region: _summary(value[:, index])
            for index, region in enumerate(REGIONS)
        }
        for method, value in errors.items()
    }


def _group_analysis(
    *,
    prediction: Tensor,
    shrinkage: Tensor,
    persistence: Tensor,
    target: Tensor,
    weights: Tensor,
    selector: Tensor,
    patient_ids: list[str],
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    candidate = region_group_errors(prediction, target, weights, selector)
    reference = region_group_errors(shrinkage, target, weights, selector)
    persistence_errors = region_group_errors(
        persistence, target, weights, selector
    )
    gate = evaluate_v4p2_gate(
        candidate,
        reference,
        patient_ids,
        seed=bootstrap_seed,
        replicates=10_000,
    )
    return (
        {
            "region_MAE": _error_summaries(
                {
                    "med_jepa_v4p2r": candidate,
                    "train_only_shrinkage": reference,
                    "latent_persistence": persistence_errors,
                }
            ),
            "paired_patient_bootstrap_sensitivity": gate,
        },
        {
            "med_jepa_v4p2r": candidate,
            "train_only_shrinkage": reference,
            "latent_persistence": persistence_errors,
        },
    )


def _locality_validation(
    t0: Tensor, weights: Tensor, probe: Mapping[str, Any]
) -> tuple[dict[str, Any], Tensor]:
    probabilities = apply_locality_probe(t0, probe)
    labels = weights[:, REGIONS.index("t0_disease_bed")] > 0
    overall = binary_ranking_metrics(probabilities, labels)
    overall["AUPRC_lift_over_prevalence"] = (
        overall["auprc"] / overall["prevalence"]
    )
    per_patient: list[dict[str, Any]] = []
    for probability, label in zip(probabilities, labels, strict=True):
        if int(label.sum()) > 0 and int((~label).sum()) > 0:
            per_patient.append(binary_ranking_metrics(probability, label))
    overall["eligible_patient_count"] = len(per_patient)
    overall["per_patient_AUPRC"] = _summary(
        torch.tensor([item["auprc"] for item in per_patient])
    )
    overall["per_patient_AUROC"] = _summary(
        torch.tensor([item["auroc"] for item in per_patient])
    )
    overall["model_fitted_or_calibrated_on_validation"] = False
    return overall, probabilities


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "manifest": args.manifest,
            "lock_report": args.artifact_lock_report,
            "lock_private": args.artifact_lock_private,
            "recovery_report": args.recovery_report,
            "recovery_private": args.recovery_private_measurements,
            "validation_root": args.parent_validation_root,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    final_output = paths["output"]
    final_private = paths["private_output"]
    if final_output.exists() or final_private.exists():
        raise FileExistsError("immutable token-locality audit output exists")
    staging_output = final_output.parent / f".{final_output.name}.staging"
    staging_private = final_private.parent / f".{final_private.name}.staging"
    if staging_output.exists() or staging_private.exists():
        raise FileExistsError("stale token-locality audit staging output exists")
    staging_output.mkdir(parents=True, mode=0o700)
    staging_private.mkdir(parents=True, mode=0o700)

    lock = _json(paths["lock_report"])
    recovery = _json(paths["recovery_report"])
    lock_evidence = lock.get("evidence", {})
    recovery_evidence = recovery.get("evidence", {})
    if (
        lock.get("schema_version") != LOCK_SCHEMA
        or lock.get("status") != "locked_train409_token_artifact_audit"
        or lock.get("validation_loaded_while_freezing_parameters") is not False
        or lock.get("test_split_loaded") is not False
        or lock.get("selected_feature_variant") != SELECTED_VARIANT
        or recovery.get("schema_version") != RECOVERY_SCHEMA
        or recovery.get("status") not in {
            "completed_posthoc_selection_recovery_gate_pass",
            "completed_posthoc_selection_recovery_gate_fail",
        }
        or recovery.get("posthoc_exploratory") is not True
        or recovery.get("test_split_loaded") is not False
        or lock_evidence.get("recovery_report_sha256")
        != _sha(paths["recovery_report"])
        or lock_evidence.get("private_lock_payload_sha256")
        != _sha(paths["lock_private"])
        or recovery_evidence.get("private_validation_measurements_sha256")
        != _sha(paths["recovery_private"])
        or lock_evidence.get("recovery_private_validation_measurements_sha256")
        != _sha(paths["recovery_private"])
        or lock_evidence.get("manifest_sha256") != _sha(paths["manifest"])
    ):
        raise ValueError("token-artifact audit release chain is invalid")

    private_lock = torch.load(
        paths["lock_private"], map_location="cpu", weights_only=True
    )
    private_recovery = torch.load(
        paths["recovery_private"], map_location="cpu", weights_only=True
    )
    if (
        not isinstance(private_lock, Mapping)
        or private_lock.get("schema_version") != LOCK_PRIVATE_SCHEMA
        or private_lock.get("selected_feature_variant") != SELECTED_VARIANT
        or private_lock.get("norm_rule") != lock.get("norm_rule")
        or not isinstance(private_lock.get("locality_probe"), Mapping)
        or not isinstance(private_recovery, Mapping)
        or private_recovery.get("schema_version") != RECOVERY_PRIVATE_SCHEMA
    ):
        raise ValueError("private token-artifact audit inputs are invalid")

    val, weights, val_merkle, val_index, extraction_contract = (
        load_existing_selected_validation_release(
            manifest_path=paths["manifest"],
            validation_root=paths["validation_root"],
            expected_index_sha256=recovery_evidence[
                "parent_validation_feature_index_sha256"
            ],
            expected_payload_merkle_sha256=recovery_evidence[
                "parent_validation_feature_payload_merkle_sha256"
            ],
        )
    )
    patient_ids = [item.patient_id for item in val]
    prediction = private_recovery.get("prediction")
    shrinkage = private_recovery.get("shrinkage_prediction")
    if (
        private_recovery.get("patient_ids") != patient_ids
        or not isinstance(prediction, Tensor)
        or not isinstance(shrinkage, Tensor)
        or prediction.shape != (88, 1024, 7, 7, 7)
        or shrinkage.shape != prediction.shape
    ):
        raise ValueError("private v4.2r validation predictions do not align")
    prediction = prediction.float()
    shrinkage = shrinkage.float()
    features = _record_features(val)
    t0, source, target = features[:, 0], features[:, 1], features[:, 2]
    norm_rule = private_lock["norm_rule"]
    source_outlier = apply_source_norm_rule(source, norm_rule)
    target_outlier = apply_source_norm_rule(target, norm_rule)
    union_outlier = source_outlier | target_outlier
    normal = ~source_outlier

    locality, locality_probabilities = _locality_validation(
        t0, weights, private_lock["locality_probe"]
    )
    all_analysis, all_errors = _group_analysis(
        prediction=prediction,
        shrinkage=shrinkage,
        persistence=source,
        target=target,
        weights=weights,
        selector=torch.ones_like(source_outlier),
        patient_ids=patient_ids,
        bootstrap_seed=20_260_811,
    )
    normal_analysis, normal_errors = _group_analysis(
        prediction=prediction,
        shrinkage=shrinkage,
        persistence=source,
        target=target,
        weights=weights,
        selector=normal,
        patient_ids=patient_ids,
        bootstrap_seed=20_270_811,
    )
    outlier_errors = {
        "med_jepa_v4p2r": region_group_errors(
            prediction, target, weights, source_outlier
        ),
        "train_only_shrinkage": region_group_errors(
            shrinkage, target, weights, source_outlier
        ),
        "latent_persistence": region_group_errors(
            source, target, weights, source_outlier
        ),
    }
    decomposition = {
        "med_jepa_v4p2r_vs_train_only_shrinkage": contribution_decomposition(
            prediction, shrinkage, target, weights, source_outlier
        ),
        "med_jepa_v4p2r_vs_latent_persistence_descriptive": (
            contribution_decomposition(
                prediction, source, target, weights, source_outlier
            )
        ),
    }

    signature = bool(norm_rule["final_norm_sparse_heavy_tail_signature"])
    background = decomposition["med_jepa_v4p2r_vs_train_only_shrinkage"][
        "valid_non_disease_bed"
    ]
    share = background["outlier_share_of_total_improvement"]
    concentrated = (
        signature
        and isinstance(share, float)
        and math.isfinite(share)
        and share > 0.5
    )
    if not signature:
        decision = "no_sparse_final_norm_signature_prioritize_locality_and_nuisance_features"
    elif concentrated:
        decision = "background_gain_concentrated_in_sparse_high_norm_tokens"
    else:
        decision = "sparse_high_norm_tokens_present_but_not_dominant_explanation"

    private_path = staging_private / "validation_measurements.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": patient_ids,
            "source_outlier_mask": source_outlier,
            "target_outlier_mask_descriptive_only": target_outlier,
            "t0_locality_probability": locality_probabilities.half(),
            "all_token_patient_region_errors": all_errors,
            "source_normal_token_patient_region_errors": normal_errors,
            "source_outlier_token_patient_region_errors": outlier_errors,
        },
        private_path,
    )
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": "completed_posthoc_train_locked_token_locality_audit",
        "scope": "posthoc_validation_sensitivity_after_failed_v4p2r_specificity_gate",
        "selected_feature_variant": SELECTED_VARIANT,
        "feature_layer_contract": lock["feature_layer_contract"],
        "patient_counts_loaded": {"train": 0, "val_reused": 88, "test": 0},
        "norm_rule": norm_rule,
        "norm_distributions": {
            "source_T1": _norm_summary(token_l2_norm(source)),
            "target_T3_descriptive_only": _norm_summary(token_l2_norm(target)),
            "prediction_T3_descriptive_only": _norm_summary(
                token_l2_norm(prediction)
            ),
        },
        "source_T1_outlier_localization": _region_prevalence(
            source_outlier, weights
        ),
        "target_T3_outlier_localization_descriptive_only": _region_prevalence(
            target_outlier, weights
        ),
        "source_or_target_outlier_localization_descriptive_only": _region_prevalence(
            union_outlier, weights
        ),
        "tumour_locality_probe": {
            "train409_grouped_OOF": lock[
                "train_only_grouped_OOF_tumour_locality_probe"
            ],
            "validation88_once_without_refit": locality,
            "label": "T0_disease_bed_fractional_occupancy_greater_than_zero",
        },
        "all_tokens_reconstructed_from_private_half_precision_predictions": all_analysis,
        "source_T1_normal_tokens_primary_sensitivity": normal_analysis,
        "source_T1_outlier_tokens_descriptive_MAE": _error_summaries(
            outlier_errors
        ),
        "improvement_contribution_decomposition": decomposition,
        "decision": {
            "recommendation": decision,
            "final_norm_sparse_heavy_tail_signature": signature,
            "background_improvement_majority_from_source_outliers": concentrated,
            "artifact_excluded_formal_gate": False,
            "retraining_authorized_by_this_audit": False,
            "next_feature_intervention_requires_new_train409_triage": True,
        },
        "protocol": {
            "outlier_threshold_fitted_on_train409_only": True,
            "locality_probe_fitted_on_train409_only": True,
            "primary_exclusion_defined_from_T1_source_only": True,
            "target_defined_exclusion_used_for_primary_estimand": False,
            "validation_model_refit_or_calibration": False,
            "posthoc_exploratory": True,
            "confirmatory_claim_permitted": False,
            "eligible_for_test_evaluation": False,
        },
        "limitations": [
            "cached_features_are_final_normalized_patch_tokens",
            "absence_of_a_final_norm_tail_does_not_exclude_hidden_layer_register_neurons",
            "validation_has_already_been_used_by_parent_and_recovery_runs",
            "artifact_excluded_results_are_sensitivity_analyses_not_a_new_formal_gate",
        ],
        "validation_extraction_contract": extraction_contract,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "manifest_sha256": _sha(paths["manifest"]),
            "artifact_lock_report_sha256": _sha(paths["lock_report"]),
            "artifact_lock_private_sha256": _sha(paths["lock_private"]),
            "recovery_report_sha256": _sha(paths["recovery_report"]),
            "recovery_private_measurements_sha256": _sha(
                paths["recovery_private"]
            ),
            "parent_validation_feature_index_sha256": _sha(val_index),
            "parent_validation_feature_payload_merkle_sha256": val_merkle,
            "private_audit_measurements_sha256": _sha(private_path),
            "artifact_audit_source_sha256": _sha(Path(__file__)),
            "token_artifact_source_sha256": _sha(
                Path(__file__).with_name("token_artifact.py")
            ),
        },
    }
    report_path = staging_output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(report_path, 0o600)
    os.replace(staging_private, final_private)
    os.replace(staging_output, final_output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "decision": decision,
                "validation_patients": 88,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--artifact-lock-report", required=True)
    parser.add_argument("--artifact-lock-private", required=True)
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--recovery-private-measurements", required=True)
    parser.add_argument("--parent-validation-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRIVATE_SCHEMA", "SCHEMA", "run"]
