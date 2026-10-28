"""One fixed validation88 generalization evaluation for frozen v4.3.2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p2.train import _record_features, _region_errors
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3p1.anchor import anchor_bank, fit_predict_coarse_rrr_anchor
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA
from med_jepa.roi_v4p3p1.train import _predict_ensemble
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.data import load_v432_validation_release
from med_jepa.roi_v4p3p2.finalize import SCHEMA as FINAL_SCHEMA
from med_jepa.roi_v4p3p2.gate import recovery_comparisons


SCHEMA = "med-jepa-roi-v4p3p2-validation88-generalization-v2"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3p2-private-validation88-v1"


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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.3.2 validation requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "recovery_config": args.recovery_config,
            "v431_config": args.v4p3p1_config,
            "final_report": args.final_fit_report,
            "checkpoint": args.checkpoint,
            "validation_parent_report": args.validation_parent_report,
            "manifest": args.manifest,
            "train_feature_root": args.train_feature_root,
            "train_feature_index": args.train_feature_index,
            "validation_feature_root": args.validation_feature_root,
            "mask_root": args.context_mask_release,
            "mask_report": args.context_mask_report,
            "mask_index": args.context_mask_index,
            "prior_failed_stderr": args.prior_failed_validation_stderr,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    for name in (
        "recovery_config",
        "v431_config",
        "final_report",
        "checkpoint",
        "validation_parent_report",
        "manifest",
        "train_feature_index",
        "mask_report",
        "mask_index",
        "prior_failed_stderr",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"v4.3.2 validation input unavailable: {name}")
    for name in ("train_feature_root", "validation_feature_root", "mask_root"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"v4.3.2 validation release unavailable: {name}")
    prior_failed_job_id = str(args.prior_failed_validation_job_id)
    prior_failed_stderr = paths["prior_failed_stderr"].read_text(
        encoding="utf-8", errors="replace"
    )
    if (
        not prior_failed_job_id.isdigit()
        or "v4.3.2 comparison inputs violate the frozen contract"
        not in prior_failed_stderr
    ):
        raise ValueError("v4.3.2 validation retry evidence is invalid")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable v4.3.2 validation output exists")
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = (
        paths["private_output"].parent / f".{paths['private_output'].name}.staging"
    )
    output_staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=False, mode=0o700)

    recovery_config = V432RecoveryConfig.load(paths["recovery_config"])
    v431_config = V431TrainingConfig.load(paths["v431_config"])
    final_report = _json(paths["final_report"])
    validation_parent = _json(paths["validation_parent_report"])
    parent_evidence = validation_parent.get("evidence", {})
    if (
        final_report.get("schema_version") != FINAL_SCHEMA
        or final_report.get("status")
        != "completed_adaptive_recovery_train409_final_fit"
        or final_report.get("checkpoint_created") is not True
        or final_report.get("validation_88_loaded") is not False
        or final_report.get("test_split_loaded") is not False
        or final_report.get("evidence", {}).get("checkpoint_sha256")
        != _sha(paths["checkpoint"])
        or final_report.get("evidence", {}).get("manifest_sha256")
        != _sha(paths["manifest"])
        or final_report.get("evidence", {}).get("feature_index_sha256")
        != _sha(paths["train_feature_index"])
        or validation_parent.get("test_split_loaded") is not False
        or parent_evidence.get("validation_feature_index_sha256") is None
        or parent_evidence.get("validation_feature_payload_merkle_sha256") is None
        or recovery_config.final_epoch_count != 18
        or v431_config.seed != recovery_config.seed
    ):
        raise ValueError("v4.3.2 checkpoint/validation release chain is invalid")

    train, train_weights, train_occupancy, train_merkle, train_occ_digest = (
        load_v4p3_train409_release(
            paths["manifest"],
            paths["train_feature_root"],
            paths["train_feature_index"],
            expected_feature_index_sha256=_sha(paths["train_feature_index"]),
            context_mask_root=paths["mask_root"],
            context_mask_report=paths["mask_report"],
            context_mask_index=paths["mask_index"],
        )
    )
    (
        val,
        val_weights,
        val_occupancy,
        val_merkle,
        val_occ_digest,
        val_index,
    ) = load_v432_validation_release(
        manifest_path=paths["manifest"],
        validation_feature_root=paths["validation_feature_root"],
        expected_validation_index_sha256=parent_evidence[
            "validation_feature_index_sha256"
        ],
        expected_validation_payload_merkle_sha256=parent_evidence[
            "validation_feature_payload_merkle_sha256"
        ],
        context_mask_root=paths["mask_root"],
        context_mask_report=paths["mask_report"],
        context_mask_index=paths["mask_index"],
    )
    if len(train) != 409 or len(val) != 88:
        raise ValueError("v4.3.2 development split cardinality changed")
    records = [*train, *val]
    features = torch.cat((_record_features(train), _record_features(val)))
    weights = torch.cat((train_weights, val_weights))
    occupancy = torch.cat((train_occupancy, val_occupancy))
    fit_indices = torch.arange(409)
    val_indices = torch.arange(409, 497)
    anchor_fit = fit_predict_coarse_rrr_anchor(
        records,
        features,
        weights,
        occupancy,
        fit_indices,
        val_indices,
        seed=recovery_config.seed + 900_001,
        device=torch.device("cuda"),
    )
    anchors = anchor_bank(497, val_indices, anchor_fit.predictions)
    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != FINAL_SCHEMA
        or checkpoint.get("final_epoch_count") != 18
        or not isinstance(checkpoint.get("model"), Mapping)
        or not isinstance(
            checkpoint.get("coarse_RRR_train409_fit_report"), Mapping
        )
    ):
        raise ValueError("v4.3.2 checkpoint payload is invalid")
    checkpoint_anchor = checkpoint["coarse_RRR_train409_fit_report"]
    checkpoint_rrr = checkpoint_anchor.get("coarse_rrr")
    observed_rrr = anchor_fit.coarse_rrr_report
    if (
        float(checkpoint_anchor.get("shrinkage_alpha"))
        != float(anchor_fit.shrinkage_alpha)
        or not isinstance(checkpoint_rrr, Mapping)
        or checkpoint_rrr.get("selected_rank") != observed_rrr.get("selected_rank")
        or checkpoint_rrr.get("selected_alpha")
        != observed_rrr.get("selected_alpha")
        or checkpoint_rrr.get("train_grouped_cv_mean_feature_mae")
        != observed_rrr.get("train_grouped_cv_mean_feature_mae")
    ):
        raise ValueError("validation-time coarse RRR refit changed from checkpoint")
    model = CoarseRRRAnchoredCoreResidualJEPA(
        v431_config.hidden_channels, v431_config.conditioned_blocks
    ).to("cuda")
    model.load_state_dict(checkpoint["model"], strict=True)
    candidate, anchor, shuffled, outside = _predict_ensemble(
        model,
        records,
        occupancy,
        anchors,
        val_indices,
        v431_config,
        seed=recovery_config.bootstrap_seed + 7_000_001,
        device=torch.device("cuda"),
    )
    target = features[val_indices, 2]
    errors = {
        "candidate": _region_errors(candidate, target, val_weights),
        "coarse_rrr": _region_errors(anchor, target, val_weights),
        "occupancy_shuffled": _region_errors(shuffled, target, val_weights),
    }
    patient_ids = [item.patient_id for item in val]
    comparisons, alignment = recovery_comparisons(
        errors["candidate"],
        errors["coarse_rrr"],
        errors["occupancy_shuffled"],
        patient_ids,
        seed=recovery_config.bootstrap_seed + 8_000_003,
        replicates=recovery_config.bootstrap_replicates,
    )
    criteria = {
        "core_vs_coarse_RRR_CI_lower_positive": (
            float(comparisons["t0_core"]["ci_low"]) > 0
        ),
        "core_patient_win_fraction_gt_half": (
            float(comparisons["t0_core"]["patient_win_fraction"]) > 0.5
        ),
        "disease_bed_mean_not_worse_than_coarse_RRR": (
            float(comparisons["t0_disease_bed"]["improvement"]) >= 0
        ),
        "ring_mean_not_worse_than_coarse_RRR": (
            float(comparisons["t0_20mm_ring"]["improvement"]) >= 0
        ),
        "full_grid_mean_not_worse_than_coarse_RRR": (
            float(comparisons["full_shared_valid_grid"]["improvement"]) >= 0
        ),
        "background_CI_lower_above_fixed_numerical_noninferiority_margin": (
            float(comparisons["valid_non_disease_bed"]["ci_low"])
            > -recovery_config.background_numerical_noninferiority_margin
        ),
        "T0_occupancy_alignment_core_CI_lower_positive": (
            float(alignment["ci_low"]) > 0
        ),
        "core_residual_exactly_zero_where_T0_core_occupancy_is_zero": (
            float(outside) == 0.0
        ),
    }
    passed = all(criteria.values())
    private_path = private_staging / "validation88_measurements.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": patient_ids,
            "patient_region_errors": errors,
            "candidate_prediction": candidate.half(),
            "coarse_RRR_prediction": anchor.half(),
            "occupancy_shuffled_prediction": shuffled.half(),
        },
        private_path,
    )
    os.chmod(private_path, 0o600)
    report = {
        "schema_version": SCHEMA,
        "status": (
            "completed_validation88_generalization_gate_pass"
            if passed
            else "completed_validation88_generalization_gate_fail"
        ),
        "scope": "one_fixed_post_checkpoint_validation88_generalization_evaluation",
        "patient_counts": {"train_for_RRR_fit": 409, "validation_evaluated": 88, "test_loaded": 0},
        "generalization_gate_passed": passed,
        "candidate_vs_coarse_RRR": comparisons,
        "T0_occupancy_alignment_candidate_vs_patient_shuffle": alignment,
        "criterion_results": criteria,
        "maximum_outside_core_support_absolute": float(outside),
        "coarse_RRR_anchor_refit": {
            "shrinkage_alpha": anchor_fit.shrinkage_alpha,
            "selected_rank": observed_rrr["selected_rank"],
            "selected_alpha": observed_rrr["selected_alpha"],
            "matched_frozen_checkpoint_fit": True,
        },
        "background_numerical_noninferiority_margin": (
            recovery_config.background_numerical_noninferiority_margin
        ),
        "checkpoint_frozen_before_validation_loading": True,
        "technical_retry_disclosure": {
            "prior_failed_slurm_job_id": prior_failed_job_id,
            "prior_attempt_loaded_validation88": True,
            "prior_attempt_produced_no_aggregate_report": True,
            "failure_stage": "post_prediction_pre_bootstrap_input_contract_check",
            "failure_reason": (
                "empty_fractional_patient_region_MAE_was_represented_as_NaN_"
                "but_the_outer_comparison_contract_required_all_finite_values"
            ),
            "checkpoint_changed_after_prior_attempt": False,
            "model_or_gate_changed_after_prior_attempt": False,
            "evaluator_only_change": (
                "paired_complete_case_filtering_per_region_allows_NaN_"
                "while_infinity_remains_prohibited"
            ),
        },
        "validation_previously_used_by_predecessor_programs": True,
        "independent_validation_claim_permitted": False,
        "confirmatory_claim_permitted": False,
        "eligible_for_test_evaluation": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "final_fit_report_sha256": _sha(paths["final_report"]),
            "checkpoint_sha256": _sha(paths["checkpoint"]),
            "validation_parent_report_sha256": _sha(paths["validation_parent_report"]),
            "prior_failed_validation_stderr_sha256": _sha(
                paths["prior_failed_stderr"]
            ),
            "train_feature_index_sha256": _sha(paths["train_feature_index"]),
            "train_feature_payload_merkle_sha256": train_merkle,
            "train_T0_occupancy_digest": train_occ_digest,
            "validation_feature_index_sha256": _sha(val_index),
            "validation_feature_payload_merkle_sha256": val_merkle,
            "validation_T0_occupancy_digest": val_occ_digest,
            "private_validation_measurements_sha256": _sha(private_path),
            "validation_source_sha256": _sha(Path(__file__)),
        },
    }
    report_path = output_staging / "aggregate_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.chmod(report_path, 0o600)
    os.replace(private_staging, paths["private_output"])
    os.replace(output_staging, paths["output"])
    print(
        json.dumps(
            {
                "status": report["status"],
                "generalization_gate_passed": passed,
                "validation_patients": 88,
                "checkpoint_frozen_before_validation_loading": True,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--v4p3p1-config", required=True)
    parser.add_argument("--final-fit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-parent-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-feature-root", required=True)
    parser.add_argument("--train-feature-index", required=True)
    parser.add_argument("--validation-feature-root", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--prior-failed-validation-job-id", required=True)
    parser.add_argument("--prior-failed-validation-stderr", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["PRIVATE_SCHEMA", "SCHEMA", "run"]
