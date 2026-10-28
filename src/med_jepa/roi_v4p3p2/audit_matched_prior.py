"""Post-hoc recipient-matched population-prior audit for frozen v4.3.2.

For each validation recipient, this audit holds the coarse-RRR anchor, T0
core/ring support, timing, protocol context, evaluation masks, and target
fixed.  It marginalizes only T0/T1 imaging content over all 409 training
patients, averages frozen raw residual predictions before scoring, and applies
the result through the recipient's exact core gate.  It never constructs an
optimizer, changes the checkpoint, or loads test90.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary
from med_jepa.roi_v4p2.train import _region_errors
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA
from med_jepa.roi_v4p3p2.audit_factorial import SCHEMA as FACTORIAL_SCHEMA
from med_jepa.roi_v4p3p2.audit_factorial import STATUS as FACTORIAL_STATUS
from med_jepa.roi_v4p3p2.audit_factorial import (
    _evaluation_masks,
    _json,
    _paired,
    _sha,
    _tensor_sha,
    _write_json,
)
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.data import load_v432_validation_release
from med_jepa.roi_v4p3p2.finalize import SCHEMA as FINAL_SCHEMA
from med_jepa.roi_v4p3p2.validate import (
    PRIVATE_SCHEMA as VALIDATION_PRIVATE_SCHEMA,
)
from med_jepa.roi_v4p3p2.validate import SCHEMA as VALIDATION_SCHEMA


SCHEMA = "med-jepa-roi-v4p3p2-posthoc-recipient-matched-prior-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3p2-private-recipient-matched-prior-audit-v1"
STATUS = "completed_posthoc_validation88_recipient_matched_prior_audit"
MATCHED_PRIOR = "MATCHED_PRIOR_train409_content_marginalized"
CC = "CC_frozen_recipient_content"
DEFAULT_DONOR_CHUNK_SIZE = 32


def _maximum_outside_core(prediction: Tensor, anchor: Tensor, core: Tensor) -> float:
    if prediction.shape != anchor.shape or core.shape != (
        prediction.shape[0],
        1,
        7,
        7,
        7,
    ):
        raise ValueError("matched-prior gate tensors do not align")
    outside = core == 0
    if not bool(outside.any()):
        return 0.0
    residual = prediction - anchor
    return float(residual[outside.expand_as(residual)].abs().max())


@torch.no_grad()
def _recipient_matched_population_prediction(
    model: Any,
    donor_source: Tensor,
    donor_history: Tensor,
    recipient_times: Tensor,
    recipient_action: Tensor,
    recipient_core: Tensor,
    recipient_ring: Tensor,
    recipient_anchors: Tensor,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
    donor_chunk_size: int = DEFAULT_DONOR_CHUNK_SIZE,
) -> tuple[Tensor, Tensor, float]:
    """Marginalize donor content under each recipient's fixed support/context.

    The function deliberately receives no target tensor.  For each recipient,
    every training donor is evaluated with recipient timing, action, core/ring,
    anchor, and mask realization.  Raw residuals are averaged over donors and
    masks before the recipient core gate is applied.
    """

    donors = int(donor_source.shape[0])
    recipients = int(recipient_anchors.shape[0])
    channels = int(recipient_anchors.shape[1])
    grid = (7, 7, 7)
    if (
        donors < 2
        or recipients < 1
        or donor_source.shape != donor_history.shape
        or donor_source.ndim != 5
        or tuple(donor_source.shape[2:]) != grid
        or recipient_anchors.ndim != 5
        or tuple(recipient_anchors.shape[2:]) != grid
        or donor_source.shape[1] != channels
        or recipient_times.shape != (recipients, 3)
        or recipient_action.shape != (recipients, 2)
        or recipient_core.shape != (recipients, 1, *grid)
        or recipient_ring.shape != recipient_core.shape
        or not 1 <= int(donor_chunk_size) <= donors
        or not torch.isfinite(donor_source).all()
        or not torch.isfinite(donor_history).all()
        or not torch.isfinite(recipient_times).all()
        or not torch.isfinite(recipient_action).all()
        or not torch.isfinite(recipient_core).all()
        or not torch.isfinite(recipient_ring).all()
        or not torch.isfinite(recipient_anchors).all()
    ):
        raise ValueError("recipient-matched population inputs violate the contract")

    model.eval()
    predictions = torch.empty_like(recipient_anchors, device="cpu", dtype=torch.float32)
    mean_raw_residuals = torch.empty_like(
        recipient_anchors, device="cpu", dtype=torch.float32
    )
    mask_draws = (
        len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction
    )
    recipient_batch_size = int(config.batch_size)

    for recipient_start in range(0, recipients, recipient_batch_size):
        recipient_stop = min(recipient_start + recipient_batch_size, recipients)
        batch = recipient_stop - recipient_start
        anchors = recipient_anchors[recipient_start:recipient_stop].to(device)
        times = recipient_times[recipient_start:recipient_stop].to(device)
        action = recipient_action[recipient_start:recipient_stop].to(device)
        core = recipient_core[recipient_start:recipient_stop].to(device)
        ring = recipient_ring[recipient_start:recipient_stop].to(device)
        masks = _evaluation_masks(
            batch,
            recipient_start,
            config,
            seed=int(seed),
            device=device,
        )
        if len(masks) != mask_draws:
            raise RuntimeError("recipient mask schedule changed")
        raw_sum = torch.zeros_like(anchors, dtype=torch.float32)

        for donor_start in range(0, donors, int(donor_chunk_size)):
            donor_stop = min(donor_start + int(donor_chunk_size), donors)
            donor_batch = donor_stop - donor_start
            source_chunk = donor_source[donor_start:donor_stop].to(device)
            history_chunk = donor_history[donor_start:donor_stop].to(device)
            source = (
                source_chunk[None]
                .expand(batch, donor_batch, -1, -1, -1, -1)
                .reshape(batch * donor_batch, channels, *grid)
            )
            history = (
                history_chunk[None]
                .expand(batch, donor_batch, -1, -1, -1, -1)
                .reshape(batch * donor_batch, channels, *grid)
            )
            repeated_anchor = (
                anchors[:, None]
                .expand(batch, donor_batch, -1, -1, -1, -1)
                .reshape(batch * donor_batch, channels, *grid)
            )
            repeated_core = (
                core[:, None]
                .expand(batch, donor_batch, -1, -1, -1, -1)
                .reshape(batch * donor_batch, 1, *grid)
            )
            repeated_ring = (
                ring[:, None]
                .expand(batch, donor_batch, -1, -1, -1, -1)
                .reshape(batch * donor_batch, 1, *grid)
            )
            source_time = times[:, 1, None].expand(batch, donor_batch).reshape(-1)
            target_time = times[:, 2, None].expand(batch, donor_batch).reshape(-1)
            repeated_action = (
                action[:, None]
                .expand(batch, donor_batch, -1)
                .reshape(batch * donor_batch, 2)
            )

            for mask in masks:
                repeated_mask = (
                    mask[:, None]
                    .expand(batch, donor_batch, -1, -1, -1, -1)
                    .reshape(batch * donor_batch, 1, *grid)
                )
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=device.type == "cuda",
                ):
                    output = model.step(
                        source,
                        repeated_anchor,
                        source_time,
                        target_time,
                        repeated_action,
                        repeated_core,
                        repeated_ring,
                        history_delta=history,
                        context_mask=repeated_mask,
                    )
                raw = output.raw_core_residual.float().reshape(
                    batch, donor_batch, channels, *grid
                )
                raw_sum.add_(raw.sum(dim=1))

        mean_raw = raw_sum.div(float(donors * mask_draws))
        prediction = anchors.float() + core.float() * mean_raw
        mean_raw_residuals[recipient_start:recipient_stop] = mean_raw.cpu()
        predictions[recipient_start:recipient_stop] = prediction.cpu()

    outside = _maximum_outside_core(
        predictions,
        recipient_anchors.float().cpu(),
        recipient_core.float().cpu(),
    )
    if not torch.isfinite(predictions).all() or not torch.isfinite(
        mean_raw_residuals
    ).all():
        raise RuntimeError("recipient-matched population prediction is malformed")
    return predictions, mean_raw_residuals, outside


def _region_summaries(errors: Mapping[str, Tensor]) -> dict[str, Any]:
    return {
        arm: {
            region: _summary(values[:, index])
            for index, region in enumerate(REGIONS)
        }
        for arm, values in errors.items()
    }


def _original_validation_consistent(
    validation_report: Mapping[str, Any],
    candidate_errors: Tensor,
    anchor_errors: Tensor,
) -> bool:
    comparisons = validation_report.get("candidate_vs_coarse_RRR")
    if not isinstance(comparisons, Mapping):
        raise TypeError("validation report lacks regional comparisons")
    for index, region in enumerate(REGIONS):
        eligible = torch.isfinite(candidate_errors[:, index]) & torch.isfinite(
            anchor_errors[:, index]
        )
        value = comparisons.get(region)
        if int(eligible.sum()) < 2 or not isinstance(value, Mapping):
            raise ValueError("validation reproduction region is unavailable")
        if (
            abs(
                float(candidate_errors[eligible, index].mean())
                - float(value.get("candidate_mean"))
            )
            > 5e-4
            or abs(
                float(anchor_errors[eligible, index].mean())
                - float(value.get("reference_mean"))
            )
            > 5e-4
        ):
            raise ValueError("frozen validation predictions changed")
    return True


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("matched-prior audit requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "recovery_config": args.recovery_config,
            "v431_config": args.v4p3p1_config,
            "final_report": args.final_fit_report,
            "checkpoint": args.checkpoint,
            "validation_report": args.validation_report,
            "validation_private": args.validation_private_measurements,
            "factorial_report": args.factorial_report,
            "manifest": args.manifest,
            "train_feature_root": args.train_feature_root,
            "train_feature_index": args.train_feature_index,
            "validation_feature_root": args.validation_feature_root,
            "mask_root": args.context_mask_release,
            "mask_report": args.context_mask_report,
            "mask_index": args.context_mask_index,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    for name in (
        "recovery_config",
        "v431_config",
        "final_report",
        "checkpoint",
        "validation_report",
        "validation_private",
        "factorial_report",
        "manifest",
        "train_feature_index",
        "mask_report",
        "mask_index",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"matched-prior input unavailable: {name}")
    for name in ("train_feature_root", "validation_feature_root", "mask_root"):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"matched-prior release unavailable: {name}")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable matched-prior output exists")

    recovery_config = V432RecoveryConfig.load(paths["recovery_config"])
    v431_config = V431TrainingConfig.load(paths["v431_config"])
    final_report = _json(paths["final_report"])
    validation_report = _json(paths["validation_report"])
    factorial_report = _json(paths["factorial_report"])
    validation_evidence = validation_report.get("evidence", {})
    final_evidence = final_report.get("evidence", {})
    factorial_evidence = factorial_report.get("evidence", {})
    model_source = Path(__file__).parents[1] / "roi_v4p3p1" / "model.py"
    if (
        final_report.get("schema_version") != FINAL_SCHEMA
        or final_report.get("status")
        != "completed_adaptive_recovery_train409_final_fit"
        or final_report.get("checkpoint_created") is not True
        or final_report.get("test_split_loaded") is not False
        or validation_report.get("schema_version") != VALIDATION_SCHEMA
        or validation_report.get("status")
        not in {
            "completed_validation88_generalization_gate_pass",
            "completed_validation88_generalization_gate_fail",
        }
        or validation_report.get("patient_counts", {}).get("validation_evaluated")
        != 88
        or validation_report.get("test_split_loaded") is not False
        or validation_report.get("confirmatory_claim_permitted") is not False
        or factorial_report.get("schema_version") != FACTORIAL_SCHEMA
        or factorial_report.get("status") != FACTORIAL_STATUS
        or factorial_report.get("test_split_loaded") is not False
        or not isinstance(validation_evidence, Mapping)
        or not isinstance(final_evidence, Mapping)
        or not isinstance(factorial_evidence, Mapping)
        or validation_evidence.get("final_fit_report_sha256")
        != _sha(paths["final_report"])
        or validation_evidence.get("checkpoint_sha256") != _sha(paths["checkpoint"])
        or validation_evidence.get("private_validation_measurements_sha256")
        != _sha(paths["validation_private"])
        or final_evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or final_evidence.get("v4p3p1_model_source_sha256") != _sha(model_source)
        or factorial_evidence.get("checkpoint_sha256") != _sha(paths["checkpoint"])
        or factorial_evidence.get("validation_report_sha256")
        != _sha(paths["validation_report"])
    ):
        raise ValueError("matched-prior evidence chain is invalid")

    checkpoint = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, Mapping)
        or checkpoint.get("schema_version") != FINAL_SCHEMA
        or checkpoint.get("final_epoch_count") != 18
        or not isinstance(checkpoint.get("model"), Mapping)
        or checkpoint.get("v4p3p1_model_config") != asdict(v431_config)
        or checkpoint.get("recovery_config") != asdict(recovery_config)
        or checkpoint.get("validation_88_loaded") is not False
        or checkpoint.get("test_split_loaded") is not False
    ):
        raise ValueError("matched-prior checkpoint payload is invalid")

    private = torch.load(
        paths["validation_private"], map_location="cpu", weights_only=True
    )
    if (
        not isinstance(private, Mapping)
        or private.get("schema_version") != VALIDATION_PRIVATE_SCHEMA
    ):
        raise ValueError("private validation measurements have the wrong schema")
    patient_ids = private.get("patient_ids")
    original_candidate = private.get("candidate_prediction")
    anchors = private.get("coarse_RRR_prediction")
    if (
        not isinstance(patient_ids, list)
        or len(patient_ids) != 88
        or len(set(patient_ids)) != 88
        or not all(isinstance(value, str) and value for value in patient_ids)
        or not isinstance(original_candidate, Tensor)
        or not isinstance(anchors, Tensor)
        or original_candidate.shape != (88, 1024, 7, 7, 7)
        or anchors.shape != original_candidate.shape
        or not torch.isfinite(original_candidate).all()
        or not torch.isfinite(anchors).all()
    ):
        raise ValueError("private validation prediction payload is malformed")
    original_candidate = original_candidate.float()
    anchors = anchors.float()

    train, train_weights, train_occupancy, train_merkle, train_occ_digest = (
        load_v4p3_train409_release(
            paths["manifest"],
            paths["train_feature_root"],
            paths["train_feature_index"],
            expected_feature_index_sha256=str(
                validation_evidence["train_feature_index_sha256"]
            ),
            context_mask_root=paths["mask_root"],
            context_mask_report=paths["mask_report"],
            context_mask_index=paths["mask_index"],
        )
    )
    val, val_weights, val_occupancy, val_merkle, val_occ_digest, val_index = (
        load_v432_validation_release(
            manifest_path=paths["manifest"],
            validation_feature_root=paths["validation_feature_root"],
            expected_validation_index_sha256=str(
                validation_evidence["validation_feature_index_sha256"]
            ),
            expected_validation_payload_merkle_sha256=str(
                validation_evidence["validation_feature_payload_merkle_sha256"]
            ),
            context_mask_root=paths["mask_root"],
            context_mask_report=paths["mask_report"],
            context_mask_index=paths["mask_index"],
        )
    )
    if (
        len(train) != 409
        or len(train_weights) != 409
        or len(val) != 88
        or [item.patient_id for item in val] != patient_ids
        or _sha(paths["train_feature_index"])
        != validation_evidence.get("train_feature_index_sha256")
        or train_merkle
        != validation_evidence.get("train_feature_payload_merkle_sha256")
        or train_occ_digest != validation_evidence.get("train_T0_occupancy_digest")
        or _sha(val_index) != validation_evidence.get("validation_feature_index_sha256")
        or val_merkle
        != validation_evidence.get("validation_feature_payload_merkle_sha256")
        or val_occ_digest != validation_evidence.get("validation_T0_occupancy_digest")
    ):
        raise ValueError("matched-prior feature release changed after validation")

    donor_source = torch.stack([item.features[1] for item in train]).float()
    donor_history = torch.stack(
        [item.features[1] - item.features[0] for item in train]
    ).float()
    recipient_times = torch.stack([item.times_days for item in val]).float()
    recipient_action = torch.stack([item.action_13 for item in val]).float()
    device = torch.device("cuda")
    model = CoarseRRRAnchoredCoreResidualJEPA(
        v431_config.hidden_channels, v431_config.conditioned_blocks
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    mask_seed = recovery_config.bootstrap_seed + 7_000_001
    matched_prediction, matched_mean_raw, maximum_outside = (
        _recipient_matched_population_prediction(
            model,
            donor_source,
            donor_history,
            recipient_times,
            recipient_action,
            val_occupancy[:, 0, None].float(),
            val_occupancy[:, 1, None].float(),
            anchors,
            v431_config,
            seed=mask_seed,
            device=device,
            donor_chunk_size=int(args.donor_chunk_size),
        )
    )

    # Targets enter only after the matched population prediction is complete.
    target = torch.stack([item.features[2] for item in val]).float()
    errors = {
        CC: _region_errors(original_candidate, target, val_weights),
        MATCHED_PRIOR: _region_errors(matched_prediction, target, val_weights),
        "coarse_RRR_anchor": _region_errors(anchors, target, val_weights),
    }
    _original_validation_consistent(
        validation_report,
        errors[CC],
        errors["coarse_RRR_anchor"],
    )
    bootstrap_seed = recovery_config.bootstrap_seed + 23_000_011
    paired = {
        region: _paired(
            errors[CC][:, index],
            errors[MATCHED_PRIOR][:, index],
            patient_ids,
            seed=bootstrap_seed + index,
            replicates=recovery_config.bootstrap_replicates,
        )
        for index, region in enumerate(REGIONS)
    }
    primary = paired["t0_core"]
    core_content = factorial_report.get("factorial_patient_level_contrasts", {}).get(
        "content_penalty_at_correct_support_SC_minus_CC", {}
    ).get("t0_core")
    core_support = factorial_report.get("factorial_patient_level_contrasts", {}).get(
        "support_penalty_at_correct_content_CS_minus_CC", {}
    ).get("t0_core")
    if not isinstance(core_content, Mapping) or not isinstance(core_support, Mapping):
        raise ValueError("bound factorial report lacks primary core contrasts")
    matched_positive = float(primary["ci_low"]) > 0.0
    content_positive = float(core_content["ci_low"]) > 0.0
    support_positive = float(core_support["ci_low"]) > 0.0

    paths["output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths["private_output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = (
        paths["private_output"].parent / f".{paths['private_output'].name}.staging"
    )
    output_staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_path = private_staging / "validation88_matched_prior_measurements.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": patient_ids,
            "arm_region_errors": errors,
            "matched_prior_prediction": matched_prediction,
            "matched_prior_mean_raw_residual": matched_mean_raw,
            "maximum_outside_recipient_core_absolute": maximum_outside,
        },
        private_path,
    )
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "scope": "posthoc_descriptive_validation88_recipient_matched_prior_audit",
        "scientific_question": (
            "whether_recipient_T0_T1_content_beats_training_population_content_"
            "marginalized_under_fixed_recipient_support_and_context"
        ),
        "patient_counts": {
            "train_content_donors": 409,
            "validation_recipients": 88,
            "test_loaded": 0,
        },
        "arms": {
            CC: "frozen_candidate_with_recipient_content_support_and_context",
            MATCHED_PRIOR: (
                "mean_train409_content_prediction_with_recipient_support_context_"
                "anchor_and_masks"
            ),
        },
        "fixed_recipient_variables": [
            "coarse_RRR_anchor",
            "T0_fractional_core_and_ring_support",
            "source_and_target_times",
            "T1_to_T3_action",
            "context_mask_realizations",
            "T3_target_during_scoring_only",
            "evaluation_regions",
            "patient_bootstrap_identity",
        ],
        "marginalized_variable": "T1_embedding_and_T1_minus_T0_history",
        "construction_contract": {
            "content_donor_split": "train409_only",
            "content_donors_per_recipient": 409,
            "recipient_support_held_fixed": True,
            "recipient_time_and_protocol_held_fixed": True,
            "recipient_anchor_held_fixed": True,
            "same_recipient_evaluation_masks_as_frozen_validation": True,
            "mask_draws_per_recipient_donor": (
                len(v431_config.mask_fractions)
                * v431_config.evaluation_mask_draws_per_fraction
            ),
            "raw_residuals_averaged_before_core_gating_and_error_scoring": True,
            "validation_target_used_to_construct_prior": False,
            "donor_chunk_size": int(args.donor_chunk_size),
        },
        "arm_region_MAE": _region_summaries(errors),
        "CC_vs_recipient_matched_prior": paired,
        "bound_factorial_core_contrasts": {
            "patient_SC_minus_CC": dict(core_content),
            "place_CS_minus_CC": dict(core_support),
        },
        "diagnostic_criterion_results": {
            "core_patient_penalty_CI_lower_positive": content_positive,
            "core_place_penalty_CI_lower_positive": support_positive,
            "core_matched_prior_penalty_CI_lower_positive": matched_positive,
            "patient_specific_imaging_content_supported_under_matched_prior": (
                content_positive and matched_positive
            ),
            "strict_P3_signature_supported": (
                content_positive and support_positive and matched_positive
            ),
            "matched_prior_residual_exactly_zero_outside_recipient_core": (
                maximum_outside == 0.0
            ),
        },
        "interpretation_contract": {
            "positive_improvement": "CC_has_lower_MAE_than_recipient_matched_prior",
            "primary_region": "t0_core",
            "stronger_patient_claim_requires": (
                "core_SC_minus_CC_CI_low_gt_0_and_core_matched_prior_minus_CC_"
                "CI_low_gt_0"
            ),
            "claim_scope": (
                "predictive_value_of_recipient_imaging_history_within_provided_"
                "support_not_biological_identity_or_causal_treatment_response"
            ),
        },
        "maximum_outside_recipient_core_absolute": maximum_outside,
        "formal_gate_declared": False,
        "checkpoint_modified": False,
        "optimizer_constructed": False,
        "retraining_performed": False,
        "validation_previously_observed": True,
        "independent_validation_claim_permitted": False,
        "confirmatory_claim_permitted": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "recovery_config_sha256": _sha(paths["recovery_config"]),
            "v4p3p1_config_sha256": _sha(paths["v431_config"]),
            "final_fit_report_sha256": _sha(paths["final_report"]),
            "checkpoint_sha256": _sha(paths["checkpoint"]),
            "validation_report_sha256": _sha(paths["validation_report"]),
            "validation_private_measurements_sha256": _sha(
                paths["validation_private"]
            ),
            "factorial_report_sha256": _sha(paths["factorial_report"]),
            "manifest_sha256": _sha(paths["manifest"]),
            "train_feature_index_sha256": _sha(paths["train_feature_index"]),
            "train_feature_payload_merkle_sha256": train_merkle,
            "train_T0_occupancy_digest": train_occ_digest,
            "validation_feature_index_sha256": _sha(val_index),
            "validation_feature_payload_merkle_sha256": val_merkle,
            "validation_T0_occupancy_digest": val_occ_digest,
            "context_mask_report_sha256": _sha(paths["mask_report"]),
            "context_mask_index_sha256": _sha(paths["mask_index"]),
            "v4p3p1_model_source_sha256": _sha(model_source),
            "matched_prior_audit_source_sha256": _sha(Path(__file__)),
            "matched_prior_prediction_sha256": _tensor_sha(matched_prediction),
            "private_matched_prior_measurements_sha256": _sha(private_path),
        },
    }
    _write_json(output_staging / "aggregate_report.json", report)
    os.replace(private_staging, paths["private_output"])
    os.replace(output_staging, paths["output"])
    print(
        json.dumps(
            {
                "status": STATUS,
                "validation_recipients": 88,
                "training_content_donors_per_recipient": 409,
                "retraining_performed": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovery-config", required=True)
    parser.add_argument("--v4p3p1-config", required=True)
    parser.add_argument("--final-fit-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-report", required=True)
    parser.add_argument("--validation-private-measurements", required=True)
    parser.add_argument("--factorial-report", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-feature-root", required=True)
    parser.add_argument("--train-feature-index", required=True)
    parser.add_argument("--validation-feature-root", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument(
        "--donor-chunk-size", type=int, default=DEFAULT_DONOR_CHUNK_SIZE
    )
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CC",
    "DEFAULT_DONOR_CHUNK_SIZE",
    "MATCHED_PRIOR",
    "PRIVATE_SCHEMA",
    "SCHEMA",
    "STATUS",
    "_recipient_matched_population_prediction",
    "run",
]
