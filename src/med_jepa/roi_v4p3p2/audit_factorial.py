"""Post-hoc patient-content x spatial-support audit for frozen v4.3.2.

The audit runs counterfactual forward passes from the immutable train409
checkpoint on the historically reused validation88 split.  It changes neither
the checkpoint nor the original validation products, never constructs an
optimizer, and exposes no test90 input.  Four cells independently permute
patient imaging content and T0 spatial support; a fifth train409-only population
residual arm tests whether an unpersonalized neural correction is sufficient.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from med_jepa.frozen_audit.statistics import paired_patient_bootstrap
from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary
from med_jepa.roi_v4.data import PatientStates
from med_jepa.roi_v4p2.masking import uniform_block_mask
from med_jepa.roi_v4p2.train import _record_features, _region_errors
from med_jepa.roi_v4p3.data import load_v4p3_train409_release
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA
from med_jepa.roi_v4p3p1.train import _derangement
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.data import load_v432_validation_release
from med_jepa.roi_v4p3p2.finalize import SCHEMA as FINAL_SCHEMA
from med_jepa.roi_v4p3p2.validate import (
    PRIVATE_SCHEMA as VALIDATION_PRIVATE_SCHEMA,
)
from med_jepa.roi_v4p3p2.validate import (
    SCHEMA as VALIDATION_SCHEMA,
)

SCHEMA = "med-jepa-roi-v4p3p2-posthoc-content-support-factorial-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p3p2-private-content-support-factorial-audit-v1"
STATUS = "completed_posthoc_validation88_content_support_factorial_audit"
DERANGEMENTS = 32
BOOTSTRAP_REPLICATES = 10_000

CC = "CC_correct_content_correct_support"
CS = "CS_correct_content_shuffled_support"
SC = "SC_shuffled_content_correct_support"
SS = "SS_shuffled_content_shuffled_support"
POP = "POP_train409_mean_raw_residual_correct_support"
ANCHOR = "coarse_RRR_anchor"


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("expected a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    rendered = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(rendered, encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _evaluation_masks(
    batch_size: int,
    start: int,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> list[Tensor]:
    masks: list[Tensor] = []
    for fraction_index, fraction in enumerate(config.mask_fractions):
        for draw in range(config.evaluation_mask_draws_per_fraction):
            masks.append(
                uniform_block_mask(
                    batch_size,
                    (7, 7, 7),
                    seed=(
                        int(seed)
                        + int(start) * 10_007
                        + fraction_index * 1_009
                        + draw * 97
                    ),
                    masked_fraction=fraction,
                ).to(device)
            )
    if len(masks) != (
        len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction
    ):
        raise RuntimeError("factorial audit mask schedule is incomplete")
    return masks


def _derangements(
    patients: int,
    *,
    seed: int,
    draws: int = DERANGEMENTS,
) -> tuple[Tensor, list[str]]:
    if patients < 2 or draws < 1:
        raise ValueError("factorial audit requires patients and derangements")
    indices = torch.arange(patients)
    rows: list[Tensor] = []
    digests: list[str] = []
    for draw in range(draws):
        permutation = _derangement(
            indices,
            seed=int(seed) + draw * 1_000_003,
        )
        if bool(torch.any(permutation == indices)):
            raise RuntimeError("factorial donor assignment contains a self-match")
        rows.append(permutation)
        digests.append(
            hashlib.sha256(permutation.numpy().tobytes(order="C")).hexdigest()
        )
    if len(set(digests)) != len(digests):
        raise RuntimeError("factorial donor derangements are not unique")
    return torch.stack(rows), digests


def _arm_inputs(
    recipient_source: Tensor,
    recipient_history: Tensor,
    recipient_core: Tensor,
    recipient_ring: Tensor,
    donor_source: Tensor,
    donor_history: Tensor,
    donor_core: Tensor,
    donor_ring: Tensor,
) -> dict[str, tuple[Tensor, Tensor, Tensor, Tensor]]:
    """Return (source, history, core, ring) for the three shuffled cells."""

    recipient_shapes = (
        recipient_source.shape,
        recipient_history.shape,
        recipient_core.shape,
        recipient_ring.shape,
    )
    donor_shapes = (
        donor_source.shape,
        donor_history.shape,
        donor_core.shape,
        donor_ring.shape,
    )
    if recipient_shapes != donor_shapes:
        raise ValueError("recipient and donor factorial tensors do not align")
    return {
        CS: (recipient_source, recipient_history, donor_core, donor_ring),
        SC: (donor_source, donor_history, recipient_core, recipient_ring),
        SS: (donor_source, donor_history, donor_core, donor_ring),
    }


def _maximum_outside_gate(output: Any) -> float:
    outside = output.core_gate == 0
    if not bool(outside.any()):
        return 0.0
    return float(
        output.gated_core_residual[outside.expand_as(output.gated_core_residual)]
        .abs()
        .max()
    )


@torch.no_grad()
def _population_average_raw_residual(
    model: CoarseRRRAnchoredCoreResidualJEPA,
    records: Sequence[PatientStates],
    input_occupancy: Tensor,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
) -> Tensor:
    """Average frozen neural raw residuals over train409 without using targets."""

    if len(records) != len(input_occupancy) or len(records) < 2:
        raise ValueError("population residual inputs do not align")
    model.eval()
    total: Tensor | None = None
    draws = len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction
    for start in range(0, len(records), config.batch_size):
        selected = records[start : start + config.batch_size]
        source = torch.stack([item.features[1] for item in selected]).to(device)
        t0 = torch.stack([item.features[0] for item in selected]).to(device)
        times = torch.stack([item.times_days for item in selected]).to(device)
        action = torch.stack([item.action_13 for item in selected]).to(device)
        core = input_occupancy[start : start + len(selected), 0, None].to(device)
        ring = input_occupancy[start : start + len(selected), 1, None].to(device)
        raw_sum = torch.zeros_like(source, dtype=torch.float32)
        for mask in _evaluation_masks(
            len(selected), start, config, seed=seed, device=device
        ):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.step(
                    source,
                    torch.zeros_like(source),
                    times[:, 1],
                    times[:, 2],
                    action,
                    core,
                    ring,
                    history_delta=source - t0,
                    context_mask=mask,
                )
            raw_sum.add_(output.raw_core_residual.float())
        batch_mean = raw_sum.div(float(draws)).cpu()
        if total is None:
            total = torch.zeros_like(batch_mean[0], dtype=torch.float64)
        total.add_(batch_mean.double().sum(0))
    if total is None:
        raise RuntimeError("population residual did not evaluate any patient")
    result = total.div(float(len(records))).float()
    if result.shape != (1024, 7, 7, 7) or not torch.isfinite(result).all():
        raise RuntimeError("population residual template is malformed")
    return result


def _population_prediction(anchor: Tensor, template: Tensor, core: Tensor) -> Tensor:
    if (
        anchor.ndim != 5
        or template.shape != anchor.shape[1:]
        or core.shape != (anchor.shape[0], 1, 7, 7, 7)
    ):
        raise ValueError("population residual application violates the audit contract")
    return anchor.float() + template.float()[None] * core.float()


@torch.no_grad()
def _factorial_region_errors(
    model: CoarseRRRAnchoredCoreResidualJEPA,
    records: Sequence[PatientStates],
    input_occupancy: Tensor,
    anchors: Tensor,
    evaluation_weights: Tensor,
    config: V431TrainingConfig,
    *,
    seed: int,
    device: torch.device,
    derangements: int = DERANGEMENTS,
    derangement_chunk_size: int = DERANGEMENTS,
) -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    Tensor,
    list[str],
    dict[str, float],
]:
    """Evaluate CC/CS/SC/SS and retain per-donor errors, not predictions."""

    patients = len(records)
    if (
        patients < 2
        or input_occupancy.shape != (patients, 3, 7, 7, 7)
        or anchors.shape != (patients, 1024, 7, 7, 7)
        or evaluation_weights.shape != (patients, len(REGIONS), 7, 7, 7)
        or derangements != DERANGEMENTS
        or not 1 <= derangement_chunk_size <= DERANGEMENTS
    ):
        raise ValueError("factorial evaluation tensors violate the frozen contract")
    model.eval()
    features = _record_features(records).float()
    times = torch.stack([item.times_days for item in records]).float()
    actions = torch.stack([item.action_13 for item in records]).float()
    targets = features[:, 2]
    permutations, digests = _derangements(
        patients,
        seed=int(seed) + 91_003,
        draws=derangements,
    )
    correct_errors = torch.full((patients, len(REGIONS)), float("nan"))
    per_draw = {
        arm: torch.full((derangements, patients, len(REGIONS)), float("nan"))
        for arm in (CS, SC, SS)
    }
    maximum_outside = {arm: 0.0 for arm in (CC, CS, SC, SS)}
    draws = len(config.mask_fractions) * config.evaluation_mask_draws_per_fraction

    for start in range(0, patients, config.batch_size):
        stop = min(start + config.batch_size, patients)
        batch = stop - start
        source = features[start:stop, 1].to(device)
        history = (features[start:stop, 1] - features[start:stop, 0]).to(device)
        batch_times = times[start:stop].to(device)
        action = actions[start:stop].to(device)
        core = input_occupancy[start:stop, 0, None].to(device)
        ring = input_occupancy[start:stop, 1, None].to(device)
        anchor = anchors[start:stop].to(device)
        masks = _evaluation_masks(batch, start, config, seed=seed, device=device)

        correct_sum = torch.zeros_like(source, dtype=torch.float32)
        for mask in masks:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model.step(
                    source,
                    anchor,
                    batch_times[:, 1],
                    batch_times[:, 2],
                    action,
                    core,
                    ring,
                    history_delta=history,
                    context_mask=mask,
                )
            correct_sum.add_(output.predicted.float())
            maximum_outside[CC] = max(
                maximum_outside[CC], _maximum_outside_gate(output)
            )
        correct_prediction = correct_sum.div(float(draws)).cpu()
        correct_errors[start:stop] = _region_errors(
            correct_prediction,
            targets[start:stop],
            evaluation_weights[start:stop],
        )

        for draw_start in range(0, derangements, derangement_chunk_size):
            draw_stop = min(draw_start + derangement_chunk_size, derangements)
            chunk = draw_stop - draw_start
            donor_indices = permutations[draw_start:draw_stop, start:stop]
            donor_flat = donor_indices.reshape(-1)
            recipient_source = source.repeat((chunk, 1, 1, 1, 1))
            recipient_history = history.repeat((chunk, 1, 1, 1, 1))
            recipient_core = core.repeat((chunk, 1, 1, 1, 1))
            recipient_ring = ring.repeat((chunk, 1, 1, 1, 1))
            donor_source = features[donor_flat, 1].to(device)
            donor_history = (features[donor_flat, 1] - features[donor_flat, 0]).to(
                device
            )
            donor_core = input_occupancy[donor_flat, 0, None].to(device)
            donor_ring = input_occupancy[donor_flat, 1, None].to(device)
            arm_inputs = _arm_inputs(
                recipient_source,
                recipient_history,
                recipient_core,
                recipient_ring,
                donor_source,
                donor_history,
                donor_core,
                donor_ring,
            )
            repeated_anchor = anchor.repeat((chunk, 1, 1, 1, 1))
            repeated_source_time = batch_times[:, 1].repeat(chunk)
            repeated_target_time = batch_times[:, 2].repeat(chunk)
            repeated_action = action.repeat((chunk, 1))
            sums = {
                arm: torch.zeros_like(recipient_source, dtype=torch.float32)
                for arm in (CS, SC, SS)
            }
            for mask in masks:
                repeated_mask = mask.repeat((chunk, 1, 1, 1, 1))
                for arm in (CS, SC, SS):
                    arm_source, arm_history, arm_core, arm_ring = arm_inputs[arm]
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = model.step(
                            arm_source,
                            repeated_anchor,
                            repeated_source_time,
                            repeated_target_time,
                            repeated_action,
                            arm_core,
                            arm_ring,
                            history_delta=arm_history,
                            context_mask=repeated_mask,
                        )
                    sums[arm].add_(output.predicted.float())
                    maximum_outside[arm] = max(
                        maximum_outside[arm], _maximum_outside_gate(output)
                    )
            repeated_target = targets[start:stop].repeat((chunk, 1, 1, 1, 1))
            repeated_weights = evaluation_weights[start:stop].repeat(
                (chunk, 1, 1, 1, 1)
            )
            for arm in (CS, SC, SS):
                prediction = sums[arm].div(float(draws)).cpu()
                errors = _region_errors(
                    prediction,
                    repeated_target,
                    repeated_weights,
                ).reshape(chunk, batch, len(REGIONS))
                per_draw[arm][draw_start:draw_stop, start:stop] = errors

    if not torch.isfinite(correct_errors).any() or any(
        not torch.isfinite(value).any() for value in per_draw.values()
    ):
        raise RuntimeError("factorial evaluation produced no finite measurements")
    arm_errors = {CC: correct_errors}
    arm_errors.update({arm: values.mean(0) for arm, values in per_draw.items()})
    return arm_errors, per_draw, permutations, digests, maximum_outside


def _paired(
    candidate: Tensor,
    reference: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    eligible = torch.isfinite(candidate) & torch.isfinite(reference)
    ids = [
        patient_id
        for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
        if keep
    ]
    if len(ids) < 2:
        raise ValueError("factorial comparison has fewer than two patients")
    return asdict(
        paired_patient_bootstrap(
            candidate[eligible],
            reference[eligible],
            ids,
            higher_is_better=False,
            n_bootstrap=replicates,
            seed=int(seed),
        )
    )


def _contrast(
    values: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    eligible = torch.isfinite(values)
    ids = [
        patient_id
        for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
        if keep
    ]
    if len(ids) < 2:
        raise ValueError("factorial contrast has fewer than two patients")
    result = paired_patient_bootstrap(
        torch.zeros(int(eligible.sum()), dtype=torch.float32),
        values[eligible],
        ids,
        higher_is_better=False,
        n_bootstrap=replicates,
        seed=int(seed),
    )
    return {
        "mean": result.improvement,
        "ci_low": result.ci_low,
        "ci_high": result.ci_high,
        "bootstrap_positive_fraction": result.bootstrap_positive_fraction,
        "patient_positive_fraction": result.patient_win_fraction,
        "confidence_level": result.confidence_level,
        "n_patients": result.n_patients,
        "n_bootstrap": result.n_bootstrap,
    }


def _factorial_contrasts(
    arm_errors: Mapping[str, Tensor],
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, dict[str, Any]]:
    required = {CC, CS, SC, SS, POP}
    if not required.issubset(arm_errors):
        raise ValueError("factorial cells are incomplete")
    shapes = {tuple(arm_errors[arm].shape) for arm in required}
    if shapes != {(len(patient_ids), len(REGIONS))}:
        raise ValueError("factorial cell errors do not align")
    result: dict[str, dict[str, Any]] = {
        name: {}
        for name in (
            "support_penalty_at_correct_content_CS_minus_CC",
            "content_penalty_at_correct_support_SC_minus_CC",
            "joint_penalty_SS_minus_CC",
            "population_penalty_POP_minus_CC",
            "support_main_effect",
            "content_main_effect",
            "content_by_support_interaction",
        )
    }
    for index, region in enumerate(REGIONS):
        cc = arm_errors[CC][:, index]
        cs = arm_errors[CS][:, index]
        sc = arm_errors[SC][:, index]
        ss = arm_errors[SS][:, index]
        pop = arm_errors[POP][:, index]
        contrasts = {
            "support_penalty_at_correct_content_CS_minus_CC": cs - cc,
            "content_penalty_at_correct_support_SC_minus_CC": sc - cc,
            "joint_penalty_SS_minus_CC": ss - cc,
            "population_penalty_POP_minus_CC": pop - cc,
            "support_main_effect": 0.5 * ((cs - cc) + (ss - sc)),
            "content_main_effect": 0.5 * ((sc - cc) + (ss - cs)),
            "content_by_support_interaction": (cs - cc) - (ss - sc),
        }
        for offset, (name, values) in enumerate(contrasts.items()):
            result[name][region] = _contrast(
                values,
                patient_ids,
                seed=int(seed) + index * 101 + offset,
                replicates=replicates,
            )
    return result


def _arm_summaries(arm_errors: Mapping[str, Tensor]) -> dict[str, Any]:
    return {
        arm: {
            region: _summary(values[:, index]) for index, region in enumerate(REGIONS)
        }
        for arm, values in arm_errors.items()
    }


def _assert_validation_reproduction(
    validation_report: Mapping[str, Any],
    generated_errors: Tensor,
    original_errors: Tensor,
    anchor_errors: Tensor,
) -> float:
    comparisons = validation_report.get("candidate_vs_coarse_RRR")
    if not isinstance(comparisons, Mapping):
        raise TypeError("validation report lacks regional comparisons")
    maximum_patient_difference = 0.0
    for index, region in enumerate(REGIONS):
        eligible = (
            torch.isfinite(generated_errors[:, index])
            & torch.isfinite(original_errors[:, index])
            & torch.isfinite(anchor_errors[:, index])
        )
        value = comparisons.get(region)
        if int(eligible.sum()) < 2 or not isinstance(value, Mapping):
            raise ValueError("validation reproduction region is unavailable")
        generated = generated_errors[eligible, index]
        original = original_errors[eligible, index]
        anchor = anchor_errors[eligible, index]
        maximum_patient_difference = max(
            maximum_patient_difference,
            float((generated - original).abs().max()),
        )
        if (
            abs(float(generated.mean()) - float(original.mean())) > 5e-4
            or abs(float(original.mean()) - float(value.get("candidate_mean"))) > 5e-4
            or abs(float(anchor.mean()) - float(value.get("reference_mean"))) > 5e-4
        ):
            raise ValueError("generated CC cell does not reproduce frozen validation")
    return maximum_patient_difference


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("factorial audit requires exactly one CUDA GPU")
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "recovery_config": args.recovery_config,
            "v431_config": args.v4p3p1_config,
            "final_report": args.final_fit_report,
            "checkpoint": args.checkpoint,
            "validation_report": args.validation_report,
            "validation_private": args.validation_private_measurements,
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
        "manifest",
        "train_feature_index",
        "mask_report",
        "mask_index",
    ):
        if not paths[name].is_file():
            raise FileNotFoundError(f"factorial audit input unavailable: {name}")
    for name in (
        "train_feature_root",
        "validation_feature_root",
        "mask_root",
    ):
        if not paths[name].is_dir():
            raise FileNotFoundError(f"factorial audit release unavailable: {name}")
    if paths["output"].exists() or paths["private_output"].exists():
        raise FileExistsError("immutable factorial audit output exists")

    recovery_config = V432RecoveryConfig.load(paths["recovery_config"])
    v431_config = V431TrainingConfig.load(paths["v431_config"])
    final_report = _json(paths["final_report"])
    validation_report = _json(paths["validation_report"])
    validation_evidence = validation_report.get("evidence", {})
    final_evidence = final_report.get("evidence", {})
    model_source = Path(__file__).parents[1] / "roi_v4p3p1" / "model.py"
    if (
        final_report.get("schema_version") != FINAL_SCHEMA
        or final_report.get("status")
        != "completed_adaptive_recovery_train409_final_fit"
        or final_report.get("checkpoint_created") is not True
        or final_report.get("validation_88_loaded") is not False
        or final_report.get("test_split_loaded") is not False
        or validation_report.get("schema_version") != VALIDATION_SCHEMA
        or validation_report.get("status")
        not in {
            "completed_validation88_generalization_gate_pass",
            "completed_validation88_generalization_gate_fail",
        }
        or validation_report.get("patient_counts", {}).get("validation_evaluated") != 88
        or validation_report.get("test_split_loaded") is not False
        or validation_report.get("confirmatory_claim_permitted") is not False
        or not isinstance(validation_evidence, Mapping)
        or not isinstance(final_evidence, Mapping)
        or validation_evidence.get("final_fit_report_sha256")
        != _sha(paths["final_report"])
        or validation_evidence.get("checkpoint_sha256") != _sha(paths["checkpoint"])
        or validation_evidence.get("private_validation_measurements_sha256")
        != _sha(paths["validation_private"])
        or final_evidence.get("manifest_sha256") != _sha(paths["manifest"])
        or final_evidence.get("v4p3p1_model_source_sha256") != _sha(model_source)
    ):
        raise ValueError("factorial audit evidence chain is invalid")

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
        raise ValueError("factorial audit checkpoint payload is invalid")

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
    anchors = anchors.float()
    original_candidate = original_candidate.float()

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
        raise ValueError("factorial audit feature release changed after validation")

    device = torch.device("cuda")
    model = CoarseRRRAnchoredCoreResidualJEPA(
        v431_config.hidden_channels, v431_config.conditioned_blocks
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    mask_seed = recovery_config.bootstrap_seed + 7_000_001
    population_seed = recovery_config.bootstrap_seed + 17_000_003
    factorial_seed = recovery_config.bootstrap_seed + 19_000_009
    population_template = _population_average_raw_residual(
        model,
        train,
        train_occupancy,
        v431_config,
        seed=population_seed,
        device=device,
    )
    arm_errors, per_draw, permutations, permutation_digests, maximum_outside = (
        _factorial_region_errors(
            model,
            val,
            val_occupancy,
            anchors,
            val_weights,
            v431_config,
            seed=mask_seed,
            device=device,
            derangements=DERANGEMENTS,
            derangement_chunk_size=int(args.derangement_chunk_size),
        )
    )
    val_features = _record_features(val).float()
    target = val_features[:, 2]
    population_prediction = _population_prediction(
        anchors,
        population_template,
        val_occupancy[:, 0, None],
    )
    arm_errors[POP] = _region_errors(population_prediction, target, val_weights)
    arm_errors[ANCHOR] = _region_errors(anchors, target, val_weights)
    population_outside = val_occupancy[:, 0, None] == 0
    population_gated = population_template[None] * val_occupancy[:, 0, None]
    maximum_outside[POP] = (
        float(
            population_gated[population_outside.expand_as(population_gated)].abs().max()
        )
        if bool(population_outside.any())
        else 0.0
    )
    original_errors = _region_errors(original_candidate, target, val_weights)
    maximum_reproduction_difference = _assert_validation_reproduction(
        validation_report,
        arm_errors[CC],
        original_errors,
        arm_errors[ANCHOR],
    )

    factorial = _factorial_contrasts(
        arm_errors,
        patient_ids,
        seed=factorial_seed,
        replicates=recovery_config.bootstrap_replicates,
    )
    paired_penalties: dict[str, Any] = {}
    for offset, arm in enumerate((CS, SC, SS, POP)):
        paired_penalties[arm] = {
            region: _paired(
                arm_errors[CC][:, index],
                arm_errors[arm][:, index],
                patient_ids,
                seed=factorial_seed + 1_000 + offset * 101 + index,
                replicates=recovery_config.bootstrap_replicates,
            )
            for index, region in enumerate(REGIONS)
        }
    core = "t0_core"
    content_positive = (
        factorial["content_penalty_at_correct_support_SC_minus_CC"][core]["ci_low"] > 0
    )
    population_positive = (
        factorial["population_penalty_POP_minus_CC"][core]["ci_low"] > 0
    )
    support_positive = (
        factorial["support_penalty_at_correct_content_CS_minus_CC"][core]["ci_low"] > 0
    )
    diagnostic_criteria = {
        "core_content_penalty_CI_lower_positive": content_positive,
        "core_population_penalty_CI_lower_positive": population_positive,
        "core_support_penalty_CI_lower_positive": support_positive,
        "patient_specific_imaging_content_supported": (
            content_positive and population_positive
        ),
        "strict_content_and_support_personalization_supported": (
            content_positive and population_positive and support_positive
        ),
        "all_cells_exactly_zero_outside_their_assigned_core_gate": all(
            value == 0.0 for value in maximum_outside.values()
        ),
    }

    paths["output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    paths["private_output"].parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    output_staging = paths["output"].parent / f".{paths['output'].name}.staging"
    private_staging = (
        paths["private_output"].parent / f".{paths['private_output'].name}.staging"
    )
    output_staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_staging.mkdir(parents=True, exist_ok=False, mode=0o700)
    private_path = private_staging / "validation88_factorial_measurements.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": patient_ids,
            "arm_region_errors": arm_errors,
            "per_derangement_region_errors": per_draw,
            "donor_index_permutations": permutations,
            "derangement_sha256": permutation_digests,
            "train409_population_mean_raw_residual": population_template,
            "maximum_outside_assigned_core_support_absolute": maximum_outside,
            "generated_CC_vs_original_patient_region_MAE_maximum_absolute_difference": (
                maximum_reproduction_difference
            ),
        },
        private_path,
    )
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "scope": (
            "posthoc_descriptive_validation88_patient_content_by_spatial_support_"
            "factorial_audit_after_gate_observation"
        ),
        "scientific_question": (
            "whether_the_frozen_neural_residual_uses_patient_specific_T0_T1_"
            "imaging_content_beyond_tumour_location_or_a_population_correction"
        ),
        "patient_counts": {
            "train_for_population_template": 409,
            "validation_evaluated": 88,
            "test_loaded": 0,
        },
        "factorial_cells": {
            CC: "recipient_content_plus_recipient_core_ring_support",
            CS: "recipient_content_plus_deranged_donor_core_ring_support",
            SC: "deranged_donor_content_plus_recipient_core_ring_support",
            SS: "same_deranged_donor_content_plus_same_donor_core_ring_support",
            POP: ("train409_mean_frozen_raw_neural_residual_plus_recipient_core_gate"),
        },
        "fixed_recipient_variables": [
            "coarse_RRR_anchor",
            "T3_target",
            "evaluation_regions",
            "source_and_target_times",
            "T1_to_T3_action",
            "context_mask_realizations",
            "patient_bootstrap_identity",
        ],
        "patient_content_definition": "T1_embedding_and_T1_minus_T0_history",
        "spatial_support_definition": (
            "T0_fractional_core_and_ring_inputs_with_core_also_serving_as_output_gate"
        ),
        "arm_region_MAE": _arm_summaries(arm_errors),
        "paired_error_penalty_relative_to_CC": paired_penalties,
        "factorial_patient_level_contrasts": factorial,
        "diagnostic_criterion_results": diagnostic_criteria,
        "interpretation_contract": {
            "positive_penalty": "the_correct_CC_cell_has_lower_MAE",
            "patient_specific_content_requires": (
                "core_SC_minus_CC_CI_low_gt_0_and_core_POP_minus_CC_CI_low_gt_0"
            ),
            "strict_content_and_support_personalization_requires": (
                "patient_specific_content_requirements_plus_core_CS_minus_CC_CI_low_gt_0"
            ),
            "claim_scope": (
                "patient_specific_imaging_content_not_unobserved_biological_identity"
            ),
            "ood_caution": (
                "off_diagonal_content_support_pairs_are_counterfactual_and_may_be_"
                "less_in_distribution_than_CC"
            ),
        },
        "derangement_contract": {
            "method": "32_deterministic_unique_whole_cohort_derangements",
            "derangements": DERANGEMENTS,
            "self_assignment_permitted": False,
            "same_donor_used_for_content_and_support_in_SS": True,
            "per_patient_errors_averaged_across_derangements_before_bootstrap": True,
            "derangement_sha256": permutation_digests,
        },
        "population_residual_contract": {
            "fit_split": "train409_only",
            "validation_target_used": False,
            "quantity": "mean_frozen_model_raw_core_residual_before_output_gate",
            "mask_draws_per_patient": (
                len(v431_config.mask_fractions)
                * v431_config.evaluation_mask_draws_per_fraction
            ),
            "recipient_application": "population_raw_residual_times_recipient_core",
            "template_sha256": _tensor_sha(population_template),
        },
        "maximum_outside_assigned_core_support_absolute": maximum_outside,
        "CC_reproduction": {
            "matched_original_mask_seed": True,
            "mean_region_MAE_tolerance": 5e-4,
            "maximum_patient_region_MAE_absolute_difference": (
                maximum_reproduction_difference
            ),
            "passed": True,
        },
        "formal_gate_declared": False,
        "checkpoint_modified": False,
        "original_validation_products_modified": False,
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
            "validation_private_measurements_sha256": _sha(paths["validation_private"]),
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
            "factorial_audit_source_sha256": _sha(Path(__file__)),
            "private_factorial_measurements_sha256": _sha(private_path),
        },
    }
    _write_json(output_staging / "aggregate_report.json", report)
    os.replace(private_staging, paths["private_output"])
    os.replace(output_staging, paths["output"])
    print(
        json.dumps(
            {
                "status": STATUS,
                "validation_patients": 88,
                "derangements": DERANGEMENTS,
                "formal_gate_declared": False,
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
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--train-feature-root", required=True)
    parser.add_argument("--train-feature-index", required=True)
    parser.add_argument("--validation-feature-root", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--derangement-chunk-size", type=int, default=DERANGEMENTS)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BOOTSTRAP_REPLICATES",
    "CC",
    "CS",
    "DERANGEMENTS",
    "POP",
    "PRIVATE_SCHEMA",
    "SC",
    "SCHEMA",
    "SS",
    "STATUS",
    "run",
]
