"""Post-hoc development-only mechanistic audit of a trained Med-JEPA v4 model.

This audit puts persistence, masked-cover Med-JEPA, and reduced-rank ridge
baselines into one reconstructed 7x7x7 frozen-token estimand.  It stratifies
errors by a T0-defined disease bed, measures direct patient retrieval, and
reports shrinkage/alignment diagnostics.  It never trains a model, never reads
the held-out test prefix, and never reconstructs future image pixels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from med_jepa.frozen_audit.features import load_mask_sidecar
from med_jepa.frozen_audit.run import (
    _fit_rrr_temporal,
    _mask_index,
)
from med_jepa.frozen_audit.statistics import (
    TrainStandardizer,
    paired_patient_bootstrap,
)
from med_jepa.roi_v4.audit_masked_cover import (
    _predict_masked_cover,
    representation_rank_diagnostics,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4.model import TokenDynamicsJEPA
from med_jepa.roi_v4.train import _record_tensor, _views


SCHEMA = "med-jepa-roi-v4-mechanistic-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4-private-mechanistic-audit-v1"
TRAINING_SCHEMA = "med-jepa-roi-v4-development-training-v1"
MASK_RELEASE_SCHEMA = "spatial-med-jepa-ispy2-mask-release-v1"
REGIONS = (
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


def _read_json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("expected a JSON object")
    return value


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_mask_release(
    root: Path,
    report_path: Path,
    index_path: Path,
    expected_patients: set[str],
) -> dict[str, tuple[Path, str]]:
    if not root.is_dir() or not report_path.is_file() or not index_path.is_file():
        raise ValueError("development context-mask release is unavailable")
    if not _inside(report_path, root) or not _inside(index_path, root):
        raise ValueError("context-mask report/index escapes its release")
    report = _read_json(report_path)
    bindings = report.get("production_bindings")
    if (
        report.get("schema_version") != MASK_RELEASE_SCHEMA
        or report.get("status") != "ok"
        or report.get("patients") != len(expected_patients)
        or report.get("index_sha256") != _sha(index_path)
        or not isinstance(bindings, Mapping)
        or bindings.get("production_mode") != "development"
        or bindings.get("full_release_includes_test_mask_payloads") is not False
        or bindings.get("test_used_for_mask_method_development") is not False
    ):
        raise ValueError("development context-mask release evidence does not match")
    return _mask_index(
        index_path,
        release_root=root,
        expected_patients=expected_patients,
    )


def _region_weights(mask: Mapping[str, Any]) -> Tensor:
    """T0-only disease regions intersected with acquisition-valid evaluation support."""

    valid = (
        mask["valid_support"]["T0"]
        & mask["valid_support"]["T1"]
        & mask["valid_support"]["T3"]
    )
    core = mask["t0"] & valid
    ring = mask["t0_bed"] & ~mask["t0"] & valid
    background = ~mask["t0_bed"] & valid
    disease = core | ring
    voxel = torch.cat((core, ring, disease, background, valid), dim=0).float()[None]
    weights = F.avg_pool3d(voxel, kernel_size=16, stride=16)[0].contiguous()
    if tuple(weights.shape) != (len(REGIONS), 7, 7, 7):
        raise ValueError("evaluation-region weights do not match the frozen feature grid")
    # Core, ring, or background can legitimately be empty for an individual
    # after intersecting the three acquisition-valid supports.  Eligibility is
    # therefore region-specific; only the complete shared-valid grid is
    # required for every released patient.
    if float(weights[-1].sum()) <= 0:
        raise ValueError("full shared-valid evaluation grid is empty")
    return weights


def _load_region_weights(
    records: Sequence[PatientStates],
    mask_records: Mapping[str, tuple[Path, str]],
) -> Tensor:
    values: list[Tensor] = []
    for record in records:
        path, digest = mask_records[record.patient_id]
        mask = load_mask_sidecar(path, expected_sha256=digest)
        # The optional T3 tumour mask is intentionally not referenced.  T3 valid
        # acquisition support only defines where token error is assessable.
        values.append(_region_weights(mask))
    return torch.stack(values)


def _view_features(features: Tensor) -> tuple[Tensor, Tensor]:
    return _views(features)


def _covariates(records: Sequence[PatientStates]) -> dict[str, Tensor]:
    time = torch.stack([record.times_days for record in records]).float()
    action = torch.stack([record.action_13 for record in records]).float()
    clinical = torch.stack([record.clinical for record in records]).float()
    dt = torch.log1p((time[:, 2] - time[:, 1]).clamp_min(0)).unsqueeze(1)
    hr = clinical[:, 0].round().long().clamp(0, 1)
    her2 = clinical[:, 1].round().long().clamp(0, 1)
    subtype_index = hr * 2 + her2
    subtype = F.one_hot(subtype_index, num_classes=4).float()
    action_interaction = (subtype[:, :, None] * action[:, None, :]).flatten(1)
    return {
        "dt": dt,
        "action": action,
        "clinical": clinical,
        "subtype": subtype,
        "subtype_action_interaction": action_interaction,
    }


def _design(
    view: Tensor,
    global_view: Tensor,
    records: Sequence[PatientStates],
    information_set: str,
) -> Tensor:
    cov = _covariates(records)
    common = (cov["dt"], cov["action"])
    if information_set == "markov_matched":
        return torch.cat((view[:, 1], *common), dim=1)
    history = (view[:, 0], view[:, 1], view[:, 1] - view[:, 0], *common)
    if information_set == "history":
        return torch.cat(history, dim=1)
    if information_set != "conditional_history":
        raise ValueError("unknown RRR information set")
    global_t1 = global_view[:, 1]
    global_delta = global_view[:, 1] - global_view[:, 0]
    subtype = cov["subtype"]
    gated_t1 = (subtype[:, :, None] * global_t1[:, None, :]).flatten(1)
    gated_delta = (subtype[:, :, None] * global_delta[:, None, :]).flatten(1)
    return torch.cat(
        (
            *history,
            cov["clinical"],
            subtype,
            cov["subtype_action_interaction"],
            gated_t1,
            gated_delta,
        ),
        dim=1,
    )


def _reconstruct_delta(delta: Tensor, view: str) -> Tensor:
    if view == "global":
        if delta.ndim != 2 or delta.shape[1] != 1024:
            raise ValueError("global RRR prediction has the wrong shape")
        return delta[:, :, None, None, None].expand(-1, -1, 7, 7, 7).contiguous()
    if view != "coarse_spatial" or delta.ndim != 2 or delta.shape[1] != 27 * 1024:
        raise ValueError("coarse-spatial RRR prediction has the wrong shape")
    coarse = delta.reshape(-1, 3, 3, 3, 1024).permute(0, 4, 1, 2, 3)
    return F.interpolate(
        coarse, size=(7, 7, 7), mode="trilinear", align_corners=False
    ).contiguous()


def _fit_rrr_suite(
    train: Sequence[PatientStates],
    val: Sequence[PatientStates],
    *,
    seed: int,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    global_train, coarse_train = _view_features(train_features)
    global_val, coarse_val = _view_features(val_features)
    train_ids = [record.patient_id for record in train]
    predictions: dict[str, Tensor] = {}
    reports: dict[str, Any] = {}
    counter = 0
    for view_name, train_view, val_view in (
        ("global", global_train, global_val),
        ("coarse_spatial", coarse_train, coarse_val),
    ):
        for information_set in (
            "markov_matched",
            "history",
            "conditional_history",
        ):
            counter += 1
            train_design = _design(
                train_view, global_train, train, information_set
            )
            val_design = _design(val_view, global_val, val, information_set)
            report, predicted_delta = _fit_rrr_temporal(
                train_design,
                train_view[:, 2] - train_view[:, 1],
                val_design,
                val_view[:, 2] - val_view[:, 1],
                train_ids,
                seed=seed + counter * 101,
            )
            name = f"{view_name}_{information_set}_rrr"
            predictions[name] = val_features[:, 1] + _reconstruct_delta(
                predicted_delta, view_name
            )
            report.update(
                {
                    "target": f"{view_name}_t3_minus_t1_frozen_feature_change",
                    "view": view_name,
                    "information_set": information_set,
                    "input_dimensions": int(train_design.shape[1]),
                    "target_view_dimensions": int(train_view.shape[2]),
                    "full_grid_reconstruction": (
                        "broadcast_global_delta_to_7cube_v1"
                        if view_name == "global"
                        else "trilinear_3cube_delta_to_7cube_v1"
                    ),
                    "strictly_information_matched_to_checkpoint": (
                        information_set == "markov_matched"
                    ),
                }
            )
            reports[name] = report
    return predictions, reports


def patient_region_mae(prediction: Tensor, target: Tensor, weights: Tensor) -> Tensor:
    if prediction.shape != target.shape or prediction.ndim != 5:
        raise ValueError("prediction and target must share [N,C,7,7,7]")
    if weights.ndim != 4 or weights.shape[0] != prediction.shape[0]:
        raise ValueError("region weights must have shape [N,7,7,7]")
    token_error = (prediction.float() - target.float()).abs().mean(dim=1)
    denominator = weights.flatten(1).sum(1)
    result = torch.full_like(denominator, float("nan"), dtype=torch.float32)
    eligible = denominator > 0
    result[eligible] = (
        (token_error[eligible] * weights[eligible]).flatten(1).sum(1)
        / denominator[eligible]
    )
    return result


def _summary(values: Tensor) -> dict[str, float | int]:
    finite = values.float()[torch.isfinite(values)]
    if finite.numel() == 0:
        return {"count": 0}
    quantiles = torch.quantile(finite, torch.tensor([0.05, 0.5, 0.95]))
    return {
        "count": int(finite.numel()),
        "mean": float(finite.mean()),
        "median": float(quantiles[1]),
        "p05": float(quantiles[0]),
        "p95": float(quantiles[2]),
    }


def _bootstrap_from_errors(
    candidate: Tensor,
    reference: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    eligible = torch.isfinite(candidate) & torch.isfinite(reference)
    if int(eligible.sum()) < 2:
        raise ValueError("region-specific paired comparison has fewer than two patients")
    eligible_ids = [
        patient_id
        for patient_id, keep in zip(patient_ids, eligible.tolist(), strict=True)
        if keep
    ]
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


def _retrieval(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    scaler: TrainStandardizer,
    eligible: Tensor,
) -> tuple[dict[str, float | int], Tensor]:
    def pool(values: Tensor) -> Tensor:
        denominator = weights.flatten(1).sum(1).unsqueeze(1)
        return (values * weights[:, None]).flatten(2).sum(2) / denominator

    if eligible.dtype != torch.bool or eligible.shape != (prediction.shape[0],):
        raise ValueError("retrieval eligibility mask is malformed")
    if int(eligible.sum()) < 2:
        raise ValueError("retrieval requires at least two region-eligible patients")
    prediction = prediction[eligible]
    target = target[eligible]
    weights = weights[eligible]
    query = scaler.transform(pool(prediction)).float()
    gallery = scaler.transform(pool(target)).float()
    query = F.normalize(query, dim=1)
    gallery = F.normalize(gallery, dim=1)
    order = torch.argsort(query @ gallery.T, dim=1, descending=True)
    truth = torch.arange(len(order))[:, None]
    ranks = torch.nonzero(order == truth, as_tuple=False)[:, 1] + 1
    return {
        "patients": int(len(ranks)),
        "top1": float((ranks <= 1).float().mean()),
        "top5": float((ranks <= 5).float().mean()),
        "mrr": float(ranks.float().reciprocal().mean()),
        "chance_top1": 1.0 / len(ranks),
        "chance_top5": min(5.0 / len(ranks), 1.0),
    }, ranks


def _weighted_pool(values: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.flatten(1).sum(1).unsqueeze(1)
    result = torch.full(
        (values.shape[0], values.shape[1]),
        float("nan"),
        dtype=torch.float32,
        device=values.device,
    )
    eligible = denominator[:, 0] > 0
    result[eligible] = (
        (values[eligible] * weights[eligible, None]).flatten(2).sum(2)
        / denominator[eligible]
    )
    return result


def _delta_diagnostics(
    prediction: Tensor,
    source: Tensor,
    target: Tensor,
    weights: Tensor,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    predicted = prediction.float() - source.float()
    actual = target.float() - source.float()
    weight = weights[:, None].float()
    dot = (predicted * actual * weight).flatten(1).sum(1)
    predicted_energy = (predicted.square() * weight).flatten(1).sum(1)
    actual_energy = (actual.square() * weight).flatten(1).sum(1)
    valid = actual_energy > torch.finfo(torch.float32).eps
    valid_cosine = valid & (
        predicted_energy > torch.finfo(torch.float32).eps
    )
    amplitude = torch.full_like(actual_energy, float("nan"))
    energy = torch.full_like(actual_energy, float("nan"))
    cosine = torch.full_like(actual_energy, float("nan"))
    amplitude[valid] = torch.sqrt(predicted_energy[valid] / actual_energy[valid])
    energy[valid] = predicted_energy[valid] / actual_energy[valid]
    cosine[valid_cosine] = dot[valid_cosine] / torch.sqrt(
        predicted_energy[valid_cosine] * actual_energy[valid_cosine]
    )
    residual = energy - cosine.square()
    aggregate_energy = float(predicted_energy.sum() / actual_energy.sum())
    aggregate_denominator = torch.sqrt(predicted_energy.sum() * actual_energy.sum())
    aggregate_cosine = (
        float(dot.sum() / aggregate_denominator)
        if float(aggregate_denominator) > torch.finfo(torch.float32).eps
        else float("nan")
    )
    public = {
        "amplitude_ratio": _summary(amplitude),
        "energy_ratio": _summary(energy),
        "cosine_alignment": _summary(cosine),
        "energy_ratio_minus_cosine_squared": _summary(residual),
        "cohort_aggregate": {
            "amplitude_ratio": math.sqrt(max(aggregate_energy, 0.0)),
            "energy_ratio": aggregate_energy,
            "cosine_alignment": (
                aggregate_cosine if math.isfinite(aggregate_cosine) else None
            ),
            "energy_ratio_minus_cosine_squared": (
                aggregate_energy - aggregate_cosine**2
                if math.isfinite(aggregate_cosine)
                else None
            ),
        },
    }
    return public, {
        "amplitude_ratio": amplitude,
        "energy_ratio": energy,
        "cosine_alignment": cosine,
        "energy_ratio_minus_cosine_squared": residual,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("mechanistic checkpoint audit requires exactly one CUDA GPU")
    training_report_path = Path(args.training_report).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    mask_root = Path(args.context_mask_release).expanduser().resolve()
    mask_report_path = Path(args.context_mask_report).expanduser().resolve()
    mask_index_path = Path(args.context_mask_index).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable mechanistic-audit output already exists")

    training_report = _read_json(training_report_path)
    if (
        training_report.get("schema_version") != TRAINING_SCHEMA
        or training_report.get("status") != "completed_development_training"
        or training_report.get("test_split_loaded") is not False
        or training_report.get("patient_counts") != {"train": 409, "val": 88, "test": 0}
    ):
        raise ValueError("training report is not an approved development-only v4 release")
    evidence = training_report.get("evidence", {})
    if evidence.get("checkpoint_sha256") != _sha(checkpoint_path):
        raise ValueError("checkpoint does not match the training report")
    if evidence.get("manifest_sha256") != _sha(manifest_path):
        raise ValueError("manifest does not match the training report")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("schema_version") != TRAINING_SCHEMA
        or not isinstance(checkpoint.get("config"), dict)
        or not isinstance(checkpoint.get("model"), dict)
    ):
        raise ValueError("checkpoint schema is invalid")
    config = checkpoint["config"]

    train = load_development_states(manifest_path, "train")
    val = load_development_states(manifest_path, "val")
    development_ids = {item.patient_id for item in (*train, *val)}
    mask_records = _validate_mask_release(
        mask_root, mask_report_path, mask_index_path, development_ids
    )
    train_weights = _load_region_weights(train, mask_records)
    val_weights = _load_region_weights(val, mask_records)

    device = torch.device("cuda")
    model = TokenDynamicsJEPA(
        hidden_channels=int(config["hidden_channels"]),
        blocks=int(config["blocks"]),
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    batch_size = int(config["batch_size"])
    seed = int(config["seed"])
    bootstrap_seed = int(config["bootstrap_seed"])
    replicates = int(config["bootstrap_replicates"])
    masked_prediction, coverage, draws = _predict_masked_cover(
        model,
        val,
        batch_size=batch_size,
        masked_fraction=float(config["masked_fraction"]),
        seed=seed + 4_000_000,
        minimum_draws=int(args.minimum_draws),
        maximum_draws=int(args.maximum_draws),
        device=device,
    )

    # Re-run the target-free prediction API with the same mask schedule while a
    # deterministic shuffled target exists only on the evaluator side.
    shuffled_prediction, _, _ = _predict_masked_cover(
        model,
        val,
        batch_size=batch_size,
        masked_fraction=float(config["masked_fraction"]),
        seed=seed + 4_000_000,
        minimum_draws=int(args.minimum_draws),
        maximum_draws=int(args.maximum_draws),
        device=device,
    )
    target_shuffle = torch.roll(torch.arange(len(val)), shifts=1)
    leakage_max_difference = float((masked_prediction - shuffled_prediction).abs().max())
    if leakage_max_difference != 0.0:
        raise RuntimeError("target-free masked-cover prediction is not reproducible")

    val_features = _record_tensor(val, "features")
    train_features = _record_tensor(train, "features")
    source = val_features[:, 1]
    target = val_features[:, 2]
    rrr_predictions, rrr_reports = _fit_rrr_suite(
        train, val, seed=bootstrap_seed + 20_000
    )
    predictions: dict[str, Tensor] = {
        "masked_cover_med_jepa": masked_prediction,
        "latent_persistence": source,
        **rrr_predictions,
    }
    val_ids = [item.patient_id for item in val]

    errors: dict[str, dict[str, Tensor]] = {}
    error_summary: dict[str, dict[str, Any]] = {}
    for method, prediction in predictions.items():
        errors[method] = {}
        error_summary[method] = {}
        for region_index, region in enumerate(REGIONS):
            values = patient_region_mae(prediction, target, val_weights[:, region_index])
            errors[method][region] = values
            error_summary[method][region] = _summary(values)

    comparisons: dict[str, dict[str, Any]] = {}
    for region_index, region in enumerate(REGIONS):
        comparisons[region] = {}
        for counter, (method, method_errors) in enumerate(errors.items()):
            if method == "masked_cover_med_jepa":
                continue
            comparisons[region][f"med_jepa_vs_{method}"] = _bootstrap_from_errors(
                errors["masked_cover_med_jepa"][region],
                method_errors[region],
                val_ids,
                seed=bootstrap_seed + region_index * 1_000 + counter,
                replicates=replicates,
            )

    rank_diagnostics: dict[str, Any] = {
        "source_t1": representation_rank_diagnostics(source, device),
        "target_t3": representation_rank_diagnostics(target, device),
        "true_delta_t3_minus_t1": representation_rank_diagnostics(target - source, device),
    }
    for method, prediction in predictions.items():
        rank_diagnostics[method] = {
            "predicted_t3": representation_rank_diagnostics(prediction, device),
            "predicted_delta_t3_minus_t1": representation_rank_diagnostics(
                prediction - source, device
            ),
        }

    disease_index = REGIONS.index("t0_disease_bed")
    train_disease_eligible = train_weights[:, disease_index].flatten(1).sum(1) > 0
    val_disease_eligible = val_weights[:, disease_index].flatten(1).sum(1) > 0
    if int(train_disease_eligible.sum()) < 2 or int(val_disease_eligible.sum()) < 2:
        raise ValueError("disease-bed retrieval cohort is too small")
    train_target_pooled = _weighted_pool(
        train_features[:, 2], train_weights[:, disease_index]
    )
    retrieval_scaler = TrainStandardizer().fit(
        train_target_pooled[train_disease_eligible]
    )
    retrieval: dict[str, Any] = {}
    retrieval_ranks: dict[str, Tensor] = {}
    for method, prediction in predictions.items():
        retrieval[method], retrieval_ranks[method] = _retrieval(
            prediction,
            target,
            val_weights[:, disease_index],
            retrieval_scaler,
            val_disease_eligible,
        )

    delta_diagnostics: dict[str, Any] = {}
    private_delta: dict[str, dict[str, Tensor]] = {}
    for method, prediction in predictions.items():
        delta_diagnostics[method] = {}
        private_delta[method] = {}
        for region in ("t0_disease_bed", "full_shared_valid_grid"):
            region_index = REGIONS.index(region)
            public, patient_values = _delta_diagnostics(
                prediction, source, target, val_weights[:, region_index]
            )
            delta_diagnostics[method][region] = public
            for name, values in patient_values.items():
                private_delta[method][f"{region}/{name}"] = values

    pcr = torch.stack([item.pcr for item in val])
    pcr_report: dict[str, Any] = {
        "role": "posthoc_descriptive_only_not_a_gate",
        "observed": int(torch.isfinite(pcr).sum()),
        "models": {},
    }
    for method, method_errors in errors.items():
        pcr_report["models"][method] = {}
        for group in (0, 1):
            keep = torch.isfinite(pcr) & (pcr == group)
            pcr_report["models"][method][f"pcr_{group}"] = _summary(
                method_errors["t0_disease_bed"][keep]
            )

    shuffled_primary = patient_region_mae(
        masked_prediction,
        target[target_shuffle],
        val_weights[:, disease_index],
    )
    leakage_control = {
        "prediction_api_accepts_target_tensor": False,
        "prediction_exactly_identical_after_evaluator_target_shuffle": True,
        "maximum_absolute_prediction_difference": leakage_max_difference,
        "matched_target_disease_bed_mae": float(
            torch.nanmean(errors["masked_cover_med_jepa"]["t0_disease_bed"])
        ),
        "shuffled_target_disease_bed_mae": float(torch.nanmean(shuffled_primary)),
        "target_shuffle": "deterministic_cyclic_patient_shift_v1",
    }

    primary_comparisons = comparisons["t0_disease_bed"]

    def relation(name: str) -> str:
        value = primary_comparisons[name]
        if float(value["ci_low"]) > 0:
            return "med_jepa_better"
        if float(value["ci_high"]) < 0:
            return "med_jepa_worse"
        return "inconclusive"

    primary_relations = {
        "persistence": relation("med_jepa_vs_latent_persistence"),
        "matched_global_rrr": relation(
            "med_jepa_vs_global_markov_matched_rrr"
        ),
        "matched_coarse_spatial_rrr": relation(
            "med_jepa_vs_coarse_spatial_markov_matched_rrr"
        ),
        "strongest_conditional_coarse_spatial_rrr": relation(
            "med_jepa_vs_coarse_spatial_conditional_history_rrr"
        ),
    }
    if any(
        value == "med_jepa_worse"
        for key, value in primary_relations.items()
        if key in {"persistence", "matched_global_rrr", "matched_coarse_spatial_rrr"}
    ):
        checkpoint_triage = "current_checkpoint_fails_at_least_one_information_matched_disease_bed_comparator"
    elif all(
        primary_relations[key] == "med_jepa_better"
        for key in ("persistence", "matched_global_rrr", "matched_coarse_spatial_rrr")
    ):
        checkpoint_triage = "disease_bed_nonlinear_signal_supported_against_all_information_matched_comparators"
    else:
        checkpoint_triage = "disease_bed_signal_not_resolved_against_all_information_matched_comparators"

    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": val_ids,
        "patient_region_errors": errors,
        "retrieval_ranks": retrieval_ranks,
        "retrieval_disease_bed_eligible": val_disease_eligible,
        "delta_diagnostics": private_delta,
        "pcr": pcr,
        "target_shuffle_indices": target_shuffle,
        "coverage_counts": coverage,
        "draws_used_by_batch": draws,
    }
    private_path = private_output / "validation_mechanistic_measurements.pt"
    torch.save(private, private_path)
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": "completed_posthoc_development_mechanistic_audit",
        "scope": "posthoc_development_only_checkpoint_audit",
        "seed": seed,
        "patient_counts": {"train": len(train), "val": len(val), "test": 0},
        "primary_estimand": {
            "name": "t0_disease_bed_weighted_full_7cube_token_mae",
            "region": "T0 FTV core plus T0 20mm ring intersected with T0/T1/T3 acquisition-valid support",
            "channel_reduction": "mean_absolute_error_across_1024_channels",
            "spatial_reduction": "fractional_occupancy_weighted_mean",
            "primary_information_matched_comparators": [
                "latent_persistence",
                "global_markov_matched_rrr",
                "coarse_spatial_markov_matched_rrr",
            ],
            "history_and_conditional_rrr_role": "stronger_information_set_descriptive_baselines",
        },
        "region_contract": {
            "order": list(REGIONS),
            "tumour_region_source_visits": ["T0"],
            "t3_tumour_support_used": False,
            "t3_acquisition_valid_support_used_for_evaluation_grid": True,
            "background_definition": "shared_valid_grid_outside_t0_20mm_disease_bed",
            "region_specific_eligibility": {
                region: {
                    "train": int(
                        (train_weights[:, index].flatten(1).sum(1) > 0).sum()
                    ),
                    "validation": int(
                        (val_weights[:, index].flatten(1).sum(1) > 0).sum()
                    ),
                }
                for index, region in enumerate(REGIONS)
            },
        },
        "model_information_sets": {
            "masked_cover_med_jepa": ["full_T1_token_grid", "delta_t_T1_to_T3", "action_T1_to_T3"],
            "markov_matched_rrr": ["corresponding_T1_view", "delta_t_T1_to_T3", "action_T1_to_T3"],
            "history_rrr": ["corresponding_T0_view", "corresponding_T1_view", "T1_minus_T0", "delta_t", "action"],
            "conditional_history_rrr": ["history_rrr", "HR", "HER2", "subtype_action_interactions", "subtype_gated_global_T1_and_delta"],
        },
        "patient_region_mae": error_summary,
        "paired_patient_bootstrap": comparisons,
        "rrr_models": rrr_reports,
        "effective_rank_diagnostics": rank_diagnostics,
        "rank_interpretation": {
            "role": "descriptive_diagnostic_not_a_pass_fail_gate",
            "fixed_retention_threshold_used": False,
            "reason": "target_noise_can_raise_effective_rank_above_the_MMSE_prediction",
        },
        "patient_retrieval_in_t0_disease_bed": retrieval,
        "delta_shrinkage_and_alignment": delta_diagnostics,
        "t3_shuffle_leakage_control": leakage_control,
        "pcr_stratified_disease_bed_error": pcr_report,
        "masked_cover_contract": {
            "masked_fraction": float(config["masked_fraction"]),
            "minimum_draws": int(args.minimum_draws),
            "maximum_draws": int(args.maximum_draws),
            "draws_used_min": min(draws),
            "draws_used_max": max(draws),
            "all_tokens_covered": bool((coverage > 0).all()),
        },
        "decision_contract": {
            "scientific_role": "mechanism_triage_for_v4p1_not_a_new_formal_gate",
            "primary_comparator_relations": primary_relations,
            "checkpoint_triage": checkpoint_triage,
            "next_required_experiment": "frozen_3dino_augmentation_noise_floor_audit_c",
            "v4_original_gate_changed": False,
            "v4p1_training_started": False,
            "rank_threshold_removed": True,
            "noise_floor_audit_required_before_architecture_change": True,
        },
        "retraining_performed": False,
        "optimizer_created": False,
        "checkpoint_parameters_modified": False,
        "future_pixel_reconstruction": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "confirmatory_claim_permitted": False,
        "evidence": {
            "training_report_sha256": _sha(training_report_path),
            "checkpoint_sha256": _sha(checkpoint_path),
            "manifest_sha256": _sha(manifest_path),
            "context_mask_report_sha256": _sha(mask_report_path),
            "context_mask_index_sha256": _sha(mask_index_path),
            "audit_source_sha256": _sha(Path(__file__)),
            "private_measurements_sha256": _sha(private_path),
        },
    }
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(report_path, 0o600)
    primary = comparisons["t0_disease_bed"]
    print(
        json.dumps(
            {
                "status": report["status"],
                "med_jepa_vs_persistence": primary["med_jepa_vs_latent_persistence"],
                "med_jepa_vs_matched_coarse_rrr": primary[
                    "med_jepa_vs_coarse_spatial_markov_matched_rrr"
                ],
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--minimum-draws", type=int, default=32)
    parser.add_argument("--maximum-draws", type=int, default=128)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "REGIONS",
    "_delta_diagnostics",
    "_reconstruct_delta",
    "patient_region_mae",
    "run",
]
