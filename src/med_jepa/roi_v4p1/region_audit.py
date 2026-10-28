"""Read-only regional specificity audit for a completed Med-JEPA v4.1 run.

The audit consumes immutable development predictions produced by the v4.1
training job and its full-token linear baselines.  It never runs a model,
reads DICOM pixels, retrains, or loads the held-out test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import (
    REGIONS,
    _delta_diagnostics,
    _load_region_weights,
    _summary,
    _validate_mask_release,
    patient_region_mae,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4.train import _record_tensor
from med_jepa.roi_v4p1.full_token_linear import (
    PRIVATE_SCHEMA as BASELINE_PRIVATE_SCHEMA,
    SCHEMA as BASELINE_SCHEMA,
)
from med_jepa.roi_v4p1.train import (
    BASELINE_NAMES,
    PRIVATE_SCHEMA as TRAINING_PRIVATE_SCHEMA,
    SCHEMA as TRAINING_SCHEMA,
)


SCHEMA = "med-jepa-roi-v4p1-regional-specificity-audit-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p1-private-regional-specificity-audit-v1"
MANIFEST_SCHEMA = "med-jepa-ispy2-manifest-build-v1"
CANDIDATE = "source_anchored_med_jepa_v4p1"
SIMPLE_BASELINES = (
    "train_mean_t3_state",
    "train_mean_t1_to_t3_delta",
    "train_only_shrinkage_to_mean_t3",
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


def fit_shrinkage_alpha(
    source: Tensor,
    target: Tensor,
    disease_weights: Tensor,
    alphas: Sequence[float],
) -> tuple[float, dict[str, float]]:
    """Select source/mean-target shrinkage using train-only leave-one-out means."""

    if source.shape != target.shape or source.ndim != 5 or len(source) < 3:
        raise ValueError("shrinkage training tensors are malformed")
    if disease_weights.shape != (len(source), *source.shape[-3:]):
        raise ValueError("shrinkage disease weights are malformed")
    if not alphas:
        raise ValueError("shrinkage alpha grid is empty")
    source = source.float()
    target = target.float()
    target_sum = target.sum(0, keepdim=True)
    leave_one_out_mean = (target_sum - target) / float(len(target) - 1)
    scores: dict[str, float] = {}
    for alpha in alphas:
        if not 0.0 <= float(alpha) <= 1.0:
            raise ValueError("shrinkage alpha must be in [0, 1]")
        prediction = float(alpha) * source + (1.0 - float(alpha)) * leave_one_out_mean
        errors = patient_region_mae(prediction, target, disease_weights)
        if int(torch.isfinite(errors).sum()) < 2:
            raise ValueError("too few train patients have an assessable disease bed")
        scores[f"{float(alpha):.4f}"] = float(torch.nanmean(errors))
    selected = min(
        (float(key) for key in scores),
        key=lambda alpha: (scores[f"{alpha:.4f}"], alpha),
    )
    return selected, scores


def disease_specificity_bootstrap(
    candidate_disease: Tensor,
    reference_disease: Tensor,
    candidate_background: Tensor,
    reference_background: Tensor,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Bootstrap the relative-improvement difference, disease minus background."""

    eligible = (
        torch.isfinite(candidate_disease)
        & torch.isfinite(reference_disease)
        & torch.isfinite(candidate_background)
        & torch.isfinite(reference_background)
    )
    if int(eligible.sum()) < 2 or replicates < 2:
        raise ValueError("specificity bootstrap has insufficient observations")
    cd = candidate_disease[eligible].double()
    rd = reference_disease[eligible].double()
    cb = candidate_background[eligible].double()
    rb = reference_background[eligible].double()

    def relative(candidate: Tensor, reference: Tensor) -> Tensor:
        return (reference - candidate) / reference.clamp_min(1e-12)

    point_disease = float(relative(cd.mean(), rd.mean()))
    point_background = float(relative(cb.mean(), rb.mean()))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randint(
        len(cd), (int(replicates), len(cd)), generator=generator
    )
    boot_disease = relative(cd[indices].mean(1), rd[indices].mean(1))
    boot_background = relative(cb[indices].mean(1), rb[indices].mean(1))
    specificity = boot_disease - boot_background
    interval = torch.quantile(
        specificity,
        torch.tensor([0.025, 0.975], dtype=specificity.dtype),
    )
    return {
        "patients": int(len(cd)),
        "bootstrap_replicates": int(replicates),
        "relative_improvement_disease_bed": point_disease,
        "relative_improvement_valid_non_disease_bed": point_background,
        "disease_minus_background_relative_improvement": (
            point_disease - point_background
        ),
        "paired_bootstrap_95ci": [float(interval[0]), float(interval[1])],
        "bootstrap_positive_fraction": float((specificity > 0).double().mean()),
        "positive_value_favors_disease_specific_gain": True,
    }


def _occupancy_summary(weights: Tensor) -> dict[str, Any]:
    if weights.ndim != 5 or weights.shape[1] != len(REGIONS):
        raise ValueError("regional weight tensor is malformed")
    flat = weights.float().flatten(2)
    full = flat[:, REGIONS.index("full_shared_valid_grid")].sum(1).clamp_min(1e-12)
    result: dict[str, Any] = {}
    for index, region in enumerate(REGIONS):
        value = flat[:, index]
        weight_sum = value.sum(1)
        nonzero = (value > 0).sum(1).float()
        effective = weight_sum.square() / value.square().sum(1).clamp_min(1e-12)
        result[region] = {
            "fractional_token_weight_sum": _summary(weight_sum),
            "nonzero_token_count": _summary(nonzero),
            "kish_effective_token_count": _summary(effective),
            "fraction_of_full_shared_valid_weight": _summary(weight_sum / full),
        }
    return result


def _patient_region_delta_rms(delta: Tensor, weights: Tensor) -> Tensor:
    if delta.ndim != 5 or weights.shape != (len(delta), *delta.shape[-3:]):
        raise ValueError("delta RMS inputs are malformed")
    token_square = delta.float().square().mean(1)
    denominator = weights.float().flatten(1).sum(1)
    result = torch.full_like(denominator, float("nan"))
    eligible = denominator > 0
    result[eligible] = torch.sqrt(
        (token_square[eligible] * weights[eligible]).flatten(1).sum(1)
        / denominator[eligible]
    )
    return result


def _load_predictions(
    training_private_path: Path,
    baseline_private_path: Path,
    expected_ids: list[str],
) -> tuple[Tensor, dict[str, Tensor]]:
    training = torch.load(
        training_private_path, map_location="cpu", weights_only=True
    )
    baseline = torch.load(
        baseline_private_path, map_location="cpu", weights_only=True
    )
    if (
        not isinstance(training, Mapping)
        or training.get("schema_version") != TRAINING_PRIVATE_SCHEMA
        or training.get("patient_ids") != expected_ids
        or not isinstance(training.get("prediction"), Tensor)
    ):
        raise ValueError("private v4.1 prediction release does not align")
    if (
        not isinstance(baseline, Mapping)
        or baseline.get("schema_version") != BASELINE_PRIVATE_SCHEMA
        or baseline.get("patient_ids") != expected_ids
        or not isinstance(baseline.get("predictions"), Mapping)
    ):
        raise ValueError("private full-token baseline release does not align")
    baseline_predictions = baseline["predictions"]
    required = ("latent_persistence", *BASELINE_NAMES)
    if not set(required).issubset(baseline_predictions):
        raise ValueError("private full-token release lacks required baselines")
    return training["prediction"].float(), {
        name: baseline_predictions[name].float() for name in required
    }


def _validate_release_chain(
    *,
    training_report: Mapping[str, Any],
    training_private_path: Path,
    baseline_report: Mapping[str, Any],
    baseline_report_path: Path,
    baseline_private_path: Path,
    manifest_path: Path,
    manifest_report: Mapping[str, Any],
    manifest_report_path: Path,
    mask_report_path: Path,
    mask_index_path: Path,
) -> None:
    training_evidence = training_report.get("evidence", {})
    baseline_evidence = baseline_report.get("evidence", {})
    if (
        training_report.get("schema_version") != TRAINING_SCHEMA
        or training_report.get("status") != "completed_development_training"
        or training_report.get("patient_counts") != {"train": 409, "val": 88, "test": 0}
        or training_report.get("test_split_loaded") is not False
        or training_evidence.get("private_measurements_sha256") != _sha(training_private_path)
        or training_evidence.get("full_token_baseline_report_sha256") != _sha(baseline_report_path)
        or training_evidence.get("full_token_baseline_private_sha256") != _sha(baseline_private_path)
        or training_evidence.get("manifest_sha256") != _sha(manifest_path)
        or training_evidence.get("manifest_build_report_sha256") != _sha(manifest_report_path)
        or training_evidence.get("context_mask_report_sha256") != _sha(mask_report_path)
        or training_evidence.get("context_mask_index_sha256") != _sha(mask_index_path)
    ):
        raise ValueError("v4.1 training release chain is not approved")
    if (
        baseline_report.get("schema_version") != BASELINE_SCHEMA
        or baseline_report.get("status") != "completed_development_full_token_linear_baselines"
        or baseline_report.get("patient_counts") != {"train": 409, "val": 88, "test": 0}
        or baseline_report.get("test_split_loaded") is not False
        or baseline_evidence.get("private_predictions_sha256") != _sha(baseline_private_path)
        or baseline_evidence.get("manifest_sha256") != _sha(manifest_path)
        or baseline_evidence.get("manifest_build_report_sha256") != _sha(manifest_report_path)
        or baseline_evidence.get("context_mask_report_sha256") != _sha(mask_report_path)
        or baseline_evidence.get("context_mask_index_sha256") != _sha(mask_index_path)
    ):
        raise ValueError("full-token baseline release chain is not approved")
    if (
        manifest_report.get("schema_version") != MANIFEST_SCHEMA
        or manifest_report.get("status") != "ok"
        or manifest_report.get("records") != 587
        or manifest_report.get("manifest_sha256") != _sha(manifest_path)
    ):
        raise ValueError("manifest release chain is not approved")


def run(args: argparse.Namespace) -> dict[str, Any]:
    training_report_path = Path(args.training_report).expanduser().resolve()
    training_private_path = Path(args.training_private).expanduser().resolve()
    baseline_report_path = Path(args.baseline_report).expanduser().resolve()
    baseline_private_path = Path(args.baseline_private).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_report_path = Path(args.manifest_build_report).expanduser().resolve()
    mask_root = Path(args.context_mask_release).expanduser().resolve()
    mask_report_path = Path(args.context_mask_report).expanduser().resolve()
    mask_index_path = Path(args.context_mask_index).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable regional-audit output already exists")

    training_report = _json(training_report_path)
    baseline_report = _json(baseline_report_path)
    manifest_report = _json(manifest_report_path)
    _validate_release_chain(
        training_report=training_report,
        training_private_path=training_private_path,
        baseline_report=baseline_report,
        baseline_report_path=baseline_report_path,
        baseline_private_path=baseline_private_path,
        manifest_path=manifest_path,
        manifest_report=manifest_report,
        manifest_report_path=manifest_report_path,
        mask_report_path=mask_report_path,
        mask_index_path=mask_index_path,
    )

    train = load_development_states(manifest_path, "train")
    val = load_development_states(manifest_path, "val")
    if len(train) != 409 or len(val) != 88:
        raise ValueError("development split counts changed")
    development_ids = {item.patient_id for item in (*train, *val)}
    mask_records = _validate_mask_release(
        mask_root, mask_report_path, mask_index_path, development_ids
    )
    train_weights = _load_region_weights(train, mask_records)
    val_weights = _load_region_weights(val, mask_records)
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    source = val_features[:, 1].float()
    target = val_features[:, 2].float()
    val_ids = [item.patient_id for item in val]
    candidate, released_baselines = _load_predictions(
        training_private_path, baseline_private_path, val_ids
    )
    if candidate.shape != target.shape or any(
        value.shape != target.shape for value in released_baselines.values()
    ):
        raise ValueError("saved validation prediction shape changed")

    alpha_grid = tuple(index / 20.0 for index in range(21))
    disease = REGIONS.index("t0_disease_bed")
    selected_alpha, alpha_scores = fit_shrinkage_alpha(
        train_features[:, 1],
        train_features[:, 2],
        train_weights[:, disease],
        alpha_grid,
    )
    train_mean_target = train_features[:, 2].float().mean(0, keepdim=True)
    train_mean_delta = (
        train_features[:, 2].float() - train_features[:, 1].float()
    ).mean(0, keepdim=True)
    predictions: dict[str, Tensor] = {
        CANDIDATE: candidate,
        **released_baselines,
        "train_mean_t3_state": train_mean_target.expand_as(target),
        "train_mean_t1_to_t3_delta": source + train_mean_delta,
        "train_only_shrinkage_to_mean_t3": (
            selected_alpha * source
            + (1.0 - selected_alpha) * train_mean_target
        ),
    }

    errors: dict[str, dict[str, Tensor]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for method, prediction in predictions.items():
        errors[method] = {}
        summaries[method] = {}
        for index, region in enumerate(REGIONS):
            values = patient_region_mae(prediction, target, val_weights[:, index])
            errors[method][region] = values
            summaries[method][region] = _summary(values)

    from med_jepa.roi_v4.audit_mechanistic import _bootstrap_from_errors

    comparisons: dict[str, dict[str, Any]] = {}
    for region_index, region in enumerate(REGIONS):
        comparisons[region] = {}
        for method_index, method in enumerate(predictions):
            if method == CANDIDATE:
                continue
            comparisons[region][f"v4p1_vs_{method}"] = _bootstrap_from_errors(
                errors[CANDIDATE][region],
                errors[method][region],
                val_ids,
                seed=int(args.bootstrap_seed) + region_index * 100 + method_index,
                replicates=int(args.bootstrap_replicates),
            )

    specificity: dict[str, Any] = {}
    disease_region = "t0_disease_bed"
    background_region = "valid_non_disease_bed"
    for method_index, method in enumerate(predictions):
        if method == CANDIDATE:
            continue
        specificity[f"v4p1_vs_{method}"] = disease_specificity_bootstrap(
            errors[CANDIDATE][disease_region],
            errors[method][disease_region],
            errors[CANDIDATE][background_region],
            errors[method][background_region],
            seed=int(args.bootstrap_seed) + 10_000 + method_index,
            replicates=int(args.bootstrap_replicates),
        )

    true_delta = target - source
    true_delta_rms: dict[str, Any] = {}
    private_true_delta: dict[str, Tensor] = {}
    for index, region in enumerate(REGIONS):
        values = _patient_region_delta_rms(true_delta, val_weights[:, index])
        private_true_delta[region] = values
        true_delta_rms[region] = _summary(values)

    delta_diagnostics: dict[str, dict[str, Any]] = {}
    private_delta: dict[str, dict[str, Tensor]] = {}
    for method, prediction in predictions.items():
        delta_diagnostics[method] = {}
        private_delta[method] = {}
        for index, region in enumerate(REGIONS):
            public, patient = _delta_diagnostics(
                prediction, source, target, val_weights[:, index]
            )
            delta_diagnostics[method][region] = public
            for name, values in patient.items():
                private_delta[method][f"{region}/{name}"] = values

    reconstructed_means = {
        region: float(torch.nanmean(errors[CANDIDATE][region]))
        for region in ("t0_disease_bed", "full_shared_valid_grid")
    }
    released_summary = training_report["validation"]["validation_region_mae"][CANDIDATE]
    reconstruction_differences = {
        region: abs(reconstructed_means[region] - float(released_summary[region]["mean"]))
        for region in reconstructed_means
    }
    if max(reconstruction_differences.values()) > 5e-4:
        raise ValueError("float16 private prediction no longer reconstructs public v4.1 MAE")

    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    private = {
        "schema_version": PRIVATE_SCHEMA,
        "patient_ids": val_ids,
        "patient_region_errors": errors,
        "patient_region_true_delta_rms": private_true_delta,
        "patient_delta_diagnostics": private_delta,
        "region_weights": val_weights,
        "selected_shrinkage_alpha": selected_alpha,
    }
    private_path = private_output / "validation_regional_measurements.pt"
    torch.save(private, private_path)
    os.chmod(private_path, 0o600)

    report = {
        "schema_version": SCHEMA,
        "status": "completed_read_only_development_regional_specificity_audit",
        "scope": "posthoc_development_only_no_retraining_test_sealed",
        "patient_counts": {"train": 409, "val": 88, "test": 0},
        "regions": list(REGIONS),
        "methods": list(predictions),
        "primary_candidate": CANDIDATE,
        "patient_region_mae": summaries,
        "paired_patient_bootstrap": comparisons,
        "disease_specificity": {
            "primary_estimand": "relative_MAE_improvement_in_t0_disease_bed_minus_relative_MAE_improvement_in_valid_non_disease_bed",
            "comparisons": specificity,
        },
        "region_occupancy": {
            "train": _occupancy_summary(train_weights),
            "validation": _occupancy_summary(val_weights),
            "weights": "T0_fractional_occupancy_intersected_with_T0_T1_T3_shared_acquisition_valid_support",
        },
        "validation_true_t1_to_t3_delta_rms": true_delta_rms,
        "delta_shrinkage_and_alignment": delta_diagnostics,
        "train_only_shrinkage_baseline": {
            "form": "alpha_times_T1_plus_one_minus_alpha_times_train_mean_T3",
            "selection": "train_only_leave_one_patient_out_t0_disease_bed_MAE",
            "selected_alpha": selected_alpha,
            "alpha_grid": list(alpha_grid),
            "train_leave_one_out_mae_by_alpha": alpha_scores,
        },
        "prediction_serialization_audit": {
            "v4p1_private_prediction_dtype": "float16",
            "comparison_dtype": "float32_after_loading",
            "maximum_allowed_public_MAE_reconstruction_difference": 5e-4,
            "observed_absolute_differences": reconstruction_differences,
        },
        "interpretation_contract": {
            "fixed_effective_rank_gate_used": False,
            "region_weighting_changed_during_training": False,
            "posthoc_exploratory": True,
            "confirmatory_claim_permitted": False,
            "next_decision": "use_region_and_delta_results_to_freeze_nuisance_suppression_and_resolution_experiments_before_any_new_training",
        },
        "retraining_performed": False,
        "model_forward_performed": False,
        "raw_dicom_read": False,
        "future_pixel_reconstruction": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "training_report_sha256": _sha(training_report_path),
            "training_private_measurements_sha256": _sha(training_private_path),
            "full_token_baseline_report_sha256": _sha(baseline_report_path),
            "full_token_baseline_private_sha256": _sha(baseline_private_path),
            "manifest_sha256": _sha(manifest_path),
            "manifest_build_report_sha256": _sha(manifest_report_path),
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
    print(
        json.dumps(
            {
                "status": report["status"],
                "regions": list(REGIONS),
                "methods": list(predictions),
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True)
    parser.add_argument("--training-private", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--baseline-private", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "disease_specificity_bootstrap",
    "fit_shrinkage_alpha",
    "run",
]
