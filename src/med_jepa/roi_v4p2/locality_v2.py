"""Train409-only regional and fractional-occupancy probes for local64 tokens."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary
from med_jepa.roi_v4.data import PatientStates
from med_jepa.roi_v4p2.data import load_selected_train_release
from med_jepa.roi_v4p2.lock import SELECTED_VARIANT
from med_jepa.roi_v4p2.recovery import SCHEMA as RECOVERY_SCHEMA
from med_jepa.roi_v4p2.token_artifact import binary_ranking_metrics


SCHEMA = "med-jepa-roi-v4p2-train409-locality-v2-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2-private-train409-locality-v2-v1"
PROBE_REGIONS = ("t0_core", "t0_20mm_ring", "t0_disease_bed")
FOLDS = 5
DIAGNOSTIC_THRESHOLDS = {
    "core_token_AUROC_minimum": 0.75,
    "core_token_AUPRC_lift_over_prevalence_minimum": 2.0,
    "core_fractional_occupancy_pooled_Spearman_minimum": 0.40,
    "core_fractional_occupancy_median_patient_Spearman_minimum": 0.20,
}


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


def _fold(patient_id: str) -> int:
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % FOLDS


def _flat_feature(record: PatientStates) -> Tensor:
    value = record.features[0].float().permute(1, 2, 3, 0).reshape(343, 1024)
    if not torch.isfinite(value).all():
        raise ValueError("T0 token features are non-finite")
    return value


def _flat_targets(weights: Tensor) -> Tensor:
    indices = [REGIONS.index(region) for region in PROBE_REGIONS]
    value = weights[indices].float().flatten(1).transpose(0, 1).contiguous()
    if value.shape != (343, len(PROBE_REGIONS)) or torch.any((value < 0) | (value > 1)):
        raise ValueError("fractional occupancy targets are invalid")
    return value


def _sufficient_statistics(
    records: Sequence[PatientStates], weights: Tensor
) -> dict[str, Tensor]:
    patients = len(records)
    regions = len(PROBE_REGIONS)
    x_count = torch.full((patients,), 343.0, dtype=torch.double)
    x_sum = torch.zeros((patients, 1024), dtype=torch.double)
    x_square = torch.zeros_like(x_sum)
    y_sum = torch.zeros((patients, regions), dtype=torch.double)
    y_square = torch.zeros_like(y_sum)
    xy_sum = torch.zeros((patients, regions, 1024), dtype=torch.double)
    class_count = torch.zeros((patients, regions, 2), dtype=torch.double)
    class_sum = torch.zeros((patients, regions, 2, 1024), dtype=torch.double)
    class_square = torch.zeros_like(class_sum)
    for index, record in enumerate(records):
        x = _flat_feature(record).double()
        y = _flat_targets(weights[index]).double()
        x_sum[index] = x.sum(0)
        x_square[index] = x.square().sum(0)
        y_sum[index] = y.sum(0)
        y_square[index] = y.square().sum(0)
        xy_sum[index] = y.transpose(0, 1) @ x
        binary = y > 0
        for region in range(regions):
            for label, selector in ((0, ~binary[:, region]), (1, binary[:, region])):
                count = int(selector.sum())
                if count == 0:
                    continue
                selected = x[selector]
                class_count[index, region, label] = count
                class_sum[index, region, label] = selected.sum(0)
                class_square[index, region, label] = selected.square().sum(0)
    return {
        "x_count": x_count,
        "x_sum": x_sum,
        "x_square": x_square,
        "y_sum": y_sum,
        "y_square": y_square,
        "xy_sum": xy_sum,
        "class_count": class_count,
        "class_sum": class_sum,
        "class_square": class_square,
    }


def _fit_binary_directions(
    statistics: Mapping[str, Tensor], indices: Tensor
) -> dict[str, Tensor]:
    regions = len(PROBE_REGIONS)
    coefficients = torch.zeros((1024, regions), dtype=torch.float32)
    intercepts = torch.zeros(regions, dtype=torch.float32)
    scales = torch.zeros(regions, dtype=torch.float32)
    prior_logits = torch.zeros(regions, dtype=torch.float32)
    for region in range(regions):
        count = statistics["class_count"][indices, region].sum(0)
        total = statistics["class_sum"][indices, region].sum(0)
        square = statistics["class_square"][indices, region].sum(0)
        if torch.any(count < 2):
            raise ValueError("binary locality fold lacks one token class")
        mean = total / count[:, None]
        within = (square - total.square() / count[:, None]).clamp_min(0).sum(0)
        variance = within / (count.sum() - 2).clamp_min(1)
        positive = variance[variance > 0]
        variance_scale = float(torch.median(positive)) if positive.numel() else 1.0
        ridge = max(variance_scale * 1e-3, 1e-8)
        coefficient = (mean[1] - mean[0]) / (variance + ridge)
        coefficient = coefficient / coefficient.norm().clamp_min(1e-12)
        prior = (count[1] / count.sum()).clamp(1e-8, 1 - 1e-8)
        coefficients[:, region] = coefficient.float()
        intercepts[region] = float(-0.5 * torch.dot(mean[1] + mean[0], coefficient))
        scales[region] = float(
            torch.sqrt((coefficient.square() * variance.clamp_min(ridge)).sum())
            .clamp_min(1e-8)
        )
        prior_logits[region] = float(torch.log(prior / (1.0 - prior)))
    return {
        "coefficient": coefficients,
        "intercept": intercepts,
        "score_scale": scales,
        "prior_logit": prior_logits,
    }


def _fit_continuous_directions(
    statistics: Mapping[str, Tensor], indices: Tensor
) -> Tensor:
    count = statistics["x_count"][indices].sum()
    x_sum = statistics["x_sum"][indices].sum(0)
    x_square = statistics["x_square"][indices].sum(0)
    y_sum = statistics["y_sum"][indices].sum(0)
    xy_sum = statistics["xy_sum"][indices].sum(0)
    x_mean = x_sum / count
    y_mean = y_sum / count
    x_variance = (x_square / count - x_mean.square()).clamp_min(0)
    positive = x_variance[x_variance > 0]
    variance_scale = float(torch.median(positive)) if positive.numel() else 1.0
    ridge = max(variance_scale * 1e-3, 1e-8)
    covariance = xy_sum / count - y_mean[:, None] * x_mean[None]
    coefficient = covariance / (x_variance[None] + ridge)
    coefficient = coefficient / coefficient.norm(dim=1, keepdim=True).clamp_min(1e-12)
    return coefficient.transpose(0, 1).float().contiguous()


def _fit_score_calibration(
    records: Sequence[PatientStates],
    weights: Tensor,
    indices: Tensor,
    directions: Tensor,
) -> tuple[Tensor, Tensor]:
    count = 0
    score_sum = torch.zeros(len(PROBE_REGIONS), dtype=torch.double)
    score_square = torch.zeros_like(score_sum)
    target_sum = torch.zeros_like(score_sum)
    score_target = torch.zeros_like(score_sum)
    for index in indices.tolist():
        score = (_flat_feature(records[index]) @ directions).double()
        target = _flat_targets(weights[index]).double()
        count += score.shape[0]
        score_sum += score.sum(0)
        score_square += score.square().sum(0)
        target_sum += target.sum(0)
        score_target += (score * target).sum(0)
    score_mean = score_sum / count
    target_mean = target_sum / count
    score_variance = (score_square / count - score_mean.square()).clamp_min(1e-12)
    covariance = score_target / count - score_mean * target_mean
    slope = covariance / score_variance
    intercept = target_mean - slope * score_mean
    return slope.float(), intercept.float()


def _average_ranks(values: Tensor) -> Tensor:
    value = values.detach().double().flatten()
    order = torch.argsort(value, stable=True)
    sorted_value = value[order]
    _, counts = torch.unique_consecutive(sorted_value, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    group_rank = (starts.double() + ends.double() - 1.0) / 2.0
    sorted_rank = torch.repeat_interleave(group_rank, counts)
    result = torch.empty_like(sorted_rank)
    result[order] = sorted_rank
    return result


def _pearson(left: Tensor, right: Tensor) -> float:
    x = left.detach().double().flatten()
    y = right.detach().double().flatten()
    if x.shape != y.shape or x.numel() < 2 or not torch.isfinite(x).all() or not torch.isfinite(y).all():
        raise ValueError("correlation inputs are invalid")
    x = x - x.mean()
    y = y - y.mean()
    denominator = x.norm() * y.norm()
    if float(denominator) == 0.0:
        raise ValueError("correlation is undefined for a constant variable")
    return float(torch.dot(x, y) / denominator)


def _spearman(left: Tensor, right: Tensor) -> float:
    return _pearson(_average_ranks(left), _average_ranks(right))


def continuous_occupancy_metrics(
    prediction: Tensor, target: Tensor
) -> dict[str, float | int | None]:
    predicted = prediction.detach().double().flatten()
    observed = target.detach().double().flatten()
    if predicted.shape != observed.shape or torch.any((predicted < 0) | (predicted > 1)):
        raise ValueError("fractional occupancy prediction is invalid")
    residual = predicted - observed
    centered = observed - observed.mean()
    total = centered.square().sum()
    if float(total) == 0.0:
        raise ValueError("fractional occupancy target is constant")
    try:
        pearson: float | None = _pearson(predicted, observed)
    except ValueError:
        pearson = None
    try:
        spearman: float | None = _spearman(predicted, observed)
    except ValueError:
        spearman = None
    return {
        "pearson": pearson,
        "spearman": spearman,
        "mae": float(residual.abs().mean()),
        "rmse": float(residual.square().mean().sqrt()),
        "r2": float(1.0 - residual.square().sum() / total),
        "n_observations": int(observed.numel()),
    }


def _patient_continuous_summary(prediction: Tensor, target: Tensor) -> dict[str, Any]:
    pearson: list[float] = []
    spearman: list[float] = []
    for predicted, observed in zip(prediction, target, strict=True):
        if float(observed.max() - observed.min()) == 0.0:
            continue
        try:
            pearson.append(_pearson(predicted, observed))
            spearman.append(_spearman(predicted, observed))
        except ValueError:
            continue
    return {
        "eligible_patient_count": len(spearman),
        "pearson": _summary(torch.tensor(pearson)),
        "spearman": _summary(torch.tensor(spearman)),
    }


def run_grouped_oof_probes(
    records: Sequence[PatientStates], weights: Tensor
) -> tuple[dict[str, Any], dict[str, Any]]:
    if len(records) != 409 or weights.shape != (409, len(REGIONS), 7, 7, 7):
        raise ValueError("locality-v2 requires the locked train409 release")
    assignments = torch.tensor([_fold(item.patient_id) for item in records])
    if set(assignments.tolist()) != set(range(FOLDS)):
        raise ValueError("one or more patient-grouped folds are empty")
    statistics = _sufficient_statistics(records, weights)
    binary_oof = torch.zeros((409, len(PROBE_REGIONS), 343), dtype=torch.float32)
    fractional_oof = torch.zeros_like(binary_oof)
    targets = torch.stack([_flat_targets(value) for value in weights]).transpose(1, 2)
    fold_counts: dict[str, int] = {}
    for fold in range(FOLDS):
        train_indices = torch.where(assignments != fold)[0]
        held_indices = torch.where(assignments == fold)[0]
        fold_counts[str(fold)] = int(held_indices.numel())
        binary = _fit_binary_directions(statistics, train_indices)
        continuous = _fit_continuous_directions(statistics, train_indices)
        slope, intercept = _fit_score_calibration(
            records, weights, train_indices, continuous
        )
        combined = torch.cat((binary["coefficient"], continuous), dim=1)
        for index in held_indices.tolist():
            score = _flat_feature(records[index]) @ combined
            binary_score = score[:, : len(PROBE_REGIONS)]
            probability = torch.sigmoid(
                (
                    binary_score
                    + binary["intercept"][None]
                )
                / binary["score_scale"][None]
                + binary["prior_logit"][None]
            )
            continuous_score = score[:, len(PROBE_REGIONS) :]
            fractional = (
                continuous_score * slope[None] + intercept[None]
            ).clamp(0, 1)
            binary_oof[index] = probability.transpose(0, 1)
            fractional_oof[index] = fractional.transpose(0, 1)

    region_reports: dict[str, Any] = {}
    for region_index, region in enumerate(PROBE_REGIONS):
        target = targets[:, region_index]
        label = target > 0
        probability = binary_oof[:, region_index]
        fractional = fractional_oof[:, region_index]
        binary_metrics = binary_ranking_metrics(probability, label)
        binary_metrics["AUPRC_lift_over_prevalence"] = (
            binary_metrics["auprc"] / binary_metrics["prevalence"]
        )
        per_patient_binary = [
            binary_ranking_metrics(value, truth)
            for value, truth in zip(probability, label, strict=True)
            if int(truth.sum()) > 0 and int((~truth).sum()) > 0
        ]
        binary_metrics["per_patient_AUROC"] = _summary(
            torch.tensor([value["auroc"] for value in per_patient_binary])
        )
        binary_metrics["per_patient_AUPRC"] = _summary(
            torch.tensor([value["auprc"] for value in per_patient_binary])
        )
        continuous = continuous_occupancy_metrics(fractional, target)
        continuous["per_patient"] = _patient_continuous_summary(
            fractional, target
        )
        try:
            integrated_pearson: float | None = _pearson(
                fractional.sum(1), target.sum(1)
            )
        except ValueError:
            integrated_pearson = None
        try:
            integrated_spearman: float | None = _spearman(
                fractional.sum(1), target.sum(1)
            )
        except ValueError:
            integrated_spearman = None
        continuous["patient_integrated_occupancy"] = {
            "pearson": integrated_pearson,
            "spearman": integrated_spearman,
        }
        region_reports[region] = {
            "binary_overlap_probe": binary_metrics,
            "fractional_occupancy_probe": continuous,
            "fractional_occupancy_target": {
                "mean": float(target.mean()),
                "positive_token_fraction": float(label.float().mean()),
                "patient_integrated_summary": _summary(target.sum(1)),
            },
        }

    core = region_reports["t0_core"]
    binary = core["binary_overlap_probe"]
    continuous = core["fractional_occupancy_probe"]
    median_patient_spearman = continuous["per_patient"]["spearman"].get(
        "median", float("nan")
    )
    pooled_spearman = continuous.get("spearman")
    checks = {
        "core_token_AUROC": (
            float(binary["auroc"])
            >= DIAGNOSTIC_THRESHOLDS["core_token_AUROC_minimum"]
        ),
        "core_token_AUPRC_lift_over_prevalence": (
            float(binary["AUPRC_lift_over_prevalence"])
            >= DIAGNOSTIC_THRESHOLDS[
                "core_token_AUPRC_lift_over_prevalence_minimum"
            ]
        ),
        "core_fractional_occupancy_pooled_Spearman": (
            isinstance(pooled_spearman, (int, float))
            and math.isfinite(float(pooled_spearman))
            and float(pooled_spearman)
            >= DIAGNOSTIC_THRESHOLDS[
                "core_fractional_occupancy_pooled_Spearman_minimum"
            ]
        ),
        "core_fractional_occupancy_median_patient_Spearman": (
            math.isfinite(float(median_patient_spearman))
            and float(median_patient_spearman)
            >= DIAGNOSTIC_THRESHOLDS[
                "core_fractional_occupancy_median_patient_Spearman_minimum"
            ]
        ),
    }
    sufficient = all(checks.values())
    public = {
        "patient_grouped_folds": FOLDS,
        "fold_patient_counts": fold_counts,
        "regions": region_reports,
        "diagnostic_contract": {
            "thresholds": DIAGNOSTIC_THRESHOLDS,
            "checks": checks,
            "core_local_information_sufficient_for_v4p3": sufficient,
            "formal_development_gate": False,
            "recommendation": (
                "proceed_to_dual_path_v4p3_train409_design"
                if sufficient
                else "change_frozen_feature_before_further_predictor_tuning"
            ),
        },
    }
    private = {
        "binary_probability_oof": binary_oof.half(),
        "fractional_occupancy_prediction_oof": fractional_oof.half(),
        "fractional_occupancy_target": targets.half(),
        "fold_assignment": assignments,
    }
    return public, private


def run(args: argparse.Namespace) -> dict[str, Any]:
    paths = {
        name: Path(value).expanduser().resolve()
        for name, value in {
            "manifest": args.manifest,
            "triage_private": args.feature_triage_private_root,
            "triage_index": args.feature_triage_private_index,
            "recovery_report": args.recovery_report,
            "output": args.output_dir,
            "private_output": args.private_output_dir,
        }.items()
    }
    final_output = paths["output"]
    final_private = paths["private_output"]
    if final_output.exists() or final_private.exists():
        raise FileExistsError("immutable locality-v2 output exists")
    staging_output = final_output.parent / f".{final_output.name}.staging"
    staging_private = final_private.parent / f".{final_private.name}.staging"
    if staging_output.exists() or staging_private.exists():
        raise FileExistsError("stale locality-v2 staging output exists")
    staging_output.mkdir(parents=True, mode=0o700)
    staging_private.mkdir(parents=True, mode=0o700)

    recovery = _json(paths["recovery_report"])
    evidence = recovery.get("evidence", {})
    if (
        recovery.get("schema_version") != RECOVERY_SCHEMA
        or recovery.get("status") not in {
            "completed_posthoc_selection_recovery_gate_pass",
            "completed_posthoc_selection_recovery_gate_fail",
        }
        or recovery.get("selected_feature_variant") != SELECTED_VARIANT
        or recovery.get("test_split_loaded") is not False
        or not isinstance(evidence, Mapping)
        or evidence.get("feature_triage_private_index_sha256")
        != _sha(paths["triage_index"])
        or evidence.get("manifest_sha256") != _sha(paths["manifest"])
    ):
        raise ValueError("locality-v2 recovery chain is invalid")
    records, weights, train_merkle = load_selected_train_release(
        paths["manifest"],
        paths["triage_private"],
        paths["triage_index"],
        expected_index_sha256=_sha(paths["triage_index"]),
    )
    if train_merkle != evidence.get("train_feature_payload_merkle_sha256"):
        raise ValueError("locality-v2 train409 release changed")
    probes, private = run_grouped_oof_probes(records, weights)
    private.update(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": [record.patient_id for record in records],
            "selected_feature_variant": SELECTED_VARIANT,
        }
    )
    private_path = staging_private / "oof_measurements.pt"
    torch.save(private, private_path)
    os.chmod(private_path, 0o600)
    report = {
        "schema_version": SCHEMA,
        "status": "completed_train409_only_locality_v2",
        "scope": "train409_grouped_OOF_feature_sufficiency_diagnostic",
        "patients": {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0},
        "selected_feature_variant": SELECTED_VARIANT,
        "probe_definition": {
            "binary": "diagonal_LDA_patient_grouped_OOF",
            "continuous": (
                "diagonal_covariance_direction_plus_train_fold_affine_calibration"
            ),
            "binary_label": "fractional_occupancy_greater_than_zero",
            "continuous_target": "fractional_occupancy_in_0_to_1",
            "T0_only": True,
        },
        "results": probes,
        "training_performed": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "formal_development_gate": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "manifest_sha256": _sha(paths["manifest"]),
            "feature_triage_private_index_sha256": _sha(paths["triage_index"]),
            "train_feature_payload_merkle_sha256": train_merkle,
            "recovery_report_sha256": _sha(paths["recovery_report"]),
            "private_oof_measurements_sha256": _sha(private_path),
            "locality_v2_source_sha256": _sha(Path(__file__)),
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
                "recommendation": probes["diagnostic_contract"]["recommendation"],
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--feature-triage-private-root", required=True)
    parser.add_argument("--feature-triage-private-index", required=True)
    parser.add_argument("--recovery-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DIAGNOSTIC_THRESHOLDS",
    "PRIVATE_SCHEMA",
    "PROBE_REGIONS",
    "SCHEMA",
    "continuous_occupancy_metrics",
    "run",
    "run_grouped_oof_probes",
]
