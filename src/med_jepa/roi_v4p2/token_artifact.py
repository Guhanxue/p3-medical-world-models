"""Train-locked token-norm and tumour-locality primitives for v4.2r.

The functions in this module are deliberately independent of model training.
They operate on the already governed, final-normalized 3DINO token release.
Consequently, a negative norm-tail result only rejects a sparse *final-token
norm* explanation; it does not rule out hidden-layer register neurons.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS, _summary
from med_jepa.roi_v4.data import PatientStates


@dataclass(frozen=True)
class BinaryRankingMetrics:
    auroc: float
    auprc: float
    brier: float
    prevalence: float
    n_observations: int
    n_positive: int
    n_negative: int


def token_l2_norm(features: Tensor) -> Tensor:
    """Return per-token channel L2 norm for ``[..., C, 7, 7, 7]``."""

    if features.ndim < 4 or tuple(features.shape[-3:]) != (7, 7, 7):
        raise ValueError("features must end in [C,7,7,7]")
    if features.shape[-4] < 2 or not torch.isfinite(features).all():
        raise ValueError("features must contain finite channel vectors")
    return torch.linalg.vector_norm(features.float(), dim=-4)


def _quantiles(values: Tensor) -> dict[str, float]:
    flat = values.detach().float().flatten()
    if flat.numel() < 2 or not torch.isfinite(flat).all() or torch.any(flat <= 0):
        raise ValueError("token norms must be finite and strictly positive")
    levels = torch.tensor(
        [0.5, 0.9, 0.95, 0.99, 0.995, 0.999], dtype=flat.dtype
    )
    result = torch.quantile(flat, levels)
    return {
        "q50": float(result[0]),
        "q90": float(result[1]),
        "q95": float(result[2]),
        "q99": float(result[3]),
        "q99_5": float(result[4]),
        "q99_9": float(result[5]),
        "maximum": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std(unbiased=False)),
        "count": int(flat.numel()),
    }


def freeze_source_norm_rule(source_norms: Tensor) -> dict[str, Any]:
    """Freeze a sparse, train-only norm rule without forcing 1% positives.

    The primary threshold is the larger of a five-MAD log-norm threshold and
    1.5 times the train T1 median norm.  Unlike a percentile cut, this rule can
    return zero outliers and does not discard all but the upper tail of a true
    1--3% artifact population.  A separate heavy-tail signature also requires
    large norm ratios, so numerical LayerNorm variation cannot be mistaken for
    the DINO outlier-token phenomenon.
    """

    flat = source_norms.detach().float().flatten()
    stats = _quantiles(flat)
    log_values = flat.log()
    median = torch.median(log_values)
    mad = torch.median((log_values - median).abs())
    robust_sigma = 1.4826 * mad
    robust_threshold = torch.exp(median + 5.0 * robust_sigma)
    amplitude_floor = 1.5 * float(torch.exp(median))
    threshold = max(float(robust_threshold), amplitude_floor)
    observed = flat > threshold
    prevalence = float(observed.float().mean())
    q99_ratio = stats["q99"] / stats["q50"]
    max_ratio = stats["maximum"] / stats["q50"]
    sparse = 0.0005 <= prevalence <= 0.05
    ratio_signature = q99_ratio >= 1.5 and max_ratio >= 2.0
    return {
        "rule": "train_T1_log_norm_median_plus_5_scaled_MAD_with_1p5x_median_floor_v1",
        "threshold": threshold,
        "log_norm_median": float(median),
        "log_norm_MAD": float(mad),
        "log_norm_scaled_MAD": float(robust_sigma),
        "robust_threshold_before_amplitude_floor": float(robust_threshold),
        "one_point_five_times_median_amplitude_floor": amplitude_floor,
        "train_source_outlier_fraction": prevalence,
        "q99_to_median_ratio": q99_ratio,
        "maximum_to_median_ratio": max_ratio,
        "sparse_prevalence_check": sparse,
        "large_norm_ratio_check": ratio_signature,
        "final_norm_sparse_heavy_tail_signature": sparse and ratio_signature,
        "norm_distribution": stats,
        "strictly_greater_than_threshold": True,
        "threshold_selected_without_validation": True,
    }


def apply_source_norm_rule(source_features: Tensor, rule: Mapping[str, Any]) -> Tensor:
    threshold = float(rule.get("threshold", float("nan")))
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("frozen token-norm threshold is invalid")
    return token_l2_norm(source_features) > threshold


def _fold(patient_id: str, folds: int) -> int:
    digest = hashlib.sha256(patient_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % folds


def _flat_t0(record: PatientStates) -> Tensor:
    value = record.features[0].float().permute(1, 2, 3, 0).reshape(-1, 1024)
    if value.shape != (343, 1024) or not torch.isfinite(value).all():
        raise ValueError("T0 token feature payload is invalid")
    return value


def _flat_disease(weights: Tensor) -> Tensor:
    if weights.shape != (len(REGIONS), 7, 7, 7):
        raise ValueError("region weight payload is invalid")
    return weights[REGIONS.index("t0_disease_bed")].flatten() > 0


def binary_ranking_metrics(probabilities: Tensor, labels: Tensor) -> dict[str, Any]:
    """Tie-aware O(n log n) AUROC/AP without a positive-negative matrix."""

    scores = probabilities.detach().double().flatten()
    target = labels.detach().bool().flatten()
    if (
        scores.shape != target.shape
        or scores.numel() < 2
        or not torch.isfinite(scores).all()
        or torch.any((scores < 0) | (scores > 1))
    ):
        raise ValueError("binary ranking inputs are invalid")
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives < 1 or negatives < 1:
        raise ValueError("binary ranking requires both classes")

    order = torch.argsort(scores, stable=True)
    sorted_scores = scores[order]
    sorted_target = target[order]
    _, counts = torch.unique_consecutive(sorted_scores, return_counts=True)
    ends = counts.cumsum(0)
    starts = ends - counts
    cumulative_positive = sorted_target.long().cumsum(0)
    group_positive = cumulative_positive[ends - 1].clone()
    prior_ends = starts - 1
    has_prior = prior_ends >= 0
    group_positive[has_prior] -= cumulative_positive[prior_ends[has_prior]]
    group_negative = counts - group_positive
    negative_before = torch.cat(
        (torch.zeros(1, dtype=torch.long), group_negative.cumsum(0)[:-1])
    )
    wins = (
        group_positive.double() * negative_before.double()
        + 0.5 * group_positive.double() * group_negative.double()
    ).sum()
    auroc = wins / float(positives * negatives)

    descending = torch.argsort(scores, descending=True, stable=True)
    desc_scores = scores[descending]
    desc_target = target[descending].double()
    group_end = torch.ones(scores.numel(), dtype=torch.bool)
    group_end[:-1] = desc_scores[:-1] != desc_scores[1:]
    cumulative = desc_target.cumsum(0)
    positions = torch.arange(1, scores.numel() + 1, dtype=torch.double)
    tp = cumulative[group_end]
    precision = tp / positions[group_end]
    recall = tp / positives
    previous = torch.cat((torch.zeros(1, dtype=torch.double), recall[:-1]))
    auprc = ((recall - previous) * precision).sum()
    value = BinaryRankingMetrics(
        auroc=float(auroc),
        auprc=float(auprc),
        brier=float((scores - target.double()).square().mean()),
        prevalence=positives / scores.numel(),
        n_observations=int(scores.numel()),
        n_positive=positives,
        n_negative=negatives,
    )
    return asdict(value)


def _class_sufficient_statistics(
    records: Sequence[PatientStates], weights: Tensor
) -> dict[str, Tensor]:
    count = torch.zeros((len(records), 2), dtype=torch.double)
    total = torch.zeros((len(records), 2, 1024), dtype=torch.double)
    square = torch.zeros_like(total)
    for index, record in enumerate(records):
        x = _flat_t0(record).double()
        positive = _flat_disease(weights[index])
        for label, mask in ((0, ~positive), (1, positive)):
            if int(mask.sum()) == 0:
                continue
            value = x[mask]
            count[index, label] = value.shape[0]
            total[index, label] = value.sum(0)
            square[index, label] = value.square().sum(0)
    return {"count": count, "sum": total, "square": square}


def _fit_diagonal_lda(
    statistics: Mapping[str, Tensor], patient_indices: Tensor
) -> dict[str, Tensor | float]:
    count = statistics["count"][patient_indices].sum(0)
    total = statistics["sum"][patient_indices].sum(0)
    square = statistics["square"][patient_indices].sum(0)
    if torch.any(count < 2):
        raise ValueError("locality fold lacks one token class")
    mean = total / count[:, None]
    within = (square - total.square() / count[:, None]).clamp_min(0).sum(0)
    denominator = (count.sum() - 2).clamp_min(1)
    variance = within / denominator
    positive_variance = variance[variance > 0]
    scale_floor = (
        float(torch.median(positive_variance)) if positive_variance.numel() else 1.0
    )
    ridge = max(scale_floor * 1e-3, 1e-8)
    coefficient = (mean[1] - mean[0]) / (variance + ridge)
    coefficient = coefficient / coefficient.norm().clamp_min(1e-12)
    prior = (count[1] / count.sum()).clamp(1e-8, 1 - 1e-8)
    intercept = -0.5 * torch.dot(mean[1] + mean[0], coefficient)
    prior_logit = torch.log(prior / (1.0 - prior))
    score_scale = torch.sqrt(
        (coefficient.square() * variance.clamp_min(ridge)).sum()
    ).clamp_min(1e-8)
    return {
        "coefficient": coefficient.float(),
        "intercept": float(intercept),
        "score_scale": float(score_scale),
        "prior_logit": float(prior_logit),
        "ridge": ridge,
        "positive_prevalence": float(prior),
    }


def apply_locality_probe(features: Tensor, model: Mapping[str, Any]) -> Tensor:
    """Apply the train-fitted diagonal probe to ``[N,C,7,7,7]`` T0 tokens."""

    if features.ndim != 5 or features.shape[1:] != (1024, 7, 7, 7):
        raise ValueError("locality probe features must be [N,1024,7,7,7]")
    coefficient = model.get("coefficient")
    if not isinstance(coefficient, Tensor) or coefficient.shape != (1024,):
        raise ValueError("locality probe coefficient is invalid")
    intercept = float(model.get("intercept", float("nan")))
    scale = float(model.get("score_scale", float("nan")))
    prior_logit = float(model.get("prior_logit", 0.0))
    if (
        not math.isfinite(intercept)
        or not math.isfinite(scale)
        or not math.isfinite(prior_logit)
        or scale <= 0
    ):
        raise ValueError("locality probe calibration is invalid")
    flat = features.float().permute(0, 2, 3, 4, 1).reshape(-1, 1024)
    score = (flat @ coefficient.float() + intercept) / scale + prior_logit
    return torch.sigmoid(score.clamp(-40, 40)).reshape(-1, 7, 7, 7)


def train_grouped_locality_probe(
    records: Sequence[PatientStates], weights: Tensor, *, folds: int = 5
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fit and audit a patient-grouped train-only T0 tumour-locality probe."""

    if len(records) != 409 or weights.shape != (409, len(REGIONS), 7, 7, 7):
        raise ValueError("locality probe requires the locked 409-patient release")
    if folds != 5:
        raise ValueError("locality probe fold count is frozen at five")
    assignments = torch.tensor([_fold(item.patient_id, folds) for item in records])
    if set(assignments.tolist()) != set(range(folds)):
        raise ValueError("hashed grouped folds are unexpectedly empty")
    stats = _class_sufficient_statistics(records, weights)
    probabilities: list[Tensor] = []
    labels: list[Tensor] = []
    patient_metrics: list[dict[str, Any]] = []
    for fold in range(folds):
        train_indices = torch.where(assignments != fold)[0]
        held_indices = torch.where(assignments == fold)[0]
        model = _fit_diagonal_lda(stats, train_indices)
        for index in held_indices.tolist():
            feature = records[index].features[0].unsqueeze(0)
            probability = apply_locality_probe(feature, model).flatten()
            label = _flat_disease(weights[index])
            probabilities.append(probability)
            labels.append(label)
            if int(label.sum()) > 0 and int((~label).sum()) > 0:
                patient_metrics.append(binary_ranking_metrics(probability, label))
    all_probability = torch.cat(probabilities)
    all_label = torch.cat(labels)
    overall = binary_ranking_metrics(all_probability, all_label)
    per_patient_auprc = torch.tensor(
        [item["auprc"] for item in patient_metrics], dtype=torch.float32
    )
    per_patient_auroc = torch.tensor(
        [item["auroc"] for item in patient_metrics], dtype=torch.float32
    )
    overall["AUPRC_lift_over_prevalence"] = (
        overall["auprc"] / overall["prevalence"]
    )
    overall["patient_grouped_folds"] = folds
    overall["eligible_patient_count"] = len(patient_metrics)
    overall["per_patient_AUPRC"] = _summary(per_patient_auprc)
    overall["per_patient_AUROC"] = _summary(per_patient_auroc)
    all_indices = torch.arange(len(records))
    fitted = _fit_diagonal_lda(stats, all_indices)
    private_model = {
        "coefficient": fitted["coefficient"],
        "intercept": fitted["intercept"],
        "score_scale": fitted["score_scale"],
        "prior_logit": fitted["prior_logit"],
        "ridge": fitted["ridge"],
        "positive_prevalence": fitted["positive_prevalence"],
    }
    return overall, private_model


def region_group_errors(
    prediction: Tensor,
    target: Tensor,
    weights: Tensor,
    selector: Tensor,
) -> Tensor:
    """Return five-region patient MAE after a source-defined token selector."""

    if selector.shape != prediction.shape[:1] + prediction.shape[-3:]:
        raise ValueError("token selector does not align")
    from med_jepa.roi_v4.audit_mechanistic import patient_region_mae

    return torch.stack(
        [
            patient_region_mae(
                prediction,
                target,
                weights[:, index] * selector.float(),
            )
            for index in range(len(REGIONS))
        ],
        dim=1,
    )


def contribution_decomposition(
    prediction: Tensor,
    reference: Tensor,
    target: Tensor,
    weights: Tensor,
    outlier: Tensor,
) -> dict[str, Any]:
    """Partition each region's weighted improvement by source-token group."""

    candidate_error = (prediction.float() - target.float()).abs().mean(1)
    reference_error = (reference.float() - target.float()).abs().mean(1)
    improvement = reference_error - candidate_error
    result: dict[str, Any] = {}
    for index, region in enumerate(REGIONS):
        weight = weights[:, index].float()
        all_weight = float(weight.sum())
        high_weight = float((weight * outlier.float()).sum())
        total_gain = float((weight * improvement).sum())
        high_gain = float((weight * improvement * outlier.float()).sum())
        normal_gain = total_gain - high_gain
        result[region] = {
            "source_outlier_weight_fraction": (
                high_weight / all_weight if all_weight > 0 else None
            ),
            "weighted_total_improvement_numerator": total_gain,
            "weighted_outlier_improvement_numerator": high_gain,
            "weighted_normal_improvement_numerator": normal_gain,
            "outlier_share_of_total_improvement": (
                high_gain / total_gain if total_gain != 0 else None
            ),
        }
    return result


__all__ = [
    "apply_locality_probe",
    "apply_source_norm_rule",
    "binary_ranking_metrics",
    "contribution_decomposition",
    "freeze_source_norm_rule",
    "region_group_errors",
    "token_l2_norm",
    "train_grouped_locality_probe",
]
