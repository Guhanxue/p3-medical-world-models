"""Development-only full-token linear dynamics baselines for Med-JEPA v4.1.

The estimator is dual (kernel) ridge with a strictly linear kernel.  It consumes
and predicts the complete 1024x7x7x7 frozen-token grid; no global broadcast,
coarse pooling, interpolation, image decoder, or held-out test record is used.
Hyperparameters are selected by grouped folds inside the 409-patient training
split, then the 88-patient validation split is evaluated once.
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
import torch.nn.functional as F
from torch import Tensor

from med_jepa.frozen_audit.statistics import TrainStandardizer
from med_jepa.roi_v4.audit_masked_cover import representation_rank_diagnostics
from med_jepa.roi_v4.audit_mechanistic import (
    REGIONS,
    _bootstrap_from_errors,
    _delta_diagnostics,
    _load_region_weights,
    _retrieval,
    _summary,
    _validate_mask_release,
    _weighted_pool,
    patient_region_mae,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4.train import _record_tensor


SCHEMA = "med-jepa-roi-v4p1-full-token-linear-baseline-v1"
PRIVATE_SCHEMA = "med-jepa-roi-v4p1-private-full-token-predictions-v1"
RIDGE_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
INFORMATION_SETS = ("markov_matched", "history", "conditional_history")


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


def _folds(size: int, *, seed: int, count: int = 5) -> list[tuple[Tensor, Tensor]]:
    if size < count:
        raise ValueError("too few patients for grouped baseline folds")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    result: list[tuple[Tensor, Tensor]] = []
    for fold in range(count):
        validation = order[fold::count]
        keep = torch.ones(size, dtype=torch.bool)
        keep[validation] = False
        result.append((torch.arange(size)[keep], validation))
    return result


def _center_train_and_cross(
    raw_train: Tensor,
    raw_cross: Tensor,
) -> tuple[Tensor, Tensor]:
    """Center a raw linear Gram matrix using only the training rows."""

    if raw_train.ndim != 2 or raw_train.shape[0] != raw_train.shape[1]:
        raise ValueError("training Gram matrix must be square")
    if raw_cross.ndim != 2 or raw_cross.shape[1] != raw_train.shape[0]:
        raise ValueError("cross Gram matrix has incompatible dimensions")
    column_mean = raw_train.mean(0, keepdim=True)
    grand_mean = raw_train.mean()
    centered_train = (
        raw_train
        - raw_train.mean(1, keepdim=True)
        - column_mean
        + grand_mean
    )
    centered_cross = (
        raw_cross
        - raw_cross.mean(1, keepdim=True)
        - column_mean
        + grand_mean
    )
    return centered_train, centered_cross


def _token_components(features: Tensor, information_set: str) -> tuple[Tensor, ...]:
    if information_set == "markov_matched":
        return (features[:, 1].flatten(1).float(),)
    if information_set in {"history", "conditional_history"}:
        return (
            features[:, 0].flatten(1).float(),
            features[:, 1].flatten(1).float(),
            (features[:, 1] - features[:, 0]).flatten(1).float(),
        )
    raise ValueError("unknown full-token information set")


def _covariates(
    records: Sequence[PatientStates], information_set: str
) -> Tensor:
    time = torch.stack([record.times_days for record in records]).float()
    action = torch.stack([record.action_13 for record in records]).float()
    dt = torch.log1p(
        ((time[:, 2] - time[:, 1]).clamp_min(0) / 30.0)
    ).unsqueeze(1)
    source_time = torch.log1p((time[:, 1].clamp_min(0) / 30.0)).unsqueeze(1)
    common = torch.cat((dt, dt.sin(), dt.cos(), source_time, action), dim=1)
    if information_set != "conditional_history":
        return common
    clinical = torch.stack([record.clinical for record in records]).float()
    hr = clinical[:, 0].round().long().clamp(0, 1)
    her2 = clinical[:, 1].round().long().clamp(0, 1)
    subtype = F.one_hot(hr * 2 + her2, num_classes=4).float()
    interactions = (subtype[:, :, None] * action[:, None, :]).flatten(1)
    return torch.cat((common, clinical, subtype, interactions), dim=1)


def _raw_kernel(
    train_features: Tensor,
    query_features: Tensor,
    train_records: Sequence[PatientStates],
    query_records: Sequence[PatientStates],
    information_set: str,
    *,
    device: torch.device,
    train_indices: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Construct equal-weight linear token and covariate-group kernels."""

    train_parts = _token_components(train_features, information_set)
    query_parts = _token_components(query_features, information_set)
    if train_indices is not None:
        train_parts = tuple(part[train_indices] for part in train_parts)
        selected_records = [train_records[int(index)] for index in train_indices]
    else:
        selected_records = list(train_records)
    groups_train: list[Tensor] = list(train_parts)
    groups_query: list[Tensor] = list(query_parts)
    train_cov = _covariates(selected_records, information_set)
    query_cov = _covariates(query_records, information_set)
    mean = train_cov.mean(0, keepdim=True)
    std = train_cov.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    groups_train.append((train_cov - mean) / std)
    groups_query.append((query_cov - mean) / std)
    raw_train = torch.zeros(
        len(selected_records), len(selected_records), dtype=torch.float32, device=device
    )
    raw_cross = torch.zeros(
        len(query_records), len(selected_records), dtype=torch.float32, device=device
    )
    for train_group, query_group in zip(groups_train, groups_query, strict=True):
        scale = max(1, train_group.shape[1])
        train_device = train_group.float().to(device)
        query_device = query_group.float().to(device)
        raw_train.add_(train_device @ train_device.T / float(scale))
        raw_cross.add_(query_device @ train_device.T / float(scale))
    raw_train.div_(len(groups_train))
    raw_cross.div_(len(groups_train))
    return raw_train.cpu(), raw_cross.cpu()


def _dual_predict(
    centered_train_kernel: Tensor,
    centered_cross_kernel: Tensor,
    train_target: Tensor,
    *,
    alpha: float,
    device: torch.device,
) -> Tensor:
    target = train_target.float().to(device)
    target_mean = target.mean(0, keepdim=True)
    centered_target = target - target_mean
    kernel = centered_train_kernel.float().to(device)
    cross = centered_cross_kernel.float().to(device)
    identity = torch.eye(kernel.shape[0], device=device, dtype=kernel.dtype)
    coefficients = torch.linalg.solve(kernel + float(alpha) * identity, centered_target)
    return (cross @ coefficients + target_mean).cpu()


def fit_full_token_kernel_ridge(
    train: Sequence[PatientStates],
    val: Sequence[PatientStates],
    *,
    information_set: str,
    seed: int,
    device: torch.device,
) -> tuple[Tensor, dict[str, Any]]:
    """Select ridge strength inside train and predict the full validation delta."""

    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    train_delta = (train_features[:, 2] - train_features[:, 1]).flatten(1)
    scores = {alpha: [] for alpha in RIDGE_ALPHAS}
    for train_index, held_index in _folds(len(train), seed=seed):
        fold_train = [train[int(index)] for index in train_index]
        fold_held = [train[int(index)] for index in held_index]
        raw_train, raw_cross = _raw_kernel(
            train_features,
            train_features[held_index],
            train,
            fold_held,
            information_set,
            device=device,
            train_indices=train_index,
        )
        kernel, cross = _center_train_and_cross(raw_train, raw_cross)
        target = train_delta[train_index]
        truth = train_delta[held_index]
        for alpha in RIDGE_ALPHAS:
            prediction = _dual_predict(
                kernel, cross, target, alpha=alpha, device=device
            )
            scores[alpha].append(float((prediction - truth).abs().mean()))
    mean_scores = {alpha: sum(values) / len(values) for alpha, values in scores.items()}
    selected = min(mean_scores, key=lambda alpha: (mean_scores[alpha], alpha))
    raw_train, raw_val = _raw_kernel(
        train_features,
        val_features,
        train,
        val,
        information_set,
        device=device,
    )
    kernel, cross = _center_train_and_cross(raw_train, raw_val)
    prediction = _dual_predict(
        kernel, cross, train_delta, alpha=selected, device=device
    ).reshape(-1, 1024, 7, 7, 7)
    report = {
        "model": "full_grid_dual_linear_kernel_ridge_v1",
        "information_set": information_set,
        "selected_alpha": selected,
        "train_grouped_5fold_full_grid_mae": {
            str(alpha): value for alpha, value in mean_scores.items()
        },
        "input_token_grid": [1024, 7, 7, 7],
        "output_delta_grid": [1024, 7, 7, 7],
        "pooling_or_interpolation": False,
        "linear_kernel": True,
    }
    return prediction, report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("full-token linear audit requires exactly one CUDA GPU")
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_report_path = Path(args.manifest_build_report).expanduser().resolve()
    mask_root = Path(args.context_mask_release).expanduser().resolve()
    mask_report_path = Path(args.context_mask_report).expanduser().resolve()
    mask_index_path = Path(args.context_mask_index).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable full-token baseline output exists")
    manifest_report = _read_json(manifest_report_path)
    if (
        manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(manifest_path)
        or manifest_report.get("records") != 587
    ):
        raise ValueError("manifest release is not approved")
    train = load_development_states(manifest_path, "train")
    val = load_development_states(manifest_path, "val")
    development_ids = {item.patient_id for item in (*train, *val)}
    mask_records = _validate_mask_release(
        mask_root, mask_report_path, mask_index_path, development_ids
    )
    train_weights = _load_region_weights(train, mask_records)
    val_weights = _load_region_weights(val, mask_records)
    train_features = _record_tensor(train, "features")
    val_features = _record_tensor(val, "features")
    source = val_features[:, 1]
    target = val_features[:, 2]
    predictions: dict[str, Tensor] = {"latent_persistence": source}
    model_reports: dict[str, Any] = {}
    device = torch.device("cuda")
    for counter, information_set in enumerate(INFORMATION_SETS):
        delta, model_report = fit_full_token_kernel_ridge(
            train,
            val,
            information_set=information_set,
            seed=int(args.seed) + counter * 1009,
            device=device,
        )
        name = f"full_token_{information_set}_linear_ridge"
        predictions[name] = source + delta
        model_reports[name] = model_report
    val_ids = [item.patient_id for item in val]
    errors: dict[str, dict[str, Tensor]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for method, prediction in predictions.items():
        errors[method] = {}
        summaries[method] = {}
        for region_index, region in enumerate(REGIONS):
            values = patient_region_mae(prediction, target, val_weights[:, region_index])
            errors[method][region] = values
            summaries[method][region] = _summary(values)
    comparisons: dict[str, Any] = {}
    primary = errors["latent_persistence"]["t0_disease_bed"]
    for counter, method in enumerate(predictions):
        if method == "latent_persistence":
            continue
        comparisons[f"{method}_vs_persistence"] = _bootstrap_from_errors(
            errors[method]["t0_disease_bed"],
            primary,
            val_ids,
            seed=int(args.bootstrap_seed) + counter,
            replicates=10_000,
        )
    disease = REGIONS.index("t0_disease_bed")
    train_target_pooled = _weighted_pool(
        train_features[:, 2], train_weights[:, disease]
    )
    train_eligible = torch.isfinite(train_target_pooled).all(1)
    val_eligible = val_weights[:, disease].flatten(1).sum(1) > 0
    scaler = TrainStandardizer().fit(train_target_pooled[train_eligible])
    retrieval: dict[str, Any] = {}
    rank: dict[str, Any] = {}
    delta_diagnostics: dict[str, Any] = {}
    private_retrieval: dict[str, Tensor] = {}
    for method, prediction in predictions.items():
        retrieval[method], private_retrieval[method] = _retrieval(
            prediction, target, val_weights[:, disease], scaler, val_eligible
        )
        rank[method] = {
            "predicted_t3": representation_rank_diagnostics(prediction, device),
            "predicted_delta": representation_rank_diagnostics(
                prediction - source, device
            ),
        }
        delta_diagnostics[method], _ = _delta_diagnostics(
            prediction, source, target, val_weights[:, disease]
        )
    output.mkdir(parents=True, mode=0o700)
    private_output.mkdir(parents=True, mode=0o700)
    private_path = private_output / "validation_full_token_predictions.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": val_ids,
            "predictions": {
                # Keep the exact float32 values used for the public MAE so the
                # later v4.1 gate cannot move because of serialization
                # quantization.
                name: value.float() for name, value in predictions.items()
            },
            "absolute_errors": errors,
            "retrieval_ranks": private_retrieval,
        },
        private_path,
    )
    report = {
        "schema_version": SCHEMA,
        "status": "completed_development_full_token_linear_baselines",
        "scope": "train_fit_validation_once_test_sealed",
        "patient_counts": {"train": len(train), "val": len(val), "test": 0},
        "models": model_reports,
        "validation_region_mae": summaries,
        "disease_bed_paired_bootstrap": comparisons,
        "patient_retrieval": retrieval,
        "representation_rank": rank,
        "delta_diagnostics": delta_diagnostics,
        "primary_estimand": "patient_mean_t0_disease_bed_weighted_full_7cube_token_mae",
        "test_split_loaded": False,
        "future_pixel_reconstruction": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "manifest_sha256": _sha(manifest_path),
            "manifest_build_report_sha256": _sha(manifest_report_path),
            "context_mask_report_sha256": _sha(mask_report_path),
            "context_mask_index_sha256": _sha(mask_index_path),
            "source_sha256": _sha(Path(__file__)),
            "private_predictions_sha256": _sha(private_path),
        },
    }
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    for path in (private_path, report_path):
        os.chmod(path, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "models": sorted(model_reports),
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--bootstrap-seed", type=int, default=20260811)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "_center_train_and_cross",
    "_dual_predict",
    "fit_full_token_kernel_ridge",
    "run",
]
