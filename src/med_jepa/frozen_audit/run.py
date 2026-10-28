"""Run the development-only frozen-3DINO sufficiency and dynamics audit.

This command is intentionally separate from every Med-JEPA training entry point.
It never updates a neural network, never reads raw DICOM, never parses the held-out
test JSONL records, and never uses a T3 mask as an input feature.  Its purpose is
to answer two cheaper questions before a v4 architecture is justified:

1. what clinically meaningful information is linearly recoverable from the frozen
   final-layer 3DINO grid; and
2. whether selected pooled/coarse-spatial linear views contain patient-level
   T0/T1-to-T3 signal that a future neural model must beat.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from med_jepa.frozen_audit.features import (
    FEATURE_GRID_SHAPE_DHW,
    FrozenFeatureContractError,
    feature_views,
    load_feature_grid,
    load_mask_sidecar,
    morphology_descriptors,
    occupancy_geometry,
    sha256_file,
    valid_sha256,
)
from med_jepa.frozen_audit.statistics import (
    TorchReducedRankRegressor,
    TorchRidgeRegressor,
    TorchSlowFeatureAnalysis,
    TrainPCA,
    TrainStandardizer,
    classification_metrics,
    grouped_ridge_cv,
    paired_patient_bootstrap,
    patient_group_kfold,
    regression_metrics,
)
from med_jepa.frozen_audit.t2_census import build_t2_census
from med_jepa.ispy2_manifest_builder import (
    ManifestBuildError,
    validate_feature_provenance,
    validate_spatial_provenance,
)


SCHEMA_VERSION = "med-jepa-ispy2-frozen-feature-audit-v1"
PRIVATE_SCHEMA_VERSION = "med-jepa-ispy2-frozen-feature-audit-private-v1"
MANIFEST_REPORT_SCHEMA_VERSION = "med-jepa-ispy2-manifest-build-v1"
MASK_REPORT_SCHEMA_VERSION = "spatial-med-jepa-ispy2-mask-release-v1"
EXPECTED_CONTEXT_VISITS = ("T0", "T1")
EXPECTED_TARGET_VISIT = "T3"
DEFAULT_COUNTS = {"train": 409, "val": 88}
RIDGE_ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0)
RRR_ALPHAS = (0.1, 1.0, 10.0)
RRR_RANKS = (4, 8, 16, 32)
MORPHOLOGY_FIELDS = (
    "surface_area_mm2",
    "sphericity",
    "maximum_axis_extent_mm",
    "connected_components_6",
)


class FrozenAuditError(ValueError):
    """Fail-closed public error for the isolated audit."""


@dataclass(frozen=True)
class _PatientRecord:
    patient_id: str
    split: str
    clinical: Tensor
    state_actions: Tensor
    next_action: Tensor
    times: Tensor
    ftv: Tensor
    pcr: float
    global_features: Tensor
    local_features: Tensor | None
    spatial_features: Tensor
    occupancy: Tensor | None
    local_exclusion_reason: str | None
    morphology_t0: dict[str, float]
    morphology_t3: dict[str, float]


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FrozenAuditError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise FrozenAuditError(f"{label} must be a JSON object")
    return value


def _development_manifest_records(
    path: Path,
    *,
    expected_counts: Mapping[str, int],
) -> list[dict[str, Any]]:
    """Read exactly the train/val prefix and never request the next JSONL line."""

    expected_rows = sum(expected_counts.values())
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        with path.open("rb") as handle:
            for row_index in range(expected_rows):
                line = handle.readline()
                if not line or not line.strip():
                    raise FrozenAuditError("development manifest prefix is incomplete")
                try:
                    value = json.loads(line)
                except (UnicodeError, json.JSONDecodeError) as error:
                    raise FrozenAuditError("development manifest record is invalid") from error
                if not isinstance(value, dict):
                    raise FrozenAuditError("development manifest record must be an object")
                expected_split = "train" if row_index < expected_counts["train"] else "val"
                patient = str(value.get("patient_id") or "").strip()
                if (
                    not patient
                    or patient in seen
                    or str(value.get("split") or "").strip().casefold() != expected_split
                ):
                    raise FrozenAuditError("development manifest prefix order is invalid")
                context = value.get("context")
                target = value.get("target")
                if (
                    not isinstance(context, list)
                    or len(context) != 2
                    or [str(item.get("visit_id")) for item in context]
                    != list(EXPECTED_CONTEXT_VISITS)
                    or not isinstance(target, dict)
                    or target.get("visit_id") != EXPECTED_TARGET_VISIT
                ):
                    raise FrozenAuditError("development visit contract is invalid")
                seen.add(patient)
                records.append(value)
    except OSError as error:
        raise FrozenAuditError("development manifest is unreadable") from error
    return records


def _mask_index(
    index_path: Path,
    *,
    release_root: Path,
    expected_patients: set[str],
) -> dict[str, tuple[Path, str]]:
    try:
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != (
                "PatientID",
                "mask_path",
                "content_sha256",
            ):
                raise FrozenAuditError("context-mask index columns do not match")
            rows = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise FrozenAuditError("context-mask index is unreadable") from error
    result: dict[str, tuple[Path, str]] = {}
    paths: set[Path] = set()
    digests: set[str] = set()
    for row in rows:
        patient = str(row.get("PatientID") or "").strip()
        if not patient or patient in result:
            raise FrozenAuditError("context-mask index contains duplicate patients")
        raw = Path(str(row.get("mask_path") or "").strip()).expanduser()
        path = raw.resolve() if raw.is_absolute() else (index_path.parent / raw).resolve()
        digest = valid_sha256(row.get("content_sha256"), field="mask index digest")
        if not _inside(path, release_root) or path in paths or digest in digests:
            raise FrozenAuditError("context-mask index path/content contract is invalid")
        if not path.is_file():
            raise FrozenAuditError("context-mask payload is unavailable")
        result[patient] = (path, digest)
        paths.add(path)
        digests.add(digest)
    if set(result) != expected_patients:
        raise FrozenAuditError("context-mask index does not exactly cover development")
    return result


def _patch_weights(mask: Mapping[str, Any]) -> tuple[Tensor | None, str | None]:
    t0 = mask["t0"]
    bed = mask["t0_bed"]
    shared = mask["valid_support"]["T0"] & mask["valid_support"]["T1"]
    core = t0 & shared
    ring = bed & ~t0 & shared
    if not bool(core.any()):
        return None, "t0_core_empty_after_t0_t1_shared_valid_support"
    if not bool(ring.any()):
        return None, "t0_ring_empty_after_t0_t1_shared_valid_support"
    voxel = torch.stack((core[0], ring[0])).float().unsqueeze(0)
    weights = F.avg_pool3d(voxel, kernel_size=16, stride=16)[0].contiguous()
    if tuple(weights.shape) != (2, *FEATURE_GRID_SHAPE_DHW):
        raise FrozenAuditError("core/ring pooling did not produce a 7-cube")
    return weights, None


def _vector(value: Any, *, field: str) -> Tensor:
    if not isinstance(value, list):
        raise FrozenAuditError(f"{field} must be a list")
    try:
        result = torch.tensor([float(item) for item in value], dtype=torch.float32)
    except (TypeError, ValueError) as error:
        raise FrozenAuditError(f"{field} must be numeric") from error
    if not bool(torch.isfinite(result).all()):
        raise FrozenAuditError(f"{field} must be finite")
    return result


def _visit_scalar(visit: Mapping[str, Any], field: str) -> float:
    try:
        value = float(visit[field])
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise FrozenAuditError(f"visit {field} is invalid") from error
    if not math.isfinite(value):
        raise FrozenAuditError(f"visit {field} is non-finite")
    return value


def _load_patient(
    record: Mapping[str, Any],
    *,
    feature_cache_root: Path,
    mask_record: tuple[Path, str],
    expected_channels: int,
    canonical_feature_provenance: dict[str, Any] | None,
    canonical_spatial_provenance: dict[str, Any] | None,
) -> tuple[_PatientRecord, dict[str, Any], dict[str, Any]]:
    visits = [*record["context"], record["target"]]
    grids: list[Tensor] = []
    provenance = canonical_feature_provenance
    spatial = canonical_spatial_provenance
    try:
        record_provenance = validate_feature_provenance(record.get("feature_provenance"))
        record_spatial = validate_spatial_provenance(record.get("spatial_provenance"))
    except (ManifestBuildError, TypeError) as error:
        raise FrozenAuditError("development manifest provenance is invalid") from error
    for visit in visits:
        raw_path = Path(str(visit.get("volume") or "")).expanduser()
        if not raw_path.is_absolute():
            raise FrozenAuditError("formal development feature paths must be absolute")
        path = raw_path.resolve()
        if not _inside(path, feature_cache_root):
            raise FrozenAuditError("development feature path escapes the cache root")
        grid, observed_provenance, observed_spatial = load_feature_grid(
            path,
            expected_sha256=visit.get("content_sha256"),
            expected_channels=expected_channels,
        )
        if observed_provenance != record_provenance:
            raise FrozenAuditError(
                "feature payload provenance disagrees with the locked manifest"
            )
        if observed_spatial != record_spatial:
            raise FrozenAuditError(
                "feature payload geometry disagrees with the locked manifest"
            )
        if provenance is None:
            provenance = observed_provenance
        elif observed_provenance != provenance:
            raise FrozenAuditError("feature provenance changes inside development")
        if spatial is None:
            spatial = observed_spatial
        elif observed_spatial != spatial:
            raise FrozenAuditError("spatial provenance changes inside development")
        grids.append(grid)
    assert provenance is not None
    assert spatial is not None
    mask = load_mask_sidecar(mask_record[0], expected_sha256=mask_record[1])
    if mask["spatial_provenance"] != spatial:
        raise FrozenAuditError(
            "context-mask geometry does not match the frozen feature geometry"
        )
    if mask["t3"] is None:
        raise FrozenAuditError("development mask sidecar lacks T3 evaluation support")
    weights, local_exclusion_reason = _patch_weights(mask)
    views = [feature_views(grid, weights) for grid in grids]
    global_features = torch.stack([item["global_mean"] for item in views])
    spatial_features = torch.stack(
        [item["spatial_pyramid_3x3x3"] for item in views]
    )
    local_features = (
        None
        if weights is None
        else torch.stack([item["core_ring_mean"] for item in views])
    )
    occupancy = None if weights is None else occupancy_geometry(weights)
    context = record["context"]
    target = record["target"]
    times = torch.tensor(
        [_visit_scalar(item, "time_days") for item in visits], dtype=torch.float32
    )
    if not bool((times[1:] > times[:-1]).all()):
        raise FrozenAuditError("development visit times are not strictly increasing")
    ftv = torch.tensor([_visit_scalar(item, "ftv") for item in visits], dtype=torch.float32)
    if torch.any(ftv < 0):
        raise FrozenAuditError("FTV must be non-negative")
    raw_pcr = target.get("pcr")
    pcr = float("nan") if raw_pcr is None else float(raw_pcr)
    if math.isfinite(pcr) and pcr not in {0.0, 1.0}:
        raise FrozenAuditError("pCR must be binary or missing")
    morphology_t0 = morphology_descriptors(
        mask["t0"], spacing_dhw_mm=mask["spacing_dhw_mm"]
    )
    morphology_t3 = morphology_descriptors(
        mask["t3"], spacing_dhw_mm=mask["spacing_dhw_mm"]
    )
    state_actions = torch.stack(
        [_vector(item.get("state_action", []), field="state_action") for item in context]
    )
    return (
        _PatientRecord(
            patient_id=str(record["patient_id"]),
            split=str(record["split"]),
            clinical=_vector(record.get("clinical", []), field="clinical"),
            state_actions=state_actions,
            next_action=_vector(record.get("next_action", []), field="next_action"),
            times=times,
            ftv=ftv,
            pcr=pcr,
            global_features=global_features,
            local_features=local_features,
            spatial_features=spatial_features,
            occupancy=occupancy,
            local_exclusion_reason=local_exclusion_reason,
            morphology_t0=morphology_t0,
            morphology_t3=morphology_t3,
        ),
        provenance,
        spatial,
    )


def _stack(records: Sequence[_PatientRecord], field: str) -> Tensor:
    values = [getattr(record, field) for record in records]
    if any(value is None for value in values):
        raise FrozenAuditError(f"{field} is unavailable for one or more selected patients")
    return torch.stack(values).float()


def _forecast_design(records: Sequence[_PatientRecord]) -> dict[str, Tensor]:
    """Build non-imaging T0/T1 forecast covariates for one fixed cohort."""

    clinical = _stack(records, "clinical")
    ftv = _stack(records, "ftv")
    times = _stack(records, "times")
    state = torch.stack(
        [torch.cat((record.state_actions.flatten(), record.next_action)) for record in records]
    )
    ftv_time = torch.cat(
        (torch.log1p(ftv[:, :2]), times[:, 1:] - times[:, :1]), dim=1
    )
    return {
        "ftv": ftv,
        "times": times,
        "ftv_time": ftv_time,
        "clinical_base": torch.cat((ftv_time, clinical, state), dim=1),
        "target": torch.log1p(ftv[:, 2]),
        "no_change": torch.log1p(ftv[:, 1]),
    }


def _fit_forecast_suite(
    *,
    train_models: Mapping[str, Tensor],
    val_models: Mapping[str, Tensor],
    target_train: Tensor,
    target_val: Tensor,
    raw_target_val: Tensor,
    no_change_prediction: Tensor,
    raw_no_change_prediction: Tensor,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    reference_model: str,
    seed: int,
    bootstrap_replicates: int,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """Fit one fixed-cohort nested forecast suite and paired comparisons."""

    if set(train_models) != set(val_models) or reference_model not in train_models:
        raise FrozenAuditError("forecast model suites must have a shared reference")
    models: dict[str, Any] = {
        "no_change": {
            "validation": asdict(regression_metrics(no_change_prediction, target_val)),
            "raw_ml_mae": float((raw_no_change_prediction - raw_target_val).abs().mean()),
        }
    }
    predictions: dict[str, Tensor] = {"no_change": no_change_prediction}
    for offset, name in enumerate(train_models, start=1):
        probe, prediction = _fit_ridge_probe(
            train_models[name],
            target_train,
            val_models[name],
            target_val,
            train_ids,
            seed=seed + offset,
        )
        raw_prediction = torch.expm1(prediction).clamp_min(0.0)
        probe["validation_raw_ml_mae"] = float(
            (raw_prediction - raw_target_val).abs().mean()
        )
        probe["vs_no_change"] = _bootstrap_error_comparison(
            prediction,
            no_change_prediction,
            target_val,
            val_ids,
            seed=seed + 100 + offset,
            n_bootstrap=bootstrap_replicates,
        )
        models[name] = probe
        predictions[name] = prediction
    increment = {
        name: _bootstrap_error_comparison(
            prediction,
            predictions[reference_model],
            target_val,
            val_ids,
            seed=seed + 200 + offset,
            n_bootstrap=bootstrap_replicates,
        )
        for offset, (name, prediction) in enumerate(predictions.items())
        if name not in {"no_change", reference_model}
    }
    return {
        "status": "ok",
        "reference_model": reference_model,
        "models": models,
        "increment_over_reference": increment,
    }, predictions


def _effective_rank(values: Tensor) -> float:
    centered = values.double() - values.double().mean(dim=0, keepdim=True)
    singular = torch.linalg.svdvals(centered)
    variance = singular.square()
    if float(variance.sum()) <= torch.finfo(torch.float64).eps:
        return 0.0
    probabilities = variance / variance.sum()
    probabilities = probabilities[probabilities > 0]
    return float(torch.exp(-(probabilities * probabilities.log()).sum()))


def _shared_channel_projection(
    global_train: Tensor,
    global_val: Tensor,
    local_train: Tensor | None,
    local_val: Tensor | None,
    spatial_train: Tensor,
    spatial_val: Tensor,
    *,
    components: int = 64,
) -> tuple[
    Tensor,
    Tensor,
    Tensor | None,
    Tensor | None,
    Tensor,
    Tensor,
    dict[str, Any],
]:
    """Fit one outcome-free channel projection on outer-train global summaries.

    The same channel basis is applied to global, core and ring summaries at every
    visit.  This reduces repeated 1024-channel SVDs in downstream grouped CV while
    preserving a common coordinate system for temporal differences.  No validation
    row or outcome participates in fitting this view definition.
    """

    if global_train.ndim != 3 or global_train.shape[2] != 1024:
        raise FrozenAuditError("global feature matrix has the wrong channel contract")
    if (local_train is None) != (local_val is None):
        raise FrozenAuditError("local feature availability must agree across splits")
    if local_train is not None and (
        local_train.ndim != 3
        or local_val is None
        or local_val.ndim != 3
        or local_train.shape[2] != 2048
        or local_val.shape[2] != 2048
    ):
        raise FrozenAuditError("local feature matrix has the wrong channel contract")
    if spatial_train.ndim != 4 or tuple(spatial_train.shape[2:]) != (27, 1024):
        raise FrozenAuditError("spatial-pyramid feature matrix has the wrong contract")
    # Include global, core and ring summaries when learning the shared channel
    # basis.  Fitting on global means alone can suppress channels whose signal is
    # localized to the disease bed or surrounding ring.
    fit_parts = [global_train.reshape(-1, 1024)]
    if local_train is not None:
        fit_parts.append(local_train.reshape(-1, 2, 1024).reshape(-1, 1024))
    fit_rows = torch.cat(fit_parts, dim=0)
    retained = min(components, fit_rows.shape[0] - 1, fit_rows.shape[1])
    scaler = TrainStandardizer().fit(fit_rows)
    pca = TrainPCA(retained).fit(scaler.transform(fit_rows))

    def project_global(values: Tensor) -> Tensor:
        shape = values.shape[:2]
        result = pca.transform(scaler.transform(values.reshape(-1, 1024)))
        return result.reshape(*shape, retained).float()

    def project_local(values: Tensor | None) -> Tensor | None:
        if values is None:
            return None
        shape = values.shape[:2]
        regions = values.reshape(*shape, 2, 1024)
        projected = pca.transform(
            scaler.transform(regions.reshape(-1, 1024))
        )
        return projected.reshape(*shape, 2 * retained).float()

    def project_spatial(values: Tensor) -> Tensor:
        shape = values.shape[:2]
        projected = pca.transform(
            scaler.transform(values.reshape(-1, 1024))
        )
        return projected.reshape(*shape, 27 * retained).float()

    assert pca.explained_variance_ratio_ is not None
    return (
        project_global(global_train),
        project_global(global_val),
        project_local(local_train),
        project_local(local_val),
        project_spatial(spatial_train),
        project_spatial(spatial_val),
        {
            "schema": "outer_train_global_core_ring_channel_pca_shared_across_views_v1",
            "fit_rows": int(fit_rows.shape[0]),
            "input_channels": 1024,
            "retained_channels": retained,
            "local_eligible_outer_train_patients": (
                0 if local_train is None else int(local_train.shape[0])
            ),
            "train_explained_variance_fraction": float(
                pca.explained_variance_ratio_.sum()
            ),
            "outcomes_used_for_projection": False,
            "validation_rows_used_for_projection": False,
        },
    )


def _fit_ridge_probe(
    train_x: Tensor,
    train_y: Tensor,
    val_x: Tensor,
    val_y: Tensor,
    train_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], Tensor]:
    folds = min(5, len(train_ids))
    minimum_fold_train = len(train_ids) - math.ceil(len(train_ids) / folds)
    # The raw 1024-channel summaries are projected once by an outcome-free
    # outer-train channel basis before reaching this function.  Keep up to 256
    # downstream components so the core/ring comparison is not artificially
    # bottlenecked to the same dimensionality as the global view.  Any required
    # downstream PCA remains fold-local and therefore cannot see validation.
    pca_components = min(256, train_x.shape[1], max(1, minimum_fold_train - 1))
    use_pca = train_x.shape[1] > pca_components
    selection = grouped_ridge_cv(
        train_x,
        train_y,
        train_ids,
        RIDGE_ALPHAS,
        n_splits=folds,
        seed=seed,
        standardize=True,
        pca_components=pca_components if use_pca else None,
    )
    scaler = TrainStandardizer().fit(train_x)
    transformed_train = scaler.transform(train_x)
    transformed_val = scaler.transform(val_x)
    pca = None
    if use_pca:
        pca = TrainPCA(pca_components).fit(transformed_train)
        transformed_train = pca.transform(transformed_train)
        transformed_val = pca.transform(transformed_val)
    model = TorchRidgeRegressor(alpha=selection.best_alpha).fit(
        transformed_train, train_y
    )
    prediction = model.predict(transformed_val).float()
    metrics = regression_metrics(prediction, val_y)
    report = {
        "model": "train_standardized_pca_ridge_v1",
        "selected_alpha": selection.best_alpha,
        "train_grouped_cv_mean_mae_by_alpha": {
            str(key): value for key, value in selection.mean_mae_by_alpha.items()
        },
        "input_dimensions": int(train_x.shape[1]),
        "retained_pca_components": pca_components if use_pca else None,
        "validation": asdict(metrics),
    }
    if pca is not None and pca.explained_variance_ratio_ is not None:
        report["train_pca_explained_variance_fraction"] = float(
            pca.explained_variance_ratio_.sum()
        )
    return report, prediction


def _bootstrap_error_comparison(
    candidate_prediction: Tensor,
    reference_prediction: Tensor,
    targets: Tensor,
    patient_ids: Sequence[str],
    *,
    seed: int,
    n_bootstrap: int,
) -> dict[str, Any]:
    candidate_error = (candidate_prediction - targets).abs()
    reference_error = (reference_prediction - targets).abs()
    if candidate_error.ndim == 2:
        candidate_error = candidate_error.mean(dim=1)
        reference_error = reference_error.mean(dim=1)
    result = paired_patient_bootstrap(
        candidate_error,
        reference_error,
        patient_ids,
        higher_is_better=False,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    return asdict(result)


def _regularized_cca_diagnostic(
    train_x: Tensor,
    train_y: Tensor,
    *,
    components: int = 64,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Train-only RCCA spectrum after independent standardized PCA projections."""

    maximum = min(components, train_x.shape[0] - 1, train_x.shape[1], train_y.shape[1])
    x_scaler = TrainStandardizer().fit(train_x)
    y_scaler = TrainStandardizer().fit(train_y)
    x_standard = x_scaler.transform(train_x)
    y_standard = y_scaler.transform(train_y)
    x_pca = TrainPCA(maximum).fit(x_standard)
    y_pca = TrainPCA(maximum).fit(y_standard)
    x = x_pca.transform(x_standard)
    y = y_pca.transform(y_standard)
    denominator = float(x.shape[0] - 1)
    cxx = x.T @ x / denominator + ridge * torch.eye(maximum, dtype=torch.float64)
    cyy = y.T @ y / denominator + ridge * torch.eye(maximum, dtype=torch.float64)
    cxy = x.T @ y / denominator

    def inverse_sqrt(matrix: Tensor) -> Tensor:
        eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
        return eigenvectors @ torch.diag(eigenvalues.clamp_min(1e-12).rsqrt()) @ eigenvectors.T

    whitened = inverse_sqrt(cxx) @ cxy @ inverse_sqrt(cyy)
    correlations = torch.linalg.svdvals(whitened).clamp(0.0, 1.0)
    return {
        "role": "train_only_regularized_cca_diagnostic_not_a_forecast_score",
        "pca_dimensions_each_view": maximum,
        "ridge": ridge,
        "top_canonical_correlations": [float(item) for item in correlations[:10]],
        "sum_squared_canonical_correlations": float(correlations.square().sum()),
    }


def _fit_rrr_temporal(
    train_context: Tensor,
    train_delta: Tensor,
    val_context: Tensor,
    val_delta: Tensor,
    train_ids: Sequence[str],
    *,
    seed: int,
) -> tuple[dict[str, Any], Tensor]:
    folds = patient_group_kfold(train_ids, n_splits=min(5, len(train_ids)), seed=seed)
    scores: dict[tuple[int, float], list[float]] = {
        (rank, alpha): [] for rank in RRR_RANKS for alpha in RRR_ALPHAS
    }
    for fold in folds:
        fold_x_train = train_context[fold.train_indices]
        fold_x_val = train_context[fold.validation_indices]
        fold_y_train = train_delta[fold.train_indices]
        fold_y_val = train_delta[fold.validation_indices]
        x_scaler = TrainStandardizer().fit(fold_x_train)
        y_scaler = TrainStandardizer().fit(fold_y_train)
        x_train = x_scaler.transform(fold_x_train)
        x_val = x_scaler.transform(fold_x_val)
        y_train = y_scaler.transform(fold_y_train)
        x_components = min(64, x_train.shape[0] - 1, x_train.shape[1])
        y_components = min(64, y_train.shape[0] - 1, y_train.shape[1])
        x_pca = TrainPCA(x_components).fit(x_train)
        y_pca = TrainPCA(y_components).fit(y_train)
        x_train = x_pca.transform(x_train)
        x_val = x_pca.transform(x_val)
        y_train_low = y_pca.transform(y_train)
        for rank, alpha in scores:
            retained_rank = min(rank, x_components, y_components)
            model = TorchReducedRankRegressor(
                retained_rank, alpha=alpha
            ).fit(x_train, y_train_low)
            prediction = y_scaler.inverse_transform(
                y_pca.inverse_transform(model.predict(x_val))
            ).float()
            scores[(rank, alpha)].append(
                float((prediction.double() - fold_y_val.double()).abs().mean())
            )
    mean_scores = {key: sum(value) / len(value) for key, value in scores.items()}
    selected_rank, selected_alpha = min(
        mean_scores, key=lambda key: (mean_scores[key], key[0], key[1])
    )
    x_scaler = TrainStandardizer().fit(train_context)
    y_scaler = TrainStandardizer().fit(train_delta)
    x_train = x_scaler.transform(train_context)
    x_val = x_scaler.transform(val_context)
    y_train = y_scaler.transform(train_delta)
    x_components = min(64, x_train.shape[0] - 1, x_train.shape[1])
    y_components = min(64, y_train.shape[0] - 1, y_train.shape[1])
    x_pca = TrainPCA(x_components).fit(x_train)
    y_pca = TrainPCA(y_components).fit(y_train)
    model = TorchReducedRankRegressor(
        min(selected_rank, x_components, y_components), alpha=selected_alpha
    ).fit(x_pca.transform(x_train), y_pca.transform(y_train))
    prediction = y_scaler.inverse_transform(
        y_pca.inverse_transform(model.predict(x_pca.transform(x_val)))
    ).float()
    report = {
        "model": "train_standardized_pca_reduced_rank_ridge_v1",
        "target": "global_t3_minus_t1_frozen_feature_change",
        "selected_rank": selected_rank,
        "selected_alpha": selected_alpha,
        "train_grouped_cv_mean_feature_mae": {
            f"rank={rank},alpha={alpha}": value
            for (rank, alpha), value in mean_scores.items()
        },
        "validation": asdict(regression_metrics(prediction, val_delta)),
    }
    return report, prediction


def _same_visit_probes(
    train: Sequence[_PatientRecord],
    val: Sequence[_PatientRecord],
    global_train: Tensor,
    global_val: Tensor,
    spatial_train: Tensor,
    spatial_val: Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    train_ids = [record.patient_id for record in train]
    result: dict[str, Any] = {}
    for visit_index, visit in ((0, "T0"), (2, "T3")):
        endpoints: dict[str, Tensor] = {
            "log1p_ftv": torch.log1p(torch.stack([record.ftv[visit_index] for record in train])),
        }
        validation: dict[str, Tensor] = {
            "log1p_ftv": torch.log1p(torch.stack([record.ftv[visit_index] for record in val])),
        }
        morphology_attr = "morphology_t0" if visit == "T0" else "morphology_t3"
        for field in MORPHOLOGY_FIELDS:
            train_values = torch.tensor(
                [getattr(record, morphology_attr)[field] for record in train], dtype=torch.float32
            )
            val_values = torch.tensor(
                [getattr(record, morphology_attr)[field] for record in val], dtype=torch.float32
            )
            if field != "sphericity":
                train_values = torch.log1p(train_values)
                val_values = torch.log1p(val_values)
            # Endpoint inclusion is determined from outer-train only.  A
            # constant validation endpoint is reported as not assessable below
            # instead of silently changing the analysis inventory after opening
            # validation outcomes.
            if bool((train_values != train_values[0]).any()):
                endpoints[field] = train_values
                validation[field] = val_values
        visit_report: dict[str, Any] = {}
        for view_name, x_train, x_val in (
            (
                "global_pooled_channel_pca",
                global_train[:, visit_index],
                global_val[:, visit_index],
            ),
            (
                "spatial_pyramid_3x3x3_channel_pca",
                spatial_train[:, visit_index],
                spatial_val[:, visit_index],
            ),
        ):
            endpoint_report: dict[str, Any] = {}
            for endpoint, train_y in endpoints.items():
                validation_y = validation[endpoint]
                if not bool((validation_y != validation_y[0]).any()):
                    endpoint_report[endpoint] = {
                        "status": "not_assessable_constant_validation_target",
                        "endpoint_selected_from_outer_train_only": True,
                        "validation_patients": int(validation_y.numel()),
                    }
                    continue
                probe, _ = _fit_ridge_probe(
                    x_train,
                    train_y,
                    x_val,
                    validation_y,
                    train_ids,
                    seed=seed,
                )
                endpoint_report[endpoint] = probe
            visit_report[view_name] = endpoint_report
        result[visit] = visit_report
    return result


def _pcr_probe(
    train_x: Tensor,
    val_x: Tensor,
    train: Sequence[_PatientRecord],
    val: Sequence[_PatientRecord],
    *,
    seed: int,
) -> tuple[dict[str, Any], Tensor | None]:
    train_keep = torch.tensor([math.isfinite(record.pcr) for record in train], dtype=torch.bool)
    val_keep = torch.tensor([math.isfinite(record.pcr) for record in val], dtype=torch.bool)
    train_y = torch.tensor([record.pcr for record in train], dtype=torch.float32)[train_keep]
    val_y = torch.tensor([record.pcr for record in val], dtype=torch.float32)[val_keep]
    inventory = {
        "train_observed": int(train_keep.sum()),
        "train_positive": int(train_y.sum()),
        "train_negative": int(train_y.numel() - train_y.sum()),
        "validation_observed": int(val_keep.sum()),
        "validation_positive": int(val_y.sum()),
        "validation_negative": int(val_y.numel() - val_y.sum()),
    }
    if train_y.unique().numel() != 2 or val_y.unique().numel() != 2:
        return {"status": "not_assessable_missing_class", "inventory": inventory}, None
    train_ids = [record.patient_id for index, record in enumerate(train) if bool(train_keep[index])]
    probe, score = _fit_ridge_probe(
        train_x[train_keep],
        train_y,
        val_x[val_keep],
        val_y,
        train_ids,
        seed=seed,
    )
    probability = score.clamp(0.0, 1.0)
    aligned_probability = torch.full((len(val),), float("nan"), dtype=torch.float32)
    aligned_probability[val_keep] = probability
    return {
        "status": "ok_exploratory_linear_probability_probe",
        "inventory": inventory,
        "selection": probe,
        "validation": asdict(classification_metrics(probability, val_y)),
        "probability_rule": "clip_ridge_linear_score_to_unit_interval_v1",
    }, aligned_probability


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return float(value)
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(_jsonable(value), indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _source_bundle_sha256() -> str:
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in ("features.py", "run.py", "statistics.py", "t2_census.py"):
        content = (root / name).read_bytes()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _safe_error_payload(error: BaseException) -> dict[str, Any]:
    """Return actionable failure provenance without paths or patient identifiers."""

    message = str(error).strip()
    safe_message = (
        message
        if (
            message
            and "/" not in message
            and "\\" not in message
            and "ISPY2-" not in message
        )
        else "frozen_audit_failed"
    )
    source_root = Path(__file__).resolve().parent
    frames: list[dict[str, Any]] = []
    for frame in traceback.extract_tb(error.__traceback__):
        try:
            source = Path(frame.filename).resolve()
            source.relative_to(source_root)
        except (OSError, ValueError):
            continue
        frames.append(
            {
                "file": source.name,
                "function": frame.name,
                "line": int(frame.lineno),
            }
        )
    return {
        "status": "error",
        "code": safe_message,
        "exception_type": type(error).__name__,
        "safe_trace": frames[-8:],
        "identifiers_or_paths_emitted": False,
    }


def run_frozen_audit(
    *,
    manifest: str | Path,
    manifest_build_report: str | Path,
    feature_index: str | Path,
    feature_cache_root: str | Path,
    context_mask_release: str | Path,
    context_mask_report: str | Path,
    context_mask_index: str | Path,
    trajectory_manifest: str | Path,
    split_csv: str | Path | None,
    private_output_dir: str | Path,
    output_report: str | Path,
    expected_train: int = DEFAULT_COUNTS["train"],
    expected_val: int = DEFAULT_COUNTS["val"],
    seed: int = 20_260_809,
    bootstrap_replicates: int = 10_000,
) -> dict[str, Any]:
    source_bundle_sha = _source_bundle_sha256()
    counts = {"train": int(expected_train), "val": int(expected_val)}
    if any(value < 5 for value in counts.values()):
        raise FrozenAuditError("each development split must contain at least five patients")
    manifest_path = Path(manifest).expanduser().resolve()
    manifest_report_path = Path(manifest_build_report).expanduser().resolve()
    feature_index_path = Path(feature_index).expanduser().resolve()
    feature_root = Path(feature_cache_root).expanduser().resolve()
    mask_root = Path(context_mask_release).expanduser().resolve()
    mask_report_path = Path(context_mask_report).expanduser().resolve()
    mask_index_path = Path(context_mask_index).expanduser().resolve()
    trajectory_path = Path(trajectory_manifest).expanduser().resolve()
    split_path = None if split_csv is None else Path(split_csv).expanduser().resolve()
    for path, label in (
        (manifest_path, "training manifest"),
        (manifest_report_path, "manifest-build report"),
        (feature_index_path, "feature index"),
        (mask_report_path, "context-mask report"),
        (mask_index_path, "context-mask index"),
        (trajectory_path, "trajectory manifest"),
    ):
        if not path.is_file():
            raise FrozenAuditError(f"{label} is unavailable")
    if not feature_root.is_dir() or not mask_root.is_dir():
        raise FrozenAuditError("required governed release root is unavailable")
    if not _inside(mask_report_path, mask_root) or not _inside(mask_index_path, mask_root):
        raise FrozenAuditError("context-mask report/index escapes its release")

    trajectory_sha = sha256_file(trajectory_path)
    manifest_report_value = _read_json(manifest_report_path, label="manifest-build report")
    manifest_sha = sha256_file(manifest_path)
    feature_index_sha = sha256_file(feature_index_path)
    manifest_roi_index_sha = manifest_report_value.get("roi_index_sha256")
    if (
        manifest_report_value.get("schema_version") != MANIFEST_REPORT_SCHEMA_VERSION
        or manifest_report_value.get("status") != "ok"
        or manifest_report_value.get("records") != 587
        or manifest_report_value.get("manifest_sha256") != manifest_sha
        or manifest_report_value.get("feature_index_sha256") != feature_index_sha
        or manifest_report_value.get("context_visits") != ["T0", "T1"]
        or manifest_report_value.get("target_visit") != "T3"
        or manifest_report_value.get("trajectory_manifest_sha256") != trajectory_sha
        or not isinstance(manifest_roi_index_sha, str)
        or valid_sha256(manifest_roi_index_sha, field="manifest ROI index digest")
        != manifest_roi_index_sha
    ):
        raise FrozenAuditError("manifest-build release evidence does not match")
    mask_report_value = _read_json(mask_report_path, label="context-mask report")
    mask_index_sha = sha256_file(mask_index_path)
    bindings = mask_report_value.get("production_bindings")
    if (
        mask_report_value.get("schema_version") != MASK_REPORT_SCHEMA_VERSION
        or mask_report_value.get("status") != "ok"
        or mask_report_value.get("patients") != sum(counts.values())
        or mask_report_value.get("index_sha256") != mask_index_sha
        or not isinstance(bindings, Mapping)
        or bindings.get("production_mode") != "development"
        or bindings.get("trajectory_sha256") != trajectory_sha
        or bindings.get("roi_index_sha256") != manifest_roi_index_sha
        or bindings.get("full_release_includes_test_mask_payloads") is not False
        or bindings.get("test_used_for_mask_method_development") is not False
    ):
        raise FrozenAuditError("development context-mask release evidence does not match")

    raw_records = _development_manifest_records(manifest_path, expected_counts=counts)
    patient_ids = {str(record["patient_id"]) for record in raw_records}
    mask_records = _mask_index(
        mask_index_path,
        release_root=mask_root,
        expected_patients=patient_ids,
    )
    loaded: list[_PatientRecord] = []
    provenance: dict[str, Any] | None = None
    spatial_provenance: dict[str, Any] | None = None
    expected_channels = int(raw_records[0].get("feature_provenance", {}).get("feature_dim", 0))
    if expected_channels != 1024:
        raise FrozenAuditError("frozen audit requires the governed 1024-channel cache")
    for record in raw_records:
        patient, provenance, spatial_provenance = _load_patient(
            record,
            feature_cache_root=feature_root,
            mask_record=mask_records[str(record["patient_id"])],
            expected_channels=expected_channels,
            canonical_feature_provenance=provenance,
            canonical_spatial_provenance=spatial_provenance,
        )
        loaded.append(patient)
    assert provenance is not None
    assert spatial_provenance is not None
    if provenance.get("roi_index_sha256") != manifest_roi_index_sha:
        raise FrozenAuditError(
            "feature-cache ROI index binding disagrees with mask/manifest releases"
        )
    train = loaded[: counts["train"]]
    val = loaded[counts["train"] :]
    if any(record.split != "train" for record in train) or any(record.split != "val" for record in val):
        raise FrozenAuditError("loaded development split order is invalid")
    train_ids = [record.patient_id for record in train]
    val_ids = [record.patient_id for record in val]

    local_train_records = [
        record for record in train if record.local_features is not None
    ]
    local_val_records = [
        record for record in val if record.local_features is not None
    ]
    local_reason_counts: dict[str, int] = {}
    for record in loaded:
        if record.local_exclusion_reason is not None:
            local_reason_counts[record.local_exclusion_reason] = (
                local_reason_counts.get(record.local_exclusion_reason, 0) + 1
            )
    local_eligibility = {
        "rule": "nonempty_t0_core_and_t0_20mm_ring_after_t0_t1_shared_valid_support_v1",
        "train_eligible": len(local_train_records),
        "train_ineligible": len(train) - len(local_train_records),
        "validation_eligible": len(local_val_records),
        "validation_ineligible": len(val) - len(local_val_records),
        "exclusion_reasons_all_development": dict(sorted(local_reason_counts.items())),
        "patient_identifiers_emitted": False,
    }

    raw_global_train = _stack(train, "global_features")
    raw_global_val = _stack(val, "global_features")
    raw_local_train = (
        None
        if not local_train_records
        else _stack(local_train_records, "local_features")
    )
    raw_local_val = (
        None if not local_val_records else _stack(local_val_records, "local_features")
    )
    raw_spatial_train = _stack(train, "spatial_features")
    raw_spatial_val = _stack(val, "spatial_features")
    (
        global_train,
        global_val,
        local_train,
        local_val,
        spatial_train,
        spatial_val,
        channel_projection,
    ) = _shared_channel_projection(
        raw_global_train,
        raw_global_val,
        raw_local_train,
        raw_local_val,
        raw_spatial_train,
        raw_spatial_val,
    )
    full_train_design = _forecast_design(train)
    full_val_design = _forecast_design(val)
    ftv_train = full_train_design["ftv"]
    ftv_val = full_val_design["ftv"]
    time_train = full_train_design["times"]
    time_val = full_val_design["times"]
    global_context_train = torch.cat(
        (global_train[:, 0], global_train[:, 1], global_train[:, 1] - global_train[:, 0]), dim=1
    )
    global_context_val = torch.cat(
        (global_val[:, 0], global_val[:, 1], global_val[:, 1] - global_val[:, 0]), dim=1
    )
    spatial_context_train = torch.cat(
        (
            spatial_train[:, 0],
            spatial_train[:, 1],
            spatial_train[:, 1] - spatial_train[:, 0],
        ),
        dim=1,
    )
    spatial_context_val = torch.cat(
        (
            spatial_val[:, 0],
            spatial_val[:, 1],
            spatial_val[:, 1] - spatial_val[:, 0],
        ),
        dim=1,
    )
    full_train_models = {
        "ftv_time_only": full_train_design["ftv_time"],
        "ftv_time_clinical_action": full_train_design["clinical_base"],
        "ftv_time_clinical_action_plus_global_frozen_features": torch.cat(
            (full_train_design["clinical_base"], global_context_train), dim=1
        ),
        "ftv_time_clinical_action_plus_spatial_pyramid_frozen_features": torch.cat(
            (full_train_design["clinical_base"], spatial_context_train), dim=1
        ),
    }
    full_val_models = {
        "ftv_time_only": full_val_design["ftv_time"],
        "ftv_time_clinical_action": full_val_design["clinical_base"],
        "ftv_time_clinical_action_plus_global_frozen_features": torch.cat(
            (full_val_design["clinical_base"], global_context_val), dim=1
        ),
        "ftv_time_clinical_action_plus_spatial_pyramid_frozen_features": torch.cat(
            (full_val_design["clinical_base"], spatial_context_val), dim=1
        ),
    }
    full_forecast, ftv_predictions = _fit_forecast_suite(
        train_models=full_train_models,
        val_models=full_val_models,
        target_train=full_train_design["target"],
        target_val=full_val_design["target"],
        raw_target_val=full_val_design["ftv"][:, 2],
        no_change_prediction=full_val_design["no_change"],
        raw_no_change_prediction=full_val_design["ftv"][:, 1],
        train_ids=train_ids,
        val_ids=val_ids,
        reference_model="ftv_time_clinical_action",
        seed=seed,
        bootstrap_replicates=bootstrap_replicates,
    )
    full_forecast["cohort"] = {
        "train_patients": len(train),
        "validation_patients": len(val),
        "eligibility": "all_development_patients",
    }

    local_forecast: dict[str, Any]
    if (
        local_train is None
        or local_val is None
        or len(local_train_records) < 5
        or len(local_val_records) < 2
    ):
        local_forecast = {
            "status": "not_assessable_insufficient_context_local_eligibility",
            "cohort": local_eligibility,
        }
    else:
        local_train_design = _forecast_design(local_train_records)
        local_val_design = _forecast_design(local_val_records)
        local_train_ids = [record.patient_id for record in local_train_records]
        local_val_ids = [record.patient_id for record in local_val_records]
        local_occupancy_train = _stack(local_train_records, "occupancy")
        local_occupancy_val = _stack(local_val_records, "occupancy")
        local_context_train = torch.cat(
            (local_train[:, 0], local_train[:, 1], local_train[:, 1] - local_train[:, 0]),
            dim=1,
        )
        local_context_val = torch.cat(
            (local_val[:, 0], local_val[:, 1], local_val[:, 1] - local_val[:, 0]),
            dim=1,
        )
        local_train_models = {
            "ftv_time_clinical_action_local_eligible_reference": local_train_design[
                "clinical_base"
            ],
            "ftv_time_clinical_action_plus_t0_mask_geometry_negative_control": torch.cat(
                (local_train_design["clinical_base"], local_occupancy_train), dim=1
            ),
            "ftv_time_clinical_action_plus_t0_defined_core_ring_frozen_features": torch.cat(
                (local_train_design["clinical_base"], local_context_train), dim=1
            ),
        }
        local_val_models = {
            "ftv_time_clinical_action_local_eligible_reference": local_val_design[
                "clinical_base"
            ],
            "ftv_time_clinical_action_plus_t0_mask_geometry_negative_control": torch.cat(
                (local_val_design["clinical_base"], local_occupancy_val), dim=1
            ),
            "ftv_time_clinical_action_plus_t0_defined_core_ring_frozen_features": torch.cat(
                (local_val_design["clinical_base"], local_context_val), dim=1
            ),
        }
        local_forecast, _ = _fit_forecast_suite(
            train_models=local_train_models,
            val_models=local_val_models,
            target_train=local_train_design["target"],
            target_val=local_val_design["target"],
            raw_target_val=local_val_design["ftv"][:, 2],
            no_change_prediction=local_val_design["no_change"],
            raw_no_change_prediction=local_val_design["ftv"][:, 1],
            train_ids=local_train_ids,
            val_ids=local_val_ids,
            reference_model="ftv_time_clinical_action_local_eligible_reference",
            seed=seed + 300,
            bootstrap_replicates=bootstrap_replicates,
        )
        local_forecast["cohort"] = local_eligibility

    ftv_forecast: dict[str, Any] = {
        "full_development_cohort": full_forecast,
        "local_core_ring_eligible_context_cohort": local_forecast,
    }

    same_visit = _same_visit_probes(
        train,
        val,
        global_train,
        global_val,
        spatial_train,
        spatial_val,
        seed=seed + 300,
    )
    pcr_same_visit_global, pcr_same_global_prediction = _pcr_probe(
        global_train[:, 2], global_val[:, 2], train, val, seed=seed + 400
    )
    pcr_same_visit_spatial, pcr_same_spatial_prediction = _pcr_probe(
        spatial_train[:, 2], spatial_val[:, 2], train, val, seed=seed + 401
    )
    pcr_forecast_global, pcr_forecast_global_prediction = _pcr_probe(
        global_context_train, global_context_val, train, val, seed=seed + 402
    )
    pcr_forecast_spatial, pcr_forecast_spatial_prediction = _pcr_probe(
        spatial_context_train, spatial_context_val, train, val, seed=seed + 403
    )

    train_delta = global_train[:, 2] - global_train[:, 1]
    val_delta = global_val[:, 2] - global_val[:, 1]
    global_rrr_report, global_rrr_prediction = _fit_rrr_temporal(
        global_context_train,
        train_delta,
        global_context_val,
        val_delta,
        train_ids,
        seed=seed + 500,
    )
    global_persistence_prediction = torch.zeros_like(val_delta)
    global_rrr_report["vs_latent_persistence"] = _bootstrap_error_comparison(
        global_rrr_prediction,
        global_persistence_prediction,
        val_delta,
        val_ids,
        seed=seed + 501,
        n_bootstrap=bootstrap_replicates,
    )
    global_rrr_report["persistence_validation"] = asdict(
        regression_metrics(global_persistence_prediction, val_delta)
    )
    global_rrr_report["representation_diagnostics"] = {
        "t3_minus_t1_target_effective_rank": _effective_rank(val_delta),
        "predicted_change_effective_rank": _effective_rank(global_rrr_prediction),
        "t1_patient_effective_rank": _effective_rank(global_val[:, 1]),
        "t3_patient_effective_rank": _effective_rank(global_val[:, 2]),
        "mean_channel_std_t1": float(global_val[:, 1].std(dim=0, unbiased=False).mean()),
        "mean_channel_std_t3": float(global_val[:, 2].std(dim=0, unbiased=False).mean()),
    }
    spatial_train_delta = spatial_train[:, 2] - spatial_train[:, 1]
    spatial_val_delta = spatial_val[:, 2] - spatial_val[:, 1]
    spatial_rrr_report, spatial_rrr_prediction = _fit_rrr_temporal(
        spatial_context_train,
        spatial_train_delta,
        spatial_context_val,
        spatial_val_delta,
        train_ids,
        seed=seed + 502,
    )
    spatial_persistence_prediction = torch.zeros_like(spatial_val_delta)
    spatial_rrr_report["target"] = (
        "coarse_spatial_pyramid_t3_minus_t1_frozen_feature_change"
    )
    spatial_rrr_report["vs_latent_persistence"] = _bootstrap_error_comparison(
        spatial_rrr_prediction,
        spatial_persistence_prediction,
        spatial_val_delta,
        val_ids,
        seed=seed + 503,
        n_bootstrap=bootstrap_replicates,
    )
    spatial_rrr_report["persistence_validation"] = asdict(
        regression_metrics(spatial_persistence_prediction, spatial_val_delta)
    )
    spatial_rrr_report["representation_diagnostics"] = {
        "t3_minus_t1_target_effective_rank": _effective_rank(spatial_val_delta),
        "predicted_change_effective_rank": _effective_rank(spatial_rrr_prediction),
        "t1_patient_effective_rank": _effective_rank(spatial_val[:, 1]),
        "t3_patient_effective_rank": _effective_rank(spatial_val[:, 2]),
        "mean_dimension_std_t1": float(
            spatial_val[:, 1].std(dim=0, unbiased=False).mean()
        ),
        "mean_dimension_std_t3": float(
            spatial_val[:, 2].std(dim=0, unbiased=False).mean()
        ),
    }
    rcca = _regularized_cca_diagnostic(global_context_train, train_delta)
    sfa_components = min(16, global_train.shape[0] * 3 - 1, global_train.shape[2])
    sfa = TorchSlowFeatureAnalysis(sfa_components).fit(global_train, time_train)
    sfa_report = {
        "role": "secondary_train_fitted_invariance_diagnostic_not_a_forecast_model",
        "components": sfa_components,
        "train_component_slowness": [float(item) for item in sfa.score_slowness(global_train, time_train)],
        "validation_component_slowness": [float(item) for item in sfa.score_slowness(global_val, time_val)],
    }

    t2_census = build_t2_census(
        trajectory_path,
        split_csv=split_path,
        released_patients_csv=mask_index_path,
    )
    pooled_latent_improvement = global_rrr_report["vs_latent_persistence"]
    spatial_latent_improvement = spatial_rrr_report["vs_latent_persistence"]
    local_increment = local_forecast.get("increment_over_reference", {}).get(
        "ftv_time_clinical_action_plus_t0_defined_core_ring_frozen_features"
    )
    t0_ftv_r2 = same_visit["T0"]["spatial_pyramid_3x3x3_channel_pca"][
        "log1p_ftv"
    ]["validation"]["r2"]
    if t0_ftv_r2 <= 0.0:
        triage = (
            "no_linear_signal_in_tested_coarse_spatial_view_run_full_grid_or_"
            "intermediate_layer_probe_before_v4"
        )
    elif (
        pooled_latent_improvement["ci_low"] > 0.0
        or spatial_latent_improvement["ci_low"] > 0.0
    ):
        triage = "linear_temporal_signal_present_in_tested_view_v4_must_beat_rrr"
    else:
        triage = (
            "no_reliable_linear_temporal_gain_in_tested_views_hold_v4_and_"
            "revisit_representation_target"
        )

    if _source_bundle_sha256() != source_bundle_sha:
        raise FrozenAuditError("audit source bundle changed during execution")

    private_destination = Path(private_output_dir).expanduser().resolve()
    if private_destination.exists():
        raise FileExistsError("private audit output already exists")
    private_destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(private_destination.parent, 0o700)
    staging = Path(tempfile.mkdtemp(prefix=f".{private_destination.name}.staging-", dir=private_destination.parent))
    os.chmod(staging, 0o700)
    promoted = False
    try:
        private_payload = {
            "schema_version": PRIVATE_SCHEMA_VERSION,
            "patient_ids": val_ids,
            "target_log1p_ftv": full_val_design["target"],
            "ftv_predictions": ftv_predictions,
            "target_global_feature_change_t3_minus_t1": val_delta,
            "rrr_predicted_global_feature_change": global_rrr_prediction,
            "target_spatial_feature_change_t3_minus_t1": spatial_val_delta,
            "rrr_predicted_spatial_feature_change": spatial_rrr_prediction,
            "pcr_same_visit_global_probability_aligned_to_patient_ids": (
                pcr_same_global_prediction
            ),
            "pcr_same_visit_spatial_probability_aligned_to_patient_ids": (
                pcr_same_spatial_prediction
            ),
            "pcr_forecast_global_probability_aligned_to_patient_ids": (
                pcr_forecast_global_prediction
            ),
            "pcr_forecast_spatial_probability_aligned_to_patient_ids": (
                pcr_forecast_spatial_prediction
            ),
            "pcr_observed_mask_aligned_to_patient_ids": torch.tensor(
                [math.isfinite(record.pcr) for record in val], dtype=torch.bool
            ),
        }
        private_path = staging / "validation_predictions.pt"
        torch.save(private_payload, private_path)
        os.chmod(private_path, 0o600)
        private_sha = sha256_file(private_path)
        os.rename(staging, private_destination)
        promoted = True
    finally:
        if not promoted:
            shutil.rmtree(staging, ignore_errors=True)

    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_development_frozen_feature_audit",
        "scope": "development_only_train_val_exploratory_triage",
        "scientific_question": (
            "whether frozen final-layer 3DINO features encode clinically relevant "
            "state in selected linear views and whether those views contain "
            "classically predictable temporal change"
        ),
        "cohort": {
            "train_patients": len(train),
            "validation_patients": len(val),
            "test_patients_loaded": 0,
        },
        "local_context_eligibility": local_eligibility,
        "same_visit_encodability": same_visit,
        "feature_view_projection": channel_projection,
        "pcr_inventory_and_probes": {
            "same_visit_t3_global_feature_probe": pcr_same_visit_global,
            "same_visit_t3_spatial_feature_probe": pcr_same_visit_spatial,
            "t0_t1_global_forecast_probe": pcr_forecast_global,
            "t0_t1_spatial_forecast_probe": pcr_forecast_spatial,
        },
        "ftv_forecast_nested_models": ftv_forecast,
        "non_neural_temporal_baselines": {
            "global_pooled_reduced_rank_ridge": global_rrr_report,
            "coarse_spatial_reduced_rank_ridge": spatial_rrr_report,
            "global_pooled_regularized_cca": rcca,
            "global_pooled_slow_feature_analysis": sfa_report,
        },
        "t2_metadata_census": t2_census,
        "triage": {
            "recommendation": triage,
            "posthoc_exploratory_not_preregistered": True,
            "t0_same_visit_coarse_spatial_log1p_ftv_r2": t0_ftv_r2,
            "global_rrr_vs_persistence_positive_lower_ci": (
                pooled_latent_improvement["ci_low"] > 0.0
            ),
            "coarse_spatial_rrr_vs_persistence_positive_lower_ci": (
                spatial_latent_improvement["ci_low"] > 0.0
            ),
            "local_imaging_increment_over_matched_local_reference_positive_lower_ci": (
                None if local_increment is None else local_increment["ci_low"] > 0.0
            ),
            "v4_authorized_by_this_report": False,
        },
        "contracts": {
            "frozen_encoder_updated": False,
            "neural_model_trained": False,
            "t3_mask_used_as_model_input": False,
            "t0_mask_geometry_is_explicit_negative_control": True,
            "all_transforms_fit_on_training_patients_only": True,
            "shared_channel_pca_fit_on_all_outer_train_before_inner_cv": True,
            "validation_opened_once_after_train_cv_selection": True,
            "patient_level_resampling": True,
            "raw_dicom_read": False,
            "pixel_data_read": False,
            "test_split_loaded": False,
            "test_training_manifest_records_parsed": False,
            "test_tensor_files_loaded": False,
            "test_image_or_outcome_data_loaded": False,
            "test_metadata_rows_may_be_counted_by_t2_census": True,
            "full_manifest_hashed_as_opaque_bytes": True,
        },
        "evidence": {
            "manifest_sha256": manifest_sha,
            "manifest_build_report_sha256": sha256_file(manifest_report_path),
            "feature_index_sha256": feature_index_sha,
            "context_mask_report_sha256": sha256_file(mask_report_path),
            "context_mask_index_sha256": mask_index_sha,
            "trajectory_manifest_sha256": trajectory_sha,
            "split_csv_sha256": None if split_path is None else sha256_file(split_path),
            "feature_provenance": provenance,
            "spatial_provenance": spatial_provenance,
            "audit_source_bundle_sha256": source_bundle_sha,
            "private_validation_predictions_sha256": private_sha,
        },
        "limitations": [
            "The final 7x7x7 3DINO grid has globally mixed tokens and cannot support a clean post-cache occlusion claim.",
            "The 3x3x3 spatial pyramid is a coarse linear view, not an oracle over the complete frozen token grid.",
            "The classical baselines test selected linear views and cannot establish that all predictable frozen-feature dynamics are exhausted.",
            "The outcome-free shared channel PCA uses all outer-train patients before downstream inner CV; this transductive preprocessing never uses validation rows or outcomes.",
            "Normalization-matched persistence for the existing Spatial Med-JEPA checkpoint is audited by a separate read-only stage.",
            "Mask morphology describes the FTV inclusion support, not a pathology-grade anatomical tumour contour.",
            "T1 canonical lesion support is absent from the frozen sidecar, so shape-change labels are T0-to-T3 only.",
            "RCCA and SFA are train-fitted diagnostics, not held-out forecast estimators.",
            "Validation is reused for exploratory model triage; no confirmatory claim is permitted.",
        ],
        "private_artifacts": {
            "validation_predictions_written": True,
            "content_sha256": private_sha,
            "mode": "0600",
            "contains_patient_identifiers": True,
            "must_not_be_synced_or_pasted": True,
        },
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "paths_emitted": False,
        "confirmatory_claim_permitted": False,
    }
    _write_json_exclusive(Path(output_report).expanduser().resolve(), report)
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--feature-index", required=True)
    parser.add_argument("--feature-cache-root", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--trajectory-manifest", required=True)
    parser.add_argument("--split-csv")
    parser.add_argument("--private-output-dir", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--expected-train", type=int, default=409)
    parser.add_argument("--expected-val", type=int, default=88)
    parser.add_argument("--seed", type=int, default=20_260_809)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    parser.add_argument(
        "--development-splits",
        nargs=2,
        default=("train", "val"),
        choices=("train", "val"),
        help="explicit sealed-test acknowledgement; must be exactly train val",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if list(args.development_splits) != ["train", "val"]:
        print("development_split_contract_invalid", file=os.sys.stderr)
        return 2
    try:
        report = run_frozen_audit(
            manifest=args.manifest,
            manifest_build_report=args.manifest_build_report,
            feature_index=args.feature_index,
            feature_cache_root=args.feature_cache_root,
            context_mask_release=args.context_mask_release,
            context_mask_report=args.context_mask_report,
            context_mask_index=args.context_mask_index,
            trajectory_manifest=args.trajectory_manifest,
            split_csv=args.split_csv,
            private_output_dir=args.private_output_dir,
            output_report=args.output_report,
            expected_train=args.expected_train,
            expected_val=args.expected_val,
            seed=args.seed,
            bootstrap_replicates=args.bootstrap_replicates,
        )
    except (
        FrozenAuditError,
        FrozenFeatureContractError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(json.dumps(_safe_error_payload(error), sort_keys=True), file=os.sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "status": report["status"],
                "scope": report["scope"],
                "train_patients": report["cohort"]["train_patients"],
                "validation_patients": report["cohort"]["validation_patients"],
                "test_split_loaded": False,
                "triage_recommendation": report["triage"]["recommendation"],
                "patient_identifiers_emitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "FrozenAuditError",
    "PRIVATE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "build_parser",
    "main",
    "run_frozen_audit",
    "_safe_error_payload",
]
