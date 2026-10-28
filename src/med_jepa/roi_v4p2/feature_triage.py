"""Training-only frozen-3DINO feature intervention triage for Med-JEPA v4.2.

All extraction, hyperparameter selection, and nested evaluation are confined to
the 409-patient training split.  The 88-patient development validation split and
held-out test split are never loaded by this process.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor, nn

from med_jepa.cache_3dino import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_INPUT_SHAPE,
    DEFAULT_PATCH_SIZE,
    THREE_DINO_REVISION,
    build_three_dino_model,
    fixed_ser_normalize,
    load_roi_payload,
    read_roi_index,
    verify_three_dino_checkout,
)
from med_jepa.frozen_audit.features import load_mask_sidecar
from med_jepa.roi_v4.audit_mechanistic import (
    REGIONS,
    _summary,
    _validate_mask_release,
    patient_region_mae,
)
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4p2.freeze_design import SCHEMA as DESIGN_SCHEMA
from med_jepa.roi_v4p2.preprocessing import (
    LOCAL_CROP_SIZES,
    interleave_shifted_feature_views,
    local_magnified_view,
    local_token_weights,
    mask_center,
    patient_visit_reference_align_ser,
    shifted_token_weights,
    symmetric_shifted_views,
    token_weights,
    voxel_regions,
)
import med_jepa.roi_v4p2.preprocessing as preprocessing_module


SCHEMA = "med-jepa-roi-v4p2-train-only-feature-triage-v2"
PRIVATE_SCHEMA = "med-jepa-roi-v4p2-private-feature-triage-v2"
PATIENT_SCHEMA = "med-jepa-roi-v4p2-private-patient-feature-variants-v2"
VARIANTS = (
    "canonical_cached_7cube",
    "symmetric_shifted_window_14cube",
    "t0_anchored_local_magnification_80_to_112",
    "t0_anchored_local_magnification_64_to_112",
    "patient_visit_reference_aligned_SER_7cube",
)
ALPHAS = tuple(index / 20.0 for index in range(21))
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


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


def _mask_roster(index_path: Path) -> set[str]:
    """Read only the governed mask roster; no validation feature is loaded."""

    try:
        with index_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != (
                "PatientID",
                "mask_path",
                "content_sha256",
            ):
                raise ValueError("context-mask index columns do not match")
            patient_ids = [str(row.get("PatientID") or "").strip() for row in reader]
    except (OSError, UnicodeError, csv.Error) as error:
        raise ValueError("context-mask roster is unreadable") from error
    if any(not patient_id for patient_id in patient_ids):
        raise ValueError("context-mask roster contains an empty patient identifier")
    if len(set(patient_ids)) != len(patient_ids):
        raise ValueError("context-mask roster contains duplicate patients")
    return set(patient_ids)


def _folds(size: int, *, seed: int, count: int) -> list[tuple[Tensor, Tensor]]:
    if size < count or count < 2:
        raise ValueError("invalid grouped fold request")
    order = torch.randperm(size, generator=torch.Generator().manual_seed(seed))
    result = []
    for fold in range(count):
        held = order[fold::count]
        keep = torch.ones(size, dtype=torch.bool)
        keep[held] = False
        result.append((torch.arange(size)[keep], held))
    return result


def _mean_target(features: Tensor, indices: Tensor) -> Tensor:
    total = torch.zeros_like(features[0, 2], dtype=torch.float32)
    for index in indices.tolist():
        total.add_(features[index, 2].float())
    return total / float(len(indices))


def _region_errors(
    prediction: Tensor, target: Tensor, weights: Tensor
) -> Tensor:
    return torch.stack(
        [
            patient_region_mae(prediction, target, weights[:, index])
            for index in range(len(REGIONS))
        ],
        dim=1,
    )


def select_alpha_leave_one_out(
    features: Tensor,
    weights: Tensor,
    train_indices: Tensor,
    *,
    device: torch.device,
    batch_size: int = 2,
) -> tuple[float, dict[str, float]]:
    """Select shrinkage alpha without using any outer-fold patient target."""

    disease = REGIONS.index("t0_disease_bed")
    target_sum = _mean_target(features, train_indices) * float(len(train_indices))
    score_sum = torch.zeros(len(ALPHAS), dtype=torch.float64)
    score_count = torch.zeros(len(ALPHAS), dtype=torch.long)
    alpha = torch.tensor(ALPHAS, device=device, dtype=torch.float32)[:, None, None, None, None, None]
    for start in range(0, len(train_indices), batch_size):
        selected = train_indices[start : start + batch_size]
        source = features[selected, 1].float().to(device)
        target = features[selected, 2].float().to(device)
        loo_mean = (
            target_sum[None].to(device) - target
        ) / float(len(train_indices) - 1)
        prediction = alpha * source[None] + (1.0 - alpha) * loo_mean[None]
        error = (prediction - target[None]).abs().mean(2)
        region = weights[selected, disease].float().to(device)
        denominator = region.flatten(1).sum(1)
        eligible = denominator > 0
        values = (
            (error[:, eligible] * region[eligible][None]).flatten(2).sum(2)
            / denominator[eligible][None]
        )
        score_sum.add_(values.double().sum(1).cpu())
        score_count.add_(int(eligible.sum()))
    scores = score_sum / score_count.clamp_min(1)
    selected_index = min(
        range(len(ALPHAS)), key=lambda index: (float(scores[index]), ALPHAS[index])
    )
    return ALPHAS[selected_index], {
        f"{value:.4f}": float(scores[index])
        for index, value in enumerate(ALPHAS)
    }


def nested_shrinkage_evaluation(
    features: Tensor,
    weights: Tensor,
    *,
    seed: int,
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    if features.ndim != 6 or features.shape[:3] != (409, 3, 1024):
        raise ValueError("feature variant must be [409,3,1024,G,G,G]")
    if weights.shape[:2] != (409, len(REGIONS)) or weights.shape[-3:] != features.shape[-3:]:
        raise ValueError("variant region weights do not align")
    shrinkage_errors = torch.full((409, len(REGIONS)), float("nan"))
    persistence_errors = torch.full_like(shrinkage_errors, float("nan"))
    fold_alphas: list[float] = []
    fold_scores: list[dict[str, float]] = []
    for fold, (outer_train, outer_held) in enumerate(
        _folds(409, seed=seed, count=5)
    ):
        alpha, scores = select_alpha_leave_one_out(
            features,
            weights,
            outer_train,
            device=device,
        )
        fold_alphas.append(alpha)
        fold_scores.append(scores)
        mean_target = _mean_target(features, outer_train)
        for start in range(0, len(outer_held), 4):
            selected = outer_held[start : start + 4]
            source = features[selected, 1].float()
            target = features[selected, 2].float()
            prediction = alpha * source + (1.0 - alpha) * mean_target[None]
            shrinkage_errors[selected] = _region_errors(
                prediction, target, weights[selected]
            )
            persistence_errors[selected] = _region_errors(
                source, target, weights[selected]
            )
    if not torch.isfinite(shrinkage_errors[:, 2:]).all():
        raise RuntimeError("nested shrinkage evaluation left missing primary errors")
    region_report: dict[str, Any] = {}
    for index, region in enumerate(REGIONS):
        candidate = shrinkage_errors[:, index]
        reference = persistence_errors[:, index]
        eligible = torch.isfinite(candidate) & torch.isfinite(reference)
        candidate_mean = float(candidate[eligible].mean())
        reference_mean = float(reference[eligible].mean())
        region_report[region] = {
            "patients": int(eligible.sum()),
            "shrinkage_MAE": candidate_mean,
            "persistence_MAE": reference_mean,
            "relative_MAE_reduction": (
                (reference_mean - candidate_mean) / max(reference_mean, 1e-12)
            ),
        }
    return {
        "outer_grouped_folds": 5,
        "alpha_selection": "leave_one_patient_out_inside_each_outer_training_fold",
        "selected_alpha_by_outer_fold": fold_alphas,
        "inner_alpha_scores_by_outer_fold": fold_scores,
        "regions": region_report,
    }, {
        "shrinkage": shrinkage_errors,
        "persistence": persistence_errors,
    }


def _standardize(train: Tensor, query: Tensor) -> tuple[Tensor, Tensor]:
    mean = train.mean(0, keepdim=True)
    std = train.std(0, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (train - mean) / std, (query - mean) / std


def _ridge_scalar_predict(
    train_x: Tensor,
    train_y: Tensor,
    query_x: Tensor,
    alpha: float,
) -> Tensor:
    train_x, query_x = _standardize(train_x.float(), query_x.float())
    kernel = train_x @ train_x.T / float(train_x.shape[1])
    cross = query_x @ train_x.T / float(train_x.shape[1])
    mean = train_y.float().mean()
    coefficients = torch.linalg.solve(
        kernel + float(alpha) * torch.eye(len(kernel)),
        train_y.float() - mean,
    )
    return cross @ coefficients + mean


def nested_scalar_probe(
    features: Tensor,
    target: Tensor,
    *,
    seed: int,
) -> dict[str, Any]:
    prediction = torch.full_like(target.float(), float("nan"))
    selected_alphas: list[float] = []
    outer = _folds(len(target), seed=seed, count=5)
    for fold, (outer_train, outer_held) in enumerate(outer):
        inner_records = outer_train
        inner_folds = _folds(len(inner_records), seed=seed + 100 + fold, count=4)
        scores = {alpha: [] for alpha in RIDGE_ALPHAS}
        for inner_train_local, inner_held_local in inner_folds:
            inner_train = inner_records[inner_train_local]
            inner_held = inner_records[inner_held_local]
            for alpha in RIDGE_ALPHAS:
                value = _ridge_scalar_predict(
                    features[inner_train],
                    target[inner_train],
                    features[inner_held],
                    alpha,
                )
                scores[alpha].append(float((value - target[inner_held]).abs().mean()))
        mean_scores = {alpha: sum(value) / len(value) for alpha, value in scores.items()}
        selected = min(mean_scores, key=lambda value: (mean_scores[value], value))
        selected_alphas.append(selected)
        prediction[outer_held] = _ridge_scalar_predict(
            features[outer_train],
            target[outer_train],
            features[outer_held],
            selected,
        )
    residual = (prediction - target.float()).square().sum()
    total = (target.float() - target.float().mean()).square().sum().clamp_min(1e-12)
    return {
        "nested_outer_folds": 5,
        "nested_inner_folds": 4,
        "selected_alpha_by_outer_fold": selected_alphas,
        "MAE": float((prediction - target.float()).abs().mean()),
        "R2": float(1.0 - residual / total),
    }


def _weighted_pool(features: Tensor, weights: Tensor) -> Tensor:
    denominator = weights.flatten(1).sum(1).unsqueeze(1).clamp_min(1e-12)
    return (
        (features.float() * weights[:, None].float()).flatten(2).sum(2)
        / denominator
    )


def _specificity_from_errors(errors: Mapping[str, Tensor]) -> dict[str, float]:
    disease = REGIONS.index("t0_disease_bed")
    background = REGIONS.index("valid_non_disease_bed")
    shrinkage = errors["shrinkage"]
    persistence = errors["persistence"]
    disease_gain = 1.0 - float(shrinkage[:, disease].mean() / persistence[:, disease].mean())
    background_gain = 1.0 - float(
        shrinkage[:, background].mean() / persistence[:, background].mean()
    )
    return {
        "disease_relative_MAE_reduction": disease_gain,
        "background_relative_MAE_reduction": background_gain,
        "disease_minus_background_relative_MAE_reduction": disease_gain - background_gain,
    }


def relative_gain_delta_bootstrap(
    candidate: Mapping[str, Tensor],
    canonical: Mapping[str, Tensor],
    *,
    seed: int,
    replicates: int = 10_000,
) -> dict[str, Any]:
    disease = REGIONS.index("t0_disease_bed")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randint(409, (replicates, 409), generator=generator)

    def gain(errors: Mapping[str, Tensor]) -> Tensor:
        shrink = errors["shrinkage"][:, disease].double()[indices].mean(1)
        persist = errors["persistence"][:, disease].double()[indices].mean(1)
        return 1.0 - shrink / persist.clamp_min(1e-12)

    candidate_gain = gain(candidate)
    canonical_gain = gain(canonical)
    difference = candidate_gain - canonical_gain
    interval = torch.quantile(
        difference, torch.tensor([0.025, 0.975], dtype=difference.dtype)
    )
    point = (
        _specificity_from_errors(candidate)["disease_relative_MAE_reduction"]
        - _specificity_from_errors(canonical)["disease_relative_MAE_reduction"]
    )
    return {
        "candidate_minus_canonical_disease_relative_MAE_reduction": point,
        "paired_bootstrap_95ci": [float(interval[0]), float(interval[1])],
        "bootstrap_positive_fraction": float((difference > 0).double().mean()),
        "replicates": replicates,
    }


@torch.no_grad()
def _extract_feature_batch(
    model: nn.Module,
    normalized: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if (
        normalized.ndim != 5
        or normalized.shape[1] != 1
        or tuple(normalized.shape[-3:]) != DEFAULT_INPUT_SHAPE
    ):
        raise ValueError("feature-extraction batch must be [B,1,112,112,112]")
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=dtype,
        enabled=device.type == "cuda",
    ):
        outputs = model.get_intermediate_layers(
            normalized.to(device), n=1, reshape=True
        )
    expected = (
        normalized.shape[0],
        DEFAULT_FEATURE_DIM,
        *(value // DEFAULT_PATCH_SIZE for value in DEFAULT_INPUT_SHAPE),
    )
    if not isinstance(outputs, (list, tuple)) or len(outputs) != 1:
        raise RuntimeError("3DINO returned an unexpected batched feature structure")
    features = outputs[0]
    if tuple(features.shape) != expected or not torch.isfinite(features).all():
        raise RuntimeError("3DINO batched feature extraction failed its contract")
    return features.detach().to(device="cpu", dtype=torch.float16).contiguous()


def _patient_output_name(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:24] + ".pt"


def _extract_patient(
    item: PatientStates,
    roi_by_visit: Mapping[str, Any],
    mask_record: tuple[Path, str],
    model: nn.Module,
    *,
    device: torch.device,
    dtype: torch.dtype,
    extraction_contract: Mapping[str, Any],
) -> dict[str, Any]:
    mask = load_mask_sidecar(mask_record[0], expected_sha256=mask_record[1])
    physical: list[Tensor] = []
    source_hashes: list[str] = []
    for visit in ("T0", "T1", "T3"):
        record = roi_by_visit[visit]
        if _sha(record.roi_path) != record.source_content_sha256:
            raise RuntimeError("governed ROI content hash mismatch")
        volume, _, _ = load_roi_payload(record.roi_path, DEFAULT_INPUT_SHAPE)
        physical.append(volume)
        source_hashes.append(record.source_content_sha256)
    physical_tensor = torch.stack(physical)
    normalized = torch.stack([fixed_ser_normalize(value) for value in physical])
    regions = voxel_regions(mask)
    center = mask_center(mask["t0_bed"].bool())
    valid = torch.stack(
        [mask["valid_support"][visit].bool() for visit in ("T0", "T1", "T3")]
    )
    aligned_physical, normalization = patient_visit_reference_align_ser(
        physical_tensor, valid, mask["t0_bed"].bool()
    )
    aligned = torch.stack(
        [fixed_ser_normalize(value) for value in aligned_physical]
    )

    features: dict[str, Tensor] = {
        "canonical_cached_7cube": item.features.half().contiguous(),
    }
    weights: dict[str, Tensor] = {
        "canonical_cached_7cube": token_weights(regions),
        "patient_visit_reference_aligned_SER_7cube": token_weights(regions),
        "symmetric_shifted_window_14cube": shifted_token_weights(regions),
    }
    # symmetric_shifted_views pads with physical zero; convert back to normalized
    # padding before feature extraction.
    shifted_visits: list[Tensor] = []
    for visit in range(3):
        views = [view - 1.0 for view in symmetric_shifted_views(normalized[visit] + 1.0)]
        extracted = _extract_feature_batch(
            model,
            torch.stack(views),
            device=device,
            dtype=dtype,
        )
        shifted_visits.append(
            interleave_shifted_feature_views(
                list(extracted.unbind(0))
            )
        )
    features["symmetric_shifted_window_14cube"] = torch.stack(shifted_visits)
    local_features = {crop_size: [] for crop_size in LOCAL_CROP_SIZES}
    for visit in range(3):
        local_views = torch.stack(
            [
                local_magnified_view(
                    normalized[visit], center, crop_size=crop_size
                )
                for crop_size in LOCAL_CROP_SIZES
            ]
        )
        local_extracted = _extract_feature_batch(
            model, local_views, device=device, dtype=dtype
        )
        for index, crop_size in enumerate(LOCAL_CROP_SIZES):
            local_features[crop_size].append(local_extracted[index])
    for crop_size in LOCAL_CROP_SIZES:
        name = f"t0_anchored_local_magnification_{crop_size}_to_112"
        features[name] = torch.stack(local_features[crop_size])
        weights[name] = local_token_weights(
            regions, center, crop_size=crop_size
        )
    features["patient_visit_reference_aligned_SER_7cube"] = _extract_feature_batch(
        model, aligned, device=device, dtype=dtype
    )
    return {
        "schema_version": PATIENT_SCHEMA,
        "patient_id": item.patient_id,
        "extraction_contract": dict(extraction_contract),
        "features": features,
        "region_weights": weights,
        "t0_ftv": item.ftv[0].float(),
        "native_t0_core_voxels": int(mask["t0"].sum()),
        "source_roi_sha256": source_hashes,
        "reference_normalization": normalization,
    }


def _load_variant(
    patient_files: Sequence[Path], variant: str
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    feature_rows = []
    weight_rows = []
    ftv = []
    core = []
    for path in patient_files:
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.get("schema_version") != PATIENT_SCHEMA:
            raise ValueError("private feature-triage patient payload is invalid")
        feature_rows.append(value["features"][variant])
        weight_rows.append(value["region_weights"][variant])
        ftv.append(value["t0_ftv"])
        core.append(float(value["native_t0_core_voxels"]))
    return (
        torch.stack(feature_rows),
        torch.stack(weight_rows),
        torch.stack(ftv).float(),
        torch.tensor(core).float(),
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("v4.2 feature triage requires exactly one CUDA GPU")
    if args.license_acknowledgement != "I_ACCEPT_CC_BY_NC_ND_4_0_FOR_AUTHORIZED_NONCOMMERCIAL_RESEARCH":
        raise ValueError("explicit 3DINO license acknowledgement is required")
    design_path = Path(args.frozen_design).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser().resolve()
    manifest_report_path = Path(args.manifest_build_report).expanduser().resolve()
    roi_root = Path(args.roi_root).expanduser().resolve()
    roi_index_path = Path(args.roi_index).expanduser().resolve()
    mask_root = Path(args.context_mask_release).expanduser().resolve()
    mask_report_path = Path(args.context_mask_report).expanduser().resolve()
    mask_index_path = Path(args.context_mask_index).expanduser().resolve()
    repo_path = Path(args.three_dino_repo).expanduser().resolve()
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    output = Path(args.output_dir).expanduser().resolve()
    private_output = Path(args.private_output_dir).expanduser().resolve()
    staging = private_output.parent / f".{private_output.name}.staging"
    if output.exists() or private_output.exists():
        raise FileExistsError("immutable v4.2 feature-triage output exists")
    design = _json(design_path)
    manifest_report = _json(manifest_report_path)
    if (
        design.get("schema_version") != DESIGN_SCHEMA
        or design.get("status") != "frozen_feature_triage_required_training_prohibited"
        or design.get("training_started") is not False
        or manifest_report.get("status") != "ok"
        or manifest_report.get("manifest_sha256") != _sha(manifest_path)
        or manifest_report.get("records") != 587
    ):
        raise ValueError("v4.2 design or manifest release is not approved")
    train = load_development_states(manifest_path, "train")
    if len(train) != 409:
        raise ValueError("feature triage must use exactly 409 training patients")
    train_ids = {item.patient_id for item in train}
    full_mask_roster = _mask_roster(mask_index_path)
    full_mask_records = _validate_mask_release(
        mask_root,
        mask_report_path,
        mask_index_path,
        full_mask_roster,
    )
    # Validate the complete 497-patient development mask release using metadata
    # only, then retain the 409 training entries.  No validation feature or mask
    # payload is loaded by the triage computation.
    if len(full_mask_records) != 497 or not train_ids.issubset(full_mask_records):
        raise ValueError("train-only context-mask selection does not align")
    mask_records = {
        patient_id: full_mask_records[patient_id] for patient_id in train_ids
    }
    roi_records = read_roi_index(
        roi_index_path, allowed_input_root=roi_root, expected_patients=587
    )
    roi_by_patient: dict[str, dict[str, Any]] = {patient_id: {} for patient_id in train_ids}
    for record in roi_records:
        if record.patient_id in roi_by_patient:
            roi_by_patient[record.patient_id][record.timepoint] = record
    if any(set(value) != {"T0", "T1", "T3"} for value in roi_by_patient.values()):
        raise ValueError("training patients lack governed ROI visits")
    verify_three_dino_checkout(repo_path, THREE_DINO_REVISION)
    model = build_three_dino_model(
        repo_path,
        checkpoint_path,
        expected_checkpoint_sha256=args.expected_checkpoint_sha256,
        checkpoint_key="teacher",
    ).to("cuda")
    model.eval()
    device = torch.device("cuda")
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.autocast_dtype]
    extraction_contract = {
        "frozen_design_sha256": _sha(design_path),
        "manifest_sha256": _sha(manifest_path),
        "context_mask_index_sha256": _sha(mask_index_path),
        "roi_index_sha256": _sha(roi_index_path),
        "three_dino_revision": THREE_DINO_REVISION,
        "three_dino_checkpoint_sha256": _sha(checkpoint_path),
        "autocast_dtype": args.autocast_dtype,
        "feature_triage_source_sha256": _sha(Path(__file__)),
        "preprocessing_source_sha256": _sha(Path(preprocessing_module.__file__)),
    }
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    patient_files: list[Path] = []
    normalization_status_counts: Counter[str] = Counter()
    for index, item in enumerate(train):
        path = staging / _patient_output_name(item.patient_id)
        if path.exists():
            value = torch.load(path, map_location="cpu", weights_only=True)
            if (
                value.get("schema_version") != PATIENT_SCHEMA
                or value.get("patient_id") != item.patient_id
                or value.get("extraction_contract") != extraction_contract
            ):
                raise ValueError("resumable feature-triage payload does not match")
        else:
            value = _extract_patient(
                item,
                roi_by_patient[item.patient_id],
                mask_records[item.patient_id],
                model,
                device=device,
                dtype=dtype,
                extraction_contract=extraction_contract,
            )
            temporary = path.with_suffix(".partial")
            torch.save(value, temporary)
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        patient_files.append(path)
        estimator = str(
            value.get("reference_normalization", {}).get("scale_estimator", "")
        )
        if estimator not in {
            "IQR_all_visits",
            "P90_minus_P10_all_visits",
            "median_only_due_to_degenerate_scale",
        }:
            raise ValueError("SER reference-normalization estimator is invalid")
        normalization_status_counts[estimator] += 1
        if (index + 1) % 10 == 0 or index + 1 == len(train):
            print(json.dumps({"patients_extracted": index + 1, "patients_expected": 409}), flush=True)

    variant_reports: dict[str, Any] = {}
    private_errors: dict[str, Any] = {}
    canonical_errors: dict[str, Tensor] | None = None
    candidate_delta: dict[str, Any] = {}
    for variant_index, variant in enumerate(VARIANTS):
        features, weights, ftv, core = _load_variant(patient_files, variant)
        nested, errors = nested_shrinkage_evaluation(
            features,
            weights,
            seed=int(args.seed) + variant_index * 1_000,
            device=device,
        )
        disease = REGIONS.index("t0_disease_bed")
        pooled = _weighted_pool(features[:, 0], weights[:, disease])
        ftv_probe = nested_scalar_probe(
            pooled, torch.log1p(ftv.clamp_min(0)), seed=int(args.seed) + variant_index * 1_000 + 101
        )
        core_probe = nested_scalar_probe(
            pooled, torch.log1p(core), seed=int(args.seed) + variant_index * 1_000 + 202
        )
        specificity = _specificity_from_errors(errors)
        variant_reports[variant] = {
            "feature_grid": list(features.shape[-3:]),
            "nested_shrinkage": nested,
            "disease_specificity": specificity,
            "same_visit_T0_FTV_probe": ftv_probe,
            "same_visit_T0_core_volume_probe": core_probe,
        }
        private_errors[variant] = errors
        if variant == "canonical_cached_7cube":
            canonical_errors = errors
        else:
            assert canonical_errors is not None
            candidate_delta[variant] = relative_gain_delta_bootstrap(
                errors,
                canonical_errors,
                seed=int(args.seed) + 50_000 + variant_index,
            )
        del features, weights

    canonical = variant_reports["canonical_cached_7cube"]
    eligible: list[str] = []
    eligibility: dict[str, Any] = {}
    for variant in VARIANTS[1:]:
        value = variant_reports[variant]
        comparison = candidate_delta[variant]
        checks = {
            "FTV_probe_R2_not_degraded_more_than_0p02": (
                value["same_visit_T0_FTV_probe"]["R2"]
                >= canonical["same_visit_T0_FTV_probe"]["R2"] - 0.02
            ),
            "core_probe_R2_not_degraded_more_than_0p02": (
                value["same_visit_T0_core_volume_probe"]["R2"]
                >= canonical["same_visit_T0_core_volume_probe"]["R2"] - 0.02
            ),
            "disease_predictability_gain_over_canonical_CI_lower_positive": (
                comparison["paired_bootstrap_95ci"][0] > 0
            ),
            "disease_specific_shrinkage_gain_positive": (
                value["disease_specificity"][
                    "disease_minus_background_relative_MAE_reduction"
                ] > 0
            ),
        }
        passed = all(checks.values())
        eligibility[variant] = {"checks": checks, "passed": passed}
        if passed:
            eligible.append(variant)
    if eligible:
        selected = max(
            eligible,
            key=lambda name: (
                candidate_delta[name]["paired_bootstrap_95ci"][0],
                candidate_delta[name][
                    "candidate_minus_canonical_disease_relative_MAE_reduction"
                ],
            ),
        )
        decision = "intervention_locked_from_train_only_nested_CV"
    else:
        selected = "canonical_cached_7cube"
        decision = "no_intervention_cleared_train_only_gate_canonical_retained"

    private_output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.replace(staging, private_output)
    index_path = private_output / "feature_variant_index.pt"
    torch.save(
        {
            "schema_version": PRIVATE_SCHEMA,
            "patient_ids": [item.patient_id for item in train],
            "patient_files": [path.name for path in patient_files],
            "extraction_contract": extraction_contract,
            "outer_fold_patient_errors": private_errors,
            "selected_variant": selected,
        },
        index_path,
    )
    os.chmod(index_path, 0o600)
    report = {
        "schema_version": SCHEMA,
        "status": "completed_train_only_feature_triage",
        "scope": "409_train_only_nested_grouped_CV_no_validation_or_test",
        "patients": {"train_loaded": 409, "validation_loaded": 0, "test_loaded": 0},
        "variants": variant_reports,
        "intervention_vs_canonical_bootstrap": candidate_delta,
        "eligibility": eligibility,
        "selection": {
            "selected_variant": selected,
            "decision": decision,
            "validation_88_used_for_selection": False,
            "true_1mm_feature_claimed": False,
        },
        "reference_normalization_qc": {
            "patient_count": sum(normalization_status_counts.values()),
            "scale_estimator_counts": dict(sorted(normalization_status_counts.items())),
            "one_estimator_shared_across_all_three_visits_per_patient": True,
        },
        "training_performed": False,
        "validation_88_loaded": False,
        "test_split_loaded": False,
        "patient_identifiers_emitted": False,
        "evidence": {
            "frozen_design_sha256": _sha(design_path),
            "manifest_sha256": _sha(manifest_path),
            "manifest_build_report_sha256": _sha(manifest_report_path),
            "ROI_index_sha256": _sha(roi_index_path),
            "context_mask_report_sha256": _sha(mask_report_path),
            "context_mask_index_sha256": _sha(mask_index_path),
            "three_dino_checkpoint_sha256": _sha(checkpoint_path),
            "source_sha256": _sha(Path(__file__)),
            "preprocessing_source_sha256": _sha(Path(preprocessing_module.__file__)),
            "private_index_sha256": _sha(index_path),
        },
    }
    output.mkdir(parents=True, mode=0o700)
    report_path = output / "aggregate_report.json"
    with report_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.chmod(report_path, 0o600)
    print(
        json.dumps(
            {
                "status": report["status"],
                "selected_variant": selected,
                "validation_88_loaded": False,
                "test_split_loaded": False,
            },
            sort_keys=True,
        )
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-design", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-build-report", required=True)
    parser.add_argument("--roi-root", required=True)
    parser.add_argument("--roi-index", required=True)
    parser.add_argument("--context-mask-release", required=True)
    parser.add_argument("--context-mask-report", required=True)
    parser.add_argument("--context-mask-index", required=True)
    parser.add_argument("--three-dino-repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--expected-checkpoint-sha256", required=True)
    parser.add_argument("--license-acknowledgement", required=True)
    parser.add_argument("--autocast-dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--private-output-dir", required=True)
    run(parser.parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "VARIANTS",
    "nested_scalar_probe",
    "nested_shrinkage_evaluation",
    "relative_gain_delta_bootstrap",
    "run",
    "select_alpha_leave_one_out",
]
