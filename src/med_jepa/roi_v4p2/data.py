"""Governed selected-feature loading for Med-JEPA v4.2."""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.cache_3dino import (
    DEFAULT_INPUT_SHAPE,
    THREE_DINO_REVISION,
    build_three_dino_model,
    fixed_ser_normalize,
    load_roi_payload,
    read_roi_index,
    verify_three_dino_checkout,
)
from med_jepa.frozen_audit.features import load_mask_sidecar
from med_jepa.roi_v4.audit_mechanistic import REGIONS, _validate_mask_release
from med_jepa.roi_v4.data import PatientStates, load_development_states
from med_jepa.roi_v4p2.feature_triage import (
    PATIENT_SCHEMA as TRIAGE_PATIENT_SCHEMA,
    PRIVATE_SCHEMA as TRIAGE_PRIVATE_SCHEMA,
    _extract_feature_batch,
    _mask_roster,
)
from med_jepa.roi_v4p2.lock import SELECTED_VARIANT
from med_jepa.roi_v4p2.preprocessing import (
    local_magnified_view,
    local_token_weights,
    mask_center,
    voxel_regions,
)


VALIDATION_SCHEMA = "med-jepa-roi-v4p2-private-validation-features-v1"
VALIDATION_PATIENT_SCHEMA = (
    "med-jepa-roi-v4p2-private-validation-patient-feature-v1"
)


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _merkle(digests: Sequence[str]) -> str:
    value = hashlib.sha256()
    for digest in digests:
        if len(digest) != 64:
            raise ValueError("invalid patient payload digest")
        value.update(bytes.fromhex(digest))
    return value.hexdigest()


def _safe_child(root: Path, name: str) -> Path:
    if not name or Path(name).name != name:
        raise ValueError("private patient payload name is unsafe")
    path = (root / name).resolve()
    if path.parent != root.resolve():
        raise ValueError("private patient payload escapes its release")
    return path


def _validate_selected_tensors(features: Tensor, weights: Tensor) -> None:
    if (
        features.shape != (3, 1024, 7, 7, 7)
        or weights.shape != (len(REGIONS), 7, 7, 7)
        or not torch.isfinite(features).all()
        or not torch.isfinite(weights).all()
        or torch.any((weights < 0) | (weights > 1))
    ):
        raise ValueError("selected feature payload violates the 7-cube contract")


def load_selected_train_release(
    manifest_path: str | Path,
    private_root: str | Path,
    private_index_path: str | Path,
    *,
    expected_index_sha256: str,
) -> tuple[list[PatientStates], Tensor, str]:
    """Load only the train-selected 64-to-112 tensors from the triage release."""

    root = Path(private_root).expanduser().resolve()
    index_path = Path(private_index_path).expanduser().resolve()
    if index_path.parent != root or _sha(index_path) != expected_index_sha256:
        raise ValueError("feature-triage private index binding does not match")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    records = load_development_states(manifest_path, "train")
    ids = [item.patient_id for item in records]
    if (
        not isinstance(index, Mapping)
        or index.get("schema_version") != TRIAGE_PRIVATE_SCHEMA
        or index.get("selected_variant") != SELECTED_VARIANT
        or index.get("patient_ids") != ids
        or len(index.get("patient_files", ())) != 409
    ):
        raise ValueError("feature-triage private index does not align to train")
    extraction_contract = index.get("extraction_contract")
    if not isinstance(extraction_contract, Mapping):
        raise ValueError("feature-triage extraction contract is unavailable")
    selected_records: list[PatientStates] = []
    weight_rows: list[Tensor] = []
    digests: list[str] = []
    for item, name in zip(records, index["patient_files"], strict=True):
        path = _safe_child(root, str(name))
        digest = _sha(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != TRIAGE_PATIENT_SCHEMA
            or value.get("patient_id") != item.patient_id
            or value.get("extraction_contract") != extraction_contract
            or not isinstance(value.get("features"), Mapping)
            or not isinstance(value.get("region_weights"), Mapping)
        ):
            raise ValueError("feature-triage patient payload does not align")
        features = value["features"][SELECTED_VARIANT].float().contiguous()
        weights = value["region_weights"][SELECTED_VARIANT].float().contiguous()
        _validate_selected_tensors(features, weights)
        selected_records.append(replace(item, features=features))
        weight_rows.append(weights)
        digests.append(digest)
    return selected_records, torch.stack(weight_rows), _merkle(digests)


def _validation_patient_name(patient_id: str) -> str:
    return hashlib.sha256(patient_id.encode("utf-8")).hexdigest()[:24] + ".pt"


def _load_validation_release(
    records: list[PatientStates],
    root: Path,
    index_path: Path,
    expected_extraction_contract: Mapping[str, Any],
) -> tuple[list[PatientStates], Tensor, str]:
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    ids = [item.patient_id for item in records]
    if (
        not isinstance(index, Mapping)
        or index.get("schema_version") != VALIDATION_SCHEMA
        or index.get("patient_ids") != ids
        or index.get("selected_variant") != SELECTED_VARIANT
        or index.get("extraction_contract")
        != dict(expected_extraction_contract)
        or len(index.get("patient_files", ())) != 88
    ):
        raise ValueError("v4.2 validation feature index does not align")
    selected: list[PatientStates] = []
    weights: list[Tensor] = []
    digests: list[str] = []
    for item, name in zip(records, index["patient_files"], strict=True):
        path = _safe_child(root, str(name))
        digest = _sha(path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            not isinstance(value, Mapping)
            or value.get("schema_version") != VALIDATION_PATIENT_SCHEMA
            or value.get("patient_id") != item.patient_id
            or value.get("extraction_contract")
            != dict(expected_extraction_contract)
        ):
            raise ValueError("v4.2 validation patient payload does not align")
        features = value.get("features")
        region_weights = value.get("region_weights")
        if not isinstance(features, Tensor) or not isinstance(region_weights, Tensor):
            raise ValueError("v4.2 validation payload lacks tensors")
        features = features.float().contiguous()
        region_weights = region_weights.float().contiguous()
        _validate_selected_tensors(features, region_weights)
        selected.append(replace(item, features=features))
        weights.append(region_weights)
        digests.append(digest)
    observed = _merkle(digests)
    if index.get("patient_payload_merkle_sha256") != observed:
        raise ValueError("v4.2 validation patient payload Merkle root changed")
    return selected, torch.stack(weights), observed


def materialize_selected_validation_release(
    *,
    manifest_path: str | Path,
    roi_root: str | Path,
    roi_index_path: str | Path,
    mask_root: str | Path,
    mask_report_path: str | Path,
    mask_index_path: str | Path,
    three_dino_repo: str | Path,
    checkpoint_path: str | Path,
    expected_checkpoint_sha256: str,
    output_root: str | Path,
    device: torch.device,
    autocast_dtype: torch.dtype,
    extraction_contract: Mapping[str, Any],
) -> tuple[list[PatientStates], Tensor, str, Path]:
    """Open validation only after train-only selection, then extract the locked view."""

    output = Path(output_root).expanduser().resolve()
    index_path = output / "validation_feature_index.pt"
    records = load_development_states(manifest_path, "val")
    if index_path.is_file():
        selected, weights, merkle = _load_validation_release(
            records, output, index_path, extraction_contract
        )
        return selected, weights, merkle, index_path
    if output.exists():
        raise ValueError("incomplete immutable validation feature release exists")
    staging = output.parent / f".{output.name}.staging"
    staging.mkdir(parents=True, exist_ok=True, mode=0o700)
    val_ids = {item.patient_id for item in records}
    full_roster = _mask_roster(Path(mask_index_path))
    mask_records = _validate_mask_release(
        Path(mask_root),
        Path(mask_report_path),
        Path(mask_index_path),
        full_roster,
    )
    if len(mask_records) != 497 or not val_ids.issubset(mask_records):
        raise ValueError("validation mask release does not align")
    roi_records = read_roi_index(
        roi_index_path,
        allowed_input_root=roi_root,
        expected_patients=587,
    )
    roi_by_patient: dict[str, dict[str, Any]] = {
        patient_id: {} for patient_id in val_ids
    }
    for record in roi_records:
        if record.patient_id in roi_by_patient:
            roi_by_patient[record.patient_id][record.timepoint] = record
    if any(set(value) != {"T0", "T1", "T3"} for value in roi_by_patient.values()):
        raise ValueError("validation patients lack governed ROI visits")
    verify_three_dino_checkout(three_dino_repo, THREE_DINO_REVISION)
    model = build_three_dino_model(
        three_dino_repo,
        checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        checkpoint_key="teacher",
    ).to(device)
    model.eval()
    names: list[str] = []
    digests: list[str] = []
    for counter, item in enumerate(records):
        name = _validation_patient_name(item.patient_id)
        path = staging / name
        if not path.exists():
            mask_path, mask_sha = mask_records[item.patient_id]
            mask = load_mask_sidecar(mask_path, expected_sha256=mask_sha)
            regions = voxel_regions(mask)
            center = mask_center(mask["t0_bed"].bool())
            views: list[Tensor] = []
            source_hashes: list[str] = []
            for visit in ("T0", "T1", "T3"):
                roi = roi_by_patient[item.patient_id][visit]
                if _sha(roi.roi_path) != roi.source_content_sha256:
                    raise RuntimeError("governed validation ROI content hash mismatch")
                physical, _, _ = load_roi_payload(roi.roi_path, DEFAULT_INPUT_SHAPE)
                views.append(
                    local_magnified_view(
                        fixed_ser_normalize(physical), center, crop_size=64
                    )
                )
                source_hashes.append(roi.source_content_sha256)
            features = _extract_feature_batch(
                model,
                torch.stack(views),
                device=device,
                dtype=autocast_dtype,
            )
            weights = local_token_weights(regions, center, crop_size=64)
            _validate_selected_tensors(features, weights)
            temporary = path.with_suffix(".partial")
            torch.save(
                {
                    "schema_version": VALIDATION_PATIENT_SCHEMA,
                    "patient_id": item.patient_id,
                    "selected_variant": SELECTED_VARIANT,
                    "extraction_contract": dict(extraction_contract),
                    "features": features.half(),
                    "region_weights": weights.half(),
                    "source_roi_sha256": source_hashes,
                },
                temporary,
            )
            os.chmod(temporary, 0o600)
            os.replace(temporary, path)
        value = torch.load(path, map_location="cpu", weights_only=True)
        if (
            value.get("schema_version") != VALIDATION_PATIENT_SCHEMA
            or value.get("patient_id") != item.patient_id
            or value.get("selected_variant") != SELECTED_VARIANT
            or value.get("extraction_contract") != dict(extraction_contract)
        ):
            raise ValueError("resumable validation feature payload does not match")
        names.append(name)
        digests.append(_sha(path))
        if (counter + 1) % 10 == 0 or counter + 1 == len(records):
            print(
                {"validation_features_extracted": counter + 1, "expected": 88},
                flush=True,
            )
    merkle = _merkle(digests)
    torch.save(
        {
            "schema_version": VALIDATION_SCHEMA,
            "patient_ids": [item.patient_id for item in records],
            "patient_files": names,
            "selected_variant": SELECTED_VARIANT,
            "extraction_contract": dict(extraction_contract),
            "patient_payload_merkle_sha256": merkle,
        },
        staging / "validation_feature_index.pt",
    )
    os.chmod(staging / "validation_feature_index.pt", 0o600)
    os.replace(staging, output)
    selected, weights, observed = _load_validation_release(
        records,
        output,
        output / "validation_feature_index.pt",
        extraction_contract,
    )
    return selected, weights, observed, output / "validation_feature_index.pt"


def load_existing_selected_validation_release(
    *,
    manifest_path: str | Path,
    validation_root: str | Path,
    expected_index_sha256: str,
    expected_payload_merkle_sha256: str,
) -> tuple[list[PatientStates], Tensor, str, Path, Mapping[str, Any]]:
    """Reuse an immutable parent validation view without re-extracting features.

    This entry point exists for explicitly post-hoc recovery analyses.  It
    verifies the parent index and every patient payload before returning the
    same tensors used by the parent run; it never reads raw DICOM or ROI pixel
    data and it cannot materialize a missing validation release.
    """

    root = Path(validation_root).expanduser().resolve()
    index_path = root / "validation_feature_index.pt"
    if (
        len(expected_index_sha256) != 64
        or len(expected_payload_merkle_sha256) != 64
        or not index_path.is_file()
        or _sha(index_path) != expected_index_sha256
    ):
        raise ValueError("parent v4.2 validation feature index changed")
    index = torch.load(index_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(index, Mapping)
        or index.get("schema_version") != VALIDATION_SCHEMA
        or index.get("selected_variant") != SELECTED_VARIANT
        or not isinstance(index.get("extraction_contract"), Mapping)
        or len(index.get("patient_ids", ())) != 88
        or len(set(index.get("patient_ids", ()))) != 88
        or len(index.get("patient_files", ())) != 88
        or index.get("patient_payload_merkle_sha256")
        != expected_payload_merkle_sha256
    ):
        raise ValueError("parent v4.2 validation feature release is invalid")
    records = load_development_states(manifest_path, "val")
    selected, weights, observed = _load_validation_release(
        records,
        root,
        index_path,
        index["extraction_contract"],
    )
    if observed != expected_payload_merkle_sha256:
        raise ValueError("parent validation payload Merkle root changed")
    return (
        selected,
        weights,
        observed,
        index_path,
        dict(index["extraction_contract"]),
    )


__all__ = [
    "VALIDATION_SCHEMA",
    "load_existing_selected_validation_release",
    "load_selected_train_release",
    "materialize_selected_validation_release",
]
