"""Leakage-safe train409 data assembly for Med-JEPA v4.3."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor

from med_jepa.frozen_audit.features import load_mask_sidecar
from med_jepa.roi_v4.audit_mechanistic import REGIONS, _validate_mask_release
from med_jepa.roi_v4.data import PatientStates
from med_jepa.roi_v4p2.data import load_selected_train_release
from med_jepa.roi_v4p2.feature_triage import _mask_roster
from med_jepa.roi_v4p2.preprocessing import local_token_weights, mask_center


def _sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _t0_only_regions(mask: dict) -> Tensor:
    """Return T0-defined regions without consulting T1/T3 valid support."""

    valid = mask["valid_support"]["T0"].bool()
    core = mask["t0"].bool() & valid
    bed = mask["t0_bed"].bool()
    ring = bed & ~mask["t0"].bool() & valid
    disease = core | ring
    background = ~bed & valid
    result = torch.cat((core, ring, disease, background, valid), dim=0).float()
    if result.shape != (len(REGIONS), 112, 112, 112):
        raise ValueError("T0-only occupancy source violates the canonical ROI")
    if float(core.sum()) <= 0 or float(disease.sum()) <= 0:
        raise ValueError("T0-only tumour occupancy is empty")
    return result


def load_v4p3_train409_release(
    manifest_path: str | Path,
    feature_root: str | Path,
    feature_index_path: str | Path,
    *,
    expected_feature_index_sha256: str,
    context_mask_root: str | Path,
    context_mask_report: str | Path,
    context_mask_index: str | Path,
) -> tuple[list[PatientStates], Tensor, Tensor, str, str]:
    """Load local64 states, shared-valid evaluation weights and T0-only inputs.

    The full 497-patient development mask index is validated as metadata, but
    only the 409 train-patient mask payloads are opened.  T1/T3 valid support is
    deliberately not referenced while constructing predictor inputs.
    """

    records, evaluation_weights, feature_merkle = load_selected_train_release(
        manifest_path,
        feature_root,
        feature_index_path,
        expected_index_sha256=expected_feature_index_sha256,
    )
    index_path = Path(feature_index_path).expanduser().resolve()
    feature_index = torch.load(index_path, map_location="cpu", weights_only=True)
    extraction = feature_index.get("extraction_contract", {})
    mask_index_path = Path(context_mask_index).expanduser().resolve()
    if extraction.get("context_mask_index_sha256") != _sha(mask_index_path):
        raise ValueError("T0 occupancy mask index differs from feature extraction")

    roster = _mask_roster(mask_index_path)
    mask_records = _validate_mask_release(
        Path(context_mask_root).expanduser().resolve(),
        Path(context_mask_report).expanduser().resolve(),
        mask_index_path,
        roster,
    )
    ids = [item.patient_id for item in records]
    if len(mask_records) != 497 or not set(ids).issubset(mask_records):
        raise ValueError("T0 occupancy mask release does not align to train409")
    rows: list[Tensor] = []
    digest = hashlib.sha256()
    for item in records:
        path, expected_sha = mask_records[item.patient_id]
        mask = load_mask_sidecar(path, expected_sha256=expected_sha)
        regions = _t0_only_regions(mask)
        center = mask_center(mask["t0_bed"].bool())
        local = local_token_weights(regions, center, crop_size=64).float()
        # Core/ring are predictive T0 inputs.  T0 valid support is carried only
        # to keep padding out of the global background pooling operation.
        inputs = local[[0, 1, 4]].contiguous()
        if (
            inputs.shape != (3, 7, 7, 7)
            or not torch.isfinite(inputs).all()
            or torch.any((inputs < 0) | (inputs > 1))
            or torch.any(inputs[:2].sum(0) > inputs[2] + 1e-5)
        ):
            raise ValueError("T0-only local64 occupancy input is invalid")
        rows.append(inputs)
        digest.update(bytes.fromhex(expected_sha))
        digest.update(inputs.numpy().tobytes(order="C"))
    return (
        records,
        evaluation_weights,
        torch.stack(rows),
        feature_merkle,
        digest.hexdigest(),
    )


__all__ = ["load_v4p3_train409_release"]
