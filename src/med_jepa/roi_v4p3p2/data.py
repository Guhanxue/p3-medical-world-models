"""Validation88 loading for the checkpoint-gated v4.3.2 evaluation."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import Tensor

from med_jepa.frozen_audit.features import load_mask_sidecar
from med_jepa.roi_v4.audit_mechanistic import _validate_mask_release
from med_jepa.roi_v4.data import PatientStates
from med_jepa.roi_v4p2.data import load_existing_selected_validation_release
from med_jepa.roi_v4p2.feature_triage import _mask_roster
from med_jepa.roi_v4p2.preprocessing import local_token_weights, mask_center
from med_jepa.roi_v4p3.data import _t0_only_regions


def load_v432_validation_release(
    *,
    manifest_path: str | Path,
    validation_feature_root: str | Path,
    expected_validation_index_sha256: str,
    expected_validation_payload_merkle_sha256: str,
    context_mask_root: str | Path,
    context_mask_report: str | Path,
    context_mask_index: str | Path,
) -> tuple[list[PatientStates], Tensor, Tensor, str, str, Path]:
    records, weights, feature_merkle, feature_index, _ = (
        load_existing_selected_validation_release(
            manifest_path=manifest_path,
            validation_root=validation_feature_root,
            expected_index_sha256=expected_validation_index_sha256,
            expected_payload_merkle_sha256=(
                expected_validation_payload_merkle_sha256
            ),
        )
    )
    mask_index = Path(context_mask_index).expanduser().resolve()
    roster = _mask_roster(mask_index)
    mask_records = _validate_mask_release(
        Path(context_mask_root).expanduser().resolve(),
        Path(context_mask_report).expanduser().resolve(),
        mask_index,
        roster,
    )
    ids = [item.patient_id for item in records]
    if len(records) != 88 or len(mask_records) != 497 or not set(ids).issubset(mask_records):
        raise ValueError("v4.3.2 validation mask release does not align")
    rows: list[Tensor] = []
    digest = hashlib.sha256()
    for item in records:
        path, expected_sha = mask_records[item.patient_id]
        mask = load_mask_sidecar(path, expected_sha256=expected_sha)
        regions = _t0_only_regions(mask)
        center = mask_center(mask["t0_bed"].bool())
        local = local_token_weights(regions, center, crop_size=64).float()
        inputs = local[[0, 1, 4]].contiguous()
        if (
            inputs.shape != (3, 7, 7, 7)
            or not torch.isfinite(inputs).all()
            or torch.any((inputs < 0) | (inputs > 1))
            or torch.any(inputs[:2].sum(0) > inputs[2] + 1e-5)
        ):
            raise ValueError("v4.3.2 validation T0 occupancy is invalid")
        rows.append(inputs)
        digest.update(bytes.fromhex(expected_sha))
        digest.update(inputs.numpy().tobytes(order="C"))
    return (
        records,
        weights,
        torch.stack(rows),
        feature_merkle,
        digest.hexdigest(),
        feature_index,
    )


__all__ = ["load_v432_validation_release"]
