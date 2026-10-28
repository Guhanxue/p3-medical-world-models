"""Build context-only ROI patch weights for the isolated ROI-aware v3 arm.

The source is a private ``med_jepa_spatial`` mask sidecar, but this module does not
import that sibling package.  PyTorch deserializes the complete train/validation
sidecar (which contains future/evaluation tensors); the conversion then reads only
four context-safe tensors: T0/T1 acquisition validity, the T0 FTV-inclusion support,
and the T0 20 mm disease bed.  Future fields are never used to construct weights and
the emitted payload contains no T3 tensor, target field, outcome, or evaluation mask.

The 112-cube mask grid and the cached 3DINO 7-cube share an exact 16-voxel patch
partition.  Average pooling therefore yields continuous patch occupancy without
discarding small supports through nearest-neighbour thresholding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


SOURCE_MASK_SCHEMA_VERSION = "spatial-med-jepa-ispy2-mask-v1"
CONTEXT_WEIGHT_SCHEMA_VERSION = "med-jepa-ispy2-roi-v3-context-weights-v1"
CONTEXT_WEIGHT_TENSOR_KIND = "fixed_t0_core_ring_patch_occupancy"
CONTEXT_VISITS = ("T0", "T1")
REGION_ORDER = ("core", "ring")
ROI_SHAPE_DHW = (112, 112, 112)
PATCH_SIZE_DHW = (16, 16, 16)
FEATURE_GRID_SHAPE_DHW = (7, 7, 7)
CONTEXT_WEIGHT_INDEX_COLUMNS = (
    "PatientID",
    "split",
    "weights_path",
    "content_sha256",
)
SOURCE_MASK_INDEX_COLUMNS = ("PatientID", "mask_path", "content_sha256")
CONTEXT_WEIGHT_RELEASE_SCHEMA_VERSION = "med-jepa-ispy2-roi-v3-context-release-v1"
SOURCE_MASK_RELEASE_SCHEMA_VERSION = "spatial-med-jepa-ispy2-mask-release-v1"
SOURCE_PRODUCTION_BINDING_SCHEMA_VERSION = (
    "spatial-med-jepa-g1-production-bindings-v2"
)

_SOURCE_MASK_RELEASE_FIELDS = {
    "schema_version",
    "status",
    "patients",
    "payloads",
    "valid_supports",
    "index_sha256",
    "source_bundle_sha256",
    "partial_release",
    "formal_native_ftv_volume_checks",
    "native_ftv_volume_consistent",
    "maximum_native_ftv_absolute_difference_ml",
    "formal_source_hash_contract_verified",
    "upstream_pilot_report_sha256",
    "pilot_overlay_and_numeric_review_acknowledged",
    "patient_identifiers_in_aggregate_report",
    "paths_in_aggregate_report",
    "production_bindings",
}
_SOURCE_PRODUCTION_BINDING_FIELDS = {
    "schema_version",
    "production_mode",
    "g0_freeze_sha256",
    "g0_preflight_source_aggregate_sha256",
    "trajectory_sha256",
    "ftv_csv_sha256",
    "roi_release_report_sha256",
    "roi_index_sha256",
    "conversion_lineage_sha256",
    "legacy_manifest_sha256",
    "input_bundle_sha256",
    "pilot_selection_policy",
    "pilot_roster_digest_sha256",
    "physical_mapping_policy",
    "seg_support_canonicalization_policy",
    "physical_mapping_exception_count",
    "physical_mapping_exception_roster_digest_sha256",
    "full_release_includes_test_mask_payloads",
    "test_used_for_mask_method_development",
    "pilot_selection_pool",
    "pilot_selection_pool_patients",
}
_SOURCE_PRODUCTION_BINDING_NON_DIGEST_FIELDS = {
    "schema_version",
    "production_mode",
    "pilot_selection_policy",
    "physical_mapping_policy",
    "seg_support_canonicalization_policy",
    "physical_mapping_exception_count",
    "full_release_includes_test_mask_payloads",
    "test_used_for_mask_method_development",
    "pilot_selection_pool",
    "pilot_selection_pool_patients",
}

_PAYLOAD_FIELDS = {
    "schema_version",
    "tensor_kind",
    "context_visits",
    "region_order",
    "roi_shape_dhw",
    "patch_size_dhw",
    "feature_grid_shape_dhw",
    "roi_weights",
    "spatial_provenance",
    "construction",
    "source_mask_sha256",
}
_SAFE_SPATIAL_FIELDS = {
    "reference_visit",
    "roi_source_visits",
    "spacing_mm",
    "orientation",
    "transform_id",
    "roi_voxel_shape",
    "registration_type",
    "deformable_registration",
}
_FORBIDDEN_FIELD_FRAGMENTS = ("t3", "target", "evaluation", "outcome", "pcr")


class ContextWeightError(ValueError):
    """Raised when a context-weight artifact violates the frozen v3 contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _valid_sha256(value: Any) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContextWeightError("source_mask_sha256 must be a 64-character hex digest")
    return digest


def _bool_mask(value: Any, *, name: str) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != torch.bool:
        raise ContextWeightError(f"{name} must be a boolean tensor")
    if tuple(value.shape) != (1, *ROI_SHAPE_DHW):
        raise ContextWeightError(
            f"{name} must have shape {(1, *ROI_SHAPE_DHW)}"
        )
    return value.to(device="cpu").contiguous()


def _safe_spatial_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _SAFE_SPATIAL_FIELDS:
        raise ContextWeightError("source spatial_provenance fields do not match")
    result = deepcopy(dict(value))
    try:
        shape = tuple(int(item) for item in result["roi_voxel_shape"])
        spacing = tuple(float(item) for item in result["spacing_mm"])
    except (TypeError, ValueError, KeyError) as error:
        raise ContextWeightError("source spatial_provenance is malformed") from error
    if (
        str(result["reference_visit"]).strip().upper() != "T0"
        or [str(item).strip().upper() for item in result["roi_source_visits"]] != ["T0"]
        or shape != ROI_SHAPE_DHW
        or len(spacing) != 3
        or any(item <= 0 or not torch.isfinite(torch.tensor(item)) for item in spacing)
        or str(result["orientation"]).strip().upper() != "RAS"
        or str(result["registration_type"]).strip().casefold() != "rigid"
        or result["deformable_registration"] is not False
        or not str(result["transform_id"]).strip()
    ):
        raise ContextWeightError("source spatial_provenance violates the T0 rigid-grid contract")
    result.update(
        {
            "reference_visit": "T0",
            "roi_source_visits": ["T0"],
            "spacing_mm": list(spacing),
            "orientation": "RAS",
            "roi_voxel_shape": list(ROI_SHAPE_DHW),
            "registration_type": "rigid",
            "deformable_registration": False,
            "transform_id": str(result["transform_id"]).strip(),
        }
    )
    return result


def _reject_forbidden_fields(value: Any, *, owner: str = "payload") -> None:
    """Fail if the emitted object accidentally grows a future-bearing field."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).strip().casefold().replace("_", "")
            if any(fragment in normalized for fragment in _FORBIDDEN_FIELD_FRAGMENTS):
                raise ContextWeightError(f"{owner} contains a forbidden future field")
            _reject_forbidden_fields(item, owner=owner)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_forbidden_fields(item, owner=owner)


def _source_context_tensors(source: Mapping[str, Any]) -> tuple[Tensor, Tensor, Tensor]:
    if source.get("schema_version") != SOURCE_MASK_SCHEMA_VERSION:
        raise ContextWeightError("source mask schema_version is unsupported")
    valid = source.get("valid_support")
    if not isinstance(valid, Mapping):
        raise ContextWeightError("source mask lacks valid_support")
    valid_t0 = _bool_mask(valid.get("T0"), name="valid_support.T0")
    valid_t1 = _bool_mask(valid.get("T1"), name="valid_support.T1")
    t0_core = _bool_mask(
        source.get("t0_ftv_inclusion_support"),
        name="t0_ftv_inclusion_support",
    )
    t0_bed = _bool_mask(
        source.get("t0_disease_bed_20mm"),
        name="t0_disease_bed_20mm",
    )
    if not bool(t0_core.any()):
        raise ContextWeightError("T0 FTV-inclusion support is empty")
    if torch.any(t0_core & ~t0_bed):
        raise ContextWeightError("T0 disease bed does not contain the T0 core")
    shared_valid = valid_t0 & valid_t1
    if not bool(shared_valid.any()):
        raise ContextWeightError("T0/T1 shared acquisition-valid support is empty")
    return t0_core, t0_bed, shared_valid


def build_context_weight_payload(
    source_mask_payload: Mapping[str, Any],
    *,
    source_mask_sha256: str,
) -> dict[str, Any]:
    """Convert one full spatial sidecar into a strictly context-only 7-cube payload."""

    if not isinstance(source_mask_payload, Mapping):
        raise ContextWeightError("source mask payload must be an object")
    t0_core, t0_bed, shared_valid = _source_context_tensors(source_mask_payload)
    core = t0_core & shared_valid
    ring = t0_bed & ~t0_core & shared_valid
    if not bool(core.any()):
        raise ContextWeightError("shared-valid T0 core is empty")
    if not bool(ring.any()):
        raise ContextWeightError("shared-valid T0 ring is empty")
    if torch.any(core & ring):
        raise ContextWeightError("core and ring must be disjoint")

    full_resolution = torch.cat((core, ring), dim=0).float().unsqueeze(0)
    weights = F.avg_pool3d(
        full_resolution,
        kernel_size=PATCH_SIZE_DHW,
        stride=PATCH_SIZE_DHW,
    ).squeeze(0).contiguous()
    if tuple(weights.shape) != (len(REGION_ORDER), *FEATURE_GRID_SHAPE_DHW):
        raise ContextWeightError("pooled ROI weights have an unexpected shape")
    if not bool(torch.isfinite(weights).all()) or torch.any((weights < 0) | (weights > 1)):
        raise ContextWeightError("pooled ROI weights must be finite occupancies in [0,1]")
    if torch.any(weights.reshape(len(REGION_ORDER), -1).sum(dim=1) <= 0):
        raise ContextWeightError("each pooled ROI region must retain positive occupancy")

    payload: dict[str, Any] = {
        "schema_version": CONTEXT_WEIGHT_SCHEMA_VERSION,
        "tensor_kind": CONTEXT_WEIGHT_TENSOR_KIND,
        "context_visits": list(CONTEXT_VISITS),
        "region_order": list(REGION_ORDER),
        "roi_shape_dhw": list(ROI_SHAPE_DHW),
        "patch_size_dhw": list(PATCH_SIZE_DHW),
        "feature_grid_shape_dhw": list(FEATURE_GRID_SHAPE_DHW),
        "roi_weights": weights,
        "spatial_provenance": _safe_spatial_provenance(
            source_mask_payload.get("spatial_provenance")
        ),
        "construction": {
            "validity_rule": "valid_support_t0_intersection_t1_v1",
            "core_rule": "t0_ftv_inclusion_intersect_shared_valid_v1",
            "ring_rule": "t0_20mm_disease_bed_minus_core_intersect_shared_valid_v1",
            "pooling_rule": "aligned_average_occupancy_16cubed_v1",
            "shared_across_forecast_visits": True,
            "shared_with_prediction_query": True,
        },
        "source_mask_sha256": _valid_sha256(source_mask_sha256),
    }
    _reject_forbidden_fields(payload)
    return payload


def validate_context_weight_payload(value: Any) -> dict[str, Any]:
    """Validate and normalize a serialized context-only weight payload."""

    if not isinstance(value, Mapping) or set(value) != _PAYLOAD_FIELDS:
        raise ContextWeightError("context-weight payload fields do not match")
    _reject_forbidden_fields(value)
    if (
        value.get("schema_version") != CONTEXT_WEIGHT_SCHEMA_VERSION
        or value.get("tensor_kind") != CONTEXT_WEIGHT_TENSOR_KIND
        or list(value.get("context_visits", [])) != list(CONTEXT_VISITS)
        or list(value.get("region_order", [])) != list(REGION_ORDER)
        or tuple(value.get("roi_shape_dhw", ())) != ROI_SHAPE_DHW
        or tuple(value.get("patch_size_dhw", ())) != PATCH_SIZE_DHW
        or tuple(value.get("feature_grid_shape_dhw", ())) != FEATURE_GRID_SHAPE_DHW
    ):
        raise ContextWeightError("context-weight payload contract is unsupported")
    weights = value.get("roi_weights")
    if not isinstance(weights, Tensor) or not weights.is_floating_point():
        raise ContextWeightError("roi_weights must be a floating-point tensor")
    weights = weights.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(weights.shape) != (len(REGION_ORDER), *FEATURE_GRID_SHAPE_DHW):
        raise ContextWeightError("roi_weights must have shape [2,7,7,7]")
    if not bool(torch.isfinite(weights).all()) or torch.any((weights < 0) | (weights > 1)):
        raise ContextWeightError("roi_weights must contain finite occupancies in [0,1]")
    if torch.any(weights.reshape(len(REGION_ORDER), -1).sum(dim=1) <= 0):
        raise ContextWeightError("each ROI region must have positive occupancy")
    construction = value.get("construction")
    expected_construction = {
        "validity_rule": "valid_support_t0_intersection_t1_v1",
        "core_rule": "t0_ftv_inclusion_intersect_shared_valid_v1",
        "ring_rule": "t0_20mm_disease_bed_minus_core_intersect_shared_valid_v1",
        "pooling_rule": "aligned_average_occupancy_16cubed_v1",
        "shared_across_forecast_visits": True,
        "shared_with_prediction_query": True,
    }
    if construction != expected_construction:
        raise ContextWeightError("context-weight construction contract does not match")
    result = deepcopy(dict(value))
    result["roi_weights"] = weights
    result["spatial_provenance"] = _safe_spatial_provenance(
        value.get("spatial_provenance")
    )
    result["source_mask_sha256"] = _valid_sha256(value.get("source_mask_sha256"))
    return result


def load_context_weight_payload(path: str | Path) -> dict[str, Any]:
    try:
        value = torch.load(Path(path), map_location="cpu", weights_only=True)
    except Exception as error:
        raise ContextWeightError("context-weight payload is unreadable") from error
    return validate_context_weight_payload(value)


def load_source_mask_and_build(
    path: str | Path,
    *,
    expected_source_sha256: str,
) -> dict[str, Any]:
    """Hash-check one sibling mask sidecar before extracting context-only weights."""

    source = Path(path)
    expected = _valid_sha256(expected_source_sha256)
    try:
        observed = sha256_file(source)
    except OSError as error:
        raise ContextWeightError("source mask payload is unreadable") from error
    if observed != expected:
        raise ContextWeightError("source mask payload SHA-256 mismatch")
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ContextWeightError("source mask payload is unreadable") from error
    if not isinstance(value, Mapping):
        raise ContextWeightError("source mask payload must be an object")
    # Be explicit about the I/O boundary: the governed sibling payload is a single
    # torch archive, so loading its T0 fields also materializes its future fields.
    # Requiring their presence keeps the aggregate provenance honest; the extractor
    # below still has a closed list of four context-safe tensors.
    if not any(
        fragment in str(key).strip().casefold().replace("_", "")
        for key in value
        for fragment in _FORBIDDEN_FIELD_FRAGMENTS
    ) and not any(
        fragment in str(key).strip().casefold().replace("_", "")
        for key in value.get("valid_support", {})
        for fragment in ("t3", "target", "evaluation")
    ):
        raise ContextWeightError(
            "source mask payload lacks the expected future-bearing sidecar fields"
        )
    return build_context_weight_payload(value, source_mask_sha256=observed)


def _context_index_bytes(rows: Sequence[Mapping[str, str]]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(
        stream, fieldnames=CONTEXT_WEIGHT_INDEX_COLUMNS, lineterminator="\n"
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _write_private_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(path.parent, 0o700)
    with path.open("xb") as handle:
        os.chmod(path, 0o600)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())


def _torch_bytes(value: Mapping[str, Any]) -> bytes:
    buffer = io.BytesIO()
    torch.save(dict(value), buffer)
    return buffer.getvalue()


def _resolve_inside(raw: Any, *, index_root: Path, allowed_root: Path) -> Path:
    path = Path(str(raw or "").strip()).expanduser()
    if not str(path) or str(path) == ".":
        raise ContextWeightError("source mask index contains a blank path")
    path = path.resolve() if path.is_absolute() else (index_root / path).resolve()
    try:
        path.relative_to(allowed_root)
    except ValueError as error:
        raise ContextWeightError("source mask path escapes its allowed root") from error
    if not path.is_file():
        raise ContextWeightError("source mask payload is unavailable")
    return path


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ContextWeightError(f"{label} is unreadable") from error
    if not isinstance(value, dict):
        raise ContextWeightError(f"{label} must be a JSON object")
    return value


def _validate_source_development_release(
    report: Mapping[str, Any],
    *,
    source_index_sha256: str,
    legacy_manifest_sha256: str,
    expected_development_counts: Mapping[str, int],
) -> None:
    """Validate the sibling G1-development release without importing it."""

    expected_patients = sum(expected_development_counts.values())
    if set(report) != _SOURCE_MASK_RELEASE_FIELDS:
        raise ContextWeightError("source mask release report fields do not match")
    maximum_difference = report.get("maximum_native_ftv_absolute_difference_ml")
    if (
        isinstance(maximum_difference, bool)
        or not isinstance(maximum_difference, (int, float))
        or not math.isfinite(float(maximum_difference))
        or float(maximum_difference) < 0
    ):
        raise ContextWeightError("source mask release FTV audit is invalid")
    try:
        source_bundle = _valid_sha256(report.get("source_bundle_sha256"))
    except ContextWeightError as error:
        raise ContextWeightError("source mask source-bundle digest is invalid") from error
    if (
        report.get("schema_version") != SOURCE_MASK_RELEASE_SCHEMA_VERSION
        or report.get("status") != "ok"
        or report.get("patients") != expected_patients
        or report.get("payloads") != expected_patients
        or report.get("valid_supports") != expected_patients * 3
        or report.get("index_sha256") != source_index_sha256
        or source_bundle != report.get("source_bundle_sha256")
        or report.get("partial_release") is not False
        or report.get("formal_native_ftv_volume_checks") != expected_patients
        or report.get("native_ftv_volume_consistent") is not True
        or report.get("formal_source_hash_contract_verified") is not True
        or report.get("upstream_pilot_report_sha256") is not None
        or report.get("pilot_overlay_and_numeric_review_acknowledged") is not False
        or report.get("patient_identifiers_in_aggregate_report") is not False
        or report.get("paths_in_aggregate_report") is not False
    ):
        raise ContextWeightError("source mask development release is not approved")

    binding = report.get("production_bindings")
    if not isinstance(binding, Mapping) or set(binding) != _SOURCE_PRODUCTION_BINDING_FIELDS:
        raise ContextWeightError("source mask production binding fields do not match")
    for field in (
        _SOURCE_PRODUCTION_BINDING_FIELDS
        - _SOURCE_PRODUCTION_BINDING_NON_DIGEST_FIELDS
    ):
        try:
            _valid_sha256(binding.get(field))
        except ContextWeightError as error:
            raise ContextWeightError(
                "source mask production binding digest is invalid"
            ) from error
    exception_count = binding.get("physical_mapping_exception_count")
    if (
        isinstance(exception_count, bool)
        or not isinstance(exception_count, int)
        or exception_count < 0
    ):
        raise ContextWeightError("source mask physical-mapping count is invalid")
    if (
        binding.get("schema_version") != SOURCE_PRODUCTION_BINDING_SCHEMA_VERSION
        or binding.get("production_mode") != "development"
        or binding.get("legacy_manifest_sha256") != legacy_manifest_sha256
        or binding.get("pilot_selection_policy")
        != "sha256_rank_spatial_med_jepa_context_mask_pilot_trainval_v1_top10"
        or binding.get("physical_mapping_policy")
        != "lineage_authorized_only_no_runtime_fallback_v1"
        or binding.get("seg_support_canonicalization_policy")
        != "native_voxel_center_occupancy_splat_v1"
        or binding.get("full_release_includes_test_mask_payloads") is not False
        or binding.get("test_used_for_mask_method_development") is not False
        or binding.get("pilot_selection_pool") != "frozen_train_val_only"
        or binding.get("pilot_selection_pool_patients") != expected_patients
    ):
        raise ContextWeightError("source mask development binding is invalid")


def _skip_json_value(text: str, start: int) -> int:
    """Lexically skip one JSON value without materializing it."""

    length = len(text)
    index = start
    while index < length and text[index].isspace():
        index += 1
    if index >= length:
        raise ContextWeightError("legacy manifest development prefix is malformed")
    if text[index] == '"':
        try:
            _, end = json.JSONDecoder().raw_decode(text, index)
        except json.JSONDecodeError as error:
            raise ContextWeightError(
                "legacy manifest development prefix is malformed"
            ) from error
        return end
    if text[index] in "[{":
        stack = ["]" if text[index] == "[" else "}"]
        index += 1
        in_string = False
        escaped = False
        while index < length and stack:
            character = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif character == "\\":
                    escaped = True
                elif character == '"':
                    in_string = False
            elif character == '"':
                in_string = True
            elif character in "[{":
                stack.append("]" if character == "[" else "}")
            elif character in "]}":
                if character != stack.pop():
                    raise ContextWeightError(
                        "legacy manifest development prefix is malformed"
                    )
            index += 1
        if stack:
            raise ContextWeightError("legacy manifest development prefix is malformed")
        return index
    value_start = index
    while index < length and text[index] not in ",}":
        index += 1
    if index == value_start:
        raise ContextWeightError("legacy manifest development prefix is malformed")
    return index


def _development_identity_from_json_prefix(raw_line: bytes) -> tuple[str, str]:
    """Read only top-level patient/split fields and stop before target."""

    try:
        text = raw_line.decode("utf-8")
    except UnicodeError as error:
        raise ContextWeightError("legacy manifest development prefix is unreadable") from error
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        raise ContextWeightError("legacy manifest development record must be an object")
    index += 1
    result: dict[str, str] = {}
    while index < len(text):
        while index < len(text) and (text[index].isspace() or text[index] == ","):
            index += 1
        if index >= len(text) or text[index] == "}":
            break
        try:
            key, index = decoder.raw_decode(text, index)
        except json.JSONDecodeError as error:
            raise ContextWeightError(
                "legacy manifest development prefix is malformed"
            ) from error
        if not isinstance(key, str):
            raise ContextWeightError("legacy manifest development key is invalid")
        while index < len(text) and text[index].isspace():
            index += 1
        if index >= len(text) or text[index] != ":":
            raise ContextWeightError("legacy manifest development prefix is malformed")
        index += 1
        if key in {"patient_id", "split"}:
            try:
                value, index = decoder.raw_decode(text, index)
            except json.JSONDecodeError as error:
                raise ContextWeightError(
                    "legacy manifest development identity is malformed"
                ) from error
            if key in result or not isinstance(value, str):
                raise ContextWeightError("legacy manifest development identity is invalid")
            result[key] = value
            if set(result) == {"patient_id", "split"}:
                return result["patient_id"].strip(), result["split"].strip().casefold()
        elif key.casefold() in {"target", "t3"}:
            raise ContextWeightError(
                "legacy manifest identity must precede every future field"
            )
        else:
            index = _skip_json_value(text, index)
    raise ContextWeightError("legacy manifest development identity is missing")


def read_development_split_prefix(
    legacy_manifest_path: str | Path,
    *,
    expected_legacy_manifest_sha256: str,
    expected_development_counts: Mapping[str, int] = {"train": 409, "val": 88},
) -> tuple[dict[str, str], str]:
    """Hash-bind a manifest and read only its ordered train/val prefix.

    No test record is parsed, and no context or target object is materialized.
    """

    manifest = Path(legacy_manifest_path).expanduser().resolve()
    expected_digest = _valid_sha256(expected_legacy_manifest_sha256)
    try:
        observed_digest = sha256_file(manifest)
    except OSError as error:
        raise ContextWeightError("legacy manifest is unreadable") from error
    if observed_digest != expected_digest:
        raise ContextWeightError("legacy manifest SHA-256 mismatch")
    if set(expected_development_counts) != {"train", "val"}:
        raise ContextWeightError("expected development counts must contain train and val")
    counts: dict[str, int] = {}
    for split in ("train", "val"):
        value = expected_development_counts[split]
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ContextWeightError("expected development counts must be positive integers")
        counts[split] = value
    expected_rows = sum(counts.values())
    split_map: dict[str, str] = {}
    try:
        with manifest.open("rb") as handle:
            for row_index in range(expected_rows):
                raw_line = handle.readline()
                if not raw_line or not raw_line.strip():
                    raise ContextWeightError(
                        "legacy manifest development prefix has wrong cardinality"
                    )
                patient, split = _development_identity_from_json_prefix(raw_line)
                expected_split = "train" if row_index < counts["train"] else "val"
                if not patient or patient in split_map or split != expected_split:
                    raise ContextWeightError(
                        "legacy manifest development prefix order is invalid"
                    )
                split_map[patient] = split
    except OSError as error:
        raise ContextWeightError("legacy manifest is unreadable") from error
    return split_map, observed_digest


def build_context_weight_release(
    source_mask_index_path: str | Path,
    output_dir: str | Path,
    *,
    allowed_source_root: str | Path,
    expected_source_index_sha256: str,
    development_split_by_patient: Mapping[str, str],
    expected_development_counts: Mapping[str, int] = {"train": 409, "val": 88},
    source_mask_release_report_sha256: str | None = None,
    legacy_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    """Create one immutable train/validation-only context-weight release.

    ``development_split_by_patient`` is deliberately supplied by the caller's
    already-governed split source.  Test patients are rejected rather than skipped,
    preventing this builder from silently changing the cohort boundary.
    """

    source_index = Path(source_mask_index_path).expanduser().resolve()
    allowed_root = Path(allowed_source_root).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve(strict=False)
    if not allowed_root.is_dir():
        raise ContextWeightError("allowed source-mask root is unavailable")
    expected_index = _valid_sha256(expected_source_index_sha256)
    try:
        observed_index = sha256_file(source_index)
    except OSError as error:
        raise ContextWeightError("source mask index is unreadable") from error
    if observed_index != expected_index:
        raise ContextWeightError("source mask index SHA-256 mismatch")
    if output.exists():
        raise ContextWeightError("context-weight output already exists")
    if not output.parent.is_dir():
        raise ContextWeightError("context-weight output parent is unavailable")

    if set(expected_development_counts) != {"train", "val"}:
        raise ContextWeightError("expected development counts must contain train and val")
    expected_counts: dict[str, int] = {}
    for split in ("train", "val"):
        count = expected_development_counts[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ContextWeightError("expected development counts must be positive integers")
        expected_counts[split] = int(count)

    split_map: dict[str, str] = {}
    for patient, split_value in development_split_by_patient.items():
        patient_id = str(patient).strip()
        split = str(split_value).strip().casefold()
        if not patient_id or patient_id in split_map:
            raise ContextWeightError("development split mapping has duplicate/blank patients")
        if split not in {"train", "val"}:
            raise ContextWeightError("development split mapping must not contain test")
        split_map[patient_id] = split
    observed_counts = {
        split: sum(value == split for value in split_map.values())
        for split in ("train", "val")
    }
    if observed_counts != expected_counts:
        raise ContextWeightError("development split mapping counts do not match")

    try:
        with source_index.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None or tuple(reader.fieldnames) != SOURCE_MASK_INDEX_COLUMNS:
                raise ContextWeightError("source mask index columns do not match")
            source_rows = list(reader)
    except (OSError, csv.Error, UnicodeError) as error:
        raise ContextWeightError("source mask index is unreadable") from error
    if len(source_rows) != sum(expected_counts.values()):
        raise ContextWeightError("source mask index does not match development cardinality")

    staging: Path | None = None
    try:
        staging = Path(
            tempfile.mkdtemp(prefix=f".{output.name}.", suffix=".tmp", dir=output.parent)
        )
        os.chmod(staging, 0o700)
        output_rows: list[dict[str, str]] = []
        seen_patients: set[str] = set()
        seen_sources: set[str] = set()
        seen_paths: set[Path] = set()
        for row in source_rows:
            patient = str(row.get("PatientID") or "").strip()
            if not patient or patient in seen_patients or patient not in split_map:
                raise ContextWeightError("source mask index patient join is invalid")
            source_digest = _valid_sha256(row.get("content_sha256"))
            source_path = _resolve_inside(
                row.get("mask_path"), index_root=source_index.parent, allowed_root=allowed_root
            )
            if source_digest in seen_sources or source_path in seen_paths:
                raise ContextWeightError("source mask path or content is reused")
            seen_patients.add(patient)
            seen_sources.add(source_digest)
            seen_paths.add(source_path)
            payload = load_source_mask_and_build(
                source_path, expected_source_sha256=source_digest
            )
            opaque = hashlib.sha256(
                b"med-jepa-roi-v3-context-weight\0" + source_digest.encode("ascii")
            ).hexdigest()[:32]
            relative = Path("weights") / f"{opaque}.pt"
            destination = staging / relative
            _write_private_bytes(destination, _torch_bytes(payload))
            output_rows.append(
                {
                    "PatientID": patient,
                    "split": split_map[patient],
                    "weights_path": str(relative),
                    "content_sha256": sha256_file(destination),
                }
            )
        if seen_patients != set(split_map):
            raise ContextWeightError("source mask index does not exactly cover development")
        output_rows.sort(
            key=lambda item: (("train", "val").index(item["split"]), item["PatientID"])
        )
        index_bytes = _context_index_bytes(output_rows)
        _write_private_bytes(staging / "context_weight_index.csv", index_bytes)
        index_sha256 = hashlib.sha256(index_bytes).hexdigest()
        report: dict[str, Any] = {
            "schema_version": CONTEXT_WEIGHT_RELEASE_SCHEMA_VERSION,
            "status": "ok_context_only_roi_weights",
            "patient_counts": expected_counts,
            "patients": len(output_rows),
            "payloads": len(output_rows),
            "test_rows": 0,
            "test_tensor_files_loaded": False,
            "target_fields_emitted": False,
            "source_future_tensors_materialized_during_conversion": True,
            "source_future_fields_used_for_weights": False,
            "source_future_fields_emitted": False,
            "source_mask_index_sha256": observed_index,
            "index_sha256": index_sha256,
            "payload_schema_version": CONTEXT_WEIGHT_SCHEMA_VERSION,
            "tensor_kind": CONTEXT_WEIGHT_TENSOR_KIND,
            "region_order": list(REGION_ORDER),
            "roi_weights_shape": [2, *FEATURE_GRID_SHAPE_DHW],
            "context_only": True,
            "patient_identifiers_emitted_in_report": False,
            "paths_emitted_in_report": False,
        }
        if source_mask_release_report_sha256 is not None:
            report["source_mask_release_report_sha256"] = _valid_sha256(
                source_mask_release_report_sha256
            )
        if legacy_manifest_sha256 is not None:
            report["legacy_manifest_sha256"] = _valid_sha256(legacy_manifest_sha256)

        _write_private_bytes(
            staging / "aggregate_report.json",
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        staging.rename(output)
        staging = None
        return report
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging)


def build_development_context_weight_release(
    *,
    source_mask_release_report_path: str | Path,
    source_mask_index_path: str | Path,
    legacy_manifest_path: str | Path,
    output_dir: str | Path,
    allowed_source_root: str | Path,
    expected_source_mask_release_report_sha256: str,
    expected_source_mask_index_sha256: str,
    expected_legacy_manifest_sha256: str,
    expected_development_counts: Mapping[str, int] = {"train": 409, "val": 88},
) -> dict[str, Any]:
    """Validate G1-development and atomically publish context-only weights."""

    source_root = Path(allowed_source_root).expanduser().resolve()
    source_report = Path(source_mask_release_report_path).expanduser().resolve()
    source_index = Path(source_mask_index_path).expanduser().resolve()
    if not source_root.is_dir():
        raise ContextWeightError("allowed source-mask root is unavailable")
    for path, label in (
        (source_report, "source mask release report"),
        (source_index, "source mask index"),
    ):
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ContextWeightError(f"{label} escapes its allowed root") from error
        if not path.is_file():
            raise ContextWeightError(f"{label} is unavailable")

    expected_report_digest = _valid_sha256(
        expected_source_mask_release_report_sha256
    )
    expected_index_digest = _valid_sha256(expected_source_mask_index_sha256)
    if sha256_file(source_report) != expected_report_digest:
        raise ContextWeightError("source mask release report SHA-256 mismatch")
    if sha256_file(source_index) != expected_index_digest:
        raise ContextWeightError("source mask index SHA-256 mismatch")

    split_map, manifest_digest = read_development_split_prefix(
        legacy_manifest_path,
        expected_legacy_manifest_sha256=expected_legacy_manifest_sha256,
        expected_development_counts=expected_development_counts,
    )
    source_release = _read_json_object(
        source_report, label="source mask release report"
    )
    _validate_source_development_release(
        source_release,
        source_index_sha256=expected_index_digest,
        legacy_manifest_sha256=manifest_digest,
        expected_development_counts=expected_development_counts,
    )
    return build_context_weight_release(
        source_index,
        output_dir,
        allowed_source_root=source_root,
        expected_source_index_sha256=expected_index_digest,
        development_split_by_patient=split_map,
        expected_development_counts=expected_development_counts,
        source_mask_release_report_sha256=expected_report_digest,
        legacy_manifest_sha256=manifest_digest,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-mask-release-report", required=True)
    parser.add_argument("--source-mask-index", required=True)
    parser.add_argument("--legacy-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allowed-source-root", required=True)
    parser.add_argument("--expected-source-mask-release-report-sha256", required=True)
    parser.add_argument("--expected-source-mask-index-sha256", required=True)
    parser.add_argument("--expected-legacy-manifest-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = build_development_context_weight_release(
        source_mask_release_report_path=args.source_mask_release_report,
        source_mask_index_path=args.source_mask_index,
        legacy_manifest_path=args.legacy_manifest,
        output_dir=args.output_dir,
        allowed_source_root=args.allowed_source_root,
        expected_source_mask_release_report_sha256=(
            args.expected_source_mask_release_report_sha256
        ),
        expected_source_mask_index_sha256=args.expected_source_mask_index_sha256,
        expected_legacy_manifest_sha256=args.expected_legacy_manifest_sha256,
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "CONTEXT_VISITS",
    "CONTEXT_WEIGHT_INDEX_COLUMNS",
    "CONTEXT_WEIGHT_RELEASE_SCHEMA_VERSION",
    "CONTEXT_WEIGHT_SCHEMA_VERSION",
    "CONTEXT_WEIGHT_TENSOR_KIND",
    "ContextWeightError",
    "FEATURE_GRID_SHAPE_DHW",
    "PATCH_SIZE_DHW",
    "REGION_ORDER",
    "ROI_SHAPE_DHW",
    "SOURCE_MASK_SCHEMA_VERSION",
    "SOURCE_MASK_RELEASE_SCHEMA_VERSION",
    "SOURCE_MASK_INDEX_COLUMNS",
    "SOURCE_PRODUCTION_BINDING_SCHEMA_VERSION",
    "build_context_weight_payload",
    "build_context_weight_release",
    "build_development_context_weight_release",
    "build_parser",
    "load_context_weight_payload",
    "load_source_mask_and_build",
    "main",
    "read_development_split_prefix",
    "sha256_file",
    "validate_context_weight_payload",
]
