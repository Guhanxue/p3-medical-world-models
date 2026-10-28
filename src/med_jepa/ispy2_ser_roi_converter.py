"""Governed conversion of I-SPY2 SER into fixed longitudinal Med-JEPA ROIs.

This is deliberately a *post-audit* converter rather than a generic DICOM image
loader.  It accepts only the hash-pinned DS3 trajectory, the reviewed zero-only FTV
release and a successful v2 T0 anchor report.  For every selected patient it:

1. derives one physical T0 crop centre from the inverse analysis mask (``raw == 0``);
2. builds the fixed 112^3, 2.1 mm RAS grid that passed the full-cohort audit;
3. resamples T0 by identity and T1/T3 using image-only rigid registration to T0;
4. uses T1/T3 masks *only after that* for future-mask coverage QC; and
5. writes immutable, opaque ROI tensors plus a governed index and private lineage.

No pixel, identifier, path, UID, date, or per-patient metric is written to the public
aggregate report or job stdout.  The governed output itself intentionally contains
the patient index required by later manifest construction and must not be copied out
of CHPC.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
import math
import os
import statistics
import struct
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from med_jepa import ispy2_dce_geometry_probe as production
from med_jepa import ispy2_roi_anchor_audit as anchor
from med_jepa.cache_3dino import (
    ROI_SCHEMA_VERSION,
    SER_INPUT_RANGE,
    SER_RANGE_TOLERANCE,
    load_roi_payload,
    read_roi_index,
    sha256_file,
)
from med_jepa.ftv_release import FTVReleaseError, validate_ftv_release
from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_seg_mapping_probe import (
    SEGMENTATION_STORAGE_UID,
    _frame_position,
    _frame_segment_numbers,
    _sequence,
)
from med_jepa.ispy2_series_probe import _inside, _raise_stale
from med_jepa.ispy2_manifest_builder import validate_spatial_provenance


SCHEMA_VERSION = "med-jepa-ispy2-ser-roi-converter-v2"
CONVERTER_VERSION = "med-jepa-ispy2-roi-converter-v2"
TIMEPOINTS = ("T0", "T1", "T3")
EXPECTED_PATIENTS = production.EXPECTED_COMPLETE_PATIENTS
EXPECTED_SPLIT_COUNTS = production.EXPECTED_SPLIT_COUNTS
EXPECTED_ANCHOR_SCHEMA_VERSION = "med-jepa-ispy2-roi-anchor-audit-v2"
EXPECTED_ROI_SPEC_SCHEMA_VERSION = "med-jepa-ispy2-ser-roi-spec-v1"
EXPECTED_ROI_ID = "ser_roi_112_2p1mm_t0bbox_v1"
EXPECTED_SHAPE_DHW = (112, 112, 112)
EXPECTED_SPACING_DHW_MM = (2.1, 2.1, 2.1)
EXPECTED_ORIENTATION = "RAS"
EXPECTED_MARGIN_MM = 20.0
ANCHOR_CENTER_RULE = "t0_inverse_mask_bbox_midpoint_patient_space_v1"
PADDING_POLICY = "zero_unitless_ser_after_dicom_rescale_v1"
TRANSFORM_ID = "ispy2_ser_t0_bbox_rigid_112_2p1mm_v1"
SERIES_DIGEST_DOMAIN = b"med-jepa-ispy2-ser-dicom-series-v1\0"
SCAN_ID_DOMAIN = b"med-jepa-ispy2-ser-scan-id-v1\0"
ROI_NAME_DOMAIN = b"med-jepa-ispy2-ser-roi-name-v1\0"
PILOT_SELECTION_DOMAIN = b"med-jepa-ispy2-ser-roi-pilot-v1\0"
FUTURE_MASK_EDGE_TOLERANCE_MM = 0.05
MAX_RIGID_CENTER_DISPLACEMENT_MM = 300.0
REGISTRATION_SAMPLING_SEED = 20260728
ROI_INDEX_FIELDS = (
    "PatientID",
    "tp_label",
    "scan_id",
    "roi_path",
    "source_content_sha256",
)
QC_EXCLUSION_FIELDS = (
    "PatientID",
    "exclusion_reason",
    "failed_timepoints",
)
CONVERSION_LINEAGE_FIELDS = (
    "PatientID",
    "tp_label",
    "scan_id",
    "source_dicom_content_sha256",
    "registration_status",
    "registration_metric_value",
    "registration_sampling_seed",
    "registration_iterations",
    "registration_center_displacement_mm",
    "registration_transform_matrix",
    "registration_transform_offset_mm",
    "registration_stop_condition",
    "future_mask_mapping_status",
    "future_mask_inside_fixed_crop",
    "future_mask_minimum_edge_margin_mm",
    "future_mask_included_voxels",
    "roi_content_sha256",
)


class ROIConversionError(RuntimeError):
    """A fail-closed error whose public text is an allowlisted safe category."""

    def __init__(self, code: str, *, metadata: Mapping[str, Any] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.metadata = dict(metadata or {})


@dataclass(frozen=True)
class VisitSource:
    """Private in-memory input values.  Never serialize this in aggregate output."""

    patient_id: str
    timepoint: str
    ser_dir: Path
    mask_dir: Path


@dataclass(frozen=True)
class SERVolume:
    """One rescaled native SER stack and private metadata needed for conversion."""

    image: Any
    ser_geometry: Mapping[str, Any]
    source_dicom_content_sha256: str
    rescale_slope: float
    rescale_intercept: float


def _import_runtime_dependencies() -> tuple[Any, Any, Any]:
    """Load optional cluster dependencies only when pixel conversion is requested."""

    try:
        import numpy as np
        import pydicom
        import SimpleITK as sitk
    except ImportError as error:  # pragma: no cover - depends on CHPC environment
        raise ROIConversionError("roi_conversion_dependencies_unavailable") from error
    return np, pydicom, sitk


def _import_numpy() -> Any:
    """Keep geometry/unit tests independent of the optional SimpleITK runtime."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - cluster package dependency
        raise ROIConversionError("numpy_dependency_unavailable") from error
    return np


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path, *, code: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _raise_stale(error)
        raise ROIConversionError(code) from None
    return digest.hexdigest()


def _finite_float(value: Any, *, code: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ROIConversionError(code) from error
    if not math.isfinite(number):
        raise ROIConversionError(code)
    return number


def _positive_int(value: Any, *, code: str) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ROIConversionError(code) from error
    if number < 1:
        raise ROIConversionError(code)
    return number


def _numeric_summary(values: Iterable[float], *, allow_negative: bool = False) -> dict[str, float]:
    observed = [float(value) for value in values]
    if not observed or any(
        not math.isfinite(value) or (not allow_negative and value < 0.0)
        for value in observed
    ):
        raise ValueError("numeric summary needs finite values")
    ordered = sorted(observed)

    def quantile(probability: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        location = probability * (len(ordered) - 1)
        lower = math.floor(location)
        upper = math.ceil(location)
        if lower == upper:
            return ordered[lower]
        fraction = location - lower
        return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction

    return {
        "min": min(ordered),
        "p05": quantile(0.05),
        "median": statistics.median(ordered),
        "p95": quantile(0.95),
        "max": max(ordered),
    }


def _safe_json_load(path: Path, *, code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ROIConversionError(code) from error
    if not isinstance(value, dict):
        raise ROIConversionError(code)
    return value


def _strict_triplet(
    value: Any,
    *,
    integer: bool,
    code: str,
) -> tuple[int, int, int] | tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ROIConversionError(code)
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in value):
            raise ROIConversionError(code)
        return tuple(int(item) for item in value)
    result = tuple(_finite_float(item, code=code) for item in value)
    if any(item <= 0.0 for item in result):
        raise ROIConversionError(code)
    return result


def load_locked_roi_spec(path: str | Path) -> dict[str, Any]:
    """Load the single formally supported first-ROI configuration.

    The stricter-than-general check avoids accidentally treating a local experiment
    JSON as a compatible production ROI definition.
    """

    source = Path(path).expanduser().resolve()
    spec = _safe_json_load(source, code="roi_spec_unreadable")
    if spec.get("schema_version") != EXPECTED_ROI_SPEC_SCHEMA_VERSION:
        raise ROIConversionError("roi_spec_schema_mismatch")
    if spec.get("roi_id") != EXPECTED_ROI_ID or str(spec.get("modality", "")).upper() != "SER":
        raise ROIConversionError("roi_spec_identity_mismatch")
    shape = _strict_triplet(spec.get("tensor_shape"), integer=True, code="roi_spec_geometry_invalid")
    spacing = _strict_triplet(spec.get("spacing_mm"), integer=False, code="roi_spec_geometry_invalid")
    if shape != EXPECTED_SHAPE_DHW or spacing != EXPECTED_SPACING_DHW_MM:
        raise ROIConversionError("roi_spec_geometry_mismatch")
    if str(spec.get("orientation", "")).strip().upper() != EXPECTED_ORIENTATION:
        raise ROIConversionError("roi_spec_orientation_mismatch")
    anchor_spec = spec.get("anchor")
    registration_spec = spec.get("longitudinal_registration")
    spatial_spec = spec.get("spatial_provenance")
    if not isinstance(anchor_spec, Mapping) or not isinstance(registration_spec, Mapping) or not isinstance(spatial_spec, Mapping):
        raise ROIConversionError("roi_spec_contract_invalid")
    if (
        str(anchor_spec.get("reference_visit", "")).upper() != "T0"
        or list(anchor_spec.get("roi_source_visits", [])) != ["T0"]
        or anchor_spec.get("mask_included_condition") != "raw_value_equals_zero"
        or anchor_spec.get("center_rule") != ANCHOR_CENTER_RULE
        or _finite_float(anchor_spec.get("minimum_margin_mm"), code="roi_spec_contract_invalid")
        != EXPECTED_MARGIN_MM
        or anchor_spec.get("out_of_bounds_padding") != PADDING_POLICY
    ):
        raise ROIConversionError("roi_spec_anchor_contract_mismatch")
    if (
        str(registration_spec.get("registration_type", "")).casefold() != "rigid"
        or registration_spec.get("t0_transform") != "identity_rigid"
        or list(registration_spec.get("moving_visits", [])) != ["T1", "T3"]
        or registration_spec.get("deformable_registration") is not False
        or registration_spec.get("future_mask_role")
        != "coverage_qc_only_never_crop_centering"
    ):
        raise ROIConversionError("roi_spec_registration_contract_mismatch")
    spatial = {
        "reference_visit": "T0",
        "roi_source_visits": ["T0"],
        "spacing_mm": list(spacing),
        "orientation": EXPECTED_ORIENTATION,
        "transform_id": str(spatial_spec.get("transform_id", "")).strip(),
        "roi_voxel_shape": list(shape),
        "registration_type": str(spatial_spec.get("registration_type", "")).casefold(),
        "deformable_registration": spatial_spec.get("deformable_registration"),
    }
    if (
        spatial["transform_id"] != TRANSFORM_ID
        or spatial["registration_type"] != "rigid"
        or spatial["deformable_registration"] is not False
    ):
        raise ROIConversionError("roi_spec_spatial_provenance_mismatch")
    try:
        spatial = validate_spatial_provenance(spatial)
    except Exception as error:
        raise ROIConversionError("roi_spec_spatial_provenance_mismatch") from error
    return {
        "spec": spec,
        "sha256": _sha256_file(source, code="roi_spec_unreadable"),
        "shape_dhw": shape,
        "spacing_dhw_mm": spacing,
        "spatial_provenance": spatial,
    }


def validate_anchor_release(
    path: str | Path,
    *,
    roi_spec: Mapping[str, Any],
    manifest_sha256: str,
    ftv_csv_sha256: str,
    ftv_audit_report_sha256: str,
) -> str:
    """Require the successful canonical-RAS v2 audit before any pixels are read."""

    source = Path(path).expanduser().resolve()
    report = _safe_json_load(source, code="roi_anchor_report_unreadable")
    if report.get("schema_version") != EXPECTED_ANCHOR_SCHEMA_VERSION or report.get("status") != "ok_anchor_geometry":
        raise ROIConversionError("roi_anchor_report_not_approved")
    cohort = report.get("cohort")
    canonical = (
        report.get("summary", {}).get("canonical_ras_grid_feasibility")
        if isinstance(report.get("summary"), Mapping)
        else None
    )
    evidence = report.get("evidence")
    if not isinstance(cohort, Mapping) or not isinstance(canonical, Mapping) or not isinstance(evidence, Mapping):
        raise ROIConversionError("roi_anchor_report_schema_invalid")
    shape = list(roi_spec["shape_dhw"])
    spacing = list(roi_spec["spacing_dhw_mm"])
    if (
        cohort.get("patients_expected") != EXPECTED_PATIENTS
        or canonical.get("orientation") != EXPECTED_ORIENTATION
        or canonical.get("roi_shape") != shape
        or canonical.get("spacing_mm") != spacing
        or canonical.get("pairs_with_support_inside_grid") != EXPECTED_PATIENTS
        or canonical.get("pairs_outside_grid") != 0
        or canonical.get("pairs_with_required_margin") != EXPECTED_PATIENTS
        or canonical.get("pairs_below_required_margin") != 0
        or _finite_float(canonical.get("required_margin_mm"), code="roi_anchor_report_schema_invalid")
        != EXPECTED_MARGIN_MM
    ):
        raise ROIConversionError("roi_anchor_report_geometry_mismatch")
    if (
        evidence.get("trajectory_manifest_sha256") != manifest_sha256
        or evidence.get("ftv_csv_sha256") != ftv_csv_sha256
        or evidence.get("ftv_audit_report_sha256") != ftv_audit_report_sha256
    ):
        raise ROIConversionError("roi_anchor_report_evidence_mismatch")
    return _sha256_file(source, code="roi_anchor_report_unreadable")


def _resolve_manifest_dicom_directory(
    raw_value: str,
    *,
    trajectory: Path,
    dicom_root: Path,
) -> Path:
    try:
        candidate = Path(raw_value).expanduser()
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (trajectory.parent / candidate).resolve()
        )
    except OSError as error:
        _raise_stale(error)
        raise ROIConversionError("trajectory_path_resolution_error") from None
    if not _inside(resolved, dicom_root):
        raise ROIConversionError("trajectory_path_outside_approved_root")
    return resolved


def read_pinned_complete_visits(
    root: str | Path,
    trajectory_path: str | Path,
    *,
    expected_patient_ids: set[str],
    expected_patients: int = EXPECTED_PATIENTS,
) -> tuple[dict[str, dict[str, VisitSource]], str]:
    """Build the three-visit conversion cohort from one pinned manifest snapshot."""

    if expected_patients != EXPECTED_PATIENTS:
        raise ROIConversionError("expected_patient_contract_mismatch")
    root_path = Path(root).expanduser().resolve()
    trajectory = Path(trajectory_path).expanduser().resolve()
    try:
        fields, rows, manifest_sha = production._read_pinned_csv_rows(
            trajectory,
            expected_sha256=production.APPROVED_TRAJECTORY_SHA256,
            hash_error_code="trajectory_hash_mismatch",
            read_error_code="trajectory_read_error",
            schema_error_code="trajectory_schema_mismatch",
        )
    except production.DCEGeometryProbeError as error:
        raise ROIConversionError(error.code) from None
    required = {"PatientID", "tp_label", "ser_dir", "mask_dir"}
    if not required.issubset(fields):
        raise ROIConversionError("trajectory_schema_mismatch")
    raw_visits: dict[str, dict[str, tuple[str, str]]] = {}
    for row in rows:
        patient_id = str(row.get("PatientID") or "").strip()
        timepoint = str(row.get("tp_label") or "").strip().upper()
        if timepoint not in TIMEPOINTS:
            continue
        if not production._PATIENT_ID.fullmatch(patient_id):
            raise ROIConversionError("invalid_patient_identifier")
        ser_dir = str(row.get("ser_dir") or "").strip()
        mask_dir = str(row.get("mask_dir") or "").strip()
        if not ser_dir or not mask_dir:
            continue
        patient_visits = raw_visits.setdefault(patient_id, {})
        if timepoint in patient_visits:
            raise ROIConversionError("duplicate_patient_timepoint")
        patient_visits[timepoint] = (ser_dir, mask_dir)
    complete_ids = {
        patient_id
        for patient_id, visits in raw_visits.items()
        if set(TIMEPOINTS).issubset(visits)
    }
    if complete_ids != expected_patient_ids or len(complete_ids) != expected_patients:
        raise ROIConversionError("complete_trajectory_cohort_mismatch")
    try:
        dicom_root = (root_path / "dicom").resolve()
        if not dicom_root.is_dir():
            raise ROIConversionError("dicom_root_unavailable")
    except OSError as error:
        _raise_stale(error)
        raise ROIConversionError("dicom_root_unavailable") from None
    result: dict[str, dict[str, VisitSource]] = {}
    used_directories: set[Path] = set()
    for patient_id in sorted(complete_ids):
        result[patient_id] = {}
        for timepoint in TIMEPOINTS:
            raw_ser, raw_mask = raw_visits[patient_id][timepoint]
            ser_dir = _resolve_manifest_dicom_directory(raw_ser, trajectory=trajectory, dicom_root=dicom_root)
            mask_dir = _resolve_manifest_dicom_directory(raw_mask, trajectory=trajectory, dicom_root=dicom_root)
            if ser_dir == mask_dir or ser_dir in used_directories or mask_dir in used_directories:
                raise ROIConversionError("duplicate_series_directory_reference")
            used_directories.update((ser_dir, mask_dir))
            result[patient_id][timepoint] = VisitSource(patient_id, timepoint, ser_dir, mask_dir)
    return result, manifest_sha


def deterministic_pilot_selection(patient_ids: Iterable[str], *, maximum: int | None) -> list[str]:
    """Select a reproducible private pilot without emitting its participant keys."""

    ordered = sorted(set(patient_ids))
    if maximum is None:
        return ordered
    if maximum < 1 or maximum > len(ordered):
        raise ROIConversionError("invalid_pilot_patient_count")
    ranked = sorted(
        ordered,
        key=lambda item: hashlib.sha256(PILOT_SELECTION_DOMAIN + item.encode("utf-8")).digest(),
    )
    return ranked[:maximum]


def _header_paths(
    headers: Sequence[Any],
    *,
    ser_dir: Path,
    dicom_root: Path,
) -> list[Path]:
    """Recover the strict header-to-file correspondence without relying on names."""

    result: list[Path] = []
    seen: set[Path] = set()
    for header in headers:
        raw_path = getattr(header, "filename", None)
        if not isinstance(raw_path, (str, os.PathLike)):
            raise ROIConversionError("ser_header_filename_unavailable")
        try:
            path = Path(raw_path).expanduser().resolve()
        except OSError as error:
            _raise_stale(error)
            raise ROIConversionError("ser_header_filename_unavailable") from None
        if path.parent != ser_dir or not _inside(path, dicom_root) or path in seen:
            raise ROIConversionError("ser_header_file_contract_mismatch")
        seen.add(path)
        result.append(path)
    if len(result) != len(headers):
        raise ROIConversionError("ser_header_file_contract_mismatch")
    return result


def _direction_matrix_from_ser(ser: Mapping[str, Any]) -> tuple[float, ...]:
    try:
        column_axis = tuple(float(value) for value in ser["orientation"][:3])
        row_axis = tuple(float(value) for value in ser["orientation"][3:])
        normal_axis = tuple(float(value) for value in ser["normal"])
    except (KeyError, TypeError, ValueError) as error:
        raise ROIConversionError("invalid_ser_physical_geometry") from error
    if any(not math.isfinite(value) for value in (*column_axis, *row_axis, *normal_axis)):
        raise ROIConversionError("invalid_ser_physical_geometry")
    # SimpleITK stores a row-major direction matrix whose *columns* are the x/y/z
    # index axes.  DICOM IOP's first vector is the column-index axis and its second
    # vector is the row-index axis.
    return (
        column_axis[0], row_axis[0], normal_axis[0],
        column_axis[1], row_axis[1], normal_axis[1],
        column_axis[2], row_axis[2], normal_axis[2],
    )


def _series_digest_from_file_bytes(files: Sequence[Path]) -> tuple[str, list[bytes]]:
    """Hash and retain each DICOM byte stream exactly once before full parsing."""

    digest = hashlib.sha256(SERIES_DIGEST_DOMAIN)
    payloads: list[bytes] = []
    for path in files:
        try:
            payload = path.read_bytes()
        except OSError as error:
            _raise_stale(error)
            raise ROIConversionError("ser_pixel_file_read_error") from None
        if not payload:
            raise ROIConversionError("empty_ser_dicom_file")
        digest.update(struct.pack(">Q", len(payload)))
        digest.update(hashlib.sha256(payload).digest())
        payloads.append(payload)
    return digest.hexdigest(), payloads


def _rescale_pixel_array(
    value: Any,
    *,
    rows: int,
    columns: int,
    slope: float,
    intercept: float,
) -> Any:
    """Convert a stored 2-D DICOM pixel plane to its rescaled SER quantity."""

    np = _import_numpy()
    array = np.asarray(value)
    if array.shape != (rows, columns) or array.dtype.kind not in "biuf":
        raise ROIConversionError("ser_pixel_shape_or_type_invalid")
    if not np.isfinite(array).all():
        raise ROIConversionError("ser_stored_pixels_nonfinite")
    converted = array.astype(np.float32, copy=False) * np.float32(slope) + np.float32(intercept)
    if not np.isfinite(converted).all():
        raise ROIConversionError("ser_rescaled_pixels_nonfinite")
    return converted


def read_rescaled_ser_volume(
    ser_dir: Path,
    *,
    dicom_root: Path,
) -> SERVolume:
    """Read one strict SER series, apply DICOM rescale, and construct its image."""

    np, _pydicom, sitk = _import_runtime_dependencies()
    headers = anchor._strict_ser_headers(ser_dir, dicom_root=dicom_root)
    ser = anchor._ser_geometry_and_rescale(headers)
    files = _header_paths(headers, ser_dir=ser_dir, dicom_root=dicom_root)
    ordered_indices = tuple(int(value) for value in ser["order"])
    if sorted(ordered_indices) != list(range(len(headers))):
        raise ROIConversionError("invalid_ser_slice_order")
    ordered_files = [files[index] for index in ordered_indices]
    digest, payloads = _series_digest_from_file_bytes(ordered_files)
    rows = int(ser["rows"])
    columns = int(ser["columns"])
    slope = _finite_float(ser["slope"], code="invalid_ser_rescale")
    intercept = _finite_float(ser["intercept"], code="invalid_ser_rescale")
    if slope == 0.0:
        raise ROIConversionError("zero_ser_rescale_slope")
    planes: list[Any] = []
    for payload in payloads:
        try:
            dataset = anchor._quiet_dcmread(io.BytesIO(payload), force=False)
            pixels = dataset.pixel_array
        except Exception:
            raise ROIConversionError("ser_pixel_decode_error") from None
        planes.append(
            _rescale_pixel_array(
                pixels,
                rows=rows,
                columns=columns,
                slope=slope,
                intercept=intercept,
            )
        )
    try:
        array = np.stack(planes, axis=0).astype(np.float32, copy=False)
    except ValueError as error:
        raise ROIConversionError("ser_stack_assembly_error") from error
    if array.shape != (len(ordered_files), rows, columns) or not np.isfinite(array).all():
        raise ROIConversionError("ser_stack_assembly_error")
    try:
        image = sitk.GetImageFromArray(array, isVector=False)
        # SER geometry stores DICOM PixelSpacing as (row, column); SimpleITK uses
        # x/y/z = (column, row, slice) for the array [z,y,x].
        image.SetSpacing((
            float(ser["spacing"][1]),
            float(ser["spacing"][0]),
            float(ser["slice_spacing_mm"]),
        ))
        first_position = tuple(float(value) for value in ser["positions"][ordered_indices[0]])
        image.SetOrigin(first_position)
        image.SetDirection(_direction_matrix_from_ser(ser))
    except Exception as error:
        raise ROIConversionError("ser_sitk_image_construction_failed") from error
    return SERVolume(
        image=image,
        ser_geometry=ser,
        source_dicom_content_sha256=digest,
        rescale_slope=slope,
        rescale_intercept=intercept,
    )


def inverse_mask_bbox_center(
    pixels: Any,
    *,
    frame_to_ser: Sequence[int],
    ser: Mapping[str, Any],
) -> tuple[float, float, float]:
    """Return the private T0 raw-zero bbox midpoint using the audited rule."""

    np = _import_numpy()
    if pixels.ndim != 3 or pixels.shape[0] != len(frame_to_ser):
        raise ROIConversionError("unexpected_seg_pixel_shape")
    included = pixels == 0
    if not bool(np.any(included)):
        raise ROIConversionError("empty_t0_inverse_mask")
    try:
        order = tuple(int(value) for value in ser["order"])
        rank = {ser_index: index for index, ser_index in enumerate(order)}
        column_axis = tuple(float(value) for value in ser["orientation"][:3])
        row_axis = tuple(float(value) for value in ser["orientation"][3:])
        normal_axis = tuple(float(value) for value in ser["normal"])
        spacing = tuple(float(value) for value in ser["spacing"])
        slice_spacing = float(ser["slice_spacing_mm"])
        ordered_origin = tuple(float(value) for value in ser["positions"][order[0]])
    except (KeyError, IndexError, TypeError, ValueError) as error:
        raise ROIConversionError("invalid_t0_anchor_geometry") from error
    frame_indices = np.flatnonzero(np.any(included, axis=(1, 2)))
    min_row, max_row = pixels.shape[1], -1
    min_column, max_column = pixels.shape[2], -1
    ranks: list[int] = []
    for frame_index in frame_indices.tolist():
        rows, columns = np.nonzero(included[frame_index])
        min_row = min(min_row, int(rows.min()))
        max_row = max(max_row, int(rows.max()))
        min_column = min(min_column, int(columns.min()))
        max_column = max(max_column, int(columns.max()))
        try:
            ranks.append(rank[int(frame_to_ser[frame_index])])
        except (KeyError, IndexError, ValueError):
            raise ROIConversionError("invalid_seg_ser_frame_mapping") from None
    if not ranks or min_row > max_row or min_column > max_column:
        raise ROIConversionError("empty_t0_inverse_mask")
    center_column = (min_column + max_column) / 2.0
    center_row = (min_row + max_row) / 2.0
    center_rank = (min(ranks) + max(ranks)) / 2.0
    center = tuple(
        ordered_origin[axis]
        + center_column * spacing[1] * column_axis[axis]
        + center_row * spacing[0] * row_axis[axis]
        + center_rank * slice_spacing * normal_axis[axis]
        for axis in range(3)
    )
    if any(not math.isfinite(value) for value in center):
        raise ROIConversionError("invalid_t0_anchor_geometry")
    return center


def canonical_ras_grid_geometry(
    center_lps_mm: Sequence[float],
    *,
    shape_dhw: Sequence[int],
    spacing_dhw_mm: Sequence[float],
) -> dict[str, tuple[float, ...] | tuple[int, ...]]:
    """Return a center-preserving even-grid geometry without importing SimpleITK.

    The tensor convention is [D,H,W], whereas DICOM/SimpleITK physical axes are
    [X,Y,Z].  For an even 112 cube, the *midpoint between the two central voxel
    centers* equals the T0 mask-bbox center.  Its physical edges are exactly
    ``center +/- shape*spacing/2``, matching the anchor-audit FOV convention.
    """

    center = tuple(_finite_float(value, code="invalid_t0_anchor_geometry") for value in center_lps_mm)
    if len(center) != 3:
        raise ROIConversionError("invalid_t0_anchor_geometry")
    shape = _strict_triplet(shape_dhw, integer=True, code="roi_grid_geometry_invalid")
    spacing_dhw = _strict_triplet(spacing_dhw_mm, integer=False, code="roi_grid_geometry_invalid")
    depth, height, width = (int(value) for value in shape)
    spacing_z, spacing_y, spacing_x = (float(value) for value in spacing_dhw)
    size_xyz = (width, height, depth)
    spacing_xyz = (spacing_x, spacing_y, spacing_z)
    # RAS anatomical axis order expressed in DICOM/SimpleITK LPS physical space.
    direction = (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)
    half_indices = tuple((size - 1) / 2.0 for size in size_xyz)
    # origin + direction @ (half_indices * spacing) = center.
    origin = (
        center[0] + half_indices[0] * spacing_xyz[0],
        center[1] + half_indices[1] * spacing_xyz[1],
        center[2] - half_indices[2] * spacing_xyz[2],
    )
    return {
        "origin_lps_mm": origin,
        "direction_lps": direction,
        "size_xyz": size_xyz,
        "spacing_xyz_mm": spacing_xyz,
        "shape_dhw": (depth, height, width),
        "spacing_dhw_mm": (spacing_z, spacing_y, spacing_x),
    }


def make_canonical_ras_reference(
    center_lps_mm: Sequence[float],
    *,
    shape_dhw: Sequence[int],
    spacing_dhw_mm: Sequence[float],
) -> Any:
    """Instantiate the fixed RAS reference image in DICOM/SimpleITK physical space."""

    _np, _pydicom, sitk = _import_runtime_dependencies()
    geometry = canonical_ras_grid_geometry(
        center_lps_mm,
        shape_dhw=shape_dhw,
        spacing_dhw_mm=spacing_dhw_mm,
    )
    try:
        reference = sitk.Image(list(geometry["size_xyz"]), sitk.sitkFloat32)
        reference.SetSpacing(geometry["spacing_xyz_mm"])
        reference.SetOrigin(geometry["origin_lps_mm"])
        reference.SetDirection(geometry["direction_lps"])
    except Exception as error:
        raise ROIConversionError("canonical_ras_grid_creation_failed") from error
    return reference


def _identity_rigid_transform() -> Any:
    _np, _pydicom, sitk = _import_runtime_dependencies()
    try:
        return sitk.Euler3DTransform()
    except Exception as error:
        raise ROIConversionError("identity_rigid_construction_failed") from error


def _transform_affine_components(transform: Any) -> tuple[Any, Any]:
    """Return a finite affine matrix/offset and reject non-rigid transforms."""

    np = _import_numpy()
    try:
        origin = np.asarray(transform.TransformPoint((0.0, 0.0, 0.0)), dtype=np.float64)
        columns = [
            np.asarray(transform.TransformPoint(tuple(1.0 if axis == column else 0.0 for axis in range(3))), dtype=np.float64)
            - origin
            for column in range(3)
        ]
        matrix = np.column_stack(columns)
    except Exception as error:
        raise ROIConversionError("registration_transform_not_affine") from error
    if not np.isfinite(matrix).all() or not np.isfinite(origin).all():
        raise ROIConversionError("registration_transform_nonfinite")
    if not np.allclose(matrix.T @ matrix, np.eye(3), atol=1e-4, rtol=1e-4) or not math.isclose(
        float(np.linalg.det(matrix)), 1.0, abs_tol=1e-4, rel_tol=1e-4
    ):
        raise ROIConversionError("registration_transform_not_rigid")
    return matrix, origin


def _image_physical_center(image: Any) -> tuple[float, float, float]:
    """Return the physical midpoint of an image's voxel-center support."""

    try:
        size = tuple(int(value) for value in image.GetSize())
        if len(size) != 3 or any(value < 1 for value in size):
            raise ValueError("invalid image size")
        center_index = tuple((value - 1) / 2.0 for value in size)
        center = tuple(
            float(value)
            for value in image.TransformContinuousIndexToPhysicalPoint(center_index)
        )
    except Exception as error:
        raise ROIConversionError("rigid_registration_transform_invalid") from error
    if len(center) != 3 or any(not math.isfinite(value) for value in center):
        raise ROIConversionError("rigid_registration_transform_invalid")
    return center


def rigid_register_to_t0(fixed: Any, moving: Any) -> tuple[Any, dict[str, Any]]:
    """Perform deterministic image-only multi-resolution rigid registration.

    There is intentionally no identity/affine/deformable fallback.  A failed optimizer
    invalidates the visit and therefore prevents publishing a mixed ROI release.
    """

    np, _pydicom, sitk = _import_runtime_dependencies()
    try:
        initial = sitk.CenteredTransformInitializer(
            fixed,
            moving,
            sitk.Euler3DTransform(),
            sitk.CenteredTransformInitializerFilter.GEOMETRY,
        )
        method = sitk.ImageRegistrationMethod()
        method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        method.SetMetricSamplingStrategy(method.REGULAR)
        # SimpleITK otherwise defaults to a wall-clock seed, even for REGULAR
        # sampling.  A governed ROI release must be reproducible across reruns and
        # independent of worker scheduling.
        method.SetMetricSamplingPercentage(0.20, REGISTRATION_SAMPLING_SEED)
        method.SetInterpolator(sitk.sitkLinear)
        method.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0,
            minStep=1e-4,
            numberOfIterations=200,
            relaxationFactor=0.5,
            gradientMagnitudeTolerance=1e-8,
        )
        method.SetOptimizerScalesFromPhysicalShift()
        method.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
        method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
        method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
        method.SetInitialTransform(initial, inPlace=False)
        transform = method.Execute(fixed, moving)
        metric = float(method.GetMetricValue())
        iterations = int(method.GetOptimizerIteration())
        stop = str(method.GetOptimizerStopConditionDescription())
    except Exception as error:
        raise ROIConversionError("rigid_registration_failed") from error
    matrix, offset = _transform_affine_components(transform)
    try:
        physical_center = _image_physical_center(fixed)
        moved_center = np.asarray(transform.TransformPoint(physical_center), dtype=np.float64)
    except Exception as error:
        if isinstance(error, ROIConversionError):
            raise
        raise ROIConversionError("rigid_registration_transform_invalid") from error
    displacement = float(np.linalg.norm(moved_center - np.asarray(physical_center)))
    if (
        not math.isfinite(metric)
        or iterations < 0
        or not stop
        or not math.isfinite(displacement)
        or displacement > MAX_RIGID_CENTER_DISPLACEMENT_MM
    ):
        raise ROIConversionError("rigid_registration_qc_failed")
    return transform, {
        "registration_status": "rigid",
        "metric_value": metric,
        "sampling_seed": REGISTRATION_SAMPLING_SEED,
        "iterations": iterations,
        "center_displacement_mm": displacement,
        # No patient-specific parameters belong in ROI spatial_provenance.  They go
        # only to the governed lineage file written below.
        "transform_matrix": matrix.reshape(-1).tolist(),
        "transform_offset_mm": offset.tolist(),
        "stop_condition": stop,
    }


def resample_ser_to_reference(source: Any, reference: Any, transform: Any) -> Any:
    """Resample a *rescaled* SER image onto the fixed grid with post-rescale zero pad."""

    _np, _pydicom, sitk = _import_runtime_dependencies()
    try:
        return sitk.Resample(
            source,
            reference,
            transform,
            sitk.sitkLinear,
            0.0,
            sitk.sitkFloat32,
        )
    except Exception as error:
        raise ROIConversionError("ser_resampling_failed") from error


def validate_resampled_ser(image: Any, *, shape_dhw: Sequence[int]) -> Any:
    """Return [1,D,H,W] float32 only if the post-rescale ROI is cache-compatible."""

    np, _pydicom, sitk = _import_runtime_dependencies()
    try:
        array = sitk.GetArrayFromImage(image).astype(np.float32, copy=False)
    except Exception as error:
        raise ROIConversionError("resampled_ser_array_unavailable") from error
    shape = tuple(int(value) for value in shape_dhw)
    if array.shape != shape or not np.isfinite(array).all():
        raise ROIConversionError("resampled_ser_shape_or_values_invalid")
    observed_minimum = float(array.min())
    observed_maximum = float(array.max())
    if (
        observed_minimum < SER_INPUT_RANGE[0] - SER_RANGE_TOLERANCE
        or observed_maximum > SER_INPUT_RANGE[1] + SER_RANGE_TOLERANCE
    ):
        raise ROIConversionError("resampled_ser_outside_fixed_ratio_range")
    if observed_maximum - observed_minimum <= SER_RANGE_TOLERANCE:
        raise ROIConversionError("resampled_ser_constant")
    try:
        import torch
    except ImportError as error:  # pragma: no cover - package hard dependency
        raise ROIConversionError("torch_dependency_unavailable") from error
    return torch.from_numpy(array.copy()).unsqueeze(0).contiguous()


def _validate_seg_common(seg: Any, *, expected_frame_of_reference: str) -> tuple[list[Any], Any, Any]:
    """Validate enough SEG geometry for direct physical support coverage QC."""

    np = _import_numpy()
    if (
        str(getattr(seg, "SOPClassUID", "")) != SEGMENTATION_STORAGE_UID
        or str(getattr(seg, "Modality", "")).strip().upper() != "SEG"
    ):
        raise ROIConversionError("selected_object_is_not_seg")
    if str(getattr(seg, "FrameOfReferenceUID", "")).strip() != expected_frame_of_reference:
        raise ROIConversionError("future_mask_frame_of_reference_mismatch")
    try:
        frames_count = int(seg.NumberOfFrames)
        rows = int(seg.Rows)
        columns = int(seg.Columns)
    except (AttributeError, TypeError, ValueError) as error:
        raise ROIConversionError("invalid_seg_matrix_or_frames") from error
    if frames_count < 1 or rows < 1 or columns < 1:
        raise ROIConversionError("invalid_seg_matrix_or_frames")
    frames = _sequence(seg, "PerFrameFunctionalGroupsSequence")
    shared_items = _sequence(seg, "SharedFunctionalGroupsSequence")
    if len(frames) != frames_count or len(shared_items) != 1:
        raise ROIConversionError("seg_functional_group_count_mismatch")
    try:
        declared_numbers = {int(getattr(item, "SegmentNumber", None)) for item in _sequence(seg, "SegmentSequence")}
    except (TypeError, ValueError):
        raise ROIConversionError("seg_segment_contract_mismatch") from None
    frame_numbers = [_frame_segment_numbers(frame) for frame in frames]
    if (
        len(declared_numbers) != 1
        or any(len(numbers) != 1 or numbers != declared_numbers for numbers in frame_numbers)
    ):
        raise ROIConversionError("seg_segment_contract_mismatch")
    try:
        fractional_value = int(getattr(seg, "MaximumFractionalValue", 0))
    except (TypeError, ValueError):
        raise ROIConversionError("unexpected_ispy2_analysis_mask_declaration") from None
    if (
        str(getattr(seg, "SegmentationType", "")).strip().upper() != "FRACTIONAL"
        or str(getattr(seg, "SegmentationFractionalType", "")).strip().upper() != "OCCUPANCY"
        or fractional_value != 255
    ):
        raise ROIConversionError("unexpected_ispy2_analysis_mask_declaration")
    try:
        pixels = seg.pixel_array
    except Exception:
        raise ROIConversionError("seg_pixel_decode_error") from None
    if (
        tuple(pixels.shape) != (frames_count, rows, columns)
        or pixels.dtype.kind not in "biu"
        or not np.isfinite(pixels).all()
        or int(pixels.min()) < 0
        or int(pixels.max()) > 255
    ):
        raise ROIConversionError("unexpected_seg_pixel_shape")
    return frames, shared_items[0], pixels


def _future_mask_geometry(seg: Any, *, expected_frame_of_reference: str) -> dict[str, Any]:
    """Decode inverse-mask plane geometry without assuming SER-exact matrices."""

    np = _import_numpy()
    frames, shared, pixels = _validate_seg_common(seg, expected_frame_of_reference=expected_frame_of_reference)
    spacings = anchor._resolved_frame_vectors(
        shared,
        frames,
        sequence_name="PixelMeasuresSequence",
        attribute="PixelSpacing",
        length=2,
    )
    orientations = anchor._resolved_frame_vectors(
        shared,
        frames,
        sequence_name="PlaneOrientationSequence",
        attribute="ImageOrientationPatient",
        length=6,
    )
    positions = [_frame_position(frame) for frame in frames]
    if any(position is None for position in positions):
        raise ROIConversionError("missing_seg_frame_position")
    concrete_positions = [tuple(float(value) for value in position) for position in positions if position is not None]
    if len(set(orientations)) != 1:
        raise ROIConversionError("nonuniform_future_seg_orientation")
    # We deliberately do not require the segmentation matrix/frame count to match
    # SER here.  That was exactly the known 1+3 exception set.  Direct physical
    # support assessment is explicit in lineage as `physical_seg_coverage_qc`.
    included = pixels == 0
    # A post-treatment visit may legitimately have zero FTV (no raw-zero
    # enhancing voxels).  Preserve that as an explicit empty support; the
    # caller cross-checks the zero count against the governed FTV release.
    # Empty future support has no spatial extent, so its crop coverage and
    # edge margin are not applicable rather than failed or fabricated.
    return {
        "frames": frames,
        "pixels": pixels,
        "included": included,
        "spacings": spacings,
        "orientations": orientations,
        "positions": concrete_positions,
        "included_voxels": int(np.count_nonzero(included)),
    }


def _output_grid_components(reference: Any) -> tuple[Any, Any, Any, Any]:
    np = _import_numpy()
    try:
        direction = np.asarray(reference.GetDirection(), dtype=np.float64).reshape(3, 3)
        origin = np.asarray(reference.GetOrigin(), dtype=np.float64)
        spacing = np.asarray(reference.GetSpacing(), dtype=np.float64)
        size = np.asarray(reference.GetSize(), dtype=np.float64)
    except Exception as error:
        raise ROIConversionError("reference_grid_geometry_invalid") from error
    if (
        not np.isfinite(direction).all()
        or not np.isfinite(origin).all()
        or not np.isfinite(spacing).all()
        or not np.isfinite(size).all()
        or np.any(spacing <= 0)
        or np.any(size < 1)
    ):
        raise ROIConversionError("reference_grid_geometry_invalid")
    return direction, origin, spacing, size


def future_mask_coverage(
    support: Mapping[str, Any],
    *,
    moving_to_fixed: Any,
    reference: Any,
) -> dict[str, Any]:
    """Check raw-zero future support after crop and registration are already fixed.

    It assesses physical voxel support directly, rather than resampling the mask into
    the crop.  This is at least as strict as nearest-neighbour mask coverage and is
    valid for the four non-exact SEG/SER pairs because it relies on their own DICOM
    Frame-of-Reference geometry rather than fabricated slice correspondences.
    """

    np = _import_numpy()
    matrix, offset = _transform_affine_components(moving_to_fixed)
    direction, origin, output_spacing, output_size = _output_grid_components(reference)
    included = support["included"]
    positions = support["positions"]
    spacings = support["spacings"]
    orientations = support["orientations"]
    if len(positions) != included.shape[0] or len(spacings) != included.shape[0] or len(orientations) != included.shape[0]:
        raise ROIConversionError("future_mask_geometry_cardinality_mismatch")
    if not bool(np.any(included)):
        return {
            "inside_fixed_crop": None,
            "minimum_edge_margin_mm": None,
            "included_voxels": 0,
        }
    lower = np.full(3, math.inf, dtype=np.float64)
    upper = np.full(3, -math.inf, dtype=np.float64)
    total = 0
    for frame_index in range(included.shape[0]):
        rows, columns = np.nonzero(included[frame_index])
        if rows.size == 0:
            continue
        try:
            position = np.asarray(positions[frame_index], dtype=np.float64)
            row_spacing, column_spacing = (float(value) for value in spacings[frame_index])
            column_axis = np.asarray(orientations[frame_index][:3], dtype=np.float64)
            row_axis = np.asarray(orientations[frame_index][3:], dtype=np.float64)
        except (TypeError, ValueError, IndexError) as error:
            raise ROIConversionError("future_mask_geometry_invalid") from error
        if (
            not np.isfinite(position).all()
            or not np.isfinite(column_axis).all()
            or not np.isfinite(row_axis).all()
            or row_spacing <= 0.0
            or column_spacing <= 0.0
        ):
            raise ROIConversionError("future_mask_geometry_invalid")
        physical = (
            position[:, None]
            + column_axis[:, None] * (columns.astype(np.float64) * column_spacing)
            + row_axis[:, None] * (rows.astype(np.float64) * row_spacing)
        )
        fixed = matrix @ physical + offset[:, None]
        continuous = direction.T @ (fixed - origin[:, None]) / output_spacing[:, None]
        # Add half a source-pixel extent in the two in-plane source directions before
        # evaluating output grid edges.  SEG storage does not always declare an
        # independent through-plane thickness, so we do not invent one.
        basis = np.column_stack((column_axis * column_spacing * 0.5, row_axis * row_spacing * 0.5))
        half_index = np.sum(np.abs(direction.T @ matrix @ basis), axis=1) / output_spacing
        lower = np.minimum(lower, np.min(continuous - half_index[:, None], axis=1))
        upper = np.maximum(upper, np.max(continuous + half_index[:, None], axis=1))
        total += int(rows.size)
    if total < 1 or not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise ROIConversionError("future_mask_coverage_invalid")
    tolerance_indices = FUTURE_MASK_EDGE_TOLERANCE_MM / output_spacing
    inside = bool(
        np.all(lower >= -0.5 - tolerance_indices)
        and np.all(upper <= output_size - 0.5 + tolerance_indices)
    )
    margin = min(
        *(((lower + 0.5) * output_spacing).tolist()),
        *(((output_size - 0.5 - upper) * output_spacing).tolist()),
    )
    return {
        "inside_fixed_crop": inside,
        "minimum_edge_margin_mm": float(margin),
        "included_voxels": total,
    }


def _future_mask_status(
    seg: Any,
    *,
    ser: Mapping[str, Any],
    moving_to_fixed: Any,
    reference: Any,
) -> dict[str, Any]:
    """Get the explicit exact/non-exact mask path and fixed-grid coverage result."""

    try:
        anchor._exact_seg_to_ser_mapping(seg, ser)
    except anchor.ROIAnchorAuditError:
        mapping_status = "physical_seg_coverage_qc"
    else:
        mapping_status = "geometry_exact"
    support = _future_mask_geometry(
        seg,
        expected_frame_of_reference=str(ser["frame_of_reference"]),
    )
    coverage = future_mask_coverage(
        support,
        moving_to_fixed=moving_to_fixed,
        reference=reference,
    )
    if int(coverage["included_voxels"]) == 0:
        mapping_status = f"{mapping_status}_empty_support"
    return {"mapping_status": mapping_status, **coverage}


def _payload(
    volume: Any,
    *,
    spatial_provenance: Mapping[str, Any],
    ser_volume: SERVolume,
    converter_source_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": ROI_SCHEMA_VERSION,
        "modality": "SER",
        "tensor_kind": "volume",
        "volume": volume,
        "spatial_provenance": dict(spatial_provenance),
        "intensity_provenance": {
            "quantity": "signal_enhancement_ratio",
            "units": "unitless_ratio",
            "dicom_rescale_applied": True,
            "dicom_rescale_slope": float(ser_volume.rescale_slope),
            "dicom_rescale_intercept": float(ser_volume.rescale_intercept),
            "value_state": "dicom_rescaled_ser",
            "prior_intensity_transform": "none",
            "converter_version": CONVERTER_VERSION,
            "converter_source_sha256": converter_source_sha256,
            "source_dicom_content_sha256": ser_volume.source_dicom_content_sha256,
        },
    }


def _scan_id(source_dicom_content_sha256: str) -> str:
    return hashlib.sha256(SCAN_ID_DOMAIN + source_dicom_content_sha256.encode("ascii")).hexdigest()


def _opaque_roi_name(source_dicom_content_sha256: str) -> str:
    return hashlib.sha256(ROI_NAME_DOMAIN + source_dicom_content_sha256.encode("ascii")).hexdigest()[:32] + ".pt"


def _write_torch_exclusive(value: Any, destination: Path) -> None:
    """Write a private tensor atomically without replacing an existing artifact."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - project dependency
        raise ROIConversionError("torch_dependency_unavailable") from error
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(destination.parent, 0o700)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(handle.name, 0o600)
            torch.save(value, handle)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, destination)
        except FileExistsError:
            raise ROIConversionError("roi_output_exists_refusing_overwrite") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _write_bytes_exclusive(path: Path, content: bytes) -> None:
    try:
        anchor._write_exclusive(path, content)
    except anchor.ROIAnchorAuditError as error:
        raise ROIConversionError(error.code) from None


def _output_path_within_governed(
    output_root: str | Path,
    *,
    governed_root: str | Path,
    dataset_root: str | Path,
) -> Path:
    root = Path(governed_root).expanduser().resolve()
    dataset = Path(dataset_root).expanduser().resolve()
    try:
        if not root.is_dir():
            raise ROIConversionError("governed_root_unavailable")
        candidate = Path(output_root).expanduser().resolve(strict=False)
        candidate.relative_to(root)
    except (OSError, ValueError) as error:
        if isinstance(error, ROIConversionError):
            raise
        raise ROIConversionError("roi_output_outside_governed_root") from error
    if candidate == root or _inside(candidate, dataset):
        raise ROIConversionError("roi_output_outside_governed_root")
    if candidate.exists():
        raise ROIConversionError("roi_output_exists_refusing_overwrite")
    return candidate


def _ftv_counts(ftv_release: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    rows = ftv_release.get("ftv_csv_rows")
    if not isinstance(rows, list):
        raise ROIConversionError("governed_ftv_release_invalid")
    result: dict[str, dict[str, int]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ROIConversionError("governed_ftv_release_invalid")
        patient = str(row.get("PatientID") or "").strip()
        if not production._PATIENT_ID.fullmatch(patient) or patient in result:
            raise ROIConversionError("governed_ftv_release_invalid")
        values: dict[str, int] = {}
        for timepoint in TIMEPOINTS:
            try:
                count = int(str(row.get(f"included_voxels_{timepoint}") or ""))
            except ValueError as error:
                raise ROIConversionError("governed_ftv_release_invalid") from error
            # Zero is a valid post-treatment FTV state.  The governed FTV
            # release itself accepts nonnegative counts, so downstream
            # conversion must not silently tighten that contract.
            if count < 0:
                raise ROIConversionError("governed_ftv_release_invalid")
            values[timepoint] = count
        result[patient] = values
    return result


def _read_seg_for_visit(visit: VisitSource, *, dicom_root: Path) -> Any:
    try:
        return anchor._read_seg(visit.mask_dir, dicom_root=dicom_root)
    except anchor.ROIAnchorAuditError as error:
        raise ROIConversionError(error.code) from None


def _convert_patient(
    visits: Mapping[str, VisitSource],
    *,
    dicom_root: Path,
    ftv_counts: Mapping[str, int],
    shape_dhw: Sequence[int],
    spacing_dhw_mm: Sequence[float],
    spatial_provenance: Mapping[str, Any],
    converter_source_sha256: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[float]]:
    """Convert all three visits in memory.  Nothing is published until all pass."""

    t0_source = visits["T0"]
    t0_ser = read_rescaled_ser_volume(t0_source.ser_dir, dicom_root=dicom_root)
    t0_seg = _read_seg_for_visit(t0_source, dicom_root=dicom_root)
    try:
        t0_mapping, t0_pixels = anchor._exact_seg_to_ser_mapping(t0_seg, t0_ser.ser_geometry)
    except anchor.ROIAnchorAuditError as error:
        raise ROIConversionError(error.code) from None
    t0_count = int((t0_pixels == 0).sum())
    if t0_count != int(ftv_counts["T0"]):
        raise ROIConversionError("t0_inverse_mask_ftv_count_mismatch")
    center = inverse_mask_bbox_center(
        t0_pixels,
        frame_to_ser=t0_mapping,
        ser=t0_ser.ser_geometry,
    )
    reference = make_canonical_ras_reference(
        center,
        shape_dhw=shape_dhw,
        spacing_dhw_mm=spacing_dhw_mm,
    )
    prepared: list[dict[str, Any]] = []
    lineage: list[dict[str, Any]] = []
    metrics: list[float] = []
    margins: list[float] = []
    for timepoint in TIMEPOINTS:
        visit = visits[timepoint]
        if timepoint == "T0":
            ser_volume = t0_ser
            transform = _identity_rigid_transform()
            registration = {
                "registration_status": "identity_rigid",
                "metric_value": None,
                "sampling_seed": None,
                "iterations": 0,
                "center_displacement_mm": 0.0,
                "transform_matrix": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                "transform_offset_mm": [0.0, 0.0, 0.0],
                "stop_condition": "identity_t0_anchor",
            }
            future = {
                "mapping_status": "geometry_exact_t0_anchor",
                "inside_fixed_crop": True,
                "minimum_edge_margin_mm": EXPECTED_MARGIN_MM,
                "included_voxels": t0_count,
            }
        else:
            ser_volume = read_rescaled_ser_volume(visit.ser_dir, dicom_root=dicom_root)
            transform, registration = rigid_register_to_t0(t0_ser.image, ser_volume.image)
            seg = _read_seg_for_visit(visit, dicom_root=dicom_root)
            try:
                moving_to_fixed = transform.GetInverse()
            except Exception as error:
                raise ROIConversionError("rigid_registration_inverse_unavailable") from error
            future = _future_mask_status(
                seg,
                ser=ser_volume.ser_geometry,
                moving_to_fixed=moving_to_fixed,
                reference=reference,
            )
            if int(future["included_voxels"]) != int(ftv_counts[timepoint]):
                raise ROIConversionError("future_inverse_mask_ftv_count_mismatch")
            if int(future["included_voxels"]) > 0 and future["inside_fixed_crop"] is not True:
                raise ROIConversionError(
                    "future_mask_outside_fixed_crop",
                    metadata={
                        "timepoint": timepoint,
                        "future_mask_mapping_status": future["mapping_status"],
                        "future_mask_minimum_edge_margin_mm": future["minimum_edge_margin_mm"],
                        "future_mask_included_voxels": future["included_voxels"],
                    },
                )
            if int(future["included_voxels"]) == 0 and (
                future["inside_fixed_crop"] is not None
                or future["minimum_edge_margin_mm"] is not None
            ):
                raise ROIConversionError("empty_future_mask_coverage_invalid")
            metrics.append(float(registration["metric_value"]))
            if future["minimum_edge_margin_mm"] is not None:
                margins.append(float(future["minimum_edge_margin_mm"]))
        resampled = resample_ser_to_reference(ser_volume.image, reference, transform)
        volume = validate_resampled_ser(resampled, shape_dhw=shape_dhw)
        payload = _payload(
            volume,
            spatial_provenance=spatial_provenance,
            ser_volume=ser_volume,
            converter_source_sha256=converter_source_sha256,
        )
        scan_id = _scan_id(ser_volume.source_dicom_content_sha256)
        prepared.append(
            {
                "timepoint": timepoint,
                "scan_id": scan_id,
                "filename": _opaque_roi_name(ser_volume.source_dicom_content_sha256),
                "payload": payload,
            }
        )
        lineage.append(
            {
                "timepoint": timepoint,
                "scan_id": scan_id,
                "source_dicom_content_sha256": ser_volume.source_dicom_content_sha256,
                "registration_status": registration["registration_status"],
                "registration_metric_value": registration["metric_value"],
                "registration_sampling_seed": registration["sampling_seed"],
                "registration_iterations": registration["iterations"],
                "registration_center_displacement_mm": registration["center_displacement_mm"],
                "registration_transform_matrix": json.dumps(registration["transform_matrix"], separators=(",", ":")),
                "registration_transform_offset_mm": json.dumps(registration["transform_offset_mm"], separators=(",", ":")),
                "registration_stop_condition": registration["stop_condition"],
                "future_mask_mapping_status": future["mapping_status"],
                # None is preserved for a legitimate zero-FTV future visit: an
                # empty support has no spatial extent to classify as inside/outside.
                "future_mask_inside_fixed_crop": future["inside_fixed_crop"],
                "future_mask_minimum_edge_margin_mm": future["minimum_edge_margin_mm"],
                "future_mask_included_voxels": future["included_voxels"],
            }
        )
    return prepared, lineage, metrics, margins


def _write_patient_outputs(
    staging: Path,
    *,
    patient_id: str,
    prepared: Sequence[Mapping[str, Any]],
    lineage: Sequence[Mapping[str, Any]],
    index_rows: list[dict[str, str]],
    lineage_rows: list[dict[str, str]],
) -> None:
    tensors = staging / "tensors"
    for payload_item, lineage_item in zip(prepared, lineage, strict=True):
        destination = tensors / str(payload_item["filename"])
        _write_torch_exclusive(payload_item["payload"], destination)
        tensor_hash = sha256_file(destination)
        index_rows.append(
            {
                "PatientID": patient_id,
                "tp_label": str(payload_item["timepoint"]),
                "scan_id": str(payload_item["scan_id"]),
                "roi_path": str(Path("tensors") / destination.name),
                # The cache contract deliberately uses this as the serialized ROI
                # content hash, not the source DICOM hash in the payload.
                "source_content_sha256": tensor_hash,
            }
        )
        lineage_rows.append(
            _lineage_csv_row(
                patient_id=patient_id,
                payload_item=payload_item,
                lineage_item=lineage_item,
                tensor_hash=tensor_hash,
            )
        )


def _lineage_csv_row(
    *,
    patient_id: str,
    payload_item: Mapping[str, Any],
    lineage_item: Mapping[str, Any],
    tensor_hash: str,
) -> dict[str, str]:
    """Translate private in-memory lineage into the locked CSV schema."""

    timepoint = str(payload_item.get("timepoint") or "")
    if (
        timepoint not in TIMEPOINTS
        or str(lineage_item.get("timepoint") or "") != timepoint
        or str(lineage_item.get("scan_id") or "")
        != str(payload_item.get("scan_id") or "")
    ):
        raise ROIConversionError("conversion_lineage_identity_mismatch")
    row = {
        "PatientID": patient_id,
        "tp_label": timepoint,
        **{
            key: "" if value is None else str(value)
            for key, value in lineage_item.items()
            if key != "timepoint"
        },
        "roi_content_sha256": tensor_hash,
    }
    if set(row) != set(CONVERSION_LINEAGE_FIELDS):
        raise ROIConversionError("conversion_lineage_schema_mismatch")
    return row


def _write_csv_exclusive(path: Path, *, fieldnames: Sequence[str], rows: Sequence[Mapping[str, str]]) -> None:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    _write_bytes_exclusive(path, output.getvalue().encode("utf-8"))


def _qc_exclusion_rows(
    failures: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Build the private, release-local roster for approved coverage exclusions.

    Patient identifiers must never enter stdout or the aggregate report.  They are
    retained only inside the governed ROI release so every downstream consumer can
    reproduce the released cohort exactly.
    """

    grouped: dict[str, set[str]] = {}
    for row in failures:
        if str(row.get("code") or "") != "future_mask_outside_fixed_crop":
            continue
        patient_id = str(row.get("patient_id") or "")
        timepoint = str(row.get("timepoint") or "")
        if not production._PATIENT_ID.fullmatch(patient_id) or timepoint not in TIMEPOINTS:
            raise ROIConversionError("qc_exclusion_metadata_invalid")
        grouped.setdefault(patient_id, set()).add(timepoint)
    return [
        {
            "PatientID": patient_id,
            "exclusion_reason": "future_mask_outside_fixed_crop",
            "failed_timepoints": ",".join(sorted(timepoints)),
        }
        for patient_id, timepoints in sorted(grouped.items())
    ]


def _validate_staged_release(
    staging: Path,
    *,
    expected_patients: int,
    shape_dhw: Sequence[int],
) -> None:
    index = staging / "roi_index.csv"
    records = read_roi_index(
        index,
        allowed_input_root=staging,
        expected_patients=expected_patients,
    )
    if len(records) != expected_patients * len(TIMEPOINTS):
        raise ROIConversionError("roi_index_cardinality_mismatch")
    canonical_spatial: dict[str, Any] | None = None
    for record in records:
        if sha256_file(record.roi_path) != record.source_content_sha256:
            raise ROIConversionError("roi_index_tensor_hash_mismatch")
        _volume, spatial, _intensity = load_roi_payload(record.roi_path, shape_dhw)
        if canonical_spatial is None:
            canonical_spatial = spatial
        elif spatial != canonical_spatial:
            raise ROIConversionError("roi_spatial_provenance_inconsistent")


def _failure_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    outside = [
        row
        for row in rows
        if str(row.get("code") or "") == "future_mask_outside_fixed_crop"
    ]
    result: dict[str, Any] = {"metadata_rows": len(rows)}
    if outside:
        margins = [
            float(row["future_mask_minimum_edge_margin_mm"])
            for row in outside
            if row.get("future_mask_minimum_edge_margin_mm") is not None
        ]
        result["future_mask_outside_fixed_crop"] = {
            "count": len(outside),
            "timepoints": dict(sorted(Counter(str(row.get("timepoint") or "missing") for row in outside).items())),
            "mapping_statuses": dict(
                sorted(Counter(str(row.get("future_mask_mapping_status") or "missing") for row in outside).items())
            ),
            "minimum_edge_margin_mm": _numeric_summary(margins, allow_negative=True) if margins else {},
            "included_voxels": _numeric_summary(
                [
                    float(row["future_mask_included_voxels"])
                    for row in outside
                    if row.get("future_mask_included_voxels") is not None
                ],
                allow_negative=False,
            ),
        }
    return result


def _aggregate_report(
    *,
    status: str,
    selected_patients: int,
    all_patients: int,
    split_counts: Mapping[str, int],
    failure_reasons: Mapping[str, int],
    registration_metrics: Sequence[float],
    future_margins: Sequence[float],
    mapping_statuses: Mapping[str, int],
    failure_metadata: Sequence[Mapping[str, Any]],
    evidence: Mapping[str, str],
    output_promoted: bool,
    qc_exclusion_count: int = 0,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "aggregate_only": True,
        "pixel_data_read": True,
        "identifiers_paths_uids_dates_or_per_patient_measurements_emitted": False,
        "cohort": {
            "patients_total": all_patients,
            "patients_selected": selected_patients,
            "patients_excluded_qc": qc_exclusion_count,
            "patients_released": selected_patients - qc_exclusion_count if output_promoted else 0,
            "visits_expected": selected_patients * len(TIMEPOINTS),
            "split_counts_full_cohort": dict(split_counts),
        },
        "roi_contract": {
            "roi_schema_version": ROI_SCHEMA_VERSION,
            "roi_voxel_shape_dhw": list(EXPECTED_SHAPE_DHW),
            "spacing_dhw_mm": list(EXPECTED_SPACING_DHW_MM),
            "orientation": EXPECTED_ORIENTATION,
            "anchor_center_rule": ANCHOR_CENTER_RULE,
            "padding": PADDING_POLICY,
            "registration_type": "rigid",
            "registration_metric_sampling_strategy": "REGULAR",
            "registration_metric_sampling_percentage": 0.20,
            "registration_sampling_seed": REGISTRATION_SAMPLING_SEED,
            "future_mask_role": "coverage_qc_only_never_crop_centering",
            "empty_future_mask_policy": (
                "coverage_not_applicable_only_when_governed_ftv_count_is_zero"
            ),
        },
        "conversion": {
            "failure_reasons": dict(sorted((str(key), int(value)) for key, value in failure_reasons.items())),
            "failure_diagnostics": _failure_diagnostics(failure_metadata),
            "qc_exclusion_policy": (
                "future_mask_outside_fixed_crop_patient_exclusion_v1"
                if qc_exclusion_count
                else None
            ),
            "future_mask_mapping_statuses": dict(sorted((str(key), int(value)) for key, value in mapping_statuses.items())),
            "rigid_registration_metric": _numeric_summary(registration_metrics, allow_negative=True)
            if registration_metrics else {},
            "future_mask_minimum_edge_margin_mm": _numeric_summary(future_margins, allow_negative=True)
            if future_margins else {},
            "release_promoted": output_promoted,
        },
        "evidence": dict(evidence),
        "next_gate": {
            "can_cache_3dino": status.startswith("ok_roi_conversion_full") and output_promoted,
            "pilot_only": status == "ok_roi_conversion_pilot",
        },
    }


def run_roi_conversion(
    root: str | Path,
    *,
    trajectory_path: str | Path,
    ftv_csv_path: str | Path,
    ftv_audit_report_path: str | Path,
    roi_spec_path: str | Path,
    anchor_report_path: str | Path,
    governed_root: str | Path,
    output_root: str | Path,
    maximum_patients: int | None = None,
    allow_future_mask_qc_exclusions: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Convert a deterministic pilot or the full cohort with fail-closed release."""

    if workers < 1 or workers > 16:
        raise ROIConversionError("invalid_roi_conversion_worker_count")
    root_path = Path(root).expanduser().resolve()
    governed = Path(governed_root).expanduser().resolve()
    output = _output_path_within_governed(output_root, governed_root=governed, dataset_root=root_path)
    try:
        ftv_release = validate_ftv_release(ftv_csv_path, ftv_audit_report_path)
    except FTVReleaseError as error:
        raise ROIConversionError("governed_ftv_release_invalid") from error
    try:
        patient_ids, split_counts, split_sha = production._read_split(root_path, EXPECTED_PATIENTS)
    except production.DCEGeometryProbeError as error:
        raise ROIConversionError(error.code) from None
    visits, manifest_sha = read_pinned_complete_visits(
        root_path,
        trajectory_path,
        expected_patient_ids=patient_ids,
    )
    roi_spec = load_locked_roi_spec(roi_spec_path)
    anchor_sha = validate_anchor_release(
        anchor_report_path,
        roi_spec=roi_spec,
        manifest_sha256=manifest_sha,
        ftv_csv_sha256=str(ftv_release["ftv_csv_sha256"]),
        ftv_audit_report_sha256=str(ftv_release["ftv_audit_report_sha256"]),
    )
    counts = _ftv_counts(ftv_release)
    if set(counts) != patient_ids:
        raise ROIConversionError("governed_ftv_patient_cohort_mismatch")
    selected = deterministic_pilot_selection(visits, maximum=maximum_patients)
    dicom_root = (root_path / "dicom").resolve()
    converter_sha = _sha256_file(Path(__file__), code="converter_source_read_error")
    evidence = {
        "trajectory_manifest_sha256": manifest_sha,
        "split_sha256": split_sha,
        "ftv_csv_sha256": str(ftv_release["ftv_csv_sha256"]),
        "ftv_audit_report_sha256": str(ftv_release["ftv_audit_report_sha256"]),
        "roi_spec_sha256": str(roi_spec["sha256"]),
        "roi_anchor_report_sha256": anchor_sha,
        "converter_source_sha256": converter_sha,
    }
    failures: Counter[str] = Counter()
    mapping_statuses: Counter[str] = Counter()
    registration_metrics: list[float] = []
    future_margins: list[float] = []
    failure_metadata: list[dict[str, Any]] = []
    index_rows: list[dict[str, str]] = []
    lineage_rows: list[dict[str, str]] = []
    output.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(output.parent, 0o700)
    promoted = False
    qc_exclusions: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(prefix=f".{output.name}.", dir=output.parent) as temporary:
        staging = Path(temporary)
        os.chmod(staging, 0o700)
        def convert_one(patient_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[float], list[float]]:
            return _convert_patient(
                visits[patient_id],
                dicom_root=dicom_root,
                ftv_counts=counts[patient_id],
                shape_dhw=roi_spec["shape_dhw"],
                spacing_dhw_mm=roi_spec["spacing_dhw_mm"],
                spatial_provenance=roi_spec["spatial_provenance"],
                converter_source_sha256=converter_sha,
            )

        # The prior full conversion allocated four CPUs but processed patients one
        # at a time.  Keep writes deterministic and bounded (one completed batch at
        # a time), while running the independent DICOM read/rigid-registration work
        # concurrently.  The Slurm script pins ITK to one thread per patient.
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="ser-roi") as executor:
            for batch_start in range(0, len(selected), workers):
                batch = selected[batch_start : batch_start + workers]
                futures = [(patient_id, executor.submit(convert_one, patient_id)) for patient_id in batch]
                for patient_id, future in futures:
                    number = batch_start + batch.index(patient_id) + 1
                    try:
                        prepared, lineage, metrics, margins = future.result()
                        _write_patient_outputs(
                            staging,
                            patient_id=patient_id,
                            prepared=prepared,
                            lineage=lineage,
                            index_rows=index_rows,
                            lineage_rows=lineage_rows,
                        )
                        registration_metrics.extend(metrics)
                        future_margins.extend(margins)
                        mapping_statuses.update(str(row["future_mask_mapping_status"]) for row in lineage if row["timepoint"] != "T0")
                    except StaleFileHandleError:
                        raise
                    except ROIConversionError as error:
                        failures[error.code] += 1
                        if error.metadata:
                            failure_metadata.append(
                                {"code": error.code, "patient_id": patient_id, **error.metadata}
                            )
                    except OSError as error:
                        _raise_stale(error)
                        failures["roi_conversion_io_error"] += 1
                    except Exception:
                        failures["unexpected_roi_conversion_error"] += 1
                    if number % 5 == 0 or number == len(selected):
                        print(
                            json.dumps(
                                {
                                    "event": "ser_roi_conversion_progress",
                                    "patients_complete": number,
                                    "patients_total": len(selected),
                                    "pixel_data_read": True,
                                    "workers": workers,
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
        qc_exclusions = _qc_exclusion_rows(failure_metadata)
        only_approved_qc_failures = (
            allow_future_mask_qc_exclusions
            and maximum_patients is None
            and bool(qc_exclusions)
            and set(failures) == {"future_mask_outside_fixed_crop"}
            and sum(failures.values()) == len(qc_exclusions)
        )
        # This is the one explicitly reviewed exception to the all-patient release
        # contract.  Retain private metadata for aggregate accounting, but do not
        # let an approved per-patient exclusion block the remaining cohort.
        if only_approved_qc_failures:
            failures.clear()
        if not failures or only_approved_qc_failures:
            try:
                if qc_exclusions:
                    _write_csv_exclusive(
                        staging / "qc_excluded_patients.csv",
                        fieldnames=QC_EXCLUSION_FIELDS,
                        rows=qc_exclusions,
                    )
                _write_csv_exclusive(
                    staging / "roi_index.csv",
                    fieldnames=ROI_INDEX_FIELDS,
                    rows=index_rows,
                )
                _write_csv_exclusive(
                    staging / "conversion_lineage.csv",
                    fieldnames=CONVERSION_LINEAGE_FIELDS,
                    rows=lineage_rows,
                )
                _validate_staged_release(
                    staging,
                    expected_patients=len(selected) - len(qc_exclusions),
                    shape_dhw=roi_spec["shape_dhw"],
                )
                os.replace(staging, output)
                promoted = True
            except ROIConversionError as error:
                failures[error.code] += 1
                if error.metadata:
                    failure_metadata.append({"code": error.code, **error.metadata})
            except OSError as error:
                _raise_stale(error)
                failures["roi_release_finalization_io_error"] += 1
            except Exception:
                failures["roi_release_finalization_error"] += 1
    full = maximum_patients is None
    status = (
        "blocked_roi_conversion"
        if failures
        else "ok_roi_conversion_full_qc_exclusions"
        if full and qc_exclusions
        else "ok_roi_conversion_full"
        if full
        else "ok_roi_conversion_pilot"
    )
    return _aggregate_report(
        status=status,
        selected_patients=len(selected),
        all_patients=EXPECTED_PATIENTS,
        split_counts=split_counts,
        failure_reasons=failures,
        registration_metrics=registration_metrics,
        future_margins=future_margins,
        mapping_statuses=mapping_statuses,
        failure_metadata=failure_metadata,
        evidence=evidence,
        output_promoted=promoted,
        qc_exclusion_count=len(qc_exclusions),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--ftv-csv", required=True)
    parser.add_argument("--ftv-audit-report", required=True)
    parser.add_argument("--roi-spec", required=True)
    parser.add_argument("--roi-anchor-report", required=True)
    parser.add_argument("--governed-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--maximum-patients",
        type=int,
        default=None,
        help="Deterministic conversion pilot size; omit only for the full 595-patient release.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Bounded patient-level conversion concurrency; use at most allocated Slurm CPUs.",
    )
    parser.add_argument(
        "--allow-future-mask-qc-exclusions",
        action="store_true",
        help=(
            "For a full release only, allow the reviewed future-mask-outside-fixed-crop "
            "condition to exclude that patient and write governed qc_excluded_patients.csv."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_roi_conversion(
            args.root,
            trajectory_path=args.trajectory,
            ftv_csv_path=args.ftv_csv,
            ftv_audit_report_path=args.ftv_audit_report,
            roi_spec_path=args.roi_spec,
            anchor_report_path=args.roi_anchor_report,
            governed_root=args.governed_root,
            output_root=args.output_root,
            maximum_patients=args.maximum_patients,
            allow_future_mask_qc_exclusions=args.allow_future_mask_qc_exclusions,
            workers=args.workers,
        )
        _write_bytes_exclusive(
            Path(args.report).expanduser(),
            (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        )
        print(json.dumps(report, sort_keys=True))
        return 0 if report["status"].startswith("ok_") else 3
    except StaleFileHandleError:
        raise
    except ROIConversionError as error:
        print(f"ERROR: {error.code}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
