"""Full-cohort, header-only geometry gate for the I-SPY2 original DCE source.

The probe implements the deliberately narrow ``original-dce-v3`` contract established
by the aggregate-only v3 diagnostic of all 1,785 governed visits.  It accepts only
classic, single-series MR objects; requires the collection's zero-origin contiguous
Temporal Position Identifier topology; validates the collection-specific public and
private timing relationships by DCE/SER role; and proves that the derived SER grid is
either identical to, or physically contained by, the pre-contrast DCE grid in the same
Frame of Reference.

Only aggregate counters are returned or written.  Patient keys, paths, UIDs, and
acquisition date/time values are used transiently and are never placed in a report.
No pixel data are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_series_probe import _inside, _raise_stale


TIMEPOINTS = ("T0", "T1", "T3")
EXPECTED_COMPLETE_PATIENTS = 595
EXPECTED_SPLIT_COUNTS = {"train": 416, "val": 89, "test": 90}
APPROVED_DESCRIPTION_PDF_SHA256 = (
    "d1c1525a962d916483b0fde69736a6717cc647ee19468690d5d591c4e077544e"
)
APPROVED_TRAJECTORY_SHA256 = (
    "11ffa254b5dc9c4f5edc3132c089ec352ee979a4b30098a1f0c7a577ec45ff7a"
)
APPROVED_SPLIT_SHA256 = (
    "b34636e5166ba7187711887afb6087b672898463ef79a02ec489fe730cb05626"
)
DESCRIPTION_PDF_RELATIVE_PATH = Path(
    "clinical/ACRIN-6698-ISPY2-DWI-and-DCE-MRI-Data-Descriptions_20210520.pdf"
)
SPLIT_RELATIVE_PATH = Path("manifests/ispy2_split.csv")
DEFAULT_TRAJECTORY_RELATIVE_PATH = Path("manifests/ispy2_trajectory_manifest.csv")

CLASSIC_MR_IMAGE_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.4"
PRIVATE_CREATOR = (0x0117, 0x0010)
PRIVATE_TEMPORAL_TOTAL = (0x0117, 0x1030)
PRIVATE_SER_INDICES = (0x0117, 0x1035)
PRIVATE_CREATOR_VALUE = "UCSF BIRP PRIVATE CREATOR 011710xx"
GEOMETRY_RELATIONS = ("geometry_exact", "same_frame_of_reference_contained")
POSITION_TOLERANCE_MM = 0.05
VECTOR_TOLERANCE = 1e-4
SPACING_RELATIVE_TOLERANCE = 0.01
EXACT_POSITION_TOLERANCE_MM = 1e-3
EXACT_SPACING_TOLERANCE_MM = 1e-4
EXACT_SPACING_RELATIVE_TOLERANCE = 1e-6
DICOM_FILENAME = re.compile(r"^[0-9]{8}\.dcm$", re.IGNORECASE)
REQUIRED_SERIES_SIDECARS = frozenset({"LICENSE", ".complete"})
APPROVED_NORMALIZED_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "dce": {
        "ispy2volserunilateralcroppedoriginaldce": "official_unilateral_cropped",
        "ispy2volserbilateraloriginaldce": "historical_bilateral_candidate",
    },
    "ser": {
        "ispy2volserunilateralcroppedser": "official_unilateral_cropped",
        "ispy2volserbilateralser": "historical_bilateral_candidate",
    },
}
_PATIENT_ID = re.compile(r"^ISPY2-[0-9]+$")

HEADER_TAGS: tuple[Any, ...] = (
    "SOPClassUID",
    "Modality",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "SeriesDescription",
    "FrameOfReferenceUID",
    "Rows",
    "Columns",
    "PixelSpacing",
    "SpacingBetweenSlices",
    "SliceThickness",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "TemporalPositionIdentifier",
    "NumberOfTemporalPositions",
    PRIVATE_CREATOR,
    PRIVATE_TEMPORAL_TOTAL,
    PRIVATE_SER_INDICES,
)


class DCEGeometryProbeError(RuntimeError):
    """Fail-closed error whose code is safe for aggregate reports and Slurm logs."""

    def __init__(self, code: str, *, safe_metadata: Mapping[str, int] | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.safe_metadata = dict(safe_metadata or {})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _raise_stale(error)
        raise DCEGeometryProbeError("evidence_file_read_error") from None
    return digest.hexdigest()


def _read_pinned_csv_rows(
    path: Path,
    *,
    expected_sha256: str,
    hash_error_code: str,
    read_error_code: str,
    schema_error_code: str,
) -> tuple[list[str], list[dict[str | None, str | list[str] | None]], str]:
    """Read and hash the same immutable byte snapshot before CSV parsing."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        _raise_stale(error)
        raise DCEGeometryProbeError(read_error_code) from None
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise DCEGeometryProbeError(hash_error_code)
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise DCEGeometryProbeError(schema_error_code)
        fieldnames = [str(value).strip() for value in reader.fieldnames]
        rows = list(reader)
    except (UnicodeError, csv.Error):
        raise DCEGeometryProbeError(schema_error_code) from None
    if any(None in row for row in rows):
        raise DCEGeometryProbeError(schema_error_code)
    return fieldnames, rows, digest


def _finite_vector(value: Any, length: int, code: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError, OverflowError) as error:
        raise DCEGeometryProbeError(code) from error
    if len(result) != length or any(not math.isfinite(item) for item in result):
        raise DCEGeometryProbeError(code)
    return result


def _positive_integer(value: Any, code: str) -> int:
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as error:
        raise DCEGeometryProbeError(code) from error
    if not math.isfinite(number) or number != result or result < 1:
        raise DCEGeometryProbeError(code)
    return result


def _nonnegative_integer(value: Any, code: str) -> int:
    try:
        number = float(value)
        result = int(number)
    except (TypeError, ValueError, OverflowError) as error:
        raise DCEGeometryProbeError(code) from error
    if not math.isfinite(number) or number != result or result < 0:
        raise DCEGeometryProbeError(code)
    return result


def _tag_value(dataset: Any, tag: tuple[int, int]) -> Any | None:
    try:
        if tag in dataset:
            value = dataset[tag]
            return getattr(value, "value", value)
    except (KeyError, TypeError, AttributeError):
        pass
    try:
        value = dataset.get(tag, None)
    except (TypeError, AttributeError):
        return None
    return getattr(value, "value", value)


def _private_creator_text(value: Any) -> str:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise DCEGeometryProbeError("invalid_private_creator") from error
    return str(value).strip().strip("\x00").strip()


def _integer_sequence(value: Any, code: str) -> tuple[int, ...]:
    if isinstance(value, bytes):
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as error:
            raise DCEGeometryProbeError(code) from error
    if isinstance(value, str):
        text = value.strip().strip("[]()")
        if not text:
            raise DCEGeometryProbeError(code)
        tokens = [item for item in re.split(r"[\\,;\s]+", text) if item]
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, Mapping)):
        tokens = list(value)
    else:
        tokens = [value]
    result: list[int] = []
    for token in tokens:
        try:
            number = float(token)
            integer = int(number)
        except (TypeError, ValueError, OverflowError) as error:
            raise DCEGeometryProbeError(code) from error
        if not math.isfinite(number) or number != integer:
            raise DCEGeometryProbeError(code)
        result.append(integer)
    if not result:
        raise DCEGeometryProbeError(code)
    return tuple(result)


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _close(left: float, right: float, *, absolute: float = POSITION_TOLERANCE_MM) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=SPACING_RELATIVE_TOLERANCE,
        abs_tol=absolute,
    )


def build_stack_geometry(headers: Sequence[Any]) -> dict[str, Any]:
    """Build a regular, possibly oblique stack using normal projections.

    The returned mapping contains UIDs only in underscore-prefixed internal fields;
    callers must never forward those fields to aggregate output.
    """

    if len(headers) < 2:
        raise DCEGeometryProbeError("insufficient_slices_for_geometry")
    rows = {_positive_integer(getattr(item, "Rows", None), "invalid_image_matrix") for item in headers}
    columns = {
        _positive_integer(getattr(item, "Columns", None), "invalid_image_matrix")
        for item in headers
    }
    if len(rows) != 1 or len(columns) != 1:
        raise DCEGeometryProbeError("nonuniform_image_matrix")
    spacings = {
        _finite_vector(getattr(item, "PixelSpacing", None), 2, "invalid_pixel_spacing")
        for item in headers
    }
    if len(spacings) != 1 or any(value <= 0 for value in next(iter(spacings))):
        raise DCEGeometryProbeError("nonuniform_pixel_spacing")
    orientations = {
        _finite_vector(
            getattr(item, "ImageOrientationPatient", None),
            6,
            "invalid_image_orientation",
        )
        for item in headers
    }
    if len(orientations) != 1:
        raise DCEGeometryProbeError("nonuniform_image_orientation")
    orientation = next(iter(orientations))
    column_axis, row_axis = orientation[:3], orientation[3:]
    if not math.isclose(
        math.sqrt(_dot(column_axis, column_axis)),
        1.0,
        rel_tol=0.0,
        abs_tol=VECTOR_TOLERANCE,
    ):
        raise DCEGeometryProbeError("invalid_image_orientation")
    if not math.isclose(
        math.sqrt(_dot(row_axis, row_axis)),
        1.0,
        rel_tol=0.0,
        abs_tol=VECTOR_TOLERANCE,
    ):
        raise DCEGeometryProbeError("invalid_image_orientation")
    if abs(_dot(column_axis, row_axis)) > VECTOR_TOLERANCE:
        raise DCEGeometryProbeError("invalid_image_orientation")
    normal = _cross(column_axis, row_axis)
    normal_norm = math.sqrt(_dot(normal, normal))
    if not math.isfinite(normal_norm) or normal_norm <= 0:
        raise DCEGeometryProbeError("invalid_image_orientation")
    normal = tuple(value / normal_norm for value in normal)
    positions = [
        _finite_vector(
            getattr(item, "ImagePositionPatient", None),
            3,
            "invalid_image_position",
        )
        for item in headers
    ]
    projected = [_dot(position, normal) for position in positions]
    order = sorted(range(len(headers)), key=projected.__getitem__)
    sorted_positions = [positions[index] for index in order]
    sorted_projected = [projected[index] for index in order]
    differences = [right - left for left, right in zip(sorted_projected, sorted_projected[1:])]
    if any(value <= POSITION_TOLERANCE_MM for value in differences):
        raise DCEGeometryProbeError("duplicate_slice_positions")
    spacing = sum(differences) / len(differences)
    if any(not _close(value, spacing) for value in differences):
        raise DCEGeometryProbeError("nonuniform_slice_spacing")
    anchor = sorted_positions[0]
    for position, projection in zip(sorted_positions, sorted_projected):
        expected = tuple(
            anchor[index] + (projection - sorted_projected[0]) * normal[index]
            for index in range(3)
        )
        if max(abs(left - right) for left, right in zip(position, expected)) > POSITION_TOLERANCE_MM:
            raise DCEGeometryProbeError("irregular_in_plane_slice_drift")
    frame_values = {str(getattr(item, "FrameOfReferenceUID", "")).strip() for item in headers}
    series_values = {str(getattr(item, "SeriesInstanceUID", "")).strip() for item in headers}
    if "" in frame_values or len(frame_values) != 1:
        raise DCEGeometryProbeError("invalid_frame_of_reference")
    # Unit-test header doubles intentionally omit SeriesInstanceUID.  Directory-level
    # production validation enforces a non-empty, single series before this helper.
    series_values.discard("")
    return {
        "slice_count": len(headers),
        "rows": next(iter(rows)),
        "columns": next(iter(columns)),
        "pixel_spacing_mm": list(next(iter(spacings))),
        "orientation": list(orientation),
        "slice_spacing_mm": spacing,
        "slice_projections_mm": sorted_projected,
        "_positions": sorted_positions,
        "_column_axis": tuple(column_axis),
        "_row_axis": tuple(row_axis),
        "_normal_axis": normal,
        "_frame_of_reference_uid": next(iter(frame_values)),
        "_series_instance_uid": next(iter(series_values), ""),
    }


def _private_values(headers: Sequence[Any], tag: tuple[int, int]) -> list[Any]:
    return [value for item in headers if (value := _tag_value(item, tag)) is not None]


def validate_dce_phases(
    headers: Sequence[Any], *, timing_source_headers: Sequence[Any] | None = None
) -> dict[str, Any]:
    """Validate the governed, role-specific temporal contract without fallbacks.

    ``headers`` are original-DCE objects.  ``timing_source_headers`` are the paired
    derived-SER objects.  The collection places private SER source ordinals only on
    the latter; merging both roles before testing presence would therefore reject the
    verified data or, worse, obscure a role reversal.
    """

    if not headers:
        raise DCEGeometryProbeError("missing_dce_headers")
    ser_timing_headers = list(timing_source_headers or ())
    if not ser_timing_headers:
        raise DCEGeometryProbeError("missing_ser_timing_source_headers")
    phase_groups: dict[int, list[Any]] = defaultdict(list)
    for item in headers:
        if getattr(item, "TemporalPositionIdentifier", None) is None:
            raise DCEGeometryProbeError("missing_temporal_position_identifier")
        phase = _nonnegative_integer(
            getattr(item, "TemporalPositionIdentifier", None),
            "invalid_temporal_position_identifier",
        )
        phase_groups[phase].append(item)
    phase_indices = sorted(phase_groups)
    zero_based = list(range(0, len(phase_indices)))
    if phase_indices != zero_based:
        if phase_indices == list(
            range(phase_indices[0], phase_indices[0] + len(phase_indices))
        ):
            raise DCEGeometryProbeError("temporal_position_origin_not_zero")
        raise DCEGeometryProbeError("noncontiguous_temporal_position_identifiers")
    phase_count = len(phase_indices)
    ntp_raw = [
        getattr(item, "NumberOfTemporalPositions", None)
        for item in headers
        if getattr(item, "NumberOfTemporalPositions", None) is not None
    ]
    if not ntp_raw:
        raise DCEGeometryProbeError("missing_number_of_temporal_positions")
    if len(ntp_raw) != len(headers):
        raise DCEGeometryProbeError("partial_number_of_temporal_positions")
    ntp_values = {
        _nonnegative_integer(value, "invalid_number_of_temporal_positions")
        for value in ntp_raw
    }
    # Diagnostic v3 established this collection-specific writer convention on every
    # governed DCE series: for zero-origin TPI, NumberOfTemporalPositions stores the
    # largest TPI (N-1), while private 0117,1030 stores the true count N.  This is not
    # generalized to other DICOM collections.
    if ntp_values != {phase_count - 1}:
        raise DCEGeometryProbeError(
            "number_of_temporal_positions_zero_origin_mismatch"
        )
    ntp_status = "zero_origin_max_tpi_consistent"

    timing_headers = [*headers, *ser_timing_headers]
    creator_values = [_tag_value(item, PRIVATE_CREATOR) for item in timing_headers]
    if any(value is None for value in creator_values):
        raise DCEGeometryProbeError("missing_private_creator")
    if any(
        _private_creator_text(value) != PRIVATE_CREATOR_VALUE
        for value in creator_values
    ):
        raise DCEGeometryProbeError("private_creator_mismatch")
    raw_totals = _private_values(timing_headers, PRIVATE_TEMPORAL_TOTAL)
    if len(raw_totals) != len(timing_headers):
        raise DCEGeometryProbeError("missing_private_temporal_total")
    total_values = {
        _positive_integer(value, "invalid_private_temporal_total")
        for value in raw_totals
    }
    if total_values != {phase_count}:
        raise DCEGeometryProbeError("private_temporal_total_mismatch")

    if _private_values(headers, PRIVATE_SER_INDICES):
        raise DCEGeometryProbeError("private_ser_indices_present_on_dce")
    raw_indices = _private_values(ser_timing_headers, PRIVATE_SER_INDICES)
    if len(raw_indices) != len(ser_timing_headers):
        raise DCEGeometryProbeError("missing_private_ser_indices")
    parsed = [_integer_sequence(value, "invalid_private_ser_indices") for value in raw_indices]
    if any(len(value) != 3 for value in parsed):
        raise DCEGeometryProbeError("invalid_private_ser_indices")
    index_values = set(parsed)
    if len(index_values) != 1:
        raise DCEGeometryProbeError("private_temporal_index_mismatch")
    ser_zero = next(iter(index_values))
    if len(ser_zero) != 3 or any(index < 0 or index >= phase_count for index in ser_zero):
        raise DCEGeometryProbeError("private_ser_indices_out_of_range")
    if list(ser_zero) != sorted(set(ser_zero)):
        raise DCEGeometryProbeError("private_ser_indices_not_strictly_increasing")
    if ser_zero[0] != 0:
        raise DCEGeometryProbeError("private_ser_precontrast_index_not_zero")
    # The collection-private values are zero-origin acquisition ordinals, not raw
    # TemporalPositionIdentifier values.  Index the validated raw TPI sequence
    # explicitly so a zero-origin series cannot silently select its first
    # post-contrast phase as the pre-contrast source.
    ser_source_temporal_position_values = [phase_indices[index] for index in ser_zero]

    geometries = {phase: build_stack_geometry(items) for phase, items in phase_groups.items()}
    slice_counts = {geometry["slice_count"] for geometry in geometries.values()}
    if len(slice_counts) != 1:
        raise DCEGeometryProbeError("dce_phase_slice_count_mismatch")
    reference = geometries[phase_indices[0]]
    if any(not _geometries_exact(reference, geometries[phase]) for phase in phase_indices[1:]):
        raise DCEGeometryProbeError("dce_phase_grid_mismatch")
    return {
        "status": "ok",
        "phase_count": phase_count,
        "phase_indices": phase_indices,
        "precontrast_temporal_position_value": ser_source_temporal_position_values[0],
        "private_ser_source_ordinals_zero_based": list(ser_zero),
        "ser_source_temporal_position_values": ser_source_temporal_position_values,
        "phase_index_source": "temporal_position_identifier",
        "temporal_position_origin": "zero_based",
        "number_of_temporal_positions_status": ntp_status,
        "private_creator_consistent": True,
        "private_timing_consistent": True,
        "private_temporal_total_status": "dce_ser_exact_phase_count",
        "private_ser_indices_role_status": "dce_absent_ser_present",
        "slices_per_phase": next(iter(slice_counts)),
        "_headers_by_temporal_position": {
            key: tuple(value) for key, value in phase_groups.items()
        },
    }


def _positions_equal(left: Sequence[Sequence[float]], right: Sequence[Sequence[float]]) -> bool:
    if len(left) != len(right):
        return False
    return all(
        max(abs(a - b) for a, b in zip(lpos, rpos)) <= EXACT_POSITION_TOLERANCE_MM
        for lpos, rpos in zip(left, right)
    )


def _exact_spacing_equal(left: float, right: float) -> bool:
    return math.isclose(
        left,
        right,
        rel_tol=EXACT_SPACING_RELATIVE_TOLERANCE,
        abs_tol=EXACT_SPACING_TOLERANCE_MM,
    )


def _geometries_exact(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return (
        left["_frame_of_reference_uid"] == right["_frame_of_reference_uid"]
        and left["rows"] == right["rows"]
        and left["columns"] == right["columns"]
        and all(
            _exact_spacing_equal(float(a), float(b))
            for a, b in zip(left["pixel_spacing_mm"], right["pixel_spacing_mm"])
        )
        and all(abs(float(a) - float(b)) <= VECTOR_TOLERANCE for a, b in zip(left["orientation"], right["orientation"]))
        and _exact_spacing_equal(
            float(left["slice_spacing_mm"]), float(right["slice_spacing_mm"])
        )
        and _positions_equal(left["_positions"], right["_positions"])
    )


def _corners(geometry: Mapping[str, Any]) -> list[tuple[float, float, float]]:
    origin = geometry["_positions"][0]
    column_axis = geometry["_column_axis"]
    row_axis = geometry["_row_axis"]
    normal = geometry["_normal_axis"]
    row_spacing, column_spacing = geometry["pixel_spacing_mm"]
    distances = (
        (-0.5 * column_spacing, (geometry["columns"] - 0.5) * column_spacing),
        (-0.5 * row_spacing, (geometry["rows"] - 0.5) * row_spacing),
        (-0.5 * geometry["slice_spacing_mm"], (geometry["slice_count"] - 0.5) * geometry["slice_spacing_mm"]),
    )
    result = []
    for dc in distances[0]:
        for dr in distances[1]:
            for dz in distances[2]:
                result.append(
                    tuple(
                        origin[index]
                        + dc * column_axis[index]
                        + dr * row_axis[index]
                        + dz * normal[index]
                        for index in range(3)
                    )
                )
    return result


def _contained(inner: Mapping[str, Any], outer: Mapping[str, Any]) -> bool:
    if inner["_frame_of_reference_uid"] != outer["_frame_of_reference_uid"]:
        return False
    if any(
        abs(abs(_dot(inner[key], outer[key])) - 1.0) > VECTOR_TOLERANCE
        for key in ("_column_axis", "_row_axis", "_normal_axis")
    ):
        return False
    origin = outer["_positions"][0]
    axes = (outer["_column_axis"], outer["_row_axis"], outer["_normal_axis"])
    outer_corners = _corners(outer)
    bounds = [
        (
            min(_dot(tuple(point[i] - origin[i] for i in range(3)), axis) for point in outer_corners),
            max(_dot(tuple(point[i] - origin[i] for i in range(3)), axis) for point in outer_corners),
        )
        for axis in axes
    ]
    for point in _corners(inner):
        delta = tuple(point[index] - origin[index] for index in range(3))
        if any(
            _dot(delta, axis) < low - POSITION_TOLERANCE_MM
            or _dot(delta, axis) > high + POSITION_TOLERANCE_MM
            for axis, (low, high) in zip(axes, bounds)
        ):
            return False
    return True


def classify_dce_ser_geometry(
    precontrast_headers: Sequence[Any], ser_headers: Sequence[Any]
) -> dict[str, Any]:
    """Classify the two geometry relations accepted by original-dce-v3."""

    pre = build_stack_geometry(precontrast_headers)
    ser = build_stack_geometry(ser_headers)
    if pre["_frame_of_reference_uid"] != ser["_frame_of_reference_uid"]:
        return {"status": "different_frame_of_reference", "admissible": False}
    if _geometries_exact(pre, ser):
        return {"status": "geometry_exact", "admissible": True}
    if _contained(ser, pre):
        return {
            "status": "same_frame_of_reference_contained",
            "admissible": True,
        }
    return {"status": "same_frame_of_reference_not_contained", "admissible": False}


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize_visit_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempted: int,
    failure_reasons: Mapping[str, int],
) -> dict[str, Any]:
    """Return an allowlist-only aggregate; unknown row fields are discarded."""

    fields = (
        "phase_count",
        "slices_per_phase",
        "phase_index_source",
        "temporal_position_origin",
        "number_of_temporal_positions_status",
        "private_creator_consistent",
        "private_timing_consistent",
        "private_temporal_total_status",
        "private_ser_indices_role_status",
        "dce_ser_description_status",
        "dce_ser_geometry_status",
        "dce_spatial_variant",
        "ser_spatial_variant",
        "admissible",
        "dce_files_enumerated",
        "dce_dicom_candidates",
        "dce_headers_read",
        "dce_sidecar_files",
        "ser_files_enumerated",
        "ser_dicom_candidates",
        "ser_headers_read",
        "ser_sidecar_files",
    )
    failures = sum(int(value) for value in failure_reasons.values())
    return {
        "visits_attempted": attempted,
        "visits_passed": len(rows),
        "visits_failed": failures,
        "accounting_consistent": len(rows) + failures == attempted,
        "failure_reasons": dict(sorted((str(key), int(value)) for key, value in failure_reasons.items())),
        "distributions": {
            field: _distribution(row[field] for row in rows if field in row)
            for field in fields
        },
    }


def _read_headers(
    path: Path,
    *,
    dicom_root: Path,
    additional_tags: Sequence[Any] = (),
) -> tuple[list[Any], dict[str, int]]:
    """Read a contract-checked series once, optionally including extra safe tags.

    The original DCE gate passes no extra tags.  Downstream governed readers can add
    header-only tags (for example SER rescale metadata) without re-enumerating and
    reparsing every source slice on NFS.
    """
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise DCEGeometryProbeError("pydicom_dependency_unavailable") from error
    try:
        with os.scandir(path) as iterator:
            entries = sorted(iterator, key=lambda item: item.name)
    except OSError as error:
        _raise_stale(error)
        raise DCEGeometryProbeError("dicom_directory_read_error") from None
    candidates: list[Path] = []
    sidecars: set[str] = set()
    for entry in entries:
        try:
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise DCEGeometryProbeError("nonregular_series_directory_entry")
        except OSError as error:
            _raise_stale(error)
            raise DCEGeometryProbeError("series_directory_stat_error") from None
        if entry.name in REQUIRED_SERIES_SIDECARS:
            sidecars.add(entry.name)
        elif DICOM_FILENAME.fullmatch(entry.name):
            candidates.append(Path(entry.path))
        elif entry.name.startswith("."):
            # Diagnostic v3 found one hidden regular entry.  Preserve its safe class
            # in the aggregate failure report, but never ignore or parse it.
            raise DCEGeometryProbeError("unexpected_hidden_regular")
        else:
            raise DCEGeometryProbeError("unexpected_series_directory_entry")
    if sidecars != REQUIRED_SERIES_SIDECARS:
        raise DCEGeometryProbeError("missing_required_series_sidecar")
    requested_tags = list(dict.fromkeys((*HEADER_TAGS, *additional_tags)))
    headers: list[Any] = []
    parse_errors = 0
    invalid_dicom = 0
    outside = 0
    for item in candidates:
        try:
            resolved = item.resolve()
        except OSError as error:
            _raise_stale(error)
            parse_errors += 1
            continue
        if not _inside(resolved, dicom_root):
            outside += 1
            continue
        try:
            headers.append(
                pydicom.dcmread(
                    resolved,
                    stop_before_pixels=True,
                    force=False,
                    specific_tags=requested_tags,
                )
            )
        except pydicom.errors.InvalidDicomError:
            invalid_dicom += 1
        except OSError as error:
            _raise_stale(error)
            parse_errors += 1
        except Exception:
            parse_errors += 1
    counts = {
        "files_enumerated": len(entries),
        "dicom_candidates": len(candidates),
        "headers_read": len(headers),
        "sidecar_files": len(sidecars),
    }
    if outside:
        raise DCEGeometryProbeError("dicom_file_outside_approved_root", safe_metadata=counts)
    if invalid_dicom:
        raise DCEGeometryProbeError("invalid_dicom_image_file", safe_metadata=counts)
    if parse_errors:
        raise DCEGeometryProbeError("dicom_header_parse_error", safe_metadata=counts)
    if not headers:
        raise DCEGeometryProbeError("no_classic_mr_headers", safe_metadata=counts)
    return headers, counts


def _validate_classic_single_series(headers: Sequence[Any], *, kind: str) -> dict[str, str]:
    if any(str(getattr(item, "SOPClassUID", "")) != CLASSIC_MR_IMAGE_STORAGE_UID for item in headers):
        raise DCEGeometryProbeError(f"{kind}_unsupported_sop_class")
    if any(str(getattr(item, "Modality", "")).strip().upper() != "MR" for item in headers):
        raise DCEGeometryProbeError(f"{kind}_not_mr")
    series = {str(getattr(item, "SeriesInstanceUID", "")).strip() for item in headers}
    if "" in series or len(series) != 1:
        raise DCEGeometryProbeError(f"{kind}_not_single_series")
    studies = {str(getattr(item, "StudyInstanceUID", "")).strip() for item in headers}
    if "" in studies or len(studies) != 1:
        raise DCEGeometryProbeError(f"{kind}_not_single_study")
    descriptions = {
        re.sub(r"[^a-z0-9]", "", str(getattr(item, "SeriesDescription", "")).casefold())
        for item in headers
    }
    if "" in descriptions or len(descriptions) != 1:
        raise DCEGeometryProbeError(f"{kind}_series_description_mismatch")
    description = next(iter(descriptions))
    variant = APPROVED_NORMALIZED_DESCRIPTIONS[kind].get(description)
    if variant is None:
        raise DCEGeometryProbeError(f"{kind}_series_description_mismatch")
    return {
        "study_instance_uid": next(iter(studies)),
        "series_instance_uid": next(iter(series)),
        "spatial_variant": variant,
    }


def _inspect_visit(dce_dir: Path, ser_dir: Path, *, dicom_root: Path, timepoint: str) -> dict[str, Any]:
    dce_headers, dce_counts = _read_headers(dce_dir, dicom_root=dicom_root)
    ser_headers, ser_counts = _read_headers(ser_dir, dicom_root=dicom_root)
    dce_identity = _validate_classic_single_series(dce_headers, kind="dce")
    ser_identity = _validate_classic_single_series(ser_headers, kind="ser")
    if dce_identity["study_instance_uid"] != ser_identity["study_instance_uid"]:
        raise DCEGeometryProbeError("dce_ser_study_mismatch")
    if dce_identity["series_instance_uid"] == ser_identity["series_instance_uid"]:
        raise DCEGeometryProbeError("dce_ser_series_uid_collision")
    if dce_identity["spatial_variant"] != ser_identity["spatial_variant"]:
        raise DCEGeometryProbeError("dce_ser_description_variant_mismatch")
    phases = validate_dce_phases(dce_headers, timing_source_headers=ser_headers)
    pre_headers = phases["_headers_by_temporal_position"][
        phases["precontrast_temporal_position_value"]
    ]
    relation = classify_dce_ser_geometry(pre_headers, ser_headers)
    if not relation["admissible"]:
        raise DCEGeometryProbeError(str(relation["status"]))
    return {
        "timepoint": timepoint,
        "phase_count": phases["phase_count"],
        "slices_per_phase": phases["slices_per_phase"],
        "phase_index_source": phases["phase_index_source"],
        "temporal_position_origin": phases["temporal_position_origin"],
        "number_of_temporal_positions_status": phases[
            "number_of_temporal_positions_status"
        ],
        "private_creator_consistent": phases["private_creator_consistent"],
        "private_timing_consistent": phases["private_timing_consistent"],
        "private_temporal_total_status": phases["private_temporal_total_status"],
        "private_ser_indices_role_status": phases[
            "private_ser_indices_role_status"
        ],
        "dce_ser_description_status": f"same_{dce_identity['spatial_variant']}",
        "dce_ser_geometry_status": relation["status"],
        "dce_spatial_variant": dce_identity["spatial_variant"],
        "ser_spatial_variant": ser_identity["spatial_variant"],
        "admissible": True,
        "dce_files_enumerated": dce_counts["files_enumerated"],
        "dce_dicom_candidates": dce_counts["dicom_candidates"],
        "dce_headers_read": dce_counts["headers_read"],
        "dce_sidecar_files": dce_counts["sidecar_files"],
        "ser_files_enumerated": ser_counts["files_enumerated"],
        "ser_dicom_candidates": ser_counts["dicom_candidates"],
        "ser_headers_read": ser_counts["headers_read"],
        "ser_sidecar_files": ser_counts["sidecar_files"],
    }


def _read_split(root: Path, expected_patients: int) -> tuple[set[str], dict[str, int], str]:
    path = root / SPLIT_RELATIVE_PATH
    fieldnames, rows, digest = _read_pinned_csv_rows(
        path,
        expected_sha256=APPROVED_SPLIT_SHA256,
        hash_error_code="split_hash_mismatch",
        read_error_code="split_read_error",
        schema_error_code="split_schema_mismatch",
    )
    patients: set[str] = set()
    counts: Counter[str] = Counter()
    if not {"PatientID", "split"}.issubset(fieldnames):
        raise DCEGeometryProbeError("split_schema_mismatch")
    for row in rows:
        patient = str(row.get("PatientID") or "").strip()
        split = str(row.get("split") or "").strip().casefold()
        if split == "validation":
            split = "val"
        if not _PATIENT_ID.fullmatch(patient) or split not in EXPECTED_SPLIT_COUNTS:
            raise DCEGeometryProbeError("split_value_mismatch")
        if patient in patients:
            raise DCEGeometryProbeError("duplicate_split_patient")
        patients.add(patient)
        counts[split] += 1
    expected_counts = EXPECTED_SPLIT_COUNTS if expected_patients == EXPECTED_COMPLETE_PATIENTS else None
    observed = {key: counts[key] for key in EXPECTED_SPLIT_COUNTS}
    if len(patients) != expected_patients or (expected_counts is not None and observed != expected_counts):
        raise DCEGeometryProbeError("split_cohort_contract_mismatch")
    return patients, observed, digest


def _read_cohort(
    root: Path, trajectory: Path, patient_ids: set[str], expected_patients: int
) -> list[tuple[str, dict[str, tuple[Path, Path]]]]:
    dicom_root = (root / "dicom").resolve()
    fieldnames, rows, _digest = _read_pinned_csv_rows(
        trajectory,
        expected_sha256=APPROVED_TRAJECTORY_SHA256,
        hash_error_code="trajectory_hash_mismatch",
        read_error_code="trajectory_read_error",
        schema_error_code="trajectory_schema_mismatch",
    )
    patients: dict[str, dict[str, tuple[Path, Path]]] = defaultdict(dict)
    used_directories: set[Path] = set()
    required = {"PatientID", "tp_label", "dce_dir", "ser_dir"}
    if not required.issubset(fieldnames):
        raise DCEGeometryProbeError("trajectory_schema_mismatch")
    for row in rows:
        timepoint = str(row.get("tp_label") or "").strip().upper()
        if timepoint not in TIMEPOINTS:
            continue
        patient = str(row.get("PatientID") or "").strip()
        if not _PATIENT_ID.fullmatch(patient):
            raise DCEGeometryProbeError("invalid_patient_identifier")
        raw_dce = str(row.get("dce_dir") or "").strip()
        raw_ser = str(row.get("ser_dir") or "").strip()
        if not raw_dce or not raw_ser:
            continue
        resolved: list[Path] = []
        for raw in (raw_dce, raw_ser):
            value = Path(raw).expanduser()
            value = (value if value.is_absolute() else trajectory.parent / value).resolve()
            if not _inside(value, dicom_root):
                raise DCEGeometryProbeError("trajectory_path_outside_approved_root")
            resolved.append(value)
        dce, ser = resolved
        if dce in used_directories or ser in used_directories or dce == ser:
            raise DCEGeometryProbeError("duplicate_series_directory_reference")
        if timepoint in patients[patient]:
            raise DCEGeometryProbeError("duplicate_patient_timepoint")
        patients[patient][timepoint] = (dce, ser)
        used_directories.update((dce, ser))
    complete = sorted(
        (patient, visits)
        for patient, visits in patients.items()
        if set(TIMEPOINTS).issubset(visits)
    )
    if len(complete) != expected_patients:
        raise DCEGeometryProbeError("complete_cohort_count_mismatch")
    if {patient for patient, _ in complete} != patient_ids:
        raise DCEGeometryProbeError("complete_cohort_split_mismatch")
    return complete


def _run_full_cohort(
    root: Path,
    trajectory: Path,
    *,
    expected_patients: int,
) -> tuple[list[dict[str, Any]], int, dict[str, int], dict[str, int]]:
    patient_ids, split_counts, _split_hash = _read_split(root, expected_patients)
    cohort = _read_cohort(root, trajectory, patient_ids, expected_patients)
    dicom_root = (root / "dicom").resolve()
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    attempted = 0
    patient_total = len(cohort)
    for patient_number, (_patient, visits) in enumerate(cohort, start=1):
        for timepoint in TIMEPOINTS:
            attempted += 1
            dce_dir, ser_dir = visits[timepoint]
            try:
                rows.append(_inspect_visit(dce_dir, ser_dir, dicom_root=dicom_root, timepoint=timepoint))
            except StaleFileHandleError:
                raise
            except DCEGeometryProbeError as error:
                failures[error.code] += 1
            except Exception:
                failures["unexpected_visit_inspection_error"] += 1
        if patient_number % 25 == 0 or patient_number == patient_total:
            # Aggregate-only progress makes a multi-hour NFS scan observable without
            # placing patient identifiers, paths, UIDs, or dates in Slurm logs.
            print(
                json.dumps(
                    {
                        "event": "dce_geometry_progress",
                        "patients_complete": patient_number,
                        "patients_total": patient_total,
                        "visits_attempted": attempted,
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
                flush=True,
            )
    return rows, attempted, dict(sorted(failures.items())), split_counts


def run_probe(
    root: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    expected_patients: int = EXPECTED_COMPLETE_PATIENTS,
) -> dict[str, Any]:
    """Run the governed all-patient/all-visit probe and return aggregate JSON data."""

    if expected_patients < 1:
        raise ValueError("expected patient count must be positive")
    root_path = Path(root).expanduser().resolve()
    trajectory = (
        Path(trajectory_path).expanduser().resolve()
        if trajectory_path is not None
        else (root_path / DEFAULT_TRAJECTORY_RELATIVE_PATH).resolve()
    )
    evidence = {
        "description_pdf_sha256": None,
        "trajectory_manifest_sha256": None,
        "split_sha256": None,
    }
    global_failures: Counter[str] = Counter()
    try:
        evidence["description_pdf_sha256"] = _sha256(root_path / DESCRIPTION_PDF_RELATIVE_PATH)
        if evidence["description_pdf_sha256"] != APPROVED_DESCRIPTION_PDF_SHA256:
            global_failures["description_pdf_hash_mismatch"] += 1
        evidence["trajectory_manifest_sha256"] = _sha256(trajectory)
        if evidence["trajectory_manifest_sha256"] != APPROVED_TRAJECTORY_SHA256:
            global_failures["trajectory_hash_mismatch"] += 1
        evidence["split_sha256"] = _sha256(root_path / SPLIT_RELATIVE_PATH)
        if evidence["split_sha256"] != APPROVED_SPLIT_SHA256:
            global_failures["split_hash_mismatch"] += 1
    except DCEGeometryProbeError as error:
        global_failures[error.code] += 1

    rows: list[dict[str, Any]] = []
    attempted = 0
    visit_failures: Counter[str] = Counter()
    observed_split_counts: dict[str, int] = {}
    # Do not follow unpinned manifests into arbitrary paths.  With pinned evidence,
    # every visit is attempted even after earlier visit-level failures.
    if not global_failures:
        try:
            rows, attempted, visit_failure_counts, observed_split_counts = _run_full_cohort(
                root_path, trajectory, expected_patients=expected_patients
            )
            visit_failures.update(visit_failure_counts)
        except DCEGeometryProbeError as error:
            global_failures[error.code] += 1
        except StaleFileHandleError:
            raise
    summary = summarize_visit_rows(
        rows,
        attempted=attempted,
        failure_reasons=visit_failures,
    )
    if not summary["accounting_consistent"]:
        global_failures["visit_accounting_mismatch"] += 1
    failures = Counter(global_failures)
    failures.update(visit_failures)
    return {
        "schema_version": "med-jepa-ispy2-dce-geometry-probe-v3",
        "status": "ok"
        if (
            not failures
            and attempted == expected_patients * len(TIMEPOINTS)
            and summary["accounting_consistent"]
            and summary["visits_passed"] == expected_patients * len(TIMEPOINTS)
        )
        else "blocked_dce_geometry",
        "converter_contract": {
            "name": "ispy2_original_dce_v3",
            "classic_mr_single_series_only": True,
            "phase_index_source": "temporal_position_identifier",
            "accepted_temporal_position_origins": ["zero_based"],
            "number_of_temporal_positions": (
                "required_on_every_DCE_object_and_equals_the_largest_zero_origin_TPI; "
                "distinct_TPI_count_must_equal_reported_value_plus_one"
            ),
            "private_creator_required": PRIVATE_CREATOR_VALUE,
            "approved_normalized_descriptions": {
                kind: sorted(values)
                for kind, values in APPROVED_NORMALIZED_DESCRIPTIONS.items()
            },
            "series_directory_contract": {
                "dicom_filename_regex": DICOM_FILENAME.pattern,
                "required_sidecars": sorted(REQUIRED_SERIES_SIDECARS),
                "unknown_or_nonregular_entries": "block",
            },
            "private_timing_contract": {
                "0117,0010": "approved creator required on every DCE and SER object",
                "0117,1030": (
                    "required on every DCE and SER object and equals distinct DCE TPI count"
                ),
                "0117,1035": (
                    "absent on every DCE object; required and integer-parseable on every "
                    "SER object"
                ),
            },
            "description_pair": (
                "same exact allowlisted category within visit: official unilateral "
                "cropped or historical bilateral candidate"
            ),
            "prohibited_phase_fallbacks": ["filename", "pixel_intensity", "acquisition_number"],
            "admissible_geometry": list(GEOMETRY_RELATIONS),
        },
        "evidence": evidence,
        "cohort": {
            "patients_expected": expected_patients,
            "required_timepoints": list(TIMEPOINTS),
            "visits_expected": expected_patients * len(TIMEPOINTS),
            "visits_attempted": attempted,
            "split_counts": observed_split_counts,
        },
        "summary": summary,
        "global_failure_reasons": dict(sorted(global_failures.items())),
        "visit_failure_reasons": dict(sorted(visit_failures.items())),
        "failure_reasons": dict(sorted(failures.items())),
        "pixel_data_read": False,
        "aggregate_only": True,
    }


def _write_report_exclusive(destination: Path, report: Mapping[str, Any]) -> None:
    """Atomically create a mode-0600 JSON report and refuse every overwrite."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(handle.name, 0o600)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory")
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-patients", type=int, default=EXPECTED_COMPLETE_PATIENTS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.expected_patients < 1:
        raise SystemExit("expected-patients must be positive")
    try:
        report = run_probe(
            args.root,
            trajectory_path=args.trajectory,
            expected_patients=args.expected_patients,
        )
        _write_report_exclusive(Path(args.output).expanduser(), report)
    except StaleFileHandleError:
        print("stale_file_handle", file=sys.stderr)
        return 2
    except FileExistsError:
        print("output_exists_refusing_overwrite", file=sys.stderr)
        return 3
    except DCEGeometryProbeError as error:
        print(error.code, file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "cohort": report["cohort"],
                "summary": report["summary"],
                "failure_reasons": report["failure_reasons"],
                "pixel_data_read": False,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
