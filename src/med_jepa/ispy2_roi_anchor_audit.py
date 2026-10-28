"""Full-cohort, aggregate-only audit for the T0 SER ROI anchor.

The first Med-JEPA ROI must be anchored using information available at baseline, not a
later-treatment mask.  This program therefore reads only the T0 SER/SEG pairs of the
595 complete I-SPY2 trajectories, validates their physical mapping, and measures the
physical extent of voxels for which the collection-specific inverse analysis mask has
``raw == 0``.  It writes no images, masks, patient identifiers, paths, dates, UIDs, or
per-patient measurements.  Its output is used to choose a fixed 112-cube field of view
before implementing longitudinal rigid registration and ROI conversion.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import logging
import math
import os
import statistics
import sys
import tempfile
import threading
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from med_jepa import ispy2_dce_geometry_probe as production
from med_jepa.ftv_release import FTVReleaseError, validate_ftv_release
from med_jepa.ispy2_ftv_audit import FTVAuditError, _select_seg_path
from med_jepa.ispy2_pixel_probe import (
    MASK_SEMANTICS,
    POSITION_TOLERANCE_MM,
    _safe_float,
    match_positions_one_to_one,
)
from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_seg_mapping_probe import (
    SEGMENTATION_STORAGE_UID,
    _float_vector,
    _frame_position,
    _frame_segment_numbers,
    _sequence,
    _vectors_close,
)
from med_jepa.ispy2_series_probe import _inside, _raise_stale


SCHEMA_VERSION = "med-jepa-ispy2-roi-anchor-audit-v2"
TIMEPOINT = "T0"
DEFAULT_ROI_SHAPE = (112, 112, 112)
DEFAULT_MARGIN_MM = 20.0
DEFAULT_CANDIDATE_SPACINGS_MM = (1.0, 1.25, 1.5, 2.0)
LOCKED_CANONICAL_SPACING_MM = (2.1, 2.1, 2.1)
CANONICAL_ORIENTATION = "RAS"
SER_EXTRA_HEADER_TAGS: tuple[Any, ...] = (
    "RescaleSlope",
    "RescaleIntercept",
    "ModalityLUTSequence",
)
ANCHOR_CENTER_RULE = "t0_inverse_mask_bbox_midpoint_patient_space_v1"
OUT_OF_BOUNDS_PADDING_POLICY = "zero_unitless_ser_after_dicom_rescale_v1"
# ``redirect_stderr``, ``logging.disable`` and the warnings filter all mutate
# process-global state.  ROI conversion may call these readers from patient worker
# threads, so the complete suppression scope must be serialized.  An RLock permits
# a guarded higher-level reader to call ``_quiet_dcmread`` safely.
_DICOM_DIAGNOSTIC_LOCK = threading.RLock()


class ROIAnchorAuditError(ValueError):
    """Fail-closed, non-identifying failure code for the anchor audit."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@contextlib.contextmanager
def _suppress_dicom_diagnostics() -> Iterable[None]:
    """Suppress path-bearing parser diagnostics without cross-thread races."""

    with _DICOM_DIAGNOSTIC_LOCK:
        previous_logging_disable = logging.root.manager.disable
        with warnings.catch_warnings(), contextlib.redirect_stderr(io.StringIO()):
            warnings.simplefilter("ignore")
            logging.disable(logging.CRITICAL + 1)
            try:
                yield
            finally:
                logging.disable(previous_logging_disable)


def _quiet_dcmread(path: Path, **kwargs: Any) -> Any:
    """Read a DICOM object without allowing parser diagnostics into cluster logs.

    The audit emits aggregate-only JSON.  pydicom normally raises structured
    exceptions, which callers convert to allowlisted categories, but some optional
    handlers can also issue warnings or logger output containing source paths.  Keep
    that incidental output private to this process.
    """

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise ROIAnchorAuditError("pydicom_dependency_unavailable") from error
    with _suppress_dicom_diagnostics():
        return pydicom.dcmread(path, **kwargs)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        _raise_stale(error)
        raise ROIAnchorAuditError("evidence_file_read_error") from None
    return digest.hexdigest()


def _quantile(values: Sequence[float], probability: float) -> float:
    """Linearly interpolate a finite sequence without a NumPy version dependency."""

    if not values or not 0.0 <= probability <= 1.0:
        raise ValueError("quantile requires values and a probability in [0,1]")
    ordered = sorted(float(value) for value in values)
    if any(not math.isfinite(value) for value in ordered):
        raise ValueError("quantile values must be finite")
    if len(ordered) == 1:
        return ordered[0]
    location = probability * (len(ordered) - 1)
    lower = math.floor(location)
    upper = math.ceil(location)
    if lower == upper:
        return ordered[lower]
    fraction = location - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _numeric_summary(values: Iterable[float]) -> dict[str, float]:
    observed = [float(value) for value in values]
    if not observed or any(not math.isfinite(value) or value < 0.0 for value in observed):
        raise ValueError("summary values must be finite and non-negative")
    return {
        "min": min(observed),
        "p05": _quantile(observed, 0.05),
        "median": statistics.median(observed),
        "p95": _quantile(observed, 0.95),
        "max": max(observed),
    }


def _finite_summary(values: Iterable[float]) -> dict[str, float]:
    """Summarize finite values that may legitimately be negative (for margins)."""

    observed = [float(value) for value in values]
    if not observed or any(not math.isfinite(value) for value in observed):
        raise ValueError("summary values must be finite")
    return {
        "min": min(observed),
        "p05": _quantile(observed, 0.05),
        "median": statistics.median(observed),
        "p95": _quantile(observed, 0.95),
        "max": max(observed),
    }


def _strict_ser_headers(ser_dir: Path, *, dicom_root: Path) -> list[Any]:
    """Read one strict SER series once, including rescale metadata."""

    try:
        # Reuse the strict directory/sidecar contract used by the DCE gate while
        # requesting the ROI-specific tags in the same pass.  This avoids a second
        # ~80-slice header traversal for every T0 series on NFS.
        with _suppress_dicom_diagnostics():
            headers, _counts = production._read_headers(
                ser_dir,
                dicom_root=dicom_root,
                additional_tags=SER_EXTRA_HEADER_TAGS,
            )
    except StaleFileHandleError:
        raise
    except production.DCEGeometryProbeError as error:
        raise ROIAnchorAuditError(error.code) from None
    except OSError as error:
        _raise_stale(error)
        raise ROIAnchorAuditError("ser_directory_read_error") from None
    try:
        production._validate_classic_single_series(headers, kind="ser")
    except production.DCEGeometryProbeError as error:
        raise ROIAnchorAuditError(error.code) from None
    return headers


def _ser_geometry_and_rescale(headers: Sequence[Any]) -> dict[str, Any]:
    if not headers:
        raise ROIAnchorAuditError("missing_ser_headers")
    try:
        geometry = production.build_stack_geometry(headers)
    except production.DCEGeometryProbeError as error:
        raise ROIAnchorAuditError(error.code) from None
    spacing = tuple(float(value) for value in geometry["pixel_spacing_mm"])
    orientation = tuple(float(value) for value in geometry["orientation"])
    positions = [
        _float_vector(getattr(item, "ImagePositionPatient", None), 3)
        for item in headers
    ]
    if any(position is None for position in positions):  # defended by geometry helper
        raise ROIAnchorAuditError("invalid_image_position")
    concrete_positions = [position for position in positions if position is not None]
    if any(_sequence(item, "ModalityLUTSequence") for item in headers):
        raise ROIAnchorAuditError("unsupported_ser_modality_lut")
    presence = {
        (
            getattr(item, "RescaleSlope", None) is not None,
            getattr(item, "RescaleIntercept", None) is not None,
        )
        for item in headers
    }
    if len(presence) != 1 or any(left != right for left, right in presence):
        raise ROIAnchorAuditError("incomplete_or_nonuniform_ser_rescale")
    try:
        slopes = {
            _safe_float(getattr(item, "RescaleSlope", None), default=1.0)
            for item in headers
        }
        intercepts = {
            _safe_float(getattr(item, "RescaleIntercept", None), default=0.0)
            for item in headers
        }
    except Exception:
        raise ROIAnchorAuditError("invalid_ser_rescale") from None
    if len(slopes) != 1 or len(intercepts) != 1:
        raise ROIAnchorAuditError("nonuniform_ser_rescale")
    slope = next(iter(slopes))
    intercept = next(iter(intercepts))
    if slope == 0.0:
        raise ROIAnchorAuditError("zero_ser_rescale_slope")
    unit_normal = tuple(float(value) for value in geometry["_normal_axis"])
    projections = [
        sum(float(component) * unit for component, unit in zip(position, unit_normal))
        for position in concrete_positions
    ]
    if len(set(projections)) != len(projections):
        raise ROIAnchorAuditError("duplicate_ser_slice_positions")
    order = tuple(sorted(range(len(headers)), key=lambda index: projections[index]))
    return {
        "rows": int(geometry["rows"]),
        "columns": int(geometry["columns"]),
        "spacing": spacing,
        "orientation": orientation,
        "positions": concrete_positions,
        "slice_spacing_mm": float(geometry["slice_spacing_mm"]),
        "slope": slope,
        "intercept": intercept,
        "rescale_defaulted": presence == {(False, False)},
        "order": order,
        "projection": projections,
        "frame_of_reference": str(geometry["_frame_of_reference_uid"]),
        "normal": unit_normal,
    }


def _read_seg(mask_dir: Path, *, dicom_root: Path) -> Any:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise ROIAnchorAuditError("pydicom_dependency_unavailable") from error
    try:
        path = _select_seg_path(mask_dir, dicom_root=dicom_root)
        return _quiet_dcmread(path, force=False)
    except StaleFileHandleError:
        raise
    except FTVAuditError as error:
        raise ROIAnchorAuditError(error.code) from None
    except OSError as error:
        _raise_stale(error)
        raise ROIAnchorAuditError("seg_object_read_error") from None
    except Exception:
        raise ROIAnchorAuditError("seg_object_parse_error") from None


def _functional_group_vector(
    container: Any,
    *,
    sequence_name: str,
    attribute: str,
    length: int,
) -> tuple[float, ...] | None:
    """Return one valid functional-group vector, or ``None`` when absent.

    A present but malformed/mixed declaration is a data error rather than permission
    to fall back to another frame or the shared group.
    """

    items = _sequence(container, sequence_name)
    if not items:
        return None
    values = [_float_vector(getattr(item, attribute, None), length) for item in items]
    if any(value is None for value in values):
        raise ROIAnchorAuditError("invalid_seg_inplane_geometry")
    concrete = [value for value in values if value is not None]
    if len(set(concrete)) != 1:
        raise ROIAnchorAuditError("nonuniform_seg_inplane_geometry")
    return concrete[0]


def _resolved_frame_vectors(
    shared: Any,
    frames: Sequence[Any],
    *,
    sequence_name: str,
    attribute: str,
    length: int,
) -> tuple[tuple[float, ...], ...]:
    """Resolve shared/per-frame geometry and require every frame to be valid."""

    shared_value = _functional_group_vector(
        shared,
        sequence_name=sequence_name,
        attribute=attribute,
        length=length,
    )
    values: list[tuple[float, ...]] = []
    for frame in frames:
        frame_value = _functional_group_vector(
            frame,
            sequence_name=sequence_name,
            attribute=attribute,
            length=length,
        )
        if frame_value is None:
            if shared_value is None:
                raise ROIAnchorAuditError("missing_seg_inplane_geometry")
            frame_value = shared_value
        values.append(frame_value)
    if not values:
        raise ROIAnchorAuditError("missing_seg_inplane_geometry")
    return tuple(values)


def _exact_seg_to_ser_mapping(
    seg: Any, ser: Mapping[str, Any]
) -> tuple[tuple[int, ...], Any]:
    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise ROIAnchorAuditError("numpy_dependency_unavailable") from error
    if (
        str(getattr(seg, "SOPClassUID", "")) != SEGMENTATION_STORAGE_UID
        or str(getattr(seg, "Modality", "")).strip().upper() != "SEG"
    ):
        raise ROIAnchorAuditError("selected_object_is_not_seg")
    try:
        frame_count = int(seg.NumberOfFrames)
        rows = int(seg.Rows)
        columns = int(seg.Columns)
    except (AttributeError, TypeError, ValueError) as error:
        raise ROIAnchorAuditError("invalid_seg_matrix_or_frames") from error
    if (
        frame_count != len(ser["positions"])
        or rows != ser["rows"]
        or columns != ser["columns"]
    ):
        raise ROIAnchorAuditError("seg_ser_matrix_or_frame_count_mismatch")
    if str(getattr(seg, "FrameOfReferenceUID", "")).strip() != ser["frame_of_reference"]:
        raise ROIAnchorAuditError("seg_frame_of_reference_mismatch")
    frames = _sequence(seg, "PerFrameFunctionalGroupsSequence")
    shared_items = _sequence(seg, "SharedFunctionalGroupsSequence")
    if len(frames) != frame_count or len(shared_items) != 1:
        raise ROIAnchorAuditError("seg_functional_group_count_mismatch")
    shared = shared_items[0]
    positions = [_frame_position(frame) for frame in frames]
    if any(position is None for position in positions):
        raise ROIAnchorAuditError("missing_seg_frame_position")
    try:
        mapping = match_positions_one_to_one(
            [position for position in positions if position is not None],
            ser["positions"],
            tolerance_mm=POSITION_TOLERANCE_MM,
        )
    except ValueError:
        raise ROIAnchorAuditError("seg_ser_position_mapping_mismatch") from None
    spacings = _resolved_frame_vectors(
        shared,
        frames,
        sequence_name="PixelMeasuresSequence",
        attribute="PixelSpacing",
        length=2,
    )
    orientations = _resolved_frame_vectors(
        shared,
        frames,
        sequence_name="PlaneOrientationSequence",
        attribute="ImageOrientationPatient",
        length=6,
    )
    if any(not _vectors_close(value, ser["spacing"]) for value in spacings) or any(
        not _vectors_close(value, ser["orientation"]) for value in orientations
    ):
        raise ROIAnchorAuditError("seg_ser_inplane_geometry_mismatch")
    segment_items = _sequence(seg, "SegmentSequence")
    try:
        declared_segment_numbers = {
            int(getattr(item, "SegmentNumber", None)) for item in segment_items
        }
    except (TypeError, ValueError):
        raise ROIAnchorAuditError("seg_segment_contract_mismatch") from None
    frame_segment_numbers = [_frame_segment_numbers(frame) for frame in frames]
    if (
        len(segment_items) != 1
        or len(declared_segment_numbers) != 1
        or any(len(numbers) != 1 for numbers in frame_segment_numbers)
        or any(numbers != declared_segment_numbers for numbers in frame_segment_numbers)
    ):
        raise ROIAnchorAuditError("seg_segment_contract_mismatch")
    try:
        fractional_value = int(getattr(seg, "MaximumFractionalValue", 0))
    except (TypeError, ValueError):
        raise ROIAnchorAuditError("unexpected_ispy2_analysis_mask_declaration") from None
    if (
        str(getattr(seg, "SegmentationType", "")).strip().upper() != "FRACTIONAL"
        or str(getattr(seg, "SegmentationFractionalType", "")).strip().upper()
        != "OCCUPANCY"
        or fractional_value != 255
    ):
        raise ROIAnchorAuditError("unexpected_ispy2_analysis_mask_declaration")
    try:
        pixels = seg.pixel_array
    except Exception:
        raise ROIAnchorAuditError("seg_pixel_decode_error") from None
    if tuple(pixels.shape) != (frame_count, rows, columns) or pixels.dtype.kind not in "biu":
        raise ROIAnchorAuditError("unexpected_seg_pixel_shape")
    if not np.isfinite(pixels).all() or int(pixels.min()) < 0 or int(pixels.max()) > 255:
        raise ROIAnchorAuditError("invalid_seg_pixel_values")
    return mapping, pixels


def anchor_extent_from_inverse_mask(
    pixels: Any,
    *,
    frame_to_ser: Sequence[int],
    ser: Mapping[str, Any],
) -> dict[str, float | int]:
    """Measure the T0 ``raw == 0`` extent without assigning any meaning to other bits."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise ROIAnchorAuditError("numpy_dependency_unavailable") from error
    if pixels.ndim != 3 or pixels.shape[0] != len(frame_to_ser):
        raise ROIAnchorAuditError("unexpected_seg_pixel_shape")
    included = pixels == 0
    included_count = int(np.count_nonzero(included))
    if included_count < 1:
        raise ROIAnchorAuditError("empty_t0_inverse_mask")
    order = tuple(int(value) for value in ser["order"])
    rank = {ser_index: index for index, ser_index in enumerate(order)}
    frame_indices = np.flatnonzero(np.any(included, axis=(1, 2)))
    if frame_indices.size == 0:
        raise ROIAnchorAuditError("empty_t0_inverse_mask")
    min_row = pixels.shape[1]
    max_row = -1
    min_column = pixels.shape[2]
    max_column = -1
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
            raise ROIAnchorAuditError("invalid_seg_ser_frame_mapping") from None
    spacing = ser["spacing"]
    projection = ser["projection"]
    if not ranks or min_row > max_row or min_column > max_column:
        raise ROIAnchorAuditError("empty_t0_inverse_mask")
    first_ser_index = order[min(ranks)]
    last_ser_index = order[max(ranks)]
    slice_extent = (
        abs(float(projection[last_ser_index]) - float(projection[first_ser_index]))
        + float(ser["slice_spacing_mm"])
    )
    row_extent = (max_row - min_row + 1) * float(spacing[0])
    column_extent = (max_column - min_column + 1) * float(spacing[1])
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in (row_extent, column_extent, slice_extent)
    ):
        raise ROIAnchorAuditError("invalid_t0_inverse_mask_extent")
    try:
        column_axis = tuple(float(value) for value in ser["orientation"][:3])
        row_axis = tuple(float(value) for value in ser["orientation"][3:])
        normal_axis = tuple(float(value) for value in ser["normal"])
        ordered_origin = tuple(float(value) for value in ser["positions"][order[0]])
    except (KeyError, TypeError, ValueError, IndexError):
        raise ROIAnchorAuditError("missing_ser_physical_anchor_geometry") from None
    if (
        len(column_axis) != 3
        or len(row_axis) != 3
        or len(normal_axis) != 3
        or len(ordered_origin) != 3
        or any(
            not math.isfinite(value)
            for value in (*column_axis, *row_axis, *normal_axis, *ordered_origin)
        )
    ):
        raise ROIAnchorAuditError("invalid_ser_physical_anchor_geometry")
    column_spacing = float(spacing[1])
    row_spacing = float(spacing[0])
    center_column = (min_column + max_column) / 2.0
    center_row = (min_row + max_row) / 2.0
    center_rank = (min(ranks) + max(ranks)) / 2.0
    center = tuple(
        ordered_origin[axis]
        + center_column * column_spacing * column_axis[axis]
        + center_row * row_spacing * row_axis[axis]
        + center_rank * float(ser["slice_spacing_mm"]) * normal_axis[axis]
        for axis in range(3)
    )
    # Measure exact raw-zero voxel support about the prescribed native-bbox midpoint,
    # projected into canonical RAS axes.  RAS and LPS differ only by x/y signs, so the
    # three absolute half-extents are identical; no individual coordinate is emitted.
    voxel_half_width = tuple(
        0.5
        * (
            abs(column_axis[axis]) * column_spacing
            + abs(row_axis[axis]) * row_spacing
            + abs(normal_axis[axis]) * float(ser["slice_spacing_mm"])
        )
        for axis in range(3)
    )
    canonical_half_extent = [0.0, 0.0, 0.0]
    for frame_index in frame_indices.tolist():
        frame_rows, frame_columns = np.nonzero(included[frame_index])
        ser_index = int(frame_to_ser[frame_index])
        try:
            position = tuple(float(value) for value in ser["positions"][ser_index])
        except (KeyError, TypeError, ValueError, IndexError):
            raise ROIAnchorAuditError("invalid_seg_ser_frame_mapping") from None
        if len(position) != 3 or any(not math.isfinite(value) for value in position):
            raise ROIAnchorAuditError("invalid_seg_ser_frame_mapping")
        for axis in range(3):
            coordinates = (
                position[axis]
                + frame_columns * column_spacing * column_axis[axis]
                + frame_rows * row_spacing * row_axis[axis]
            )
            canonical_half_extent[axis] = max(
                canonical_half_extent[axis],
                float(np.max(np.abs(coordinates - center[axis])))
                + voxel_half_width[axis],
            )
    if any(not math.isfinite(value) or value <= 0.0 for value in canonical_half_extent):
        raise ROIAnchorAuditError("invalid_canonical_anchor_extent")
    return {
        "included_voxels": included_count,
        "row_extent_mm": row_extent,
        "column_extent_mm": column_extent,
        "slice_extent_mm": slice_extent,
        "largest_extent_mm": max(row_extent, column_extent, slice_extent),
        "slice_count": len(set(ranks)),
        "canonical_ras_half_extent_x_mm": canonical_half_extent[0],
        "canonical_ras_half_extent_y_mm": canonical_half_extent[1],
        "canonical_ras_half_extent_z_mm": canonical_half_extent[2],
    }


def inspect_t0_anchor_pair(ser_dir: Path, mask_dir: Path, *, dicom_root: Path) -> dict[str, Any]:
    """Read one T0 pair and return only scalar geometry needed for aggregate audit."""

    ser_headers = _strict_ser_headers(ser_dir, dicom_root=dicom_root)
    ser = _ser_geometry_and_rescale(ser_headers)
    seg = _read_seg(mask_dir, dicom_root=dicom_root)
    mapping, pixels = _exact_seg_to_ser_mapping(seg, ser)
    extent = anchor_extent_from_inverse_mask(pixels, frame_to_ser=mapping, ser=ser)
    # This audit intentionally does not read SER PixelData.  The full converter must
    # still validate the observed rescaled value range before it writes an ROI.
    return {
        **extent,
        "ser_rescale_slope": float(ser["slope"]),
        "ser_rescale_intercept": float(ser["intercept"]),
        "ser_rescale_defaulted": bool(ser["rescale_defaulted"]),
        "slice_spacing_mm": float(ser["slice_spacing_mm"]),
        "mapping_status": "geometry_exact",
        "mask_semantics": MASK_SEMANTICS,
    }


def summarize_anchor_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempted: int,
    failures: Mapping[str, int],
    margin_mm: float,
    roi_shape: Sequence[int],
) -> dict[str, Any]:
    if attempted != len(rows) + sum(int(value) for value in failures.values()):
        raise ValueError("anchor-audit accounting is inconsistent")
    if not rows:
        return {
            "pairs_attempted": attempted,
            "pairs_passed": 0,
            "pairs_failed": sum(int(value) for value in failures.values()),
            "failure_reasons": dict(sorted((str(key), int(value)) for key, value in failures.items())),
            "distributions": {},
            "extent_mm": {},
            "crop_feasibility": {},
        }
    extents = {
        name: [float(row[name]) for row in rows]
        for name in ("row_extent_mm", "column_extent_mm", "slice_extent_mm", "largest_extent_mm")
    }
    canonical_half_extent = {
        axis: [float(row[f"canonical_ras_half_extent_{axis}_mm"]) for row in rows]
        for axis in ("x", "y", "z")
    }
    shape = tuple(int(value) for value in roi_shape)
    if len(shape) != 3 or min(shape) < 1 or not math.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("invalid ROI crop specification")
    required_largest = [value + 2.0 * margin_mm for value in extents["largest_extent_mm"]]
    candidate_coverage = {
        f"{spacing:g}": {
            "field_of_view_mm": min(shape) * spacing,
            "pairs_covered": sum(value <= min(shape) * spacing for value in required_largest),
            "pairs_exceeding_fov": sum(value > min(shape) * spacing for value in required_largest),
        }
        for spacing in DEFAULT_CANDIDATE_SPACINGS_MM
    }
    canonical_spacing = tuple(float(value) for value in LOCKED_CANONICAL_SPACING_MM)
    if len(canonical_spacing) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in canonical_spacing
    ):
        raise ValueError("invalid canonical crop spacing")
    canonical_half_fov = tuple(
        shape[axis] * canonical_spacing[axis] / 2.0 for axis in range(3)
    )
    canonical_margins = [
        min(
            canonical_half_fov[axis] - canonical_half_extent[axis_name][row_index]
            for axis, axis_name in enumerate(("x", "y", "z"))
        )
        for row_index in range(len(rows))
    ]
    canonical_support_covered = sum(margin >= 0.0 for margin in canonical_margins)
    canonical_margin_covered = sum(
        margin >= margin_mm for margin in canonical_margins
    )
    return {
        "pairs_attempted": attempted,
        "pairs_passed": len(rows),
        "pairs_failed": sum(int(value) for value in failures.values()),
        "failure_reasons": dict(sorted((str(key), int(value)) for key, value in failures.items())),
        "distributions": {
            field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
            for field in ("mapping_status", "mask_semantics", "ser_rescale_defaulted")
        },
        "extent_mm": {name: _numeric_summary(values) for name, values in extents.items()},
        "included_voxels": _numeric_summary(float(row["included_voxels"]) for row in rows),
        "slice_count": _numeric_summary(float(row["slice_count"]) for row in rows),
        "crop_feasibility": {
            "roi_shape": list(shape),
            "t0_margin_mm": margin_mm,
            "required_largest_extent_with_margin_mm": _numeric_summary(required_largest),
            "minimum_isotropic_spacing_mm": _numeric_summary(
                value / min(shape) for value in required_largest
            ),
            "candidate_isotropic_spacing_coverage": candidate_coverage,
        },
        "canonical_ras_grid_feasibility": {
            "orientation": CANONICAL_ORIENTATION,
            "roi_shape": list(shape),
            "spacing_mm": list(canonical_spacing),
            "field_of_view_mm": [
                shape[axis] * canonical_spacing[axis] for axis in range(3)
            ],
            "half_extent_about_t0_bbox_center_mm": {
                axis: _numeric_summary(values)
                for axis, values in canonical_half_extent.items()
            },
            "minimum_edge_margin_mm": _finite_summary(canonical_margins),
            "required_margin_mm": margin_mm,
            "pairs_with_support_inside_grid": canonical_support_covered,
            "pairs_outside_grid": len(rows) - canonical_support_covered,
            "pairs_with_required_margin": canonical_margin_covered,
            "pairs_below_required_margin": len(rows) - canonical_margin_covered,
        },
    }


def _resolve_manifest_dicom_directory(
    raw_value: str,
    *,
    trajectory: Path,
    dicom_root: Path,
) -> Path:
    candidate = Path(raw_value).expanduser()
    try:
        resolved = (
            candidate.resolve()
            if candidate.is_absolute()
            else (trajectory.parent / candidate).resolve()
        )
    except OSError as error:
        _raise_stale(error)
        raise ROIAnchorAuditError("trajectory_path_resolution_error") from None
    if not _inside(resolved, dicom_root):
        raise ROIAnchorAuditError("trajectory_path_outside_approved_root")
    return resolved


def _read_pinned_t0_pairs(
    root: Path,
    trajectory: Path,
    *,
    expected_patient_ids: set[str],
    expected_patients: int,
) -> tuple[list[tuple[str, Path, Path]], str]:
    """Build selected T0 pairs from the same hash-pinned manifest byte snapshot.

    T1/T3 strings establish the declared complete cohort but are never resolved or
    read here.  The returned patient keys remain in-memory only, so they can be joined
    to the governed FTV release without leaking to the aggregate report.
    """

    try:
        fieldnames, manifest_rows, manifest_sha = production._read_pinned_csv_rows(
            trajectory,
            expected_sha256=production.APPROVED_TRAJECTORY_SHA256,
            hash_error_code="trajectory_hash_mismatch",
            read_error_code="trajectory_read_error",
            schema_error_code="trajectory_schema_mismatch",
        )
    except production.DCEGeometryProbeError as error:
        raise ROIAnchorAuditError(error.code) from None
    required = {"PatientID", "tp_label", "ser_dir", "mask_dir"}
    if not required.issubset(fieldnames):
        raise ROIAnchorAuditError("trajectory_schema_mismatch")

    visits: dict[str, dict[str, tuple[str, str]]] = {}
    for row in manifest_rows:
        timepoint = str(row.get("tp_label") or "").strip().upper()
        if timepoint not in production.TIMEPOINTS:
            continue
        patient_id = str(row.get("PatientID") or "").strip()
        if not production._PATIENT_ID.fullmatch(patient_id):
            raise ROIAnchorAuditError("invalid_patient_identifier")
        ser_dir = str(row.get("ser_dir") or "").strip()
        mask_dir = str(row.get("mask_dir") or "").strip()
        if not ser_dir or not mask_dir:
            continue
        if timepoint in visits.setdefault(patient_id, {}):
            raise ROIAnchorAuditError("duplicate_patient_timepoint")
        visits[patient_id][timepoint] = (ser_dir, mask_dir)

    required_timepoints = set(production.TIMEPOINTS)
    complete_patient_ids = {
        patient_id
        for patient_id, patient_visits in visits.items()
        if required_timepoints.issubset(patient_visits)
    }
    if (
        len(complete_patient_ids) != expected_patients
        or complete_patient_ids != expected_patient_ids
    ):
        raise ROIAnchorAuditError("complete_trajectory_cohort_mismatch")

    dicom_root = (root / "dicom").resolve()
    try:
        if not dicom_root.is_dir():
            raise ROIAnchorAuditError("dicom_root_unavailable")
    except OSError as error:
        _raise_stale(error)
        raise ROIAnchorAuditError("dicom_root_unavailable") from None
    t0_pairs: list[tuple[str, Path, Path]] = []
    used_directories: set[Path] = set()
    for patient_id in sorted(complete_patient_ids):
        raw_ser, raw_mask = visits[patient_id][TIMEPOINT]
        ser_dir = _resolve_manifest_dicom_directory(
            raw_ser, trajectory=trajectory, dicom_root=dicom_root
        )
        mask_dir = _resolve_manifest_dicom_directory(
            raw_mask, trajectory=trajectory, dicom_root=dicom_root
        )
        if ser_dir == mask_dir or ser_dir in used_directories or mask_dir in used_directories:
            raise ROIAnchorAuditError("duplicate_series_directory_reference")
        used_directories.update((ser_dir, mask_dir))
        t0_pairs.append((patient_id, ser_dir, mask_dir))
    return t0_pairs, manifest_sha


def _ftv_t0_counts(ftv_release: Mapping[str, Any]) -> dict[str, int]:
    """Return verified per-patient T0 counts for an in-memory lineage check only."""

    rows = ftv_release.get("ftv_csv_rows")
    if not isinstance(rows, list):
        raise ROIAnchorAuditError("governed_ftv_release_invalid")
    counts: dict[str, int] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ROIAnchorAuditError("governed_ftv_release_invalid")
        patient_id = str(row.get("PatientID") or "").strip()
        try:
            count = int(str(row.get("included_voxels_T0") or ""))
        except ValueError:
            raise ROIAnchorAuditError("governed_ftv_release_invalid") from None
        if not production._PATIENT_ID.fullmatch(patient_id) or count < 0 or patient_id in counts:
            raise ROIAnchorAuditError("governed_ftv_release_invalid")
        counts[patient_id] = count
    return counts


def run_anchor_audit(
    root: str | Path,
    *,
    trajectory_path: str | Path,
    ftv_csv_path: str | Path,
    ftv_audit_report_path: str | Path,
    expected_patients: int = production.EXPECTED_COMPLETE_PATIENTS,
    margin_mm: float = DEFAULT_MARGIN_MM,
    roi_shape: Sequence[int] = DEFAULT_ROI_SHAPE,
) -> dict[str, Any]:
    if expected_patients != production.EXPECTED_COMPLETE_PATIENTS:
        raise ROIAnchorAuditError("expected_patient_contract_mismatch")
    if not math.isfinite(margin_mm) or margin_mm < 0.0:
        raise ROIAnchorAuditError("invalid_t0_margin_mm")
    root_path = Path(root).expanduser().resolve()
    trajectory = Path(trajectory_path).expanduser().resolve()
    try:
        ftv_release = validate_ftv_release(ftv_csv_path, ftv_audit_report_path)
    except FTVReleaseError:
        raise ROIAnchorAuditError("governed_ftv_release_invalid") from None
    patient_ids, split_counts, split_sha = production._read_split(root_path, expected_patients)
    t0_pairs, trajectory_sha = _read_pinned_t0_pairs(
        root_path,
        trajectory,
        expected_patient_ids=patient_ids,
        expected_patients=expected_patients,
    )
    ftv_t0_counts = _ftv_t0_counts(ftv_release)
    if set(ftv_t0_counts) != patient_ids:
        raise ROIAnchorAuditError("governed_ftv_patient_cohort_mismatch")
    dicom_root = (root_path / "dicom").resolve()
    rows: list[dict[str, Any]] = []
    failures: Counter[str] = Counter()
    for number, (patient_id, ser_dir, mask_dir) in enumerate(t0_pairs, start=1):
        try:
            row = inspect_t0_anchor_pair(ser_dir, mask_dir, dicom_root=dicom_root)
            if int(row["included_voxels"]) != ftv_t0_counts[patient_id]:
                failures["t0_inverse_mask_ftv_count_mismatch"] += 1
            else:
                rows.append(row)
        except StaleFileHandleError:
            raise
        except ROIAnchorAuditError as error:
            failures[error.code] += 1
        except OSError as error:
            _raise_stale(error)
            failures["dicom_io_error"] += 1
        except Exception:
            failures["unexpected_anchor_inspection_error"] += 1
        if number % 25 == 0 or number == len(t0_pairs):
            print(
                json.dumps(
                    {
                        "event": "roi_anchor_progress",
                        "t0_pairs_complete": number,
                        "t0_pairs_total": len(t0_pairs),
                        "pixel_data_read": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    summary = summarize_anchor_rows(
        rows,
        attempted=len(t0_pairs),
        failures=failures,
        margin_mm=margin_mm,
        roi_shape=roi_shape,
    )
    canonical_grid = summary.get("canonical_ras_grid_feasibility", {})
    canonical_margin_passed = (
        canonical_grid.get("pairs_with_required_margin") == expected_patients
        and canonical_grid.get("pairs_below_required_margin") == 0
    )
    status = (
        "ok_anchor_geometry"
        if not failures and len(rows) == expected_patients and canonical_margin_passed
        else "blocked_anchor_geometry"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": status,
        "selection": {"timepoint": TIMEPOINT, "anchor_source": "t0_inverse_mask_only"},
        "cohort": {
            "patients_expected": expected_patients,
            "t0_pairs_expected": expected_patients,
            "split_counts": split_counts,
        },
        "summary": summary,
        "mask_contract": {
            "semantics": MASK_SEMANTICS,
            "included_condition": "raw_value_equals_zero",
            "generic_fractional_occupancy_interpretation_used": False,
        },
        "evidence": {
            "trajectory_manifest_sha256": trajectory_sha,
            "split_sha256": split_sha,
            "ftv_csv_sha256": ftv_release["ftv_csv_sha256"],
            "ftv_audit_report_sha256": ftv_release["ftv_audit_report_sha256"],
            "converter_source_sha256": _sha256(Path(__file__)),
        },
        "pixel_data_read": True,
        "aggregate_only": True,
        "identifiers_paths_uids_dates_or_per_patient_measurements_emitted": False,
        "crop_anchor_contract": {
            "center_rule": ANCHOR_CENTER_RULE,
            "extent_source": "t0_inverse_mask_raw_value_equals_zero_bbox",
            "out_of_bounds_padding": OUT_OF_BOUNDS_PADDING_POLICY,
            "canonical_orientation": CANONICAL_ORIENTATION,
            "canonical_spacing_mm": list(LOCKED_CANONICAL_SPACING_MM),
        },
        "next_gate": {
            "required": "lock_bbox_centered_t0_roi_grid_then_validate_rigid_t1_t3_registration",
            "can_create_roi_release": False,
        },
    }


def _write_exclusive(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        os.chmod(handle.name, 0o600)
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        raise ROIAnchorAuditError("output_exists_refusing_overwrite") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--ftv-csv", required=True)
    parser.add_argument("--ftv-audit-report", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-patients", type=int, default=production.EXPECTED_COMPLETE_PATIENTS)
    parser.add_argument("--t0-margin-mm", type=float, default=DEFAULT_MARGIN_MM)
    parser.add_argument("--roi-shape", type=int, nargs=3, default=DEFAULT_ROI_SHAPE)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_anchor_audit(
            args.root,
            trajectory_path=args.trajectory,
            ftv_csv_path=args.ftv_csv,
            ftv_audit_report_path=args.ftv_audit_report,
            expected_patients=args.expected_patients,
            margin_mm=args.t0_margin_mm,
            roi_shape=args.roi_shape,
        )
        content = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
        _write_exclusive(Path(args.output).expanduser(), content)
    except StaleFileHandleError:
        print("stale_file_handle", file=sys.stderr)
        return 2
    except (ROIAnchorAuditError, production.DCEGeometryProbeError) as error:
        print(getattr(error, "code", "roi_anchor_audit_error"), file=sys.stderr)
        return 3
    except OSError:
        print("roi_anchor_audit_os_error", file=sys.stderr)
        return 3
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "status": report["status"],
                "cohort": report["cohort"],
                "summary": report["summary"],
                "pixel_data_read": True,
            },
            sort_keys=True,
        )
    )
    return 0 if report["status"] == "ok_anchor_geometry" else 3


if __name__ == "__main__":
    raise SystemExit(main())
