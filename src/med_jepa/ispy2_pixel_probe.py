"""Read-only, aggregate pixel smoke probe for ISPY2 SER and DICOM SEG pairs.

The probe is intentionally narrower than an ROI converter.  It selects the same
small, deterministic sample of complete T0/T1/T3 trajectories at every timepoint,
maps SEG frames to SER slices by patient-space position, decodes pixels, and emits
only aggregate QC.  It never writes image data and never emits patient, path, study,
series, SOP-instance, or frame-of-reference identifiers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_seg_mapping_probe import (
    SEGMENTATION_STORAGE_UID,
    _float_vector,
    _frame_position,
    _frame_segment_numbers,
    _nested_vector,
    _sequence,
    _vectors_close,
    audit_manifest_pairs,
)
from med_jepa.ispy2_series_probe import _raise_stale


TIMEPOINTS = ("T0", "T1", "T3")
POSITION_TOLERANCE_MM = 0.01
SER_VALUE_RANGE = (0.0, 3.0)
SER_RANGE_TOLERANCE = 1e-4
MASK_SEMANTICS_RELATIVE_PATH = Path(
    "clinical/Analysis-mask-files-description.v20211020.docx"
)
APPROVED_MASK_SEMANTICS_SHA256 = (
    "0115d28ce5d11cc134f663ac111731a6df85b08945c9be17a9ac209b36cecb47"
)
MASK_SEMANTICS = "inverse_bit_encoded_ftv_analysis_mask_zero_is_included"
FTV_EXCLUSION_BITS = {
    "pe_threshold": 1,
    "mnc_filter": 2,
    "background_mask": 8,
    "manual_voi": 32,
    "omit_regions": 64,
}
FTV_ALLOWED_BITS = sum(FTV_EXCLUSION_BITS.values())


class PixelProbeError(ValueError):
    """A fail-closed error whose message is a non-identifying category."""

    def __init__(
        self,
        code: str,
        *,
        safe_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.safe_metadata = dict(safe_metadata or {})


@dataclass(frozen=True)
class TransferSyntax:
    name: str
    compressed: bool


def evenly_spaced_indices(total: int, count: int) -> tuple[int, ...]:
    """Return deterministic indices spanning a cohort without exposing its keys."""

    if total < 1 or count < 1 or count > total:
        raise ValueError("sample count must be positive and no larger than cohort")
    if count == 1:
        return (total // 2,)
    # Integer arithmetic avoids platform-dependent rounding and always includes
    # both ends.  With count <= total the returned indices are strictly increasing.
    return tuple(index * (total - 1) // (count - 1) for index in range(count))


def match_positions_one_to_one(
    observed: Sequence[Sequence[float]],
    expected: Sequence[Sequence[float]],
    *,
    tolerance_mm: float = POSITION_TOLERANCE_MM,
) -> tuple[int, ...]:
    """Map every observed position to one unique expected position."""

    if tolerance_mm <= 0 or len(observed) != len(expected):
        raise ValueError("position sets must have equal size and positive tolerance")
    remaining = set(range(len(expected)))
    mapping: list[int] = []
    for raw_position in observed:
        position = tuple(float(value) for value in raw_position)
        if len(position) != 3 or any(not math.isfinite(value) for value in position):
            raise ValueError("positions must be finite 3-vectors")
        candidates = [
            index
            for index in remaining
            if len(expected[index]) == 3
            and max(
                abs(left - float(right))
                for left, right in zip(position, expected[index])
            )
            <= tolerance_mm
        ]
        if len(candidates) != 1:
            raise ValueError("physical positions do not have a unique one-to-one mapping")
        match = candidates[0]
        mapping.append(match)
        remaining.remove(match)
    if remaining:
        raise ValueError("physical position mapping is incomplete")
    return tuple(mapping)


def rescaled_value_range(
    stored_minimum: float,
    stored_maximum: float,
    slope: float,
    intercept: float,
) -> tuple[float, float]:
    """Return the finite range after a linear DICOM rescale, including negative slope."""

    values = tuple(float(value) for value in (stored_minimum, stored_maximum, slope, intercept))
    if any(not math.isfinite(value) for value in values):
        raise ValueError("rescale inputs must be finite")
    if stored_minimum > stored_maximum or slope == 0:
        raise ValueError("stored range is invalid or rescale slope is zero")
    endpoints = (stored_minimum * slope + intercept, stored_maximum * slope + intercept)
    return min(endpoints), max(endpoints)


def inverse_ftv_mask_value_statistics(
    values: Iterable[int],
    *,
    voxel_volume_mm3: float,
) -> dict[str, int | float | str | dict[str, int]]:
    """Interpret the collection-specific inverse, bit-encoded I-SPY2 FTV mask.

    The source collection documentation takes precedence over the generic DICOM SEG
    ``FRACTIONAL/OCCUPANCY`` declaration: a stored value of zero is included in the
    measured FTV, while non-zero values are one or more exclusion-reason bits.
    Unknown bits fail closed instead of being treated as fractional occupancy.
    """

    counts: Counter[int] = Counter()
    for raw in values:
        try:
            value = int(raw)
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError("FTV mask pixels must be finite integers") from error
        if value != raw or value < 0 or value > 255:
            raise ValueError("FTV mask pixels are outside the uint8 range")
        counts[value] += 1
    return _inverse_ftv_mask_count_statistics(
        counts, voxel_volume_mm3=voxel_volume_mm3
    )


def _inverse_ftv_mask_count_statistics(
    counts: Mapping[int, int],
    *,
    voxel_volume_mm3: float,
) -> dict[str, int | float | str | dict[str, int]]:
    """Count-table implementation that does not expand a full 3D mask."""

    if not math.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError("voxel volume must be finite and positive")
    if not counts:
        raise ValueError("FTV mask pixel stream is empty")
    if any(
        not isinstance(value, int)
        or not isinstance(count, int)
        or value < 0
        or value > 255
        or count < 1
        for value, count in counts.items()
    ):
        raise ValueError("FTV mask count table is invalid")
    if any(value & ~FTV_ALLOWED_BITS for value in counts):
        raise ValueError("FTV mask contains an undocumented exclusion bit")

    total = sum(counts.values())
    included = counts.get(0, 0)
    excluded = total - included
    multiple_reasons = sum(
        count
        for value, count in counts.items()
        if value != 0 and value & (value - 1)
    )
    bit_counts = {
        name: sum(count for value, count in counts.items() if value & bit)
        for name, bit in FTV_EXCLUSION_BITS.items()
    }
    return {
        "mask_semantics": MASK_SEMANTICS,
        "raw_unique_count": len(counts),
        "raw_value_set": ",".join(map(str, sorted(counts))),
        "included_ftv_voxels": included,
        "included_ftv_fraction": included / total,
        "included_ftv_volume_ml": included * voxel_volume_mm3 / 1000.0,
        "excluded_voxels": excluded,
        "excluded_fraction": excluded / total,
        "multiple_exclusion_reason_voxels": multiple_reasons,
        "exclusion_bit_voxels": bit_counts,
        "total_voxels": total,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        _raise_stale(error)
        raise PixelProbeError("mask_semantics_document_read_error") from None
    return digest.hexdigest()


def _safe_float(value: Any, *, default: float | None = None) -> float:
    if value is None and default is not None:
        result = default
    else:
        try:
            result = float(value)
        except (TypeError, ValueError) as error:
            raise PixelProbeError("invalid_numeric_dicom_attribute") from error
    if not math.isfinite(result):
        raise PixelProbeError("nonfinite_numeric_dicom_attribute")
    return result


def _transfer_syntax(dataset: Any) -> TransferSyntax:
    file_meta = getattr(dataset, "file_meta", None)
    syntax = getattr(file_meta, "TransferSyntaxUID", None)
    if syntax is None:
        raise PixelProbeError("missing_transfer_syntax")
    name = str(getattr(syntax, "name", "")).strip()
    raw = str(syntax).strip()
    if not name or name == raw or name.casefold().startswith("unknown"):
        # Raw UIDs are deliberately not emitted, even for an unknown syntax.
        raise PixelProbeError("unknown_transfer_syntax")
    try:
        compressed = bool(syntax.is_compressed)
    except (AttributeError, ValueError) as error:
        raise PixelProbeError("invalid_transfer_syntax") from error
    return TransferSyntax(name=name, compressed=compressed)


def _read_objects(path: Path) -> list[Any]:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise RuntimeError("pixel probe requires pydicom") from error
    try:
        files = sorted(item for item in path.iterdir() if item.is_file())
    except OSError as error:
        _raise_stale(error)
        raise PixelProbeError("dicom_directory_read_error") from None
    objects: list[Any] = []
    for file_path in files:
        try:
            objects.append(pydicom.dcmread(file_path, force=False))
        except pydicom.errors.InvalidDicomError:
            # Known sidecars are ignored only when strict DICOM parsing rejects them.
            continue
        except OSError as error:
            _raise_stale(error)
            raise PixelProbeError("dicom_file_read_error") from None
        except Exception:
            # Do not let parser messages containing source filenames reach logs.
            raise PixelProbeError("dicom_parse_error") from None
    return objects


def _decode_pixels(dataset: Any, *, kind: str) -> Any:
    try:
        pixels = dataset.pixel_array
    except Exception:  # pydicom handlers expose several exception types
        raise PixelProbeError(f"{kind}_pixel_decode_error") from None
    if pixels is None:
        raise PixelProbeError(f"{kind}_pixel_decode_error")
    return pixels


def _cross(left: Sequence[float], right: Sequence[float]) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _slice_spacing_mm(
    positions: Sequence[Sequence[float]], orientation: Sequence[float]
) -> float:
    if len(positions) < 2 or len(orientation) != 6:
        raise PixelProbeError("insufficient_geometry_for_slice_spacing")
    normal = _cross(orientation[:3], orientation[3:])
    norm = math.sqrt(sum(value * value for value in normal))
    if not math.isfinite(norm) or norm <= 0:
        raise PixelProbeError("invalid_image_orientation")
    unit = tuple(value / norm for value in normal)
    projected = sorted(
        sum(float(a) * b for a, b in zip(position, unit)) for position in positions
    )
    differences = [right - left for left, right in zip(projected, projected[1:])]
    if any(not math.isfinite(value) or value <= 0 for value in differences):
        raise PixelProbeError("duplicate_or_invalid_slice_positions")
    spacing = sum(differences) / len(differences)
    if max(abs(value - spacing) for value in differences) > max(0.01, spacing * 0.01):
        raise PixelProbeError("nonuniform_slice_spacing")
    return spacing


def _safe_pair_metadata(
    ser_syntax: TransferSyntax, seg_syntax: TransferSyntax
) -> dict[str, Any]:
    return {
        "ser_transfer_syntax": ser_syntax.name,
        "ser_compressed": ser_syntax.compressed,
        "seg_transfer_syntax": seg_syntax.name,
        "seg_compressed": seg_syntax.compressed,
        "ser_decode_status": "not_attempted",
        "seg_decode_status": "not_attempted",
    }


def inspect_pixel_pair(ser_dir: Path, mask_dir: Path) -> dict[str, Any]:
    """Decode and summarize one pair; returned data are aggregate-safe scalars."""

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise RuntimeError("pixel probe requires numpy") from error

    objects = _read_objects(ser_dir)
    ser_objects = [
        item
        for item in objects
        if str(getattr(item, "Modality", "")).upper() == "MR"
        and getattr(item, "Rows", None) is not None
        and getattr(item, "Columns", None) is not None
        and hasattr(item, "PixelData")
    ]
    seg_objects = [
        item
        for item in _read_objects(mask_dir)
        if str(getattr(item, "SOPClassUID", "")) == SEGMENTATION_STORAGE_UID
        and str(getattr(item, "Modality", "")).upper() == "SEG"
        and hasattr(item, "PixelData")
    ]
    if not ser_objects:
        raise PixelProbeError("no_usable_ser_images")
    if len(seg_objects) != 1:
        raise PixelProbeError("seg_object_count_mismatch")
    seg = seg_objects[0]

    ser_syntaxes = {_transfer_syntax(item) for item in ser_objects}
    if len(ser_syntaxes) != 1:
        raise PixelProbeError("mixed_ser_transfer_syntax")
    ser_syntax = next(iter(ser_syntaxes))
    seg_syntax = _transfer_syntax(seg)
    safe_metadata = _safe_pair_metadata(ser_syntax, seg_syntax)

    try:
        ser_rows = {int(item.Rows) for item in ser_objects}
        ser_columns = {int(item.Columns) for item in ser_objects}
        seg_rows = int(seg.Rows)
        seg_columns = int(seg.Columns)
        frame_count = int(seg.NumberOfFrames)
    except (AttributeError, TypeError, ValueError) as error:
        raise PixelProbeError(
            "missing_matrix_or_frame_attributes", safe_metadata=safe_metadata
        ) from error
    if min(*ser_rows, *ser_columns, seg_rows, seg_columns, frame_count) < 1:
        raise PixelProbeError(
            "nonpositive_matrix_or_frame_attributes", safe_metadata=safe_metadata
        )
    if len(ser_rows) != 1 or len(ser_columns) != 1:
        raise PixelProbeError("nonuniform_ser_matrix", safe_metadata=safe_metadata)
    if seg_rows not in ser_rows or seg_columns not in ser_columns:
        raise PixelProbeError("ser_seg_matrix_mismatch", safe_metadata=safe_metadata)

    series_uids = {
        str(getattr(item, "SeriesInstanceUID", "")).strip() for item in ser_objects
    }
    frame_refs = {
        str(getattr(item, "FrameOfReferenceUID", "")).strip() for item in ser_objects
    }
    if "" in series_uids or len(series_uids) != 1:
        raise PixelProbeError("ser_series_identity_mismatch", safe_metadata=safe_metadata)
    if "" in frame_refs or len(frame_refs) != 1:
        raise PixelProbeError("ser_frame_of_reference_mismatch", safe_metadata=safe_metadata)
    if str(getattr(seg, "FrameOfReferenceUID", "")).strip() not in frame_refs:
        raise PixelProbeError("seg_frame_of_reference_mismatch", safe_metadata=safe_metadata)

    ser_spacings = {
        value
        for item in ser_objects
        if (value := _float_vector(getattr(item, "PixelSpacing", None), 2)) is not None
    }
    ser_orientations = {
        value
        for item in ser_objects
        if (value := _float_vector(getattr(item, "ImageOrientationPatient", None), 6))
        is not None
    }
    ser_positions = [
        _float_vector(getattr(item, "ImagePositionPatient", None), 3)
        for item in ser_objects
    ]
    if len(ser_spacings) != 1 or len(ser_orientations) != 1:
        raise PixelProbeError("nonuniform_ser_inplane_geometry", safe_metadata=safe_metadata)
    if any(position is None for position in ser_positions):
        raise PixelProbeError("missing_ser_slice_position", safe_metadata=safe_metadata)
    concrete_ser_positions = [position for position in ser_positions if position is not None]

    frames = _sequence(seg, "PerFrameFunctionalGroupsSequence")
    shared_items = _sequence(seg, "SharedFunctionalGroupsSequence")
    if len(frames) != frame_count or len(shared_items) != 1:
        raise PixelProbeError("seg_functional_group_count_mismatch", safe_metadata=safe_metadata)
    shared = shared_items[0]
    seg_positions = [_frame_position(frame) for frame in frames]
    if any(position is None for position in seg_positions):
        raise PixelProbeError("missing_seg_frame_position", safe_metadata=safe_metadata)
    concrete_seg_positions = [position for position in seg_positions if position is not None]
    try:
        mapping = match_positions_one_to_one(concrete_seg_positions, concrete_ser_positions)
    except ValueError as error:
        raise PixelProbeError(
            "seg_ser_position_mapping_mismatch", safe_metadata=safe_metadata
        ) from error
    if frame_count != len(ser_objects) or len(mapping) != frame_count:
        raise PixelProbeError("seg_ser_frame_count_mismatch", safe_metadata=safe_metadata)

    spacing = _nested_vector(
        shared,
        frames,
        sequence_name="PixelMeasuresSequence",
        attribute="PixelSpacing",
        length=2,
    )
    orientation = _nested_vector(
        shared,
        frames,
        sequence_name="PlaneOrientationSequence",
        attribute="ImageOrientationPatient",
        length=6,
    )
    ser_spacing = next(iter(ser_spacings))
    ser_orientation = next(iter(ser_orientations))
    if not _vectors_close(spacing, ser_spacing) or not _vectors_close(
        orientation, ser_orientation
    ):
        raise PixelProbeError("seg_ser_inplane_geometry_mismatch", safe_metadata=safe_metadata)

    segment_items = _sequence(seg, "SegmentSequence")
    try:
        segment_numbers = {
            int(item.SegmentNumber)
            for item in segment_items
            if getattr(item, "SegmentNumber", None) is not None
        }
    except (TypeError, ValueError) as error:
        raise PixelProbeError(
            "invalid_seg_segment_number", safe_metadata=safe_metadata
        ) from error
    frame_segment_numbers = set().union(
        *(_frame_segment_numbers(frame) for frame in frames)
    )
    if len(segment_items) != 1 or len(segment_numbers) != 1:
        raise PixelProbeError("seg_segment_count_mismatch", safe_metadata=safe_metadata)
    if frame_segment_numbers != segment_numbers:
        raise PixelProbeError("seg_frame_segment_number_mismatch", safe_metadata=safe_metadata)

    segmentation_type = str(getattr(seg, "SegmentationType", "")).strip().upper()
    fractional_type = str(
        getattr(seg, "SegmentationFractionalType", "not_applicable")
    ).strip().upper()
    if segmentation_type == "BINARY":
        maximum_value = 1
    elif segmentation_type == "FRACTIONAL":
        if fractional_type not in {"OCCUPANCY", "PROBABILITY"}:
            raise PixelProbeError(
                "unsupported_segmentation_fractional_type",
                safe_metadata=safe_metadata,
            )
        try:
            maximum_value = int(seg.MaximumFractionalValue)
        except (AttributeError, TypeError, ValueError) as error:
            raise PixelProbeError(
                "invalid_seg_fractional_maximum", safe_metadata=safe_metadata
            ) from error
        if maximum_value < 1:
            raise PixelProbeError("invalid_seg_fractional_maximum", safe_metadata=safe_metadata)
    else:
        raise PixelProbeError("unsupported_segmentation_type", safe_metadata=safe_metadata)
    if (
        segmentation_type != "FRACTIONAL"
        or fractional_type != "OCCUPANCY"
        or maximum_value != 255
    ):
        raise PixelProbeError(
            "unexpected_ispy2_analysis_mask_declaration",
            safe_metadata=safe_metadata,
        )

    if any(_sequence(item, "ModalityLUTSequence") for item in ser_objects):
        raise PixelProbeError("unsupported_ser_modality_lut", safe_metadata=safe_metadata)
    rescale_presence = {
        (
            getattr(item, "RescaleSlope", None) is not None,
            getattr(item, "RescaleIntercept", None) is not None,
        )
        for item in ser_objects
    }
    if any(
        slope_present != intercept_present
        for slope_present, intercept_present in rescale_presence
    ):
        raise PixelProbeError("incomplete_ser_rescale", safe_metadata=safe_metadata)
    if len(rescale_presence) != 1:
        raise PixelProbeError(
            "nonuniform_ser_rescale_presence", safe_metadata=safe_metadata
        )
    rescale_defaulted = rescale_presence == {(False, False)}
    try:
        slopes = {
            _safe_float(getattr(item, "RescaleSlope", None), default=1.0)
            for item in ser_objects
        }
        intercepts = {
            _safe_float(getattr(item, "RescaleIntercept", None), default=0.0)
            for item in ser_objects
        }
    except PixelProbeError as error:
        raise PixelProbeError(error.code, safe_metadata=safe_metadata) from None
    if len(slopes) != 1 or len(intercepts) != 1:
        raise PixelProbeError("nonuniform_ser_rescale", safe_metadata=safe_metadata)
    slope = next(iter(slopes))
    intercept = next(iter(intercepts))
    if slope == 0:
        raise PixelProbeError("zero_ser_rescale_slope", safe_metadata=safe_metadata)

    stored_minimum = math.inf
    stored_maximum = -math.inf
    safe_metadata["ser_decode_status"] = "attempted"
    for item in ser_objects:
        try:
            pixels = _decode_pixels(item, kind="ser")
        except PixelProbeError as error:
            raise PixelProbeError(
                error.code,
                safe_metadata={**safe_metadata, "ser_decode_status": "failed"},
            ) from None
        if tuple(pixels.shape) != (next(iter(ser_rows)), next(iter(ser_columns))):
            raise PixelProbeError("unexpected_ser_pixel_shape", safe_metadata=safe_metadata)
        current_minimum = float(np.min(pixels))
        current_maximum = float(np.max(pixels))
        if not math.isfinite(current_minimum) or not math.isfinite(current_maximum):
            raise PixelProbeError("nonfinite_ser_pixels", safe_metadata=safe_metadata)
        stored_minimum = min(stored_minimum, current_minimum)
        stored_maximum = max(stored_maximum, current_maximum)
    safe_metadata["ser_decode_status"] = "success"

    safe_metadata["seg_decode_status"] = "attempted"
    try:
        seg_pixels = _decode_pixels(seg, kind="seg")
    except PixelProbeError as error:
        raise PixelProbeError(
            error.code,
            safe_metadata={**safe_metadata, "seg_decode_status": "failed"},
        ) from None
    safe_metadata["seg_decode_status"] = "success"
    if tuple(seg_pixels.shape) != (frame_count, seg_rows, seg_columns):
        raise PixelProbeError("unexpected_seg_pixel_shape", safe_metadata=safe_metadata)
    if getattr(seg_pixels.dtype, "kind", "") not in "biu":
        raise PixelProbeError("noninteger_seg_pixels", safe_metadata=safe_metadata)

    unique_values, unique_counts = np.unique(seg_pixels, return_counts=True)
    raw_value_counts = {
        int(value): int(count)
        for value, count in zip(unique_values.tolist(), unique_counts.tolist())
    }
    if any(value < 0 or value > maximum_value for value in raw_value_counts):
        raise PixelProbeError("invalid_seg_pixel_values", safe_metadata=safe_metadata)

    slice_spacing = _slice_spacing_mm(concrete_ser_positions, ser_orientation)
    if any(value <= 0 for value in ser_spacing):
        raise PixelProbeError(
            "nonpositive_ser_pixel_spacing", safe_metadata=safe_metadata
    )
    voxel_volume_mm3 = ser_spacing[0] * ser_spacing[1] * slice_spacing
    try:
        mask_stats = _inverse_ftv_mask_count_statistics(
            raw_value_counts,
            voxel_volume_mm3=voxel_volume_mm3,
        )
    except ValueError as error:
        raise PixelProbeError(
            "invalid_inverse_ftv_mask_values", safe_metadata=safe_metadata
        ) from error
    rescaled_minimum, rescaled_maximum = rescaled_value_range(
        stored_minimum, stored_maximum, slope, intercept
    )
    if (
        rescaled_minimum < SER_VALUE_RANGE[0] - SER_RANGE_TOLERANCE
        or rescaled_maximum > SER_VALUE_RANGE[1] + SER_RANGE_TOLERANCE
    ):
        raise PixelProbeError(
            "ser_rescaled_range_outside_fixed_contract",
            safe_metadata=safe_metadata,
        )
    if rescaled_maximum - rescaled_minimum <= SER_RANGE_TOLERANCE:
        raise PixelProbeError("constant_ser_volume", safe_metadata=safe_metadata)
    exclusion_counts = mask_stats["exclusion_bit_voxels"]
    if not isinstance(exclusion_counts, Mapping):
        raise PixelProbeError("invalid_inverse_ftv_mask_statistics")
    return {
        **safe_metadata,
        "ser_decode_success": True,
        "seg_decode_success": True,
        "ser_rescale_slope": slope,
        "ser_rescale_intercept": intercept,
        "ser_rescale_defaulted": rescale_defaulted,
        "ser_stored_minimum": stored_minimum,
        "ser_stored_maximum": stored_maximum,
        "ser_rescaled_minimum": rescaled_minimum,
        "ser_rescaled_maximum": rescaled_maximum,
        "segmentation_type": segmentation_type,
        "segmentation_fractional_type": fractional_type,
        "maximum_fractional_value": maximum_value,
        "mask_semantics": str(mask_stats["mask_semantics"]),
        "mask_semantics_override_applied": True,
        "seg_raw_unique_count": int(mask_stats["raw_unique_count"]),
        "seg_raw_value_set": str(mask_stats["raw_value_set"]),
        "seg_included_ftv_voxels": int(mask_stats["included_ftv_voxels"]),
        "seg_included_ftv_fraction": float(mask_stats["included_ftv_fraction"]),
        "seg_included_ftv_volume_ml": float(mask_stats["included_ftv_volume_ml"]),
        "seg_excluded_voxels": int(mask_stats["excluded_voxels"]),
        "seg_excluded_fraction": float(mask_stats["excluded_fraction"]),
        "seg_multiple_exclusion_reason_voxels": int(
            mask_stats["multiple_exclusion_reason_voxels"]
        ),
        **{
            f"seg_exclusion_{name}_voxels": int(exclusion_counts[name])
            for name in FTV_EXCLUSION_BITS
        },
        "voxel_volume_mm3": voxel_volume_mm3,
        "slice_spacing_mm": slice_spacing,
        "physical_position_mapping_exact": True,
    }


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def _range(values: Iterable[int | float]) -> dict[str, int | float]:
    observed = list(values)
    if not observed or any(not math.isfinite(float(value)) for value in observed):
        raise ValueError("aggregate range requires finite values")
    return {"min": min(observed), "max": max(observed)}


def summarize_pixel_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    attempted: int,
    failure_reasons: Mapping[str, int],
    failure_metadata: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Aggregate rows without retaining patient-level or filesystem values."""

    if attempted < len(rows) or attempted != len(rows) + sum(failure_reasons.values()):
        raise ValueError("attempt accounting is inconsistent")
    safe_rows = [*rows, *failure_metadata]
    distributions = {
        field: _distribution(row[field] for row in safe_rows if field in row)
        for field in (
            "ser_transfer_syntax",
            "ser_compressed",
            "seg_transfer_syntax",
            "seg_compressed",
            "ser_decode_status",
            "seg_decode_status",
        )
    }
    distributions.update(
        {
            field: _distribution(row[field] for row in rows)
            for field in (
                "ser_decode_success",
                "seg_decode_success",
                "ser_rescale_defaulted",
                "segmentation_type",
                "segmentation_fractional_type",
                "maximum_fractional_value",
                "mask_semantics",
                "mask_semantics_override_applied",
                "seg_raw_value_set",
                "physical_position_mapping_exact",
            )
        }
    )
    numeric_fields = (
        "ser_rescale_slope",
        "ser_rescale_intercept",
        "ser_stored_minimum",
        "ser_stored_maximum",
        "ser_rescaled_minimum",
        "ser_rescaled_maximum",
        "seg_raw_unique_count",
        "seg_included_ftv_voxels",
        "seg_included_ftv_fraction",
        "seg_included_ftv_volume_ml",
        "seg_excluded_voxels",
        "seg_excluded_fraction",
        "seg_multiple_exclusion_reason_voxels",
        *(f"seg_exclusion_{name}_voxels" for name in FTV_EXCLUSION_BITS),
        "voxel_volume_mm3",
        "slice_spacing_mm",
    )
    ranges = (
        {field: _range(row[field] for row in rows) for field in numeric_fields}
        if rows
        else {}
    )
    return {
        "pairs_attempted": attempted,
        "pairs_passed": len(rows),
        "pairs_failed": sum(failure_reasons.values()),
        "failure_reasons": dict(sorted(failure_reasons.items())),
        "distributions": distributions,
        "ranges": ranges,
    }


def run_pixel_probe(
    root: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    max_pairs_per_timepoint: int = 3,
) -> dict[str, Any]:
    root_path = Path(root).expanduser().resolve()
    path_contract, pairs = audit_manifest_pairs(
        root_path, trajectory_path, timepoints=TIMEPOINTS
    )
    outside = sum(
        group.get("outside_ds3_dicom", 0)
        for point, group in path_contract.items()
        if point in TIMEPOINTS
    )
    report: dict[str, Any] = {
        "schema_version": "med-jepa-ispy2-pixel-probe-v2",
        "path_contract": path_contract,
        "outside_ds3_dicom_total": outside,
        "pixel_data_read": False,
        "pixels_or_identifiers_written": False,
        "sample_strategy": "deterministic_evenly_spaced_complete_trajectory_cohort",
        "groups": {},
    }
    semantics_document = root_path / MASK_SEMANTICS_RELATIVE_PATH
    if not semantics_document.is_file():
        report["mask_semantics_contract"] = {
            "source": MASK_SEMANTICS_RELATIVE_PATH.as_posix(),
            "status": "missing",
        }
        report["status"] = "blocked_missing_mask_semantics_document"
        return report
    observed_digest = _sha256(semantics_document)
    report["mask_semantics_contract"] = {
        "source": MASK_SEMANTICS_RELATIVE_PATH.as_posix(),
        "source_sha256": observed_digest,
        "expected_source_sha256": APPROVED_MASK_SEMANTICS_SHA256,
        "interpretation": MASK_SEMANTICS,
        "included_value": 0,
        "exclusion_bits": FTV_EXCLUSION_BITS,
        "generic_fractional_occupancy_interpretation_used": False,
    }
    if observed_digest != APPROVED_MASK_SEMANTICS_SHA256:
        report["status"] = "blocked_mask_semantics_document_digest_mismatch"
        return report
    if outside:
        report["status"] = "blocked_wrong_manifest_root"
        return report
    cohort_sizes = {len(pairs.get(point, [])) for point in TIMEPOINTS}
    if len(cohort_sizes) != 1 or not cohort_sizes:
        report["status"] = "blocked_incomplete_trajectory_alignment"
        return report
    cohort_size = next(iter(cohort_sizes))
    if cohort_size < max_pairs_per_timepoint:
        report["status"] = "blocked_insufficient_complete_trajectories"
        return report
    indices = evenly_spaced_indices(cohort_size, max_pairs_per_timepoint)

    all_passed = True
    any_decode_attempted = False
    for point in TIMEPOINTS:
        rows: list[dict[str, Any]] = []
        failures: Counter[str] = Counter()
        failure_metadata: list[Mapping[str, Any]] = []
        for index in indices:
            ser_dir, mask_dir = pairs[point][index]
            try:
                rows.append(inspect_pixel_pair(ser_dir, mask_dir))
                any_decode_attempted = True
            except PixelProbeError as error:
                failures[error.code] += 1
                failure_metadata.append(error.safe_metadata)
                if any(
                    error.safe_metadata.get(field, "not_attempted")
                    != "not_attempted"
                    for field in ("ser_decode_status", "seg_decode_status")
                ):
                    any_decode_attempted = True
            except OSError as error:
                _raise_stale(error)
                failures["dicom_io_error"] += 1
        group = summarize_pixel_rows(
            rows,
            attempted=len(indices),
            failure_reasons=failures,
            failure_metadata=failure_metadata,
        )
        report["groups"][point] = group
        if failures or len(rows) != len(indices):
            all_passed = False
    report["pixel_data_read"] = any_decode_attempted
    report["status"] = "ok" if all_passed else "blocked_pixel_probe"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--max-pairs-per-timepoint", type=int, default=3)
    parser.add_argument("--output", help="Create this aggregate JSON report; refuses overwrite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_pairs_per_timepoint < 1:
        raise SystemExit("max-pairs-per-timepoint must be positive")
    try:
        report = run_pixel_probe(
            args.root,
            trajectory_path=args.trajectory,
            max_pairs_per_timepoint=args.max_pairs_per_timepoint,
        )
    except StaleFileHandleError as error:
        print(str(error), file=sys.stderr)
        return 2
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        destination = Path(args.output).expanduser()
        with destination.open("x", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    print(rendered)
    return 0 if report["status"] == "ok" else 3


if __name__ == "__main__":
    raise SystemExit(main())
