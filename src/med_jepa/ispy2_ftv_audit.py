"""Governed, full-cohort audit of I-SPY2 inverse analysis masks and FTV.

This command is deliberately separate from ROI conversion.  It reads the approved
DS3 trajectory manifest and DICOM SEG objects without modifying the source tree,
interprets the collection-specific analysis mask exactly as documented (zero is an
included FTV voxel; non-zero values are exclusion-reason bits), and writes:

* one patient-level CSV inside an explicitly governed output root only after all
  mask-contract gates pass; and
* one aggregate-only JSON report that contains no patient identifiers or paths.

The generic DICOM ``FRACTIONAL/OCCUPANCY`` declaration is *not* used as a
probability.  The independently reviewed collection document is pinned by SHA-256,
and undocumented pixel bits are handled by an explicit, evidence-bound release
policy after an aggregate full-cohort diagnostic.  That policy never assigns a
meaning to an undocumented bit or adds it to the documented allowlist.
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
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree as ET
from zipfile import BadZipFile, ZipFile

from med_jepa.ispy2_pixel_probe import (
    APPROVED_MASK_SEMANTICS_SHA256,
    FTV_ALLOWED_BITS,
    FTV_EXCLUSION_BITS,
    MASK_SEMANTICS,
    _inverse_ftv_mask_count_statistics,
    _slice_spacing_mm,
)
from med_jepa.ftv_release import (
    APPROVED_CLINICAL_WORKBOOK_SHA256,
    EXPECTED_CONSUMER_CONTRACT,
    POLICY_PERMITTED_UNDOCUMENTED_BITS,
    ZERO_ONLY_FTV_RELEASE_POLICY_ID,
)
from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_seg_mapping_probe import (
    SEGMENTATION_STORAGE_UID,
    _float_vector,
    _frame_position,
    _frame_segment_numbers,
    _sequence,
)
from med_jepa.ispy2_series_probe import _inside, _raise_stale


TIMEPOINTS = ("T0", "T1", "T3")
EXPECTED_COMPLETE_PATIENTS = 595
EXPECTED_SPLIT_COUNTS = {"train": 416, "val": 89, "test": 90}
APPROVED_MASK_DOCUMENT_SHA256 = APPROVED_MASK_SEMANTICS_SHA256
APPROVED_TRAJECTORY_SHA256 = (
    "11ffa254b5dc9c4f5edc3132c089ec352ee979a4b30098a1f0c7a577ec45ff7a"
)
APPROVED_SPLIT_SHA256 = (
    "b34636e5166ba7187711887afb6087b672898463ef79a02ec489fe730cb05626"
)
MASK_DOCUMENT_RELATIVE_PATH = Path(
    "clinical/Analysis-mask-files-description.v20211020.docx"
)
CLINICAL_WORKBOOK_RELATIVE_PATH = Path("clinical/Multi-feature-MRI-NACT-Data.xlsx")
SPLIT_RELATIVE_PATH = Path("manifests/ispy2_split.csv")
CLINICAL_ID_COLUMN = "CLINICAL-TRIAL-SUBJECT-ID"
CLINICAL_FTV_COLUMNS = {
    "T0": "VOLUME_TUM_BLU_V10",
    "T1": "VOLUME_TUM_BLU_V20",
    "T3": "VOLUME_TUM_BLU_V40",
}
CLINICAL_VALIDATION_THRESHOLDS = {
    "minimum_pairs_per_timepoint": 270,
    "minimum_pearson_correlation": 0.995,
    "minimum_median_ratio": 0.98,
    "maximum_median_ratio": 1.02,
    "maximum_mape_percent": 5.0,
}
UNDOCUMENTED_SINGLE_BITS = tuple(
    1 << index
    for index in range(8)
    if not (FTV_ALLOWED_BITS & (1 << index))
)
FTV_RELEASE_TOLERATED_UNDOCUMENTED_BIT_UNION = 16
if POLICY_PERMITTED_UNDOCUMENTED_BITS != (
    FTV_RELEASE_TOLERATED_UNDOCUMENTED_BIT_UNION,
):  # pragma: no cover - source-level contract guard
    raise RuntimeError("FTV release policy constants disagree")


class FTVAuditError(RuntimeError):
    """A fail-closed error with a message safe to put in a Slurm log."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


_PATIENT_ID = re.compile(r"^ISPY2-[0-9]+$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError("file_read_error") from None
    return digest.hexdigest()


def _safe_resolve(path: Path, *, code: str) -> Path:
    try:
        return path.expanduser().resolve()
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError(code) from None


def _require_directory(path: Path, *, code: str) -> None:
    try:
        if not path.is_dir():
            raise FTVAuditError(code)
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError(code) from None


def _cohort_from_manifest(
    root: Path,
    trajectory_path: Path,
    *,
    expected_patients: int,
    expected_patient_ids: set[str] | None = None,
) -> list[tuple[str, dict[str, Path]]]:
    """Return complete patients internally while reporting only aggregate counts.

    Patient identifiers are needed solely for the governed CSV and workbook join.
    Callers must never include the returned keys in logs or aggregate reports.
    """

    if expected_patients < 1:
        raise ValueError("expected patient count must be positive")
    dicom_root = _safe_resolve(root / "dicom", code="dicom_root_unavailable")
    _require_directory(dicom_root, code="dicom_root_unavailable")
    patients: dict[str, dict[str, Path]] = defaultdict(dict)
    used_mask_directories: set[Path] = set()
    try:
        with trajectory_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"PatientID", "tp_label", "mask_dir"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise FTVAuditError("trajectory_schema_mismatch")
            for row in reader:
                timepoint = str(row.get("tp_label") or "").strip().upper()
                if timepoint not in TIMEPOINTS:
                    continue
                patient_id = str(row.get("PatientID") or "").strip()
                raw_path = str(row.get("mask_dir") or "").strip()
                if not _PATIENT_ID.fullmatch(patient_id):
                    raise FTVAuditError("invalid_patient_identifier")
                if not raw_path:
                    continue
                candidate = Path(raw_path).expanduser()
                candidate = _safe_resolve(
                    candidate if candidate.is_absolute() else trajectory_path.parent / candidate,
                    code="mask_path_resolution_error",
                )
                if not _inside(candidate, dicom_root):
                    raise FTVAuditError("mask_path_outside_approved_ds3_dicom")
                if candidate in used_mask_directories:
                    raise FTVAuditError("duplicate_mask_directory_reference")
                if timepoint in patients[patient_id]:
                    raise FTVAuditError("duplicate_patient_timepoint")
                patients[patient_id][timepoint] = candidate
                used_mask_directories.add(candidate)
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError("trajectory_read_error") from None

    required_points = set(TIMEPOINTS)
    complete = sorted(
        (patient_id, visits)
        for patient_id, visits in patients.items()
        if required_points.issubset(visits)
    )
    if len(complete) != expected_patients:
        raise FTVAuditError("complete_cohort_count_mismatch")
    if expected_patient_ids is not None and {
        patient_id for patient_id, _ in complete
    } != expected_patient_ids:
        raise FTVAuditError("complete_cohort_split_mismatch")
    return complete


def _approved_split_patient_ids(
    root: Path, *, expected_patients: int
) -> tuple[set[str], dict[str, int], str]:
    """Read the source split and return its exact, non-identifying cohort contract."""

    path = root / SPLIT_RELATIVE_PATH
    counts: Counter[str] = Counter()
    patient_ids: set[str] = set()
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"PatientID", "split"}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise FTVAuditError("split_schema_mismatch")
            for row in reader:
                patient_id = str(row.get("PatientID") or "").strip()
                split = str(row.get("split") or "").strip().casefold()
                if not _PATIENT_ID.fullmatch(patient_id):
                    raise FTVAuditError("invalid_split_patient_identifier")
                if split == "validation":
                    split = "val"
                if split not in EXPECTED_SPLIT_COUNTS:
                    raise FTVAuditError("invalid_split_assignment")
                if patient_id in patient_ids:
                    raise FTVAuditError("duplicate_split_patient")
                patient_ids.add(patient_id)
                counts[split] += 1
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError("split_read_error") from None
    observed = {name: counts[name] for name in EXPECTED_SPLIT_COUNTS}
    if len(patient_ids) != expected_patients or observed != EXPECTED_SPLIT_COUNTS:
        raise FTVAuditError("split_cohort_contract_mismatch")
    observed_sha256 = _sha256(path)
    if observed_sha256 != APPROVED_SPLIT_SHA256:
        raise FTVAuditError("split_hash_mismatch")
    return patient_ids, observed, observed_sha256


def _select_seg_path(mask_dir: Path, *, dicom_root: Path) -> Path:
    """Select exactly one strict DICOM SEG object and ignore non-DICOM sidecars."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster dependency
        raise FTVAuditError("pydicom_dependency_unavailable") from error

    try:
        candidates = sorted(item for item in mask_dir.iterdir() if item.is_file())
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError("mask_directory_read_error") from None
    selected: list[Path] = []
    for candidate in candidates:
        resolved = _safe_resolve(candidate, code="seg_path_resolution_error")
        if not _inside(resolved, dicom_root):
            raise FTVAuditError("seg_file_outside_approved_ds3_dicom")
        try:
            header = pydicom.dcmread(
                resolved,
                stop_before_pixels=True,
                force=False,
                specific_tags=(
                    "SOPClassUID",
                    "Modality",
                    "Rows",
                    "Columns",
                    "NumberOfFrames",
                ),
            )
        except pydicom.errors.InvalidDicomError:
            continue
        except OSError as error:
            _raise_stale(error)
            raise FTVAuditError("dicom_header_read_error") from None
        except Exception:
            raise FTVAuditError("dicom_header_parse_error") from None
        if (
            str(getattr(header, "SOPClassUID", "")) == SEGMENTATION_STORAGE_UID
            and str(getattr(header, "Modality", "")).strip().upper() == "SEG"
            and getattr(header, "Rows", None) is not None
            and getattr(header, "Columns", None) is not None
            and getattr(header, "NumberOfFrames", None) is not None
        ):
            selected.append(resolved)
    if len(selected) != 1:
        raise FTVAuditError("seg_object_count_mismatch")
    return selected[0]


def _positive_float(value: Any, *, code: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise FTVAuditError(code) from error
    if not math.isfinite(result) or result <= 0:
        raise FTVAuditError(code)
    return result


def _nested_positive_float(
    shared: Any,
    frames: Sequence[Any],
    *,
    sequence_name: str,
    attribute: str,
) -> float | None:
    for container in (shared, *frames):
        if container is None:
            continue
        for item in _sequence(container, sequence_name):
            value = getattr(item, attribute, None)
            if value is None:
                continue
            try:
                result = float(value)
            except (TypeError, ValueError, OverflowError):
                return None
            return result if math.isfinite(result) and result > 0 else None
    return None


def _all_nested_vectors(
    shared: Any,
    frames: Sequence[Any],
    *,
    sequence_name: str,
    attribute: str,
    length: int,
) -> set[tuple[float, ...]]:
    values: set[tuple[float, ...]] = set()
    for container in (shared, *frames):
        if container is None:
            continue
        for item in _sequence(container, sequence_name):
            value = _float_vector(getattr(item, attribute, None), length)
            if value is None:
                raise FTVAuditError("invalid_seg_functional_group_geometry")
            values.add(value)
    return values


def _seg_voxel_volume_mm3(dataset: Any, *, frame_count: int) -> tuple[float, float]:
    frames = _sequence(dataset, "PerFrameFunctionalGroupsSequence")
    shared_items = _sequence(dataset, "SharedFunctionalGroupsSequence")
    if len(frames) != frame_count or len(shared_items) != 1:
        raise FTVAuditError("seg_functional_group_count_mismatch")
    shared = shared_items[0]
    spacing_values = _all_nested_vectors(
        shared,
        frames,
        sequence_name="PixelMeasuresSequence",
        attribute="PixelSpacing",
        length=2,
    )
    orientation_values = _all_nested_vectors(
        shared,
        frames,
        sequence_name="PlaneOrientationSequence",
        attribute="ImageOrientationPatient",
        length=6,
    )
    if len(spacing_values) != 1 or len(orientation_values) != 1:
        raise FTVAuditError("nonuniform_seg_functional_group_geometry")
    spacing = next(iter(spacing_values))
    orientation = next(iter(orientation_values))
    positions = [_frame_position(frame) for frame in frames]
    if any(value <= 0 for value in spacing):
        raise FTVAuditError("invalid_seg_pixel_spacing")
    if any(position is None for position in positions):
        raise FTVAuditError("missing_seg_frame_geometry")
    try:
        geometric_spacing = _slice_spacing_mm(
            [position for position in positions if position is not None], orientation
        )
    except Exception as error:
        # PixelProbeError and geometry ValueError are deliberately collapsed so no
        # lower-level message can reveal a source path.
        raise FTVAuditError("invalid_seg_slice_geometry") from error
    declared_spacing = _nested_positive_float(
        shared,
        frames,
        sequence_name="PixelMeasuresSequence",
        attribute="SpacingBetweenSlices",
    )
    if declared_spacing is not None and not math.isclose(
        declared_spacing,
        geometric_spacing,
        rel_tol=0.01,
        abs_tol=0.01,
    ):
        raise FTVAuditError("seg_slice_spacing_disagreement")
    voxel_volume = spacing[0] * spacing[1] * geometric_spacing
    return _positive_float(voxel_volume, code="invalid_seg_voxel_volume"), geometric_spacing


def inverse_mask_statistics_from_counts(
    raw_counts: Mapping[int, int],
    *,
    voxel_volume_mm3: float,
    allow_undocumented_bits: bool = False,
) -> dict[str, Any]:
    """Validate counts and compute FTV, optionally retaining unknown-bit diagnostics.

    ``allow_undocumented_bits`` is diagnostic only.  It does not add any bit to the
    approved mask contract and gives no undocumented bit a semantic name.  Every
    non-zero voxel remains excluded from the candidate FTV.  A later release gate may
    separate that zero-only FTV claim from the still-blocked exclusion taxonomy only
    when the aggregate evidence satisfies the pinned policy.
    """

    if any(
        not isinstance(value, int)
        or not isinstance(count, int)
        or value < 0
        or value > 255
        or count < 1
        for value, count in raw_counts.items()
    ):
        raise FTVAuditError("invalid_seg_pixel_count_table")
    undocumented_raw_counts = {
        value: count
        for value, count in raw_counts.items()
        if value & ~FTV_ALLOWED_BITS
    }
    undocumented_only_raw_counts = {
        value: count
        for value, count in undocumented_raw_counts.items()
        if value != 0 and value & FTV_ALLOWED_BITS == 0
    }
    if undocumented_raw_counts and not allow_undocumented_bits:
        raise FTVAuditError("undocumented_ftv_exclusion_bit")
    if undocumented_raw_counts:
        if not math.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
            raise FTVAuditError("invalid_inverse_ftv_mask")
        total = sum(raw_counts.values())
        included = raw_counts.get(0, 0)
        stats: dict[str, Any] = {
            "mask_semantics": MASK_SEMANTICS,
            "raw_unique_count": len(raw_counts),
            "raw_value_set": ",".join(map(str, sorted(raw_counts))),
            "included_ftv_voxels": included,
            "included_ftv_fraction": included / total,
            "included_ftv_volume_ml": included * voxel_volume_mm3 / 1000.0,
            "excluded_voxels": total - included,
            "excluded_fraction": (total - included) / total,
            "multiple_exclusion_reason_voxels": sum(
                count
                for value, count in raw_counts.items()
                if value != 0 and value & (value - 1)
            ),
            "exclusion_bit_voxels": {
                name: sum(
                    count for value, count in raw_counts.items() if value & bit
                )
                for name, bit in FTV_EXCLUSION_BITS.items()
            },
            "total_voxels": total,
        }
    else:
        try:
            stats = _inverse_ftv_mask_count_statistics(
                raw_counts,
                voxel_volume_mm3=voxel_volume_mm3,
            )
        except ValueError as error:
            raise FTVAuditError("invalid_inverse_ftv_mask") from error
    stats["undocumented_raw_code_voxels"] = dict(
        sorted(undocumented_raw_counts.items())
    )
    stats["undocumented_only_raw_code_voxels"] = dict(
        sorted(undocumented_only_raw_counts.items())
    )
    stats["undocumented_only_voxels"] = int(
        sum(undocumented_only_raw_counts.values())
    )
    stats["contains_undocumented_only_voxels"] = bool(
        undocumented_only_raw_counts
    )
    stats["undocumented_single_bit_voxels"] = {
        bit: sum(count for value, count in raw_counts.items() if value & bit)
        for bit in UNDOCUMENTED_SINGLE_BITS
    }
    stats["contains_undocumented_bits"] = bool(undocumented_raw_counts)
    return stats


def inspect_inverse_mask(
    mask_dir: Path,
    *,
    dicom_root: Path,
    expected_patient_id: str,
) -> dict[str, Any]:
    """Decode one SEG and return aggregate-safe values plus its FTV."""

    try:
        import numpy as np
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster dependencies
        raise FTVAuditError("pixel_dependency_unavailable") from error

    seg_path = _select_seg_path(mask_dir, dicom_root=dicom_root)
    try:
        dataset = pydicom.dcmread(seg_path, force=False)
    except OSError as error:
        _raise_stale(error)
        raise FTVAuditError("seg_object_read_error") from None
    except Exception:
        raise FTVAuditError("seg_object_parse_error") from None

    if (
        str(getattr(dataset, "SOPClassUID", "")) != SEGMENTATION_STORAGE_UID
        or str(getattr(dataset, "Modality", "")).strip().upper() != "SEG"
    ):
        raise FTVAuditError("selected_object_is_not_seg")
    dicom_patient_id = str(getattr(dataset, "PatientID", "")).strip()
    if dicom_patient_id:
        normalized_dicom_patient_id = _normalize_patient_id(dicom_patient_id)
        if normalized_dicom_patient_id != expected_patient_id:
            raise FTVAuditError("dicom_manifest_patient_mismatch")
    sop_instance_uid = str(getattr(dataset, "SOPInstanceUID", "")).strip()
    if not sop_instance_uid:
        raise FTVAuditError("missing_seg_sop_instance_uid")
    seg_identity_sha256 = hashlib.sha256(sop_instance_uid.encode("utf-8")).hexdigest()
    try:
        rows = int(dataset.Rows)
        columns = int(dataset.Columns)
        frame_count = int(dataset.NumberOfFrames)
    except (AttributeError, TypeError, ValueError) as error:
        raise FTVAuditError("invalid_seg_matrix_or_frames") from error
    if min(rows, columns, frame_count) < 1:
        raise FTVAuditError("invalid_seg_matrix_or_frames")

    segmentation_type = str(getattr(dataset, "SegmentationType", "")).upper()
    fractional_type = str(
        getattr(dataset, "SegmentationFractionalType", "")
    ).upper()
    try:
        maximum_fractional_value = int(dataset.MaximumFractionalValue)
    except (AttributeError, TypeError, ValueError) as error:
        raise FTVAuditError("invalid_seg_fractional_declaration") from error
    if (
        segmentation_type != "FRACTIONAL"
        or fractional_type != "OCCUPANCY"
        or maximum_fractional_value != 255
    ):
        raise FTVAuditError("unexpected_ispy2_analysis_mask_declaration")
    try:
        stored_pixel_contract = (
            int(dataset.BitsAllocated),
            int(dataset.BitsStored),
            int(dataset.HighBit),
            int(dataset.PixelRepresentation),
        )
    except (AttributeError, TypeError, ValueError) as error:
        raise FTVAuditError("invalid_seg_stored_pixel_contract") from error
    if stored_pixel_contract != (8, 8, 7, 0):
        raise FTVAuditError("unexpected_seg_stored_pixel_contract")

    segments = _sequence(dataset, "SegmentSequence")
    frames = _sequence(dataset, "PerFrameFunctionalGroupsSequence")
    try:
        segment_numbers = {
            int(item.SegmentNumber)
            for item in segments
            if getattr(item, "SegmentNumber", None) is not None
        }
    except (TypeError, ValueError) as error:
        raise FTVAuditError("invalid_seg_segment_number") from error
    frame_segment_numbers = set().union(
        *(_frame_segment_numbers(frame) for frame in frames)
    ) if frames else set()
    if (
        len(segments) != 1
        or len(segment_numbers) != 1
        or frame_segment_numbers != segment_numbers
    ):
        raise FTVAuditError("seg_segment_contract_mismatch")

    try:
        pixels = dataset.pixel_array
    except Exception:
        raise FTVAuditError("seg_pixel_decode_error") from None
    if tuple(pixels.shape) != (frame_count, rows, columns):
        raise FTVAuditError("unexpected_seg_pixel_shape")
    if getattr(pixels.dtype, "kind", "") not in "biu":
        raise FTVAuditError("noninteger_seg_pixels")
    unique, counts = np.unique(pixels, return_counts=True)
    raw_counts = {
        int(value): int(count)
        for value, count in zip(unique.tolist(), counts.tolist())
    }
    voxel_volume_mm3, slice_spacing_mm = _seg_voxel_volume_mm3(
        dataset, frame_count=frame_count
    )
    stats = inverse_mask_statistics_from_counts(
        raw_counts,
        voxel_volume_mm3=voxel_volume_mm3,
        allow_undocumented_bits=True,
    )
    bit_counts = stats["exclusion_bit_voxels"]
    if not isinstance(bit_counts, Mapping):
        raise FTVAuditError("invalid_inverse_ftv_statistics")
    return {
        "raw_counts": raw_counts,
        "bit_counts": {name: int(bit_counts[name]) for name in FTV_EXCLUSION_BITS},
        "undocumented_raw_code_counts": {
            int(value): int(count)
            for value, count in stats["undocumented_raw_code_voxels"].items()
        },
        "undocumented_single_bit_counts": {
            int(bit): int(count)
            for bit, count in stats["undocumented_single_bit_voxels"].items()
        },
        "contains_undocumented_bits": bool(stats["contains_undocumented_bits"]),
        "undocumented_only_raw_code_counts": {
            int(value): int(count)
            for value, count in stats[
                "undocumented_only_raw_code_voxels"
            ].items()
        },
        "undocumented_only_voxels": int(stats["undocumented_only_voxels"]),
        "contains_undocumented_only_voxels": bool(
            stats["contains_undocumented_only_voxels"]
        ),
        "raw_value_signature": str(stats["raw_value_set"]),
        "included_voxels": int(stats["included_ftv_voxels"]),
        "excluded_voxels": int(stats["excluded_voxels"]),
        "multiple_exclusion_reason_voxels": int(
            stats["multiple_exclusion_reason_voxels"]
        ),
        "total_voxels": int(stats["total_voxels"]),
        "undocumented_voxels": int(
            sum(stats["undocumented_raw_code_voxels"].values())
        ),
        "undocumented_volume_ml": float(
            sum(stats["undocumented_raw_code_voxels"].values())
            * voxel_volume_mm3
            / 1000.0
        ),
        "ftv_ml": float(stats["included_ftv_volume_ml"]),
        "voxel_volume_mm3": voxel_volume_mm3,
        "slice_spacing_mm": slice_spacing_mm,
        "dicom_patient_id_present": bool(dicom_patient_id),
        "seg_identity_sha256": seg_identity_sha256,
    }


def _numeric_summary(values: Iterable[int | float]) -> dict[str, float | int]:
    observed = [float(value) for value in values]
    if not observed or any(not math.isfinite(value) for value in observed):
        raise ValueError("numeric summaries require finite observations")
    return {
        "min": min(observed),
        "median": statistics.median(observed),
        "mean": statistics.fmean(observed),
        "max": max(observed),
    }


def summarize_visits(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create an aggregate-only summary of decoded mask rows."""

    raw_counts: Counter[int] = Counter()
    bit_counts: Counter[str] = Counter()
    undocumented_raw_counts: Counter[int] = Counter()
    undocumented_only_raw_counts: Counter[int] = Counter()
    undocumented_single_bit_counts: Counter[int] = Counter()
    signatures: Counter[str] = Counter()
    masks_with_undocumented_only_voxels = 0
    for row in rows:
        raw_counts.update(row["raw_counts"])
        bit_counts.update(row["bit_counts"])
        row_undocumented = {
            int(code): int(count)
            for code, count in row.get("undocumented_raw_code_counts", {}).items()
        }
        undocumented_raw_counts.update(row_undocumented)
        row_undocumented_only = {
            code: count
            for code, count in row_undocumented.items()
            if code != 0 and code & FTV_ALLOWED_BITS == 0
        }
        undocumented_only_raw_counts.update(row_undocumented_only)
        masks_with_undocumented_only_voxels += bool(row_undocumented_only)
        undocumented_single_bit_counts.update(
            row.get("undocumented_single_bit_counts", {})
        )
        signatures[str(row["raw_value_signature"])] += 1
    total_voxels = int(
        sum(int(row["included_voxels"]) + int(row["excluded_voxels"]) for row in rows)
    )
    undocumented_voxels = int(sum(undocumented_raw_counts.values()))
    undocumented_only_voxels = int(sum(undocumented_only_raw_counts.values()))
    undocumented_volume_ml = float(
        sum(
            int(sum(row.get("undocumented_raw_code_counts", {}).values()))
            * float(row["voxel_volume_mm3"])
            / 1000.0
            for row in rows
        )
    )
    return {
        "masks_audited": len(rows),
        "total_voxels": total_voxels,
        "raw_code_voxels": {
            str(code): count for code, count in sorted(raw_counts.items())
        },
        "raw_value_set_signatures": dict(sorted(signatures.items())),
        "exclusion_bit_voxels": {
            name: int(bit_counts[name]) for name in FTV_EXCLUSION_BITS
        },
        "undocumented_raw_code_voxels": {
            str(code): count
            for code, count in sorted(undocumented_raw_counts.items())
        },
        "undocumented_single_bit_voxels": {
            str(bit): int(undocumented_single_bit_counts[bit])
            for bit in UNDOCUMENTED_SINGLE_BITS
        },
        "masks_with_undocumented_bits": sum(
            bool(row.get("contains_undocumented_bits", False)) for row in rows
        ),
        "undocumented_only_raw_code_voxels": {
            str(code): count
            for code, count in sorted(undocumented_only_raw_counts.items())
        },
        "undocumented_only_voxels": undocumented_only_voxels,
        "masks_with_undocumented_only_voxels": int(
            masks_with_undocumented_only_voxels
        ),
        "undocumented_nonzero_voxels": undocumented_voxels,
        "undocumented_nonzero_voxel_fraction": (
            undocumented_voxels / total_voxels if total_voxels else None
        ),
        "undocumented_nonzero_volume_ml": undocumented_volume_ml,
        "diagnostic_upper_bound_if_all_undocumented_voxels_were_included_ml": (
            float(sum(float(row["ftv_ml"]) for row in rows))
            + undocumented_volume_ml
        ),
        "included_ftv_voxels": int(sum(row["included_voxels"] for row in rows)),
        "excluded_voxels": int(sum(row["excluded_voxels"] for row in rows)),
        "multiple_exclusion_reason_voxels": int(
            sum(row["multiple_exclusion_reason_voxels"] for row in rows)
        ),
        "ftv_ml": _numeric_summary(row["ftv_ml"] for row in rows),
        "voxel_volume_mm3": _numeric_summary(
            row["voxel_volume_mm3"] for row in rows
        ),
        "slice_spacing_mm": _numeric_summary(
            row["slice_spacing_mm"] for row in rows
        ),
        "dicom_patient_id_present": {
            "true": sum(bool(row["dicom_patient_id_present"]) for row in rows),
            "false": sum(not bool(row["dicom_patient_id_present"]) for row in rows),
        },
    }


_XLSX_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _xlsx_shared_strings(archive: ZipFile) -> list[str]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return [
        "".join(node.text or "" for node in item.iter(f"{_XLSX_NS}t"))
        for item in root.iter(f"{_XLSX_NS}si")
    ]


def _xlsx_column_index(reference: str) -> int:
    letters = "".join(character for character in reference if character.isalpha())
    if not letters:
        raise ValueError("XLSX cell reference has no column")
    index = 0
    for character in letters.upper():
        index = index * 26 + ord(character) - ord("A") + 1
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared: Sequence[str]) -> Any:
    kind = cell.attrib.get("t", "")
    if kind == "inlineStr":
        return "".join(
            node.text or "" for node in cell.iter(f"{_XLSX_NS}t")
        )
    node = cell.find(f"{_XLSX_NS}v")
    if node is None or node.text is None:
        return None
    if kind == "s":
        try:
            return shared[int(node.text)]
        except (IndexError, ValueError):
            raise FTVAuditError("clinical_workbook_shared_string_error") from None
    return node.text


def _xlsx_rows(archive: ZipFile, member: str, shared: Sequence[str]) -> list[dict[int, Any]]:
    try:
        root = ET.fromstring(archive.read(member))
    except (KeyError, ET.ParseError) as error:
        raise FTVAuditError("clinical_workbook_sheet_parse_error") from error
    result: list[dict[int, Any]] = []
    for row in root.iter(f"{_XLSX_NS}row"):
        values: dict[int, Any] = {}
        for cell in row.findall(f"{_XLSX_NS}c"):
            reference = str(cell.attrib.get("r", ""))
            values[_xlsx_column_index(reference)] = _xlsx_cell_value(cell, shared)
        result.append(values)
    return result


def _normalize_patient_id(value: Any) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper().startswith("ISPY2-"):
        suffix = text.split("-", 1)[1].strip()
    else:
        suffix = text
    try:
        numeric = float(suffix)
    except ValueError:
        return None
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0:
        return None
    return f"ISPY2-{int(numeric)}"


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _pearson(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    centered_left = [value - mean_left for value in left]
    centered_right = [value - mean_right for value in right]
    denominator = math.sqrt(
        sum(value * value for value in centered_left)
        * sum(value * value for value in centered_right)
    )
    if denominator == 0:
        return None
    return sum(a * b for a, b in zip(centered_left, centered_right)) / denominator


def clinical_validation_gates(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the predeclared concordance gate to every required timepoint."""

    per_timepoint: dict[str, bool] = {}
    for timepoint in TIMEPOINTS:
        values = metrics.get(timepoint, {})
        pairs = values.get("pairs")
        correlation = values.get("pearson_correlation")
        ratio = values.get("median_computed_to_reference_ratio")
        mape = values.get("mape_percent")
        try:
            passed = (
                int(pairs) >= CLINICAL_VALIDATION_THRESHOLDS["minimum_pairs_per_timepoint"]
                and float(correlation)
                >= CLINICAL_VALIDATION_THRESHOLDS["minimum_pearson_correlation"]
                and CLINICAL_VALIDATION_THRESHOLDS["minimum_median_ratio"]
                <= float(ratio)
                <= CLINICAL_VALIDATION_THRESHOLDS["maximum_median_ratio"]
                and float(mape)
                <= CLINICAL_VALIDATION_THRESHOLDS["maximum_mape_percent"]
            )
        except (TypeError, ValueError, OverflowError):
            passed = False
        per_timepoint[timepoint] = passed
    return {
        "thresholds": dict(CLINICAL_VALIDATION_THRESHOLDS),
        "per_timepoint": per_timepoint,
        "all_passed": all(per_timepoint.values()),
    }


def audit_gate_decision(
    *,
    clinical_ftv_passed: bool,
    clinical_workbook_hash_pinned: bool,
    clinical_workbook_hash_matches: bool,
    undocumented_single_bit_voxels: Mapping[int, int],
    undocumented_only_voxels: int,
    masks_with_undocumented_only_voxels: int,
) -> dict[str, Any]:
    """Return independent FTV, provenance, taxonomy, and release decisions.

    The one special release state concerns only the literal ``raw == 0`` FTV rule.
    Bit 16 remains undocumented and keeps the taxonomy blocked.  It is tolerated for
    FTV publication only when it is the sole observed unknown primitive bit and no
    voxel is excluded solely by undocumented bits.
    """

    if clinical_workbook_hash_matches and not clinical_workbook_hash_pinned:
        raise ValueError("an unpinned clinical workbook cannot match a pinned digest")
    if (
        isinstance(undocumented_only_voxels, bool)
        or not isinstance(undocumented_only_voxels, int)
        or undocumented_only_voxels < 0
        or isinstance(masks_with_undocumented_only_voxels, bool)
        or not isinstance(masks_with_undocumented_only_voxels, int)
        or masks_with_undocumented_only_voxels < 0
    ):
        raise ValueError("undocumented-only counts must be non-negative integers")
    normalized_bit_counts: dict[int, int] = {}
    for raw_bit, raw_count in undocumented_single_bit_voxels.items():
        if (
            isinstance(raw_bit, bool)
            or not isinstance(raw_bit, int)
            or raw_bit < 1
            or raw_bit > 128
            or raw_bit & (raw_bit - 1)
            or isinstance(raw_count, bool)
            or not isinstance(raw_count, int)
            or raw_count < 0
        ):
            raise ValueError("invalid undocumented single-bit count table")
        normalized_bit_counts[raw_bit] = raw_count
    observed_bits = sorted(
        bit for bit, count in normalized_bit_counts.items() if count > 0
    )
    unsupported_bits = sorted(
        set(observed_bits).difference(POLICY_PERMITTED_UNDOCUMENTED_BITS)
    )
    undocumented_only_present = (
        undocumented_only_voxels > 0
        or masks_with_undocumented_only_voxels > 0
    )
    zero_only_policy_passed = not unsupported_bits and not undocumented_only_present
    taxonomy_blocked = bool(observed_bits)
    bit16_only_release = observed_bits == list(POLICY_PERMITTED_UNDOCUMENTED_BITS)

    if not clinical_ftv_passed:
        status = "blocked_clinical_ftv_validation"
    elif not clinical_workbook_hash_pinned:
        status = "blocked_unpinned_clinical_workbook"
    elif clinical_workbook_hash_pinned and not clinical_workbook_hash_matches:
        status = "blocked_clinical_workbook_hash_mismatch"
    elif unsupported_bits:
        status = "blocked_undocumented_ftv_exclusion_bits"
    elif undocumented_only_present:
        status = "blocked_undocumented_only_ftv_exclusions"
    elif bit16_only_release:
        status = "ok_ftv_pass_taxonomy_blocked"
    else:
        status = "ok"
    releasable = status in {"ok", "ok_ftv_pass_taxonomy_blocked"}
    blockers = [
        name
        for name, blocked in (
            ("clinical_ftv_validation", not clinical_ftv_passed),
            ("clinical_workbook_unpinned", not clinical_workbook_hash_pinned),
            (
                "clinical_workbook_hash_mismatch",
                clinical_workbook_hash_pinned
                and not clinical_workbook_hash_matches,
            ),
            ("unsupported_undocumented_ftv_exclusion_bits", bool(unsupported_bits)),
            ("undocumented_only_ftv_exclusions", undocumented_only_present),
        )
        if blocked
    ]
    return {
        "status": status,
        "candidate_ftv": "pass" if clinical_ftv_passed else "blocked",
        "clinical_workbook_provenance": (
            "pass" if clinical_workbook_hash_matches else "blocked"
        ),
        "exclusion_taxonomy": "blocked" if taxonomy_blocked else "pass",
        "zero_only_ftv_release_policy": (
            "pass" if zero_only_policy_passed else "blocked"
        ),
        "observed_undocumented_single_bits": observed_bits,
        "patient_csv_publication": "approved" if releasable else "withheld",
        "release_scope": (
            "zero_rule_ftv_only"
            if status == "ok_ftv_pass_taxonomy_blocked"
            else "ftv_and_documented_taxonomy"
            if status == "ok"
            else "none"
        ),
        "taxonomy_claim_publication": (
            "approved" if status == "ok" else "withheld"
        ),
        "publication_blockers": blockers,
    }


def validate_against_clinical_workbook(
    workbook: Path,
    patient_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate FTV against the clinical XLSX using only standard-library readers."""

    required_headers = {CLINICAL_ID_COLUMN, *CLINICAL_FTV_COLUMNS.values()}
    try:
        with ZipFile(workbook) as archive:
            shared = _xlsx_shared_strings(archive)
            sheet_members = sorted(
                name
                for name in archive.namelist()
                if name.startswith("xl/worksheets/") and name.endswith(".xml")
            )
            selected_rows: list[dict[int, Any]] | None = None
            selected_header: dict[str, int] | None = None
            selected_header_index = -1
            for member in sheet_members:
                rows = _xlsx_rows(archive, member, shared)
                for index, row in enumerate(rows[:50]):
                    header = {
                        str(value).strip(): column
                        for column, value in row.items()
                        if value is not None
                    }
                    if required_headers.issubset(header):
                        if selected_rows is not None:
                            raise FTVAuditError("clinical_workbook_sheet_ambiguous")
                        selected_rows = rows
                        selected_header = header
                        selected_header_index = index
                        break
    except FileNotFoundError:
        return {"status": "not_available"}
    except (OSError, BadZipFile, ET.ParseError) as error:
        if isinstance(error, OSError):
            _raise_stale(error)
        return {"status": "unreadable_or_invalid"}

    if selected_rows is None or selected_header is None:
        return {"status": "required_columns_not_found"}
    reference: dict[str, dict[str, float]] = {}
    duplicate_ids = 0
    for raw_row in selected_rows[selected_header_index + 1 :]:
        patient_id = _normalize_patient_id(
            raw_row.get(selected_header[CLINICAL_ID_COLUMN])
        )
        if patient_id is None:
            continue
        if patient_id in reference:
            duplicate_ids += 1
            continue
        reference[patient_id] = {
            timepoint: value
            for timepoint, column_name in CLINICAL_FTV_COLUMNS.items()
            if (
                value := _optional_float(raw_row.get(selected_header[column_name]))
            ) is not None
        }
    if duplicate_ids:
        return {
            "status": "duplicate_subject_rows",
            "duplicate_subject_rows": duplicate_ids,
        }

    metrics: dict[str, Any] = {}
    matched_subjects: set[str] = set()
    for timepoint in TIMEPOINTS:
        computed: list[float] = []
        expected: list[float] = []
        key = f"cFTV_{timepoint}"
        for row in patient_rows:
            patient_id = str(row["PatientID"])
            reference_value = reference.get(patient_id, {}).get(timepoint)
            if reference_value is None:
                continue
            computed.append(float(row[key]))
            expected.append(reference_value)
            matched_subjects.add(patient_id)
        nonzero = [
            (actual, target)
            for actual, target in zip(computed, expected)
            if abs(target) > 1e-6
        ]
        metrics[timepoint] = {
            "pairs": len(computed),
            "pearson_correlation": _pearson(computed, expected),
            "mae_ml": statistics.fmean(
                abs(actual - target)
                for actual, target in zip(computed, expected)
            ) if computed else None,
            "median_computed_to_reference_ratio": statistics.median(
                actual / target for actual, target in nonzero
            ) if nonzero else None,
            "mape_percent": 100.0 * statistics.fmean(
                abs(actual - target) / abs(target)
                for actual, target in nonzero
            ) if nonzero else None,
        }
    gates = clinical_validation_gates(metrics)
    return {
        "status": "ok" if gates["all_passed"] else "failed_thresholds",
        "matched_patients_any_timepoint": len(matched_subjects),
        "metrics": metrics,
        "gates": gates,
    }


def _render_patient_csv(rows: Sequence[Mapping[str, Any]]) -> bytes:
    fields = (
        "PatientID",
        *(f"cFTV_{point}" for point in TIMEPOINTS),
        *(f"included_voxels_{point}" for point in TIMEPOINTS),
        *(f"voxel_volume_mm3_{point}" for point in TIMEPOINTS),
    )
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows({field: row[field] for field in fields} for row in rows)
    return output.getvalue().encode("utf-8")


def _write_artifacts_exclusive(
    csv_path: Path,
    csv_content: bytes,
    report_path: Path,
    report_content: bytes,
) -> None:
    """Create both outputs with mode 0600 and never replace an existing artifact.

    The aggregate release descriptor is linked first and the identifier-bearing CSV is
    linked last.  A node failure can therefore leave a harmless descriptor without its
    payload, but cannot leave a newly published patient CSV without a descriptor.  A
    consumer must require both files and verify the CSV digest recorded in the report.
    """

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: list[Path] = []
    published: list[Path] = []
    try:
        artifacts = (
            (report_path, report_content),
            (csv_path, csv_content),
        )
        for path, content in artifacts:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=f".{path.name}.",
                suffix=".tmp",
                dir=path.parent,
                delete=False,
            ) as handle:
                temporary.append(Path(handle.name))
                os.chmod(handle.name, 0o600)
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
        for staged, (destination, _) in zip(temporary, artifacts):
            try:
                os.link(staged, destination)
            except FileExistsError:
                raise FTVAuditError("output_exists_refusing_overwrite") from None
            published.append(destination)
    except Exception:
        # Remove only final names published by this invocation. A pre-existing
        # artifact that triggered O_EXCL is never touched.
        for path in published:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        for path in temporary:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass


def _write_report_exclusive(report_path: Path, report_content: bytes) -> None:
    """Publish one aggregate report atomically without creating a patient CSV."""

    report_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{report_path.name}.",
            suffix=".tmp",
            dir=report_path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            os.chmod(handle.name, 0o600)
            handle.write(report_content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, report_path)
        except FileExistsError:
            raise FTVAuditError("output_exists_refusing_overwrite") from None
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


def run_ftv_audit(
    root: str | Path,
    *,
    trajectory_path: str | Path,
    governed_root: str | Path,
    output_csv: str | Path,
    output_report: str | Path,
    expected_patients: int = EXPECTED_COMPLETE_PATIENTS,
) -> dict[str, Any]:
    """Run the complete audit, write governed artifacts, and return safe aggregates."""

    dataset_root = _safe_resolve(Path(root), code="dataset_root_unavailable")
    _require_directory(dataset_root, code="dataset_root_unavailable")
    trajectory = _safe_resolve(Path(trajectory_path), code="trajectory_unavailable")
    governed = _safe_resolve(Path(governed_root), code="governed_root_unavailable")
    csv_path = _safe_resolve(Path(output_csv), code="output_path_resolution_error")
    report_path = _safe_resolve(Path(output_report), code="output_path_resolution_error")
    if not _inside(csv_path, governed):
        raise FTVAuditError("patient_csv_outside_governed_root")
    if _inside(csv_path, dataset_root) or _inside(report_path, dataset_root):
        raise FTVAuditError("output_inside_read_only_dataset")
    if csv_path == report_path:
        raise FTVAuditError("output_paths_collide")
    if csv_path.exists() or report_path.exists():
        raise FTVAuditError("output_exists_refusing_overwrite")

    document = dataset_root / MASK_DOCUMENT_RELATIVE_PATH
    observed_document_sha256 = _sha256(document)
    if observed_document_sha256 != APPROVED_MASK_DOCUMENT_SHA256:
        raise FTVAuditError("mask_semantics_document_hash_mismatch")
    observed_trajectory_sha256 = _sha256(trajectory)
    if observed_trajectory_sha256 != APPROVED_TRAJECTORY_SHA256:
        raise FTVAuditError("trajectory_manifest_hash_mismatch")
    split_patient_ids, split_counts, split_sha256 = _approved_split_patient_ids(
        dataset_root, expected_patients=expected_patients
    )
    cohort = _cohort_from_manifest(
        dataset_root,
        trajectory,
        expected_patients=expected_patients,
        expected_patient_ids=split_patient_ids,
    )
    dicom_root = _safe_resolve(
        dataset_root / "dicom", code="dicom_root_unavailable"
    )

    patient_rows: list[dict[str, Any]] = []
    visits_by_timepoint: dict[str, list[dict[str, Any]]] = {
        point: [] for point in TIMEPOINTS
    }
    seg_identities: set[str] = set()
    for patient_id, visits in cohort:
        patient_row: dict[str, Any] = {"PatientID": patient_id}
        for timepoint in TIMEPOINTS:
            visit = inspect_inverse_mask(
                visits[timepoint],
                dicom_root=dicom_root,
                expected_patient_id=patient_id,
            )
            seg_identity = str(visit["seg_identity_sha256"])
            if seg_identity in seg_identities:
                raise FTVAuditError("duplicate_seg_sop_instance_uid")
            seg_identities.add(seg_identity)
            visits_by_timepoint[timepoint].append(visit)
            patient_row[f"cFTV_{timepoint}"] = visit["ftv_ml"]
            patient_row[f"included_voxels_{timepoint}"] = visit["included_voxels"]
            patient_row[f"voxel_volume_mm3_{timepoint}"] = visit["voxel_volume_mm3"]
        patient_rows.append(patient_row)

    workbook = dataset_root / CLINICAL_WORKBOOK_RELATIVE_PATH
    observed_clinical_workbook_sha256 = _sha256(workbook)
    clinical_workbook_hash_pinned = APPROVED_CLINICAL_WORKBOOK_SHA256 is not None
    clinical_workbook_hash_matches = (
        clinical_workbook_hash_pinned
        and observed_clinical_workbook_sha256
        == APPROVED_CLINICAL_WORKBOOK_SHA256
    )
    validation = validate_against_clinical_workbook(workbook, patient_rows)
    group_reports = {
        timepoint: summarize_visits(visits_by_timepoint[timepoint])
        for timepoint in TIMEPOINTS
    }
    masks_with_undocumented_bits = sum(
        int(group_reports[timepoint]["masks_with_undocumented_bits"])
        for timepoint in TIMEPOINTS
    )
    aggregate_undocumented_bit_counts = {
        bit: sum(
            int(
                group_reports[timepoint]["undocumented_single_bit_voxels"][
                    str(bit)
                ]
            )
            for timepoint in TIMEPOINTS
        )
        for bit in UNDOCUMENTED_SINGLE_BITS
    }
    observed_undocumented_single_bits = sorted(
        bit for bit, count in aggregate_undocumented_bit_counts.items() if count > 0
    )
    observed_undocumented_bit_union = 0
    for bit in observed_undocumented_single_bits:
        observed_undocumented_bit_union |= bit
    undocumented_only_voxels = sum(
        int(group_reports[timepoint]["undocumented_only_voxels"])
        for timepoint in TIMEPOINTS
    )
    masks_with_undocumented_only_voxels = sum(
        int(
            group_reports[timepoint][
                "masks_with_undocumented_only_voxels"
            ]
        )
        for timepoint in TIMEPOINTS
    )
    clinical_ftv_passed = validation.get("status") == "ok"
    gate_decision = audit_gate_decision(
        clinical_ftv_passed=clinical_ftv_passed,
        clinical_workbook_hash_pinned=clinical_workbook_hash_pinned,
        clinical_workbook_hash_matches=clinical_workbook_hash_matches,
        undocumented_single_bit_voxels=aggregate_undocumented_bit_counts,
        undocumented_only_voxels=undocumented_only_voxels,
        masks_with_undocumented_only_voxels=(
            masks_with_undocumented_only_voxels
        ),
    )
    status = str(gate_decision["status"])
    report: dict[str, Any] = {
        "schema_version": "med-jepa-ispy2-ftv-audit-v3",
        "status": status,
        "cohort": {
            "complete_patients": len(patient_rows),
            "required_timepoints": list(TIMEPOINTS),
            "masks_audited": sum(len(rows) for rows in visits_by_timepoint.values()),
            "unique_seg_objects": len(seg_identities),
            "split_counts": split_counts,
        },
        "input_contract": {
            "trajectory_manifest_sha256": observed_trajectory_sha256,
            "split_sha256": split_sha256,
            "clinical_workbook_sha256": observed_clinical_workbook_sha256,
            "clinical_workbook_sha256_pinned_in_source": (
                clinical_workbook_hash_pinned
            ),
            "clinical_workbook_sha256_matches_pinned": (
                clinical_workbook_hash_matches
            ),
        },
        "mask_contract": {
            "semantics": MASK_SEMANTICS,
            "included_condition": "raw_value_equals_zero",
            "allowed_exclusion_bits": dict(FTV_EXCLUSION_BITS),
            "allowed_bit_union": FTV_ALLOWED_BITS,
            "undocumented_single_bits": list(UNDOCUMENTED_SINGLE_BITS),
            "document_sha256": observed_document_sha256,
        },
        "undocumented_bit_audit": {
            "present": bool(observed_undocumented_single_bits),
            "masks_with_undocumented_bits": masks_with_undocumented_bits,
            "undocumented_single_bit_voxels": {
                str(bit): count
                for bit, count in sorted(
                    aggregate_undocumented_bit_counts.items()
                )
            },
            "observed_undocumented_single_bits": (
                observed_undocumented_single_bits
            ),
            "observed_undocumented_bit_union": (
                observed_undocumented_bit_union
            ),
            "undocumented_only_voxels": undocumented_only_voxels,
            "masks_with_undocumented_only_voxels": (
                masks_with_undocumented_only_voxels
            ),
            "only_reviewed_bit16": observed_undocumented_single_bits == [
                FTV_RELEASE_TOLERATED_UNDOCUMENTED_BIT_UNION
            ],
            "unknown_bits_added_to_allowlist": False,
            "candidate_ftv_rule": "raw_value_equals_zero",
        },
        "zero_only_ftv_release_policy": {
            "policy_id": ZERO_ONLY_FTV_RELEASE_POLICY_ID,
            "decision": gate_decision["zero_only_ftv_release_policy"],
            "candidate_ftv_inclusion_rule": "raw_value_equals_zero",
            "permitted_undocumented_single_bits": list(
                POLICY_PERMITTED_UNDOCUMENTED_BITS
            ),
            "observed_undocumented_single_bits": (
                observed_undocumented_single_bits
            ),
            "requires_undocumented_only_voxels_zero": True,
            "undocumented_only_voxels": undocumented_only_voxels,
            "masks_with_undocumented_only_voxels": (
                masks_with_undocumented_only_voxels
            ),
            "assigns_semantics_to_undocumented_bits": False,
        },
        "gate_decision": gate_decision,
        "groups": group_reports,
        "clinical_ftv_validation": validation,
        "raw_dataset_modified": False,
    }
    if gate_decision["patient_csv_publication"] != "approved":
        report["patient_csv"] = {
            "written": False,
            "candidate_rows_retained_in_memory_only": len(patient_rows),
            "blocked_reason": status,
        }
        report_content = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
        _write_report_exclusive(report_path, report_content)
        return report

    csv_content = _render_patient_csv(patient_rows)
    csv_sha256 = hashlib.sha256(csv_content).hexdigest()
    report["patient_csv"] = {
        "written": True,
        "rows": len(patient_rows),
        "sha256": csv_sha256,
        "contains_identifiers": True,
        "governed_output": True,
        "consumer_contract": EXPECTED_CONSUMER_CONTRACT,
    }
    report_content = (
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    _write_artifacts_exclusive(csv_path, csv_content, report_path, report_content)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit the complete DS3 I-SPY2 inverse-mask FTV cohort"
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ISPY2_ROOT)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--governed-root", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = run_ftv_audit(
            args.root,
            trajectory_path=args.trajectory,
            governed_root=args.governed_root,
            output_csv=args.output_csv,
            output_report=args.output_report,
        )
    except StaleFileHandleError:
        print("ERROR: stale_nfs_file_handle", file=sys.stderr)
        return 2
    except FTVAuditError as error:
        print(f"ERROR: {error.code}", file=sys.stderr)
        return 3
    except Exception as error:  # Last-resort redaction: never print source paths/IDs.
        print(f"ERROR: unexpected_{type(error).__name__}", file=sys.stderr)
        return 4
    publication_approved = (
        report.get("gate_decision", {}).get("patient_csv_publication")
        == "approved"
    )
    if not publication_approved:
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "complete_patients": report["cohort"]["complete_patients"],
                    "masks_audited": report["cohort"]["masks_audited"],
                    "masks_with_undocumented_bits": report[
                        "undocumented_bit_audit"
                    ]["masks_with_undocumented_bits"],
                    "clinical_ftv_validation": report["clinical_ftv_validation"][
                        "status"
                    ],
                    "patient_csv_written": bool(
                        report.get("patient_csv", {}).get("written", False)
                    ),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 3
    print(
        json.dumps(
            {
                "status": report["status"],
                "complete_patients": report["cohort"]["complete_patients"],
                "masks_audited": report["cohort"]["masks_audited"],
                "patient_csv_written": bool(
                    report.get("patient_csv", {}).get("written", False)
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
