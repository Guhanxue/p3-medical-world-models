"""Aggregate-only probe of DICOM SEG frame mapping to ISPY2 SER slices.

This is the final header-only gate before reading pixels.  It does not assume that
SEG frame order equals filesystem or SER slice order.  A sampled pair is accepted
only when every SEG frame maps to a SER image by referenced SOP Instance UID, or by
physical position in the same Frame of Reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError
from med_jepa.ispy2_series_probe import DEFAULT_TRAJECTORY, _inside, _raise_stale


SEGMENTATION_STORAGE_UID = "1.2.840.10008.5.1.4.1.1.66.4"
PAIR_COLUMNS = ("ser_dir", "mask_dir")


def _sequence(value: Any, name: str) -> list[Any]:
    sequence = getattr(value, name, None)
    if sequence is None:
        return []
    try:
        return list(sequence)
    except TypeError:
        return []


def _float_vector(value: Any, length: int) -> tuple[float, ...] | None:
    try:
        result = tuple(float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if len(result) != length or any(not math.isfinite(item) for item in result):
        return None
    return result


def _frame_source_uids(frame: Any) -> set[str]:
    result: set[str] = set()
    for derivation in _sequence(frame, "DerivationImageSequence"):
        for source in _sequence(derivation, "SourceImageSequence"):
            value = str(getattr(source, "ReferencedSOPInstanceUID", "")).strip()
            if value:
                result.add(value)
    return result


def _frame_position(frame: Any) -> tuple[float, float, float] | None:
    for plane in _sequence(frame, "PlanePositionSequence"):
        position = _float_vector(getattr(plane, "ImagePositionPatient", None), 3)
        if position is not None:
            return position
    return None


def _frame_segment_numbers(frame: Any) -> set[int]:
    result: set[int] = set()
    for identification in _sequence(frame, "SegmentIdentificationSequence"):
        try:
            result.add(int(identification.ReferencedSegmentNumber))
        except (AttributeError, TypeError, ValueError):
            continue
    return result


def _nested_vector(
    shared: Any,
    frames: Sequence[Any],
    *,
    sequence_name: str,
    attribute: str,
    length: int,
) -> tuple[float, ...] | None:
    containers = [shared, *frames]
    for container in containers:
        if container is None:
            continue
        for item in _sequence(container, sequence_name):
            value = _float_vector(getattr(item, attribute, None), length)
            if value is not None:
                return value
    return None


def _positions_match_count(
    observed: Iterable[tuple[float, float, float]],
    expected: Iterable[tuple[float, float, float]],
    *,
    tolerance_mm: float = 0.01,
) -> int:
    # Greedy removal makes this a one-to-one count rather than allowing duplicate
    # SEG positions to repeatedly match a single source slice.
    remaining = list(expected)
    matches = 0
    for position in observed:
        match_index = next(
            (
                index
                for index, candidate in enumerate(remaining)
                if max(abs(left - right) for left, right in zip(position, candidate))
                <= tolerance_mm
            ),
            None,
        )
        if match_index is not None:
            matches += 1
            remaining.pop(match_index)
    return matches


def _vectors_close(
    left: tuple[float, ...] | None,
    right: tuple[float, ...] | None,
    *,
    tolerance: float = 1e-4,
) -> bool:
    return (
        left is not None
        and right is not None
        and len(left) == len(right)
        and max(abs(a - b) for a, b in zip(left, right)) <= tolerance
    )


def audit_manifest_pairs(
    root: str | Path,
    trajectory_path: str | Path | None = None,
    *,
    timepoints: Sequence[str] = ("T0", "T1", "T3"),
) -> tuple[dict[str, Any], dict[str, list[tuple[Path, Path]]]]:
    dataset_root = Path(root).expanduser().resolve()
    dicom_root = (dataset_root / "dicom").resolve()
    manifest = (
        Path(trajectory_path).expanduser().resolve()
        if trajectory_path is not None
        else dataset_root / DEFAULT_TRAJECTORY
    )
    counters: dict[str, Counter[str]] = defaultdict(Counter)
    patient_pairs: dict[str, dict[str, tuple[Path, Path]]] = defaultdict(dict)
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"tp_label", *PAIR_COLUMNS}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("trajectory manifest lacks tp_label/SER/mask columns")
            normalized_fields = {
                "".join(character for character in name.casefold() if character.isalnum()): name
                for name in reader.fieldnames
            }
            patient_column = next(
                (
                    normalized_fields[name]
                    for name in ("patientid", "patient")
                    if name in normalized_fields
                ),
                None,
            )
            if patient_column is None:
                raise ValueError("trajectory manifest lacks a patient identifier column")
            for row in reader:
                timepoint = str(row.get("tp_label") or "").strip().upper()
                if timepoint not in timepoints:
                    continue
                patient_id = str(row.get(patient_column) or "").strip()
                if not patient_id:
                    counters[timepoint]["missing_patient_id"] += 1
                    continue
                raw_values = [str(row.get(column) or "").strip() for column in PAIR_COLUMNS]
                if any(not value for value in raw_values):
                    counters[timepoint]["missing_pair_path"] += 1
                    continue
                resolved: list[Path] = []
                for raw in raw_values:
                    candidate = Path(raw).expanduser()
                    candidate = (
                        candidate.resolve()
                        if candidate.is_absolute()
                        else (manifest.parent / candidate).resolve()
                    )
                    resolved.append(candidate)
                if any(not _inside(path, dicom_root) for path in resolved):
                    counters[timepoint]["outside_ds3_dicom"] += 1
                    continue
                if timepoint in patient_pairs[patient_id]:
                    raise ValueError("trajectory manifest duplicates a patient/timepoint pair")
                counters[timepoint]["inside_ds3_pair"] += 1
                patient_pairs[patient_id][timepoint] = (resolved[0], resolved[1])
    except OSError as error:
        _raise_stale(error)
        raise
    required_points = set(timepoints)
    complete_patients = sorted(
        patient_id
        for patient_id, visits in patient_pairs.items()
        if required_points.issubset(visits)
    )
    pairs: dict[str, list[tuple[Path, Path]]] = {
        point: [patient_pairs[patient_id][point] for patient_id in complete_patients]
        for point in timepoints
    }
    path_report = {
        point: dict(sorted(counters[point].items())) for point in timepoints
    }
    path_report["complete_trajectory_cohort"] = {
        "patients": len(complete_patients),
        "required_timepoints": len(timepoints),
    }
    return path_report, pairs


def _dicom_objects(path: Path) -> list[Any]:
    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - cluster-only dependency
        raise RuntimeError("pydicom is required for the SEG mapping probe") from error
    try:
        files = sorted(item for item in path.iterdir() if item.is_file())
    except OSError as error:
        _raise_stale(error)
        raise
    objects: list[Any] = []
    for file_path in files:
        try:
            objects.append(pydicom.dcmread(file_path, stop_before_pixels=True, force=False))
        except pydicom.errors.InvalidDicomError:
            continue
        except OSError as error:
            _raise_stale(error)
            raise
    return objects


def inspect_ser_seg_pair(ser_dir: Path, mask_dir: Path) -> dict[str, Any]:
    ser_objects = [
        item
        for item in _dicom_objects(ser_dir)
        if str(getattr(item, "Modality", "")).upper() == "MR"
        and getattr(item, "Rows", None) is not None
        and getattr(item, "Columns", None) is not None
        and str(getattr(item, "SOPInstanceUID", "")).strip()
    ]
    seg_objects = [
        item
        for item in _dicom_objects(mask_dir)
        if str(getattr(item, "SOPClassUID", "")) == SEGMENTATION_STORAGE_UID
        and str(getattr(item, "Modality", "")).upper() == "SEG"
    ]
    if not ser_objects:
        raise ValueError("SER directory has no usable MR image objects")
    if len(seg_objects) != 1:
        raise ValueError("mask directory must contain exactly one DICOM SEG object")
    seg = seg_objects[0]

    ser_uids = {str(item.SOPInstanceUID) for item in ser_objects}
    if len(ser_uids) != len(ser_objects):
        raise ValueError("SER image SOP Instance UIDs are not unique")
    ser_rows = {int(item.Rows) for item in ser_objects}
    ser_columns = {int(item.Columns) for item in ser_objects}
    ser_series_uids = {
        str(getattr(item, "SeriesInstanceUID", "")).strip() for item in ser_objects
    } - {""}
    ser_spacings = {
        value
        for item in ser_objects
        if (value := _float_vector(getattr(item, "PixelSpacing", None), 2)) is not None
    }
    ser_orientations = {
        value
        for item in ser_objects
        if (
            value := _float_vector(getattr(item, "ImageOrientationPatient", None), 6)
        )
        is not None
    }
    ser_positions = [
        position
        for item in ser_objects
        if (position := _float_vector(getattr(item, "ImagePositionPatient", None), 3))
        is not None
    ]
    ser_frame_refs = {
        str(getattr(item, "FrameOfReferenceUID", "")).strip() for item in ser_objects
    } - {""}

    try:
        frame_count = int(seg.NumberOfFrames)
        seg_rows = int(seg.Rows)
        seg_columns = int(seg.Columns)
    except (AttributeError, TypeError, ValueError) as error:
        raise ValueError("DICOM SEG lacks required frame/matrix attributes") from error
    frames = _sequence(seg, "PerFrameFunctionalGroupsSequence")
    shared_items = _sequence(seg, "SharedFunctionalGroupsSequence")
    shared = shared_items[0] if len(shared_items) == 1 else None
    source_sets = [_frame_source_uids(frame) for frame in frames]
    positions = [_frame_position(frame) for frame in frames]
    segment_sets = [_frame_segment_numbers(frame) for frame in frames]
    referenced_frames = sum(bool(values) for values in source_sets)
    uid_matched_frames = sum(bool(values.intersection(ser_uids)) for values in source_sets)
    valid_positions = [position for position in positions if position is not None]
    position_matches = _positions_match_count(valid_positions, ser_positions)
    segment_numbers = set().union(*segment_sets) if segment_sets else set()
    segment_items = _sequence(seg, "SegmentSequence")
    segment_count = len(segment_items)
    algorithm_types = {
        str(getattr(item, "SegmentAlgorithmType", "missing")).upper()
        for item in segment_items
    }
    seg_frame_ref = str(getattr(seg, "FrameOfReferenceUID", "")).strip()
    frame_of_reference_match = (
        bool(seg_frame_ref) and len(ser_frame_refs) == 1 and seg_frame_ref in ser_frame_refs
    )
    matrix_match = len(ser_rows) == 1 and len(ser_columns) == 1 and (
        seg_rows in ser_rows and seg_columns in ser_columns
    )
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

    spacing_match = len(ser_spacings) == 1 and _vectors_close(
        spacing, next(iter(ser_spacings), None)
    )
    orientation_match = len(ser_orientations) == 1 and _vectors_close(
        orientation, next(iter(ser_orientations), None)
    )
    geometry_exact = (
        len(frames) == frame_count
        and len(valid_positions) == frame_count
        and position_matches == frame_count
        and frame_count == len(ser_objects)
        and frame_of_reference_match
        and matrix_match
        and spacing_match
        and orientation_match
    )

    mapping_status = "unmapped"
    uid_exact = referenced_frames == frame_count and uid_matched_frames == frame_count
    if geometry_exact and uid_exact:
        mapping_status = "uid_and_geometry_exact"
    elif geometry_exact:
        mapping_status = "geometry_exact"

    segmentation_type = str(getattr(seg, "SegmentationType", "missing")).upper()
    fractional_type = str(
        getattr(seg, "SegmentationFractionalType", "not_applicable")
    ).upper()
    try:
        maximum_fractional_value = int(seg.MaximumFractionalValue)
    except (AttributeError, TypeError, ValueError):
        maximum_fractional_value = 0
    semantics_ok = segmentation_type == "BINARY" or (
        segmentation_type == "FRACTIONAL" and maximum_fractional_value > 0
    )
    usable = (
        mapping_status in {"uid_and_geometry_exact", "geometry_exact"}
        and segment_count == 1
        and len(segment_numbers) == 1
        and semantics_ok
    )
    return {
        "_frame_of_reference_uid": next(iter(ser_frame_refs), "")
        if len(ser_frame_refs) == 1
        else "",
        "ser_images": len(ser_objects),
        "ser_rows": next(iter(ser_rows), 0) if len(ser_rows) == 1 else 0,
        "ser_columns": next(iter(ser_columns), 0) if len(ser_columns) == 1 else 0,
        "seg_rows": seg_rows,
        "seg_columns": seg_columns,
        "ser_single_series": len(ser_series_uids) == 1,
        "ser_uniform_pixel_spacing": len(ser_spacings) == 1,
        "ser_uniform_orientation": len(ser_orientations) == 1,
        "seg_frames": frame_count,
        "per_frame_groups": len(frames),
        "segment_count": segment_count,
        "frame_segment_numbers": len(segment_numbers),
        "segmentation_type": segmentation_type,
        "segmentation_fractional_type": fractional_type,
        "maximum_fractional_value": maximum_fractional_value,
        "segment_algorithm_types": ",".join(sorted(algorithm_types)),
        "referenced_frames": referenced_frames,
        "uid_matched_frames": uid_matched_frames,
        "position_frames": len(valid_positions),
        "position_matched_frames": position_matches,
        "frame_of_reference_match": frame_of_reference_match,
        "matrix_match": matrix_match,
        "functional_pixel_spacing": spacing is not None,
        "functional_orientation": orientation is not None,
        "pixel_spacing_match": spacing_match,
        "orientation_match": orientation_match,
        "mapping_status": mapping_status,
        "usable_for_roi": usable,
    }


def _distribution(values: Iterable[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def summarize_pair_rows(rows: Sequence[Mapping[str, Any]], errors: int) -> dict[str, Any]:
    fields = (
        "ser_images",
        "ser_rows",
        "ser_columns",
        "seg_rows",
        "seg_columns",
        "ser_single_series",
        "ser_uniform_pixel_spacing",
        "ser_uniform_orientation",
        "seg_frames",
        "per_frame_groups",
        "segment_count",
        "frame_segment_numbers",
        "segmentation_type",
        "segmentation_fractional_type",
        "maximum_fractional_value",
        "segment_algorithm_types",
        "referenced_frames",
        "uid_matched_frames",
        "position_frames",
        "position_matched_frames",
        "frame_of_reference_match",
        "matrix_match",
        "functional_pixel_spacing",
        "functional_orientation",
        "pixel_spacing_match",
        "orientation_match",
        "mapping_status",
        "usable_for_roi",
    )
    return {
        "pairs_inspected": len(rows),
        "inspection_errors": errors,
        "distributions": {
            field: _distribution(row[field] for row in rows) for field in fields
        },
    }


def run_mapping_probe(
    root: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    max_pairs_per_timepoint: int = 8,
) -> dict[str, Any]:
    path_contract, pairs = audit_manifest_pairs(root, trajectory_path)
    outside = sum(group.get("outside_ds3_dicom", 0) for group in path_contract.values())
    report: dict[str, Any] = {
        "schema_version": "med-jepa-ispy2-seg-mapping-probe-v1",
        "path_contract": path_contract,
        "outside_ds3_dicom_total": outside,
        "pixel_data_read": False,
        "groups": {},
    }
    if outside:
        report["status"] = "blocked_wrong_manifest_root"
        return report
    all_usable = True
    rows_by_timepoint: dict[str, list[dict[str, Any]]] = {}
    for timepoint in ("T0", "T1", "T3"):
        rows: list[dict[str, Any]] = []
        errors = 0
        for ser_dir, mask_dir in pairs.get(timepoint, [])[:max_pairs_per_timepoint]:
            try:
                rows.append(inspect_ser_seg_pair(ser_dir, mask_dir))
            except (OSError, ValueError):
                errors += 1
        group = summarize_pair_rows(rows, errors)
        rows_by_timepoint[timepoint] = rows
        report["groups"][timepoint] = group
        if (
            errors
            or len(rows) != max_pairs_per_timepoint
            or any(not bool(row["usable_for_roi"]) for row in rows)
        ):
            all_usable = False
    aligned_count = min(
        (len(rows_by_timepoint.get(point, [])) for point in ("T0", "T1", "T3")),
        default=0,
    )
    same_longitudinal_frame = sum(
        len(
            {
                rows_by_timepoint[point][index]["_frame_of_reference_uid"]
                for point in ("T0", "T1", "T3")
            }
            - {""}
        )
        == 1
        for index in range(aligned_count)
    )
    report["longitudinal_geometry"] = {
        "patients_inspected": aligned_count,
        "same_frame_of_reference_all_visits": same_longitudinal_frame,
    }
    report["status"] = "ok" if all_usable else "blocked_seg_mapping"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory")
    parser.add_argument("--max-pairs-per-timepoint", type=int, default=8)
    parser.add_argument("--output", help="Create this aggregate JSON report; refuses overwrite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_pairs_per_timepoint < 1:
        raise SystemExit("max-pairs-per-timepoint must be positive")
    try:
        report = run_mapping_probe(
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
