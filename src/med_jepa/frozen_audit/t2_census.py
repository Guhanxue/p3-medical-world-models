"""Aggregate-only census of ISPY2 T2 metadata availability.

This module deliberately reads CSV bytes and filesystem metadata only.  It never
opens a DICOM object, tensor, image, or feature payload.  Patient identifiers,
dates, and path values are retained transiently in memory solely to compute
aggregate counts and are never written to the report or stdout.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import errno
import hashlib
import io
import json
import math
import os
import re
import stat
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPORT_SCHEMA_VERSION = "med-jepa-ispy2-t2-metadata-census-v1"
TIMEPOINTS = ("T0", "T1", "T2", "T3")
SPLITS = ("train", "val", "test")
SPLIT_BUCKETS = (*SPLITS, "unassigned")
TRIPLET = ("T0", "T1", "T3")
QUARTET = TIMEPOINTS

REQUIRED_TRAJECTORY_COLUMNS = frozenset(
    {"PatientID", "tp_label", "ser_dir", "mask_dir"}
)
PATH_COLUMNS = ("ser_dir", "mask_dir", "pe2_dir", "dce_dir")
NUMERIC_TIME_COLUMNS = (
    "delta_t_days",
    "time_days",
    "days_from_t0",
    "delta_days",
)
DATE_TIME_COLUMNS = (
    "acquisition_datetime",
    "scan_datetime",
    "acquisition_date",
    "scan_date",
    "exam_date",
    "study_date",
)

_PATIENT_ID = re.compile(r"^ISPY2-[0-9]+$")
_MISSING = {"", "na", "n/a", "nan", "none", "null"}


class T2CensusError(ValueError):
    """Fail-closed input or output contract violation."""

    def __init__(self, code: str) -> None:
        if not re.fullmatch(r"[a-z0-9_]+", code):
            raise ValueError("invalid census error code")
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class _CSVSnapshot:
    fields: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _TimeSource:
    column: str
    kind: str


@dataclass(frozen=True)
class _TrajectoryRow:
    patient: str
    timepoint: str
    raw: Mapping[str, str]
    path_status: Mapping[str, str]
    time_value: float | None


def _missing(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in _MISSING


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_csv_snapshot(path: str | Path, *, label: str) -> _CSVSnapshot:
    source = Path(path).expanduser()
    try:
        payload = source.read_bytes()
    except OSError as error:
        raise T2CensusError(f"{label}_csv_unavailable") from error
    try:
        text = payload.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise T2CensusError(f"{label}_csv_not_utf8") from error
    if "\x00" in text:
        raise T2CensusError(f"{label}_csv_contains_nul")
    try:
        reader = csv.DictReader(io.StringIO(text, newline=""))
        raw_fields = reader.fieldnames
        if raw_fields is None:
            raise T2CensusError(f"{label}_csv_missing_header")
        fields = tuple(str(value) for value in raw_fields)
        if (
            not fields
            or any(not value or value != value.strip() for value in fields)
            or len(fields) != len(set(fields))
        ):
            raise T2CensusError(f"{label}_csv_invalid_header")
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise T2CensusError(f"{label}_csv_row_width_mismatch")
            rows.append({field: str(row[field]) for field in fields})
    except csv.Error as error:
        raise T2CensusError(f"{label}_csv_parse_error") from error
    if not rows:
        raise T2CensusError(f"{label}_csv_empty")
    return _CSVSnapshot(
        fields=fields,
        rows=tuple(rows),
        sha256=_sha256_bytes(payload),
        size_bytes=len(payload),
    )


def _patient(value: Any, *, label: str) -> str:
    patient = str(value or "").strip()
    if not _PATIENT_ID.fullmatch(patient):
        raise T2CensusError(f"{label}_invalid_patient_id")
    return patient


def _timepoint(value: Any, *, label: str) -> str:
    point = str(value or "").strip().upper()
    if point not in TIMEPOINTS:
        raise T2CensusError(f"{label}_invalid_timepoint")
    return point


def _split(value: Any, *, label: str) -> str:
    split = str(value or "").strip().casefold()
    if split not in SPLITS:
        raise T2CensusError(f"{label}_invalid_split")
    return split


def _select_time_source(fields: Sequence[str]) -> _TimeSource | None:
    present = set(fields)
    for column in NUMERIC_TIME_COLUMNS:
        if column in present:
            return _TimeSource(column=column, kind="relative_days")
    for column in DATE_TIME_COLUMNS:
        if column in present:
            return _TimeSource(column=column, kind="calendar")
    return None


def _parse_calendar(value: str) -> float | None:
    text = value.strip()
    candidates = (text, text.rstrip("Z"))
    for candidate in candidates:
        try:
            return dt.datetime.fromisoformat(candidate).timestamp()
        except (ValueError, OverflowError, OSError):
            pass
    for pattern in (
        "%Y%m%d",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%Y%m%d%H%M%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return dt.datetime.strptime(text, pattern).timestamp()
        except (ValueError, OverflowError, OSError):
            continue
    return None


def _parse_time(value: Any, source: _TimeSource | None) -> float | None:
    if source is None or _missing(value):
        return None
    text = str(value).strip()
    if source.kind == "relative_days":
        try:
            result = float(text)
        except ValueError:
            return None
        return result if math.isfinite(result) else None
    return _parse_calendar(text)


def _path_status(value: Any, *, base: Path) -> str:
    if _missing(value):
        return "missing_value"
    try:
        candidate = Path(str(value).strip()).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        metadata = os.stat(candidate)
    except FileNotFoundError:
        return "not_found"
    except PermissionError:
        return "permission_denied"
    except OSError as error:
        if error.errno == errno.ESTALE:
            return "stale_file_handle"
        return "os_error"
    if stat.S_ISDIR(metadata.st_mode):
        return "exists_directory"
    if stat.S_ISREG(metadata.st_mode):
        return "exists_file"
    return "exists_other"


def _parse_split_snapshot(snapshot: _CSVSnapshot) -> dict[str, str]:
    required = {"PatientID", "split"}
    if not required.issubset(snapshot.fields):
        raise T2CensusError("split_csv_schema_mismatch")
    result: dict[str, str] = {}
    for row in snapshot.rows:
        patient = _patient(row["PatientID"], label="split_csv")
        if patient in result:
            raise T2CensusError("split_csv_duplicate_patient")
        result[patient] = _split(row["split"], label="split_csv")
    return result


def _parse_released_snapshot(
    snapshot: _CSVSnapshot,
) -> tuple[set[str], dict[str, str]]:
    if "PatientID" not in snapshot.fields:
        raise T2CensusError("released_csv_schema_mismatch")
    has_timepoint = "tp_label" in snapshot.fields
    has_split = "split" in snapshot.fields
    released: set[str] = set()
    inline_splits: dict[str, str] = {}
    seen_rows: set[tuple[str, str | None]] = set()
    for row in snapshot.rows:
        patient = _patient(row["PatientID"], label="released_csv")
        point = (
            _timepoint(row["tp_label"], label="released_csv")
            if has_timepoint
            else None
        )
        key = (patient, point)
        if key in seen_rows:
            raise T2CensusError("released_csv_duplicate_patient_timepoint")
        if not has_timepoint and patient in released:
            raise T2CensusError("released_csv_duplicate_patient")
        seen_rows.add(key)
        released.add(patient)
        if has_split:
            current = _split(row["split"], label="released_csv")
            previous = inline_splits.get(patient)
            if previous is not None and previous != current:
                raise T2CensusError("released_csv_inconsistent_split")
            inline_splits[patient] = current
    return released, inline_splits


def _merge_splits(
    external: Mapping[str, str], inline: Mapping[str, str]
) -> dict[str, str]:
    result = dict(external)
    for patient, split in inline.items():
        if patient in result and result[patient] != split:
            raise T2CensusError("released_and_split_csv_disagree")
        result[patient] = split
    return result


def _parse_trajectory(
    snapshot: _CSVSnapshot,
    *,
    base: Path,
) -> tuple[list[_TrajectoryRow], _TimeSource | None, tuple[str, ...]]:
    if not REQUIRED_TRAJECTORY_COLUMNS.issubset(snapshot.fields):
        raise T2CensusError("trajectory_csv_schema_mismatch")
    time_source = _select_time_source(snapshot.fields)
    available_paths = tuple(column for column in PATH_COLUMNS if column in snapshot.fields)
    rows: list[_TrajectoryRow] = []
    for row in snapshot.rows:
        patient = _patient(row["PatientID"], label="trajectory_csv")
        point = _timepoint(row["tp_label"], label="trajectory_csv")
        rows.append(
            _TrajectoryRow(
                patient=patient,
                timepoint=point,
                raw=row,
                path_status={
                    column: _path_status(row[column], base=base)
                    for column in available_paths
                },
                time_value=_parse_time(
                    row.get(time_source.column) if time_source is not None else None,
                    time_source,
                ),
            )
        )
    return rows, time_source, available_paths


def _bucket(patient: str, splits: Mapping[str, str]) -> str:
    return splits.get(patient, "unassigned")


def _ordered_counts(counter: Mapping[str, int]) -> dict[str, int]:
    return {name: int(counter.get(name, 0)) for name in SPLIT_BUCKETS}


def _rows_and_patients_by_split(
    rows: Iterable[_TrajectoryRow], splits: Mapping[str, str]
) -> dict[str, dict[str, int]]:
    row_counts: Counter[str] = Counter()
    patients: dict[str, set[str]] = {name: set() for name in SPLIT_BUCKETS}
    for row in rows:
        split = _bucket(row.patient, splits)
        row_counts[split] += 1
        patients[split].add(row.patient)
    return {
        name: {"rows": int(row_counts[name]), "patients": len(patients[name])}
        for name in SPLIT_BUCKETS
    }


def _complete_set_summary(
    grouped: Mapping[str, Mapping[str, Sequence[_TrajectoryRow]]],
    required: Sequence[str],
    splits: Mapping[str, str],
    *,
    time_source: _TimeSource | None,
) -> dict[str, Any]:
    row_complete: list[str] = []
    exact: list[str] = []
    exact_ser: list[str] = []
    exact_ser_mask: list[str] = []
    usable_ser: list[str] = []
    usable_ser_mask: list[str] = []
    required_set = set(required)
    for patient, visits in grouped.items():
        if required_set.issubset(visits):
            row_complete.append(patient)
            if all(len(visits[point]) == 1 for point in required):
                exact.append(patient)
                rows = [visits[point][0] for point in required]
                ser_ready = all(
                    row.path_status.get("ser_dir") == "exists_directory"
                    for row in rows
                )
                mask_ready = all(
                    row.path_status.get("mask_dir") == "exists_directory"
                    for row in rows
                )
                if ser_ready:
                    exact_ser.append(patient)
                if ser_ready and mask_ready:
                    exact_ser_mask.append(patient)
                values = [row.time_value for row in rows]
                chronology_ready = (
                    time_source is not None
                    and all(value is not None for value in values)
                    and all(
                        float(left) < float(right)
                        for left, right in zip(values, values[1:])
                    )
                )
                if ser_ready and chronology_ready:
                    usable_ser.append(patient)
                if ser_ready and mask_ready and chronology_ready:
                    usable_ser_mask.append(patient)

    def counts(patients: Iterable[str]) -> dict[str, int]:
        return _ordered_counts(
            Counter(_bucket(patient, splits) for patient in patients)
        )

    return {
        "required_timepoints": list(required),
        "row_complete_patients": len(row_complete),
        "row_complete_by_split": counts(row_complete),
        "exactly_one_row_per_timepoint_patients": len(exact),
        "exactly_one_row_per_timepoint_by_split": counts(exact),
        "exact_ser_path_patients": len(exact_ser),
        "exact_ser_path_by_split": counts(exact_ser),
        "exact_ser_and_mask_path_patients": len(exact_ser_mask),
        "exact_ser_and_mask_path_by_split": counts(exact_ser_mask),
        "usable_exact_ser_patients": (
            len(usable_ser) if time_source is not None else None
        ),
        "usable_exact_ser_by_split": (
            counts(usable_ser) if time_source is not None else None
        ),
        "usable_exact_ser_and_mask_patients": (
            len(usable_ser_mask) if time_source is not None else None
        ),
        "usable_exact_ser_and_mask_by_split": (
            counts(usable_ser_mask) if time_source is not None else None
        ),
        "usable_exact_definition": (
            "exactly_one_row_per_required_timepoint_and_required_paths_exist_"
            "and_time_is_strictly_increasing"
        ),
        "usable_exact_status": (
            "assessed" if time_source is not None else "not_assessable_without_time_column"
        ),
    }


def _chronology_summary(
    grouped: Mapping[str, Mapping[str, Sequence[_TrajectoryRow]]],
    required: Sequence[str],
    splits: Mapping[str, str],
) -> dict[str, Any]:
    statuses = (
        "strictly_increasing",
        "not_strictly_increasing",
        "missing_or_invalid_time",
        "duplicate_timepoint",
    )
    overall: Counter[str] = Counter()
    by_split: dict[str, Counter[str]] = {
        name: Counter() for name in SPLIT_BUCKETS
    }
    required_set = set(required)
    for patient, visits in grouped.items():
        if not required_set.issubset(visits):
            continue
        if any(len(visits[point]) != 1 for point in required):
            status = "duplicate_timepoint"
        else:
            values = [visits[point][0].time_value for point in required]
            if any(value is None for value in values):
                status = "missing_or_invalid_time"
            else:
                numeric = [float(value) for value in values if value is not None]
                status = (
                    "strictly_increasing"
                    if all(left < right for left, right in zip(numeric, numeric[1:]))
                    else "not_strictly_increasing"
                )
        overall[status] += 1
        by_split[_bucket(patient, splits)][status] += 1
    return {
        "required_timepoints": list(required),
        "complete_patients_assessed_or_categorized": sum(overall.values()),
        "status_counts": {name: int(overall[name]) for name in statuses},
        "status_counts_by_split": {
            split: {name: int(by_split[split][name]) for name in statuses}
            for split in SPLIT_BUCKETS
        },
    }


def _population_report(
    rows: Sequence[_TrajectoryRow],
    *,
    splits: Mapping[str, str],
    time_source: _TimeSource | None,
    available_paths: Sequence[str],
    declared_patients: set[str] | None = None,
) -> dict[str, Any]:
    observed_patients = {row.patient for row in rows}
    grouped: dict[str, dict[str, list[_TrajectoryRow]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in rows:
        grouped[row.patient][row.timepoint].append(row)

    timepoints: dict[str, Any] = {}
    for point in TIMEPOINTS:
        selected = [row for row in rows if row.timepoint == point]
        selected_groups = [
            visits[point]
            for visits in grouped.values()
            if point in visits and len(visits[point]) > 1
        ]
        timepoints[point] = {
            "rows": len(selected),
            "patients": len({row.patient for row in selected}),
            "duplicate_patient_timepoint_groups": len(selected_groups),
            "duplicate_excess_rows": sum(len(group) - 1 for group in selected_groups),
            "by_split": _rows_and_patients_by_split(selected, splits),
        }

    path_report: dict[str, Any] = {}
    for point in TIMEPOINTS:
        path_report[point] = {}
        for column in available_paths:
            counts = Counter(
                row.path_status[column]
                for row in rows
                if row.timepoint == point
            )
            path_report[point][column] = dict(sorted(counts.items()))

    chronology: dict[str, Any]
    if time_source is None:
        chronology = {
            "status": "not_available",
            "time_column": None,
            "time_kind": None,
        }
    else:
        chronology = {
            "status": "assessed",
            "time_column": time_source.column,
            "time_kind": time_source.kind,
            "triplet_t0_t1_t3": _chronology_summary(
                grouped, TRIPLET, splits
            ),
            "quartet_t0_t1_t2_t3": _chronology_summary(
                grouped, QUARTET, splits
            ),
        }

    duplicate_groups = sum(
        1
        for visits in grouped.values()
        for visit_rows in visits.values()
        if len(visit_rows) > 1
    )
    duplicate_excess = sum(
        len(visit_rows) - 1
        for visits in grouped.values()
        for visit_rows in visits.values()
        if len(visit_rows) > 1
    )
    report: dict[str, Any] = {
        "rows": len(rows),
        "patients_observed": len(observed_patients),
        "rows_and_patients_by_split": _rows_and_patients_by_split(rows, splits),
        "timepoints": timepoints,
        "duplicate_patient_timepoint": {
            "groups": duplicate_groups,
            "excess_rows": duplicate_excess,
        },
        "completeness": {
            "triplet_t0_t1_t3": _complete_set_summary(
                grouped, TRIPLET, splits, time_source=time_source
            ),
            "quartet_t0_t1_t2_t3": _complete_set_summary(
                grouped, QUARTET, splits, time_source=time_source
            ),
        },
        "chronology": chronology,
        "path_existence": {
            "columns_inspected": list(available_paths),
            "status_counts_by_timepoint": path_report,
            "values_emitted": False,
        },
    }
    if declared_patients is not None:
        report["patients_declared"] = len(declared_patients)
        report["declared_patients_without_trajectory_rows"] = len(
            declared_patients.difference(observed_patients)
        )
    return report


def build_t2_census(
    trajectory_csv: str | Path,
    *,
    split_csv: str | Path | None = None,
    released_patients_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Build an aggregate report from immutable CSV snapshots in memory."""

    trajectory_path = Path(trajectory_csv).expanduser()
    trajectory_snapshot = _read_csv_snapshot(
        trajectory_path, label="trajectory"
    )
    split_snapshot = (
        _read_csv_snapshot(split_csv, label="split")
        if split_csv is not None
        else None
    )
    released_snapshot = (
        _read_csv_snapshot(released_patients_csv, label="released")
        if released_patients_csv is not None
        else None
    )

    external_splits = (
        _parse_split_snapshot(split_snapshot) if split_snapshot is not None else {}
    )
    if released_snapshot is None:
        released_patients: set[str] | None = None
        inline_splits: dict[str, str] = {}
    else:
        released_patients, inline_splits = _parse_released_snapshot(
            released_snapshot
        )
    splits = _merge_splits(external_splits, inline_splits)
    rows, time_source, available_paths = _parse_trajectory(
        trajectory_snapshot,
        base=trajectory_path.parent,
    )
    trajectory_patients = {row.patient for row in rows}

    populations: dict[str, Any] = {
        "trajectory_all": _population_report(
            rows,
            splits=splits,
            time_source=time_source,
            available_paths=available_paths,
        )
    }
    if released_patients is not None:
        released_rows = [
            row for row in rows if row.patient in released_patients
        ]
        populations["released"] = _population_report(
            released_rows,
            splits=splits,
            time_source=time_source,
            available_paths=available_paths,
            declared_patients=released_patients,
        )

    split_patients = set(splits)
    evidence = {
        "trajectory_csv_sha256": trajectory_snapshot.sha256,
        "trajectory_csv_size_bytes": trajectory_snapshot.size_bytes,
        "split_csv_provided": split_snapshot is not None,
        "split_csv_sha256": (
            split_snapshot.sha256 if split_snapshot is not None else None
        ),
        "released_patients_csv_provided": released_snapshot is not None,
        "released_patients_csv_sha256": (
            released_snapshot.sha256 if released_snapshot is not None else None
        ),
    }
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "status": "ok_metadata_only_t2_census",
        "aggregate_only": True,
        "pixel_data_read": False,
        "tensor_data_read": False,
        "dicom_headers_read": False,
        "source_files_modified": False,
        "identifiers_paths_uids_dates_or_per_patient_values_emitted": False,
        "input_contract": {
            "decision_role": "descriptive_census_not_a_production_release_gate",
            "trajectory_rows": len(trajectory_snapshot.rows),
            "trajectory_patients": len(trajectory_patients),
            "split_patients": len(split_patients),
            "split_patients_without_trajectory_rows": len(
                split_patients.difference(trajectory_patients)
            ),
            "trajectory_patients_without_split": len(
                trajectory_patients.difference(split_patients)
            ),
            "released_patients": (
                len(released_patients)
                if released_patients is not None
                else None
            ),
            "released_patients_without_split": (
                len(released_patients.difference(split_patients))
                if released_patients is not None
                else None
            ),
            "released_patients_without_trajectory_rows": (
                len(released_patients.difference(trajectory_patients))
                if released_patients is not None
                else None
            ),
        },
        "evidence": evidence,
        "populations": populations,
    }


def write_report_exclusive(path: str | Path, report: Mapping[str, Any]) -> str:
    """Create one immutable mode-0600 JSON report and return its SHA-256."""

    destination = Path(path).expanduser()
    try:
        content = (
            json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise T2CensusError("report_not_json_serializable") from error
    descriptor: int | None = None
    created = False
    try:
        descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
        created = True
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb") as handle:
            descriptor = None
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except FileExistsError as error:
        raise T2CensusError("output_already_exists") from error
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        if created:
            try:
                destination.unlink()
            except OSError:
                pass
        raise T2CensusError("output_write_failed") from error
    return _sha256_bytes(content)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit T0/T1/T2/T3 metadata availability without reading pixels or tensors."
        )
    )
    parser.add_argument("--trajectory-csv", required=True)
    parser.add_argument("--split-csv")
    parser.add_argument("--released-patients-csv")
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        report = build_t2_census(
            args.trajectory_csv,
            split_csv=args.split_csv,
            released_patients_csv=args.released_patients_csv,
        )
        digest = write_report_exclusive(args.output, report)
    except T2CensusError as error:
        print(
            json.dumps(
                {"status": "error", "error_code": error.code},
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {
                "aggregate_only": True,
                "output_sha256": digest,
                "status": report["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
