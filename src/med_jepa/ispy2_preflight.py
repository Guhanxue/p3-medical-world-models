"""Read-only preflight for the governed ISPY2 metadata mount.

The report intentionally contains schema and aggregate counts only.  It never emits a
cell value other than a tightly sanitized timepoint category.
"""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import json
import re
import stat
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_ISPY2_ROOT = Path("data/ispy2")

REQUIRED_METADATA_FILES: tuple[Path, ...] = (
    Path("manifests/ispy2_patient_table.csv"),
    Path("manifests/ispy2_trajectory_manifest.csv"),
    Path("manifests/ispy2_split.csv"),
)

OPTIONAL_METADATA_FILES: tuple[Path, ...] = (
    Path("manifests/ispy2_series.json"),
    Path("clinical_codes/ispy2_ftv_computed.csv"),
)

KNOWN_METADATA_FILES: tuple[Path, ...] = REQUIRED_METADATA_FILES + OPTIONAL_METADATA_FILES

_MISSING_TOKENS = {"", "na", "n/a", "nan", "none", "null"}
_TIMEPOINT_COLUMN_NAMES = {
    "timepoint",
    "time_point",
    "visit",
    "visit_id",
    "n_timepoints",
}
_SAFE_TIMEPOINT = re.compile(r"^T\d+$", flags=re.IGNORECASE)
_SAFE_COUNT = re.compile(r"^\d+$")


class StaleFileHandleError(RuntimeError):
    """Raised when the CHPC NFS mount returns ``ESTALE``."""


class PreflightRootError(RuntimeError):
    """Raised when the requested dataset root is unavailable or not a directory."""


def _is_stale(error: OSError) -> bool:
    return error.errno == errno.ESTALE or "stale file handle" in str(error).lower()


def _raise_stale(error: OSError, path: Path) -> None:
    if _is_stale(error):
        raise StaleFileHandleError(
            f"CHPC NFS returned a stale file handle while reading {path}. "
            "Start a fresh login session or compute node and rerun this read-only "
            "preflight. If ESTALE persists, contact CHPC/storage support; this tool "
            "made no changes to ISPY2."
        ) from error


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    return isinstance(value, str) and value.strip().casefold() in _MISSING_TOKENS


def _safe_timepoint(value: Any) -> str:
    if _is_missing(value):
        return "<missing>"
    text = str(value).strip()
    if _SAFE_TIMEPOINT.fullmatch(text):
        return text.upper()
    if _SAFE_COUNT.fullmatch(text):
        return text
    return "<other>"


def _timepoint_columns(columns: Iterable[str]) -> set[str]:
    return {
        column
        for column in columns
        if column.strip().casefold().replace("-", "_") in _TIMEPOINT_COLUMN_NAMES
    }


def _summarize_rows(
    rows: Iterable[Mapping[str, Any]], columns: Sequence[str]
) -> dict[str, Any]:
    missing = Counter({column: 0 for column in columns})
    timepoint_names = _timepoint_columns(columns)
    timepoints: dict[str, Counter[str]] = {
        column: Counter() for column in columns if column in timepoint_names
    }
    row_count = 0
    for row in rows:
        row_count += 1
        for column in columns:
            value = row.get(column)
            if _is_missing(value):
                missing[column] += 1
            if column in timepoints:
                timepoints[column][_safe_timepoint(value)] += 1

    missingness = {
        column: {
            "missing_count": missing[column],
            "missing_rate": round(missing[column] / row_count, 6) if row_count else None,
        }
        for column in columns
    }
    return {
        "row_count": row_count,
        "columns": list(columns),
        "missingness": missingness,
        "timepoint_stats": {
            column: dict(sorted(counts.items()))
            for column, counts in timepoints.items()
        },
    }


def _csv_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError("CSV has no header")
            columns = [str(column) for column in reader.fieldnames]
            if len(set(columns)) != len(columns):
                raise ValueError("CSV contains duplicate column names")
            return _summarize_rows(reader, columns)
    except OSError as error:
        _raise_stale(error, path)
        raise


def _json_records(value: Any) -> tuple[list[Mapping[str, Any]], list[str]]:
    records: list[Mapping[str, Any]]
    if isinstance(value, list):
        records = [item for item in value if isinstance(item, Mapping)]
        if len(records) != len(value):
            raise ValueError("JSON top-level list must contain objects only")
    elif isinstance(value, Mapping):
        nested = next(
            (
                value[key]
                for key in ("records", "series", "data")
                if key in value and isinstance(value[key], list)
            ),
            None,
        )
        if nested is not None:
            records = [item for item in nested if isinstance(item, Mapping)]
            if len(records) != len(nested):
                raise ValueError("JSON record list must contain objects only")
        elif value and all(isinstance(item, Mapping) for item in value.values()):
            # Patient/series identifiers are often dictionary keys.  Treat only the
            # values as rows so those identifiers never enter the report.
            records = list(value.values())
        else:
            records = [value]
    else:
        raise ValueError("JSON root must be an object or list of objects")

    columns = sorted({str(key) for record in records for key in record})
    normalized = [{str(key): item for key, item in record.items()} for record in records]
    return normalized, columns


def _json_summary(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except OSError as error:
        _raise_stale(error, path)
        raise
    records, columns = _json_records(value)
    return _summarize_rows(records, columns)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as error:
        _raise_stale(error, path)
        raise
    return digest.hexdigest()


def inspect_metadata_file(root: Path, relative_path: Path) -> dict[str, Any]:
    """Inspect one allowlisted file without returning any data-row values."""

    path = root / relative_path
    base: dict[str, Any] = {
        "path": relative_path.as_posix(),
        "format": relative_path.suffix.lower().lstrip("."),
        "readable": False,
    }
    try:
        file_stat = path.stat()
    except FileNotFoundError:
        return {**base, "status": "missing"}
    except OSError as error:
        _raise_stale(error, path)
        return {**base, "status": "unreadable", "error_type": type(error).__name__}

    if not stat.S_ISREG(file_stat.st_mode):
        return {**base, "status": "not_a_file", "size_bytes": file_stat.st_size}

    try:
        digest = _sha256(path)
        if relative_path.suffix.lower() == ".csv":
            summary = _csv_summary(path)
        elif relative_path.suffix.lower() == ".json":
            summary = _json_summary(path)
        else:  # Defensive: the public allowlist currently contains only CSV and JSON.
            raise ValueError("unsupported metadata format")
    except StaleFileHandleError:
        raise
    except OSError as error:
        _raise_stale(error, path)
        return {
            **base,
            "status": "parse_error",
            "readable": False,
            "size_bytes": file_stat.st_size,
            "sha256": digest if "digest" in locals() else None,
            "error_type": type(error).__name__,
        }
    except (UnicodeError, csv.Error, json.JSONDecodeError, ValueError) as error:
        return {
            **base,
            "status": "parse_error",
            "readable": True,
            "size_bytes": file_stat.st_size,
            "sha256": digest if "digest" in locals() else None,
            "error_type": type(error).__name__,
        }

    return {
        **base,
        "status": "ok",
        "readable": True,
        "size_bytes": file_stat.st_size,
        "sha256": digest,
        **summary,
    }


def run_preflight(
    root: str | Path = DEFAULT_ISPY2_ROOT,
    files: Sequence[Path] = KNOWN_METADATA_FILES,
) -> dict[str, Any]:
    """Return a PHI-minimized, read-only metadata report."""

    root_path = Path(root).expanduser()
    try:
        root_stat = root_path.stat()
    except FileNotFoundError as error:
        raise PreflightRootError(f"ISPY2 root does not exist: {root_path}") from error
    except OSError as error:
        _raise_stale(error, root_path)
        raise PreflightRootError(
            f"ISPY2 root is not readable ({type(error).__name__}): {root_path}"
        ) from error
    if not stat.S_ISDIR(root_stat.st_mode):
        raise PreflightRootError(f"ISPY2 root is not a directory: {root_path}")

    reports = [inspect_metadata_file(root_path, Path(item)) for item in files]
    required_paths = {item.as_posix() for item in REQUIRED_METADATA_FILES}
    for report in reports:
        report["required"] = str(report["path"]) in required_paths
    statuses = Counter(str(report["status"]) for report in reports)
    required_reports = [report for report in reports if bool(report["required"])]
    required_problem_files = sum(
        1 for report in required_reports if report["status"] != "ok"
    )
    return {
        "root": str(root_path),
        "read_only": True,
        "patient_values_emitted": False,
        "files": reports,
        "summary": {
            "expected_files": len(reports),
            "required_files": len(required_reports),
            "ok_files": statuses.get("ok", 0),
            "problem_files": len(reports) - statuses.get("ok", 0),
            "required_problem_files": required_problem_files,
            "status_counts": dict(sorted(statuses.items())),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        default=str(DEFAULT_ISPY2_ROOT),
        help=f"ISPY2 root (default: {DEFAULT_ISPY2_ROOT})",
    )
    parser.add_argument(
        "--compact", action="store_true", help="Emit compact rather than indented JSON"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = run_preflight(args.root)
    except (StaleFileHandleError, PreflightRootError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            report,
            indent=None if args.compact else 2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0 if report["summary"]["required_problem_files"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
