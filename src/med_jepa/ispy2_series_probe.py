"""Read-only, aggregate DICOM-header probe for the ISPY2 SER/mask connection.

The probe is deliberately run before implementing a voxel converter.  It verifies
that the trajectory manifest points into the approved DS3 tree and summarizes enough
DICOM geometry to choose a reader, orientation, spacing, and fixed ROI contract.  It
never reads pixel data and never emits patient, study, series, or filesystem values.
"""

from __future__ import annotations

import argparse
import csv
import errno
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from med_jepa.ispy2_preflight import DEFAULT_ISPY2_ROOT, StaleFileHandleError


DEFAULT_TRAJECTORY = Path("manifests/ispy2_trajectory_manifest.csv")
SERIES_COLUMNS = {"ser": "ser_dir", "mask": "mask_dir"}
HEADER_TAGS = (
    "Rows",
    "Columns",
    "NumberOfFrames",
    "PixelSpacing",
    "SliceThickness",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "ImagePositionPatient",
    "InstanceNumber",
    "TemporalPositionIdentifier",
    "NumberOfTemporalPositions",
    "PhotometricInterpretation",
    "BitsAllocated",
    "Modality",
    "SOPClassUID",
)


def _is_stale(error: OSError) -> bool:
    return error.errno == errno.ESTALE or "stale file handle" in str(error).lower()


def _raise_stale(error: OSError) -> None:
    if _is_stale(error):
        raise StaleFileHandleError(
            "CHPC NFS returned a stale file handle during the DICOM header probe. "
            "Start a fresh login or compute-node session and rerun; no data were changed."
        ) from error


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def audit_manifest_paths(
    root: str | Path,
    trajectory_path: str | Path | None = None,
    *,
    timepoints: Sequence[str] = ("T0", "T1", "T3"),
) -> tuple[dict[str, Any], dict[tuple[str, str], list[Path]]]:
    """Audit roots and return approved paths grouped by kind/timepoint."""

    dataset_root = Path(root).expanduser().resolve()
    approved_dicom_root = (dataset_root / "dicom").resolve()
    manifest = (
        Path(trajectory_path).expanduser().resolve()
        if trajectory_path is not None
        else dataset_root / DEFAULT_TRAJECTORY
    )
    counters: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    approved: dict[tuple[str, str], list[Path]] = defaultdict(list)
    try:
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"tp_label", *SERIES_COLUMNS.values()}
            if reader.fieldnames is None or not required.issubset(reader.fieldnames):
                raise ValueError("trajectory manifest lacks tp_label/SER/mask columns")
            for row in reader:
                timepoint = str(row["tp_label"]).strip().upper()
                if timepoint not in timepoints:
                    continue
                for kind, column in SERIES_COLUMNS.items():
                    raw = str(row.get(column) or "").strip()
                    key = (kind, timepoint)
                    if not raw:
                        counters[key]["missing"] += 1
                        continue
                    candidate = Path(raw).expanduser()
                    candidate = (
                        candidate.resolve()
                        if candidate.is_absolute()
                        else (manifest.parent / candidate).resolve()
                    )
                    if not _inside(candidate, approved_dicom_root):
                        counters[key]["outside_ds3_dicom"] += 1
                        continue
                    counters[key]["inside_ds3_dicom"] += 1
                    approved[key].append(candidate)
    except OSError as error:
        _raise_stale(error)
        raise

    report = {
        f"{kind}/{timepoint}": dict(sorted(counters[(kind, timepoint)].items()))
        for kind in SERIES_COLUMNS
        for timepoint in timepoints
    }
    return report, approved


def _rounded_sequence(value: Any, length: int) -> str:
    try:
        values = tuple(round(float(item), 5) for item in value)
    except (TypeError, ValueError):
        return "missing"
    return ",".join(map(str, values)) if len(values) == length else "invalid"


def _scalar(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return str(int(value))
    except (TypeError, ValueError):
        return "invalid"


def _float_scalar(value: Any) -> str:
    if value is None:
        return "missing"
    try:
        return str(round(float(value), 5))
    except (TypeError, ValueError):
        return "invalid"


def _distribution(values: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))


def _is_image_header(dataset: Any) -> bool:
    """Reject force-parsed sidecars while accepting ordinary and multi-frame images."""

    return (
        _scalar(getattr(dataset, "Rows", None)) not in {"missing", "invalid"}
        and _scalar(getattr(dataset, "Columns", None)) not in {"missing", "invalid"}
    )


def inspect_series_directory(
    path: Path,
    *,
    max_files: int,
) -> dict[str, Any]:
    """Inspect DICOM headers only. The returned values contain geometry, never IDs."""

    try:
        import pydicom
    except ImportError as error:  # pragma: no cover - exercised on the cluster
        raise RuntimeError(
            "pydicom is required for the read-only series probe; install the cluster extra"
        ) from error
    try:
        files = sorted(item for item in path.iterdir() if item.is_file())
    except OSError as error:
        _raise_stale(error)
        raise
    if not files:
        raise ValueError("series directory contains no files")
    headers = []
    forced_reads = 0
    unreadable_files = 0
    for file_path in files[:max_files]:
        try:
            header = pydicom.dcmread(
                file_path,
                stop_before_pixels=True,
                force=False,
                specific_tags=list(HEADER_TAGS),
            )
            if _is_image_header(header):
                headers.append(header)
            else:
                unreadable_files += 1
        except pydicom.errors.InvalidDicomError:
            # Some ISPY2 objects are valid datasets without the optional Part 10
            # preamble. Fall back for that variation, but require core image tags so
            # that JSON/text sidecars cannot silently become the representative image.
            try:
                header = pydicom.dcmread(
                    file_path,
                    stop_before_pixels=True,
                    force=True,
                    specific_tags=list(HEADER_TAGS),
                )
                if _is_image_header(header):
                    headers.append(header)
                    forced_reads += 1
                else:
                    unreadable_files += 1
            except pydicom.errors.InvalidDicomError:
                unreadable_files += 1
            except OSError as error:
                _raise_stale(error)
                unreadable_files += 1
        except OSError as error:
            _raise_stale(error)
            unreadable_files += 1
    if not headers:
        raise ValueError("series directory has no readable DICOM headers")
    first = headers[0]
    temporal_positions = {
        _scalar(getattr(header, "TemporalPositionIdentifier", None))
        for header in headers
    }
    z_positions = {
        _float_scalar(
            (getattr(header, "ImagePositionPatient", [None, None, None]) or [None] * 3)[2]
        )
        for header in headers
    }
    return {
        "file_count": len(files),
        "headers_read": len(headers),
        "forced_header_reads": forced_reads,
        "unreadable_files": unreadable_files,
        "rows": _scalar(getattr(first, "Rows", None)),
        "columns": _scalar(getattr(first, "Columns", None)),
        "number_of_frames": _scalar(getattr(first, "NumberOfFrames", None)),
        "pixel_spacing": _rounded_sequence(getattr(first, "PixelSpacing", None), 2),
        "slice_thickness": _float_scalar(getattr(first, "SliceThickness", None)),
        "spacing_between_slices": _float_scalar(
            getattr(first, "SpacingBetweenSlices", None)
        ),
        "orientation": _rounded_sequence(
            getattr(first, "ImageOrientationPatient", None), 6
        ),
        "temporal_position_count": len(temporal_positions - {"missing", "invalid"}),
        "z_position_count": len(z_positions - {"missing", "invalid"}),
        "photometric": str(getattr(first, "PhotometricInterpretation", "missing")),
        "bits_allocated": _scalar(getattr(first, "BitsAllocated", None)),
        "modality": str(getattr(first, "Modality", "missing")),
        "sop_class": str(getattr(first, "SOPClassUID", "missing")),
    }


def summarize_probe_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"series_inspected": 0}
    categorical = (
        "file_count",
        "headers_read",
        "forced_header_reads",
        "unreadable_files",
        "rows",
        "columns",
        "number_of_frames",
        "pixel_spacing",
        "slice_thickness",
        "spacing_between_slices",
        "orientation",
        "temporal_position_count",
        "z_position_count",
        "photometric",
        "bits_allocated",
        "modality",
        "sop_class",
    )
    return {
        "series_inspected": len(rows),
        "distributions": {
            name: _distribution(str(row[name]) for row in rows) for name in categorical
        },
    }


def run_probe(
    root: str | Path,
    *,
    trajectory_path: str | Path | None = None,
    max_series_per_group: int = 8,
    max_files_per_series: int = 512,
) -> dict[str, Any]:
    path_report, approved = audit_manifest_paths(root, trajectory_path)
    outside_count = sum(
        values.get("outside_ds3_dicom", 0) for values in path_report.values()
    )
    report: dict[str, Any] = {
        "schema_version": "med-jepa-ispy2-series-probe-v1",
        "path_contract": path_report,
        "outside_ds3_dicom_total": outside_count,
        "pixel_data_read": False,
        "groups": {},
    }
    if outside_count:
        report["status"] = "blocked_wrong_manifest_root"
        return report
    for key, paths in sorted(approved.items()):
        rows = []
        errors = 0
        for path in paths[:max_series_per_group]:
            try:
                rows.append(inspect_series_directory(path, max_files=max_files_per_series))
            except (OSError, ValueError):
                errors += 1
        group = summarize_probe_rows(rows)
        group["inspection_errors"] = errors
        report["groups"][f"{key[0]}/{key[1]}"] = group
    report["status"] = "ok"
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(DEFAULT_ISPY2_ROOT))
    parser.add_argument("--trajectory")
    parser.add_argument("--max-series-per-group", type=int, default=8)
    parser.add_argument("--max-files-per-series", type=int, default=512)
    parser.add_argument("--output", help="Create this JSON report; refuses overwrite")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.max_series_per_group < 1 or args.max_files_per_series < 1:
        raise SystemExit("probe limits must be positive")
    try:
        report = run_probe(
            args.root,
            trajectory_path=args.trajectory,
            max_series_per_group=args.max_series_per_group,
            max_files_per_series=args.max_files_per_series,
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
    return 3 if report["status"] != "ok" else 0


if __name__ == "__main__":
    raise SystemExit(main())
