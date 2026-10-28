"""Validate and merge sharded 3DINO cache indices without exposing row values."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

from med_jepa.cache_3dino import OUTPUT_INDEX_COLUMNS, _valid_sha256, sha256_file
from med_jepa.ispy2_manifest_builder import (
    ManifestBuildError,
    validate_feature_provenance,
    validate_spatial_provenance,
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _atomic_json(value: Any, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing provenance sidecar")
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def merge_feature_indices(
    cache_root: str | Path,
    output_path: str | Path,
    *,
    expected_num_shards: int,
    expected_patients: int,
    expected_timepoints: Sequence[str] = ("T0", "T1", "T3"),
    verify_file_hashes: bool = True,
) -> dict[str, Any]:
    root = Path(cache_root).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if expected_num_shards < 1:
        raise ValueError("expected_num_shards must be positive")
    if destination.exists():
        raise FileExistsError("refusing to overwrite an existing merged feature index")
    feature_sidecar = destination.with_name(
        f"{destination.stem}.feature_provenance.json"
    )
    spatial_sidecar = destination.with_name(
        f"{destination.stem}.spatial_provenance.json"
    )
    if feature_sidecar.exists() or spatial_sidecar.exists():
        raise FileExistsError("refusing to overwrite an existing provenance sidecar")
    shard_dirs = [
        root / f"shard-{index:05d}-of-{expected_num_shards:05d}"
        for index in range(expected_num_shards)
    ]
    if any(not path.is_dir() for path in shard_dirs):
        raise ValueError("one or more expected cache shard directories are missing")
    unexpected = sorted(
        path for path in root.glob("shard-*-of-*") if path not in set(shard_dirs)
    )
    if unexpected:
        raise ValueError("cache root contains shard directories from another run")

    rows: list[dict[str, str]] = []
    reports: list[dict[str, Any]] = []
    seen_visits: set[tuple[str, str]] = set()
    seen_scans: set[str] = set()
    seen_feature_hashes: set[str] = set()
    seen_source_hashes: set[str] = set()
    patient_visits: dict[str, set[str]] = {}
    for expected_index, shard_dir in enumerate(shard_dirs):
        with (shard_dir / "report.json").open("r", encoding="utf-8") as handle:
            report = json.load(handle)
        if (
            report.get("schema_version") != "med-jepa-3dino-cache-v2"
            or report.get("shard_index") != expected_index
            or report.get("num_shards") != expected_num_shards
        ):
            raise ValueError("cache shard report does not match its declared position")
        try:
            feature_meta = validate_feature_provenance(report.get("feature_provenance"))
            spatial_meta = validate_spatial_provenance(report.get("spatial_provenance"))
        except (ManifestBuildError, TypeError) as error:
            raise ValueError("cache shard report has invalid provenance") from error
        spatial_digest = hashlib.sha256(
            json.dumps(
                spatial_meta,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if report.get("spatial_provenance_sha256") != spatial_digest:
            raise ValueError("cache shard spatial provenance digest mismatch")
        if (
            feature_meta["feature_dim"] != report.get("feature_dim")
            or feature_meta["feature_grid_shape"] != report.get("feature_grid_shape")
            or feature_meta["input_shape"] != report.get("input_shape")
            or feature_meta["checkpoint_sha256"] != report.get("checkpoint_sha256")
            or feature_meta.get("checkpoint_key") != report.get("checkpoint_key")
            or feature_meta["source_revision"] != report.get("source_revision")
        ):
            raise ValueError("cache shard feature provenance disagrees with its report")
        report["feature_provenance"] = feature_meta
        report["spatial_provenance"] = spatial_meta
        reports.append(report)
        with (shard_dir / "feature_index.csv").open(
            "r", encoding="utf-8-sig", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != OUTPUT_INDEX_COLUMNS:
                raise ValueError("cache shard index columns do not match the contract")
            shard_rows = list(reader)
        if len(shard_rows) != int(report.get("cached_records", -1)):
            raise ValueError("cache shard row count does not match its report")
        for row in shard_rows:
            patient_id = row["PatientID"].strip()
            timepoint = row["tp_label"].strip().upper()
            scan_id = row["scan_id"].strip()
            feature_digest = row["content_sha256"].strip().lower()
            source_digest = row["source_content_sha256"].strip().lower()
            if not patient_id or not scan_id or timepoint not in expected_timepoints:
                raise ValueError("cache index contains an invalid identifier/timepoint")
            if not _valid_sha256(feature_digest) or not _valid_sha256(source_digest):
                raise ValueError("cache index contains an invalid SHA-256")
            visit = (patient_id, timepoint)
            if visit in seen_visits:
                raise ValueError("cache index contains a duplicate patient visit")
            if scan_id in seen_scans:
                raise ValueError("cache index contains a duplicate scan_id")
            if feature_digest in seen_feature_hashes or source_digest in seen_source_hashes:
                raise ValueError("cache index contains duplicate tensor content")
            feature_path = Path(row["feature_path"]).expanduser().resolve()
            if not _inside(feature_path, root) or feature_path.parent != shard_dir:
                raise ValueError("feature path escapes its declared cache shard")
            if not feature_path.is_file():
                raise FileNotFoundError("a declared feature tensor is missing")
            if verify_file_hashes and sha256_file(feature_path) != feature_digest:
                raise RuntimeError("a cached feature tensor failed SHA-256 verification")
            seen_visits.add(visit)
            seen_scans.add(scan_id)
            seen_feature_hashes.add(feature_digest)
            seen_source_hashes.add(source_digest)
            patient_visits.setdefault(patient_id, set()).add(timepoint)
            rows.append({name: row[name] for name in OUTPUT_INDEX_COLUMNS})

    invariant_keys = (
        "schema_version",
        "num_shards",
        "input_shape",
        "feature_grid_shape",
        "feature_dim",
        "checkpoint_sha256",
        "checkpoint_key",
        "source_revision",
        "hf_revision",
        "feature_provenance",
        "spatial_provenance",
        "spatial_provenance_sha256",
    )
    baseline = {key: reports[0].get(key) for key in invariant_keys}
    if any({key: report.get(key) for key in invariant_keys} != baseline for report in reports):
        raise ValueError("cache shards were produced with inconsistent extractor settings")
    expected_visit_set = set(expected_timepoints)
    if any(visits != expected_visit_set for visits in patient_visits.values()):
        raise ValueError("merged cache does not contain every expected visit per patient")
    if expected_patients is not None and len(patient_visits) != expected_patients:
        raise ValueError(
            f"merged cache patient count mismatch: expected {expected_patients}, "
            f"observed {len(patient_visits)}"
        )
    rows.sort(key=lambda row: (row["PatientID"], row["tp_label"]))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    _atomic_json(baseline["feature_provenance"], feature_sidecar)
    _atomic_json(baseline["spatial_provenance"], spatial_sidecar)
    return {
        "schema_version": "med-jepa-merged-feature-index-v1",
        "patients": len(patient_visits),
        "records": len(rows),
        "timepoints": list(expected_timepoints),
        "num_shards": expected_num_shards,
        "feature_dim": baseline["feature_dim"],
        "feature_grid_shape": baseline["feature_grid_shape"],
        "checkpoint_sha256": baseline["checkpoint_sha256"],
        "source_revision": baseline["source_revision"],
        "spatial_provenance_sha256": baseline["spatial_provenance_sha256"],
        "index_sha256": sha256_file(destination),
        "feature_provenance_sha256": sha256_file(feature_sidecar),
        "spatial_sidecar_sha256": sha256_file(spatial_sidecar),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--expected-num-shards", required=True, type=int)
    parser.add_argument("--expected-patients", type=int, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = merge_feature_indices(
        args.cache_root,
        args.output,
        expected_num_shards=args.expected_num_shards,
        expected_patients=args.expected_patients,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
