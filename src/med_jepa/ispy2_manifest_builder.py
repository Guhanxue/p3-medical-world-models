"""Fail-closed builder for the confirmatory ISPY2 cached-feature manifest.

The builder runs where the governed DS3 metadata and feature cache are available.  It
joins only the preregistered patient split with the T0/T1/T3 trajectory, FTV and cached
feature rows, verifies serialized feature hashes, and writes one deterministic JSONL
record per patient.  Diagnostics deliberately contain aggregate counts only.
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
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

from med_jepa.ftv_release import FTVReleaseError, validate_ftv_release


DEFAULT_ISPY2_ROOT = Path("data/ispy2")
DEFAULT_EXPECTED_SPLIT_COUNTS: dict[str, int] = {
    "train": 416,
    "val": 89,
    "test": 90,
}

SPLIT_RELATIVE_PATH = Path("manifests/ispy2_split.csv")
SCHEMA_VERSION = "med-jepa.ispy2.t0-t1-to-t3.v1"
DEFAULT_MODALITY = "SER"
DEFAULT_TENSOR_KIND = "cached_features"
MANIFEST_BUILD_REPORT_SCHEMA_VERSION = "med-jepa-ispy2-manifest-build-v1"
CLINICAL_FEATURES = ("hr_positive", "her2_positive")
ACTION_FEATURES = ("phase_1_exposed", "phase_2_planned")
CONTEXT_TIMEPOINTS = ("T0", "T1")
TARGET_TIMEPOINT = "T3"
REQUIRED_TIMEPOINTS = (*CONTEXT_TIMEPOINTS, TARGET_TIMEPOINT)
ALLOWED_TIMEPOINTS = {"T0", "T1", "T2", "T3"}
ALLOWED_SPLITS = ("train", "val", "test")
ROI_RELEASE_SCHEMA_VERSION = "med-jepa-ispy2-ser-roi-converter-v2"
ROI_RELEASE_STATUS_FULL = "ok_roi_conversion_full"
ROI_RELEASE_STATUS_QC = "ok_roi_conversion_full_qc_exclusions"
ROI_QC_EXCLUSION_POLICY = "future_mask_outside_fixed_crop_patient_exclusion_v1"
ROI_QC_EXCLUSION_REASON = "future_mask_outside_fixed_crop"
ROI_QC_ROSTER_FIELDS = (
    "PatientID",
    "exclusion_reason",
    "failed_timepoints",
)

FEATURE_INDEX_COLUMNS = (
    "PatientID",
    "tp_label",
    "scan_id",
    "feature_path",
    "content_sha256",
    "source_content_sha256",
)

_MISSING = {"", "na", "n/a", "nan", "none", "null"}
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_PATIENT_ID = re.compile(r"^ISPY2-[0-9]+$")
_PATIENT_ID_IN_TEXT = re.compile(r"(?<![A-Za-z0-9])ISPY2-[0-9]+(?![A-Za-z0-9])")
_SOURCE_REVISION = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")
_SPATIAL_ONLY_FIELDS = {
    "reference_visit",
    "roi_source_visits",
    "spacing_mm",
    "orientation",
    "transform_id",
    "roi_voxel_shape",
    "crop_shape",
}
_REQUIRED_FEATURE_PROVENANCE = {
    "extractor",
    "source_revision",
    "checkpoint_sha256",
    "roi_index_sha256",
    "layer",
    "feature_dim",
    "feature_grid_shape",
    "input_shape",
    "normalization",
}


class ManifestBuildError(ValueError):
    """Raised when an input violates the preregistered manifest contract."""


@dataclass(frozen=True)
class ROIReleaseCohort:
    """Validated private cohort state plus aggregate-safe release provenance."""

    excluded_patient_ids: frozenset[str]
    patients_selected: int
    patients_released: int
    roi_report_sha256: str
    qc_roster_sha256: str | None
    qc_exclusion_policy: str | None

    def public_summary(self) -> dict[str, Any]:
        return {
            "validated": True,
            "roi_report_sha256": self.roi_report_sha256,
            "qc_exclusion_roster_sha256": self.qc_roster_sha256,
            "patients_selected": self.patients_selected,
            "patients_excluded_qc": len(self.excluded_patient_ids),
            "patients_released": self.patients_released,
            "qc_exclusion_policy": self.qc_exclusion_policy,
        }


def _normal_header(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _is_missing(value: Any) -> bool:
    return value is None or str(value).strip().casefold() in _MISSING


def _read_csv(
    path: Path, *, label: str
) -> tuple[list[str], list[dict[str, str]], str]:
    try:
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None:
            raise ManifestBuildError(f"{label} CSV has no header")
        fields = [str(field).strip() for field in reader.fieldnames]
        normalized = [_normal_header(field) for field in fields]
        if any(not field for field in fields) or len(set(normalized)) != len(fields):
            raise ManifestBuildError(
                f"{label} CSV has blank or duplicate normalized column names"
            )
        rows: list[dict[str, str]] = []
        malformed_rows = 0
        for row in reader:
            if None in row:
                malformed_rows += 1
                continue
            rows.append(
                {
                    str(key).strip(): "" if value is None else str(value).strip()
                    for key, value in row.items()
                }
            )
        if malformed_rows:
            raise ManifestBuildError(
                f"{label} CSV contains malformed rows (count={malformed_rows})"
            )
    except FileNotFoundError as error:
        raise ManifestBuildError(f"required {label} CSV is missing") from error
    except (OSError, UnicodeError, csv.Error) as error:
        # Do not echo a data-bearing path or row value in diagnostics.
        raise ManifestBuildError(f"required {label} CSV is unreadable") from error
    if not rows:
        raise ManifestBuildError(f"required {label} CSV is empty")
    return fields, rows, hashlib.sha256(content).hexdigest()


def _column(
    fields: Sequence[str],
    aliases: Sequence[str],
    *,
    label: str,
    required: bool = True,
) -> str | None:
    wanted = {_normal_header(alias) for alias in aliases}
    matches = [field for field in fields if _normal_header(field) in wanted]
    if len(matches) > 1:
        raise ManifestBuildError(f"{label} CSV has ambiguous columns for one field")
    if not matches:
        if required:
            raise ManifestBuildError(f"{label} CSV lacks a required column")
        return None
    return matches[0]


def _require_exact_feature_columns(fields: Sequence[str]) -> dict[str, str]:
    normalized = {_normal_header(field): field for field in fields}
    result: dict[str, str] = {}
    for required in FEATURE_INDEX_COLUMNS:
        field = normalized.get(_normal_header(required))
        if field is None:
            raise ManifestBuildError(
                "feature index lacks one or more required contract columns"
            )
        result[required] = field
    return result


def _patient_id(value: Any, *, label: str) -> str:
    if _is_missing(value):
        raise ManifestBuildError(f"{label} contains a missing patient identifier")
    return str(value).strip()


def _timepoint(value: Any, *, label: str) -> str:
    point = str(value).strip().upper()
    if point not in ALLOWED_TIMEPOINTS:
        raise ManifestBuildError(f"{label} contains an unsupported timepoint")
    return point


def _finite_float(value: Any, *, field: str, allow_missing: bool = False) -> float | None:
    if _is_missing(value):
        if allow_missing:
            return None
        raise ManifestBuildError(f"{field} contains missing values")
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as error:
        raise ManifestBuildError(f"{field} contains non-numeric values") from error
    if not math.isfinite(parsed):
        raise ManifestBuildError(f"{field} contains non-finite values")
    return parsed


def _binary(value: Any, *, field: str, allow_missing: bool = False) -> int | None:
    if _is_missing(value):
        if allow_missing:
            return None
        raise ManifestBuildError(f"{field} contains missing values")
    text = str(value).strip().casefold().replace("_", " ")
    positive = {
        "1",
        "1.0",
        "true",
        "yes",
        "y",
        "positive",
        "pos",
        "+",
        "hr+",
        "her2+",
    }
    negative = {
        "0",
        "0.0",
        "false",
        "no",
        "n",
        "negative",
        "neg",
        "-",
        "hr-",
        "her2-",
    }
    if text in positive:
        return 1
    if text in negative:
        return 0
    raise ManifestBuildError(f"{field} contains values outside the binary contract")


def _validate_sha256(value: Any, *, field: str) -> str:
    text = str(value).strip().lower()
    if not _SHA256.fullmatch(text):
        raise ManifestBuildError(f"{field} must contain 64-character SHA-256 hex digests")
    return text


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _release_count(value: Any, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManifestBuildError(f"ROI release {field} must be an integer")
    if value < 0 or (not allow_zero and value == 0):
        raise ManifestBuildError(f"ROI release {field} is outside the count contract")
    return int(value)


def _contains_patient_identifier(value: Any) -> bool:
    if isinstance(value, str):
        return _PATIENT_ID_IN_TEXT.search(value) is not None
    if isinstance(value, Mapping):
        return any(
            _contains_patient_identifier(key) or _contains_patient_identifier(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(_contains_patient_identifier(item) for item in value)
    return False


def _read_roi_release_report(path: str | Path) -> tuple[dict[str, Any], str]:
    source = Path(path).expanduser()
    try:
        content = source.read_bytes()
        text = content.decode("utf-8")
        value = json.loads(text)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestBuildError("ROI release report is unreadable") from error
    report = _json_object(value, field="ROI release report")
    if _contains_patient_identifier(report):
        raise ManifestBuildError("ROI release report violates the aggregate-only contract")
    return report, hashlib.sha256(content).hexdigest()


def _read_qc_exclusion_roster(
    path: str | Path,
) -> tuple[frozenset[str], str]:
    source = Path(path).expanduser()
    try:
        content = source.read_bytes()
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if tuple(reader.fieldnames or ()) != ROI_QC_ROSTER_FIELDS:
            raise ManifestBuildError(
                "QC exclusion roster columns do not match the release contract"
            )
        rows = list(reader)
    except ManifestBuildError:
        raise
    except (OSError, UnicodeError, csv.Error) as error:
        raise ManifestBuildError("QC exclusion roster is unreadable") from error
    if not rows or any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ManifestBuildError("QC exclusion roster is empty or malformed")
    patients: set[str] = set()
    duplicates = 0
    invalid_rows = 0
    for row in rows:
        patient = str(row["PatientID"]).strip()
        reason = str(row["exclusion_reason"]).strip()
        raw_timepoints = str(row["failed_timepoints"]).strip()
        timepoints = raw_timepoints.split(",") if raw_timepoints else []
        if (
            not _PATIENT_ID.fullmatch(patient)
            or reason != ROI_QC_EXCLUSION_REASON
            or not timepoints
            or any(point not in {"T1", "T3"} for point in timepoints)
            or raw_timepoints != ",".join(sorted(set(timepoints)))
        ):
            invalid_rows += 1
            continue
        if patient in patients:
            duplicates += 1
        patients.add(patient)
    if invalid_rows or duplicates:
        raise ManifestBuildError(
            "QC exclusion roster violates the reviewed contract: "
            f"invalid_rows={invalid_rows}, duplicate_patients={duplicates}"
        )
    return frozenset(patients), hashlib.sha256(content).hexdigest()


def validate_roi_release_cohort(
    roi_report_path: str | Path,
    *,
    qc_exclusion_path: str | Path | None,
    expected_full_split_counts: Mapping[str, int],
    expected_evidence_sha256: Mapping[str, str],
) -> ROIReleaseCohort:
    """Validate one promoted full ROI release without emitting private identifiers."""

    expected_counts = _validate_expected_split_counts(expected_full_split_counts)
    expected_total = sum(expected_counts.values())
    report, report_sha256 = _read_roi_release_report(roi_report_path)
    if (
        report.get("schema_version") != ROI_RELEASE_SCHEMA_VERSION
        or report.get("status") not in {ROI_RELEASE_STATUS_FULL, ROI_RELEASE_STATUS_QC}
        or report.get("aggregate_only") is not True
        or report.get("pixel_data_read") is not True
        or report.get(
            "identifiers_paths_uids_dates_or_per_patient_measurements_emitted"
        )
        is not False
    ):
        raise ManifestBuildError("ROI release report is not an approved full release")
    cohort = report.get("cohort")
    conversion = report.get("conversion")
    next_gate = report.get("next_gate")
    evidence = report.get("evidence")
    roi_contract = report.get("roi_contract")
    if not all(
        isinstance(value, Mapping)
        for value in (cohort, conversion, next_gate, evidence, roi_contract)
    ):
        raise ManifestBuildError("ROI release report lacks required contract objects")
    assert isinstance(cohort, Mapping)
    assert isinstance(conversion, Mapping)
    assert isinstance(next_gate, Mapping)
    assert isinstance(evidence, Mapping)
    assert isinstance(roi_contract, Mapping)
    selected = _release_count(cohort.get("patients_selected"), field="patients_selected")
    total = _release_count(cohort.get("patients_total"), field="patients_total")
    excluded = _release_count(
        cohort.get("patients_excluded_qc"),
        field="patients_excluded_qc",
        allow_zero=True,
    )
    released = _release_count(cohort.get("patients_released"), field="patients_released")
    visits = _release_count(cohort.get("visits_expected"), field="visits_expected")
    if (
        total != expected_total
        or selected != expected_total
        or released != selected - excluded
        or visits != selected * len(REQUIRED_TIMEPOINTS)
        or cohort.get("split_counts_full_cohort") != expected_counts
        or conversion.get("release_promoted") is not True
        or conversion.get("failure_reasons") != {}
        or next_gate.get("can_cache_3dino") is not True
        or next_gate.get("pilot_only") is not False
        or roi_contract.get("roi_schema_version") != "med-jepa-ispy2-roi-v2"
    ):
        raise ManifestBuildError("ROI release report cohort or promotion contract mismatch")
    required_evidence = {
        "trajectory_manifest_sha256",
        "split_sha256",
        "ftv_csv_sha256",
        "ftv_audit_report_sha256",
        "roi_spec_sha256",
        "roi_anchor_report_sha256",
        "converter_source_sha256",
    }
    if not required_evidence.issubset(evidence) or any(
        not _SHA256.fullmatch(str(evidence[key])) for key in required_evidence
    ):
        raise ManifestBuildError("ROI release report evidence is incomplete or invalid")
    expected_evidence_keys = {
        "trajectory_manifest_sha256",
        "split_sha256",
        "ftv_csv_sha256",
        "ftv_audit_report_sha256",
    }
    if set(expected_evidence_sha256) != expected_evidence_keys:
        raise ManifestBuildError(
            "ROI release evidence expectations do not match the consumer contract"
        )
    normalized_expected_evidence = {
        key: _validate_sha256(value, field=f"expected ROI evidence {key}")
        for key, value in expected_evidence_sha256.items()
    }
    if any(
        str(evidence[key]).lower() != normalized_expected_evidence[key]
        for key in expected_evidence_keys
    ):
        raise ManifestBuildError(
            "ROI release evidence does not match the selected trajectory/split/FTV inputs"
        )

    expected_status = ROI_RELEASE_STATUS_QC if excluded else ROI_RELEASE_STATUS_FULL
    expected_policy = ROI_QC_EXCLUSION_POLICY if excluded else None
    if (
        report.get("status") != expected_status
        or conversion.get("qc_exclusion_policy") != expected_policy
    ):
        raise ManifestBuildError("ROI release status disagrees with its QC exclusion count")
    diagnostics = conversion.get("failure_diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("metadata_rows") != excluded:
        raise ManifestBuildError("ROI release QC diagnostics disagree with its cohort")

    if excluded:
        if qc_exclusion_path is None:
            raise ManifestBuildError("QC exclusion roster is required by the ROI release")
        excluded_patients, roster_sha256 = _read_qc_exclusion_roster(
            qc_exclusion_path
        )
        if len(excluded_patients) != excluded:
            raise ManifestBuildError(
                "QC exclusion roster count disagrees with the ROI release report"
            )
    else:
        if qc_exclusion_path is not None:
            raise ManifestBuildError(
                "QC exclusion roster must be omitted for a release without exclusions"
            )
        excluded_patients = frozenset()
        roster_sha256 = None
    return ROIReleaseCohort(
        excluded_patient_ids=excluded_patients,
        patients_selected=selected,
        patients_released=released,
        roi_report_sha256=report_sha256,
        qc_roster_sha256=roster_sha256,
        qc_exclusion_policy=expected_policy,
    )


def _verify_cached_feature_payload(
    path: Path,
    *,
    expected_provenance: Mapping[str, Any],
    expected_source_hash: str,
    expected_spatial_provenance: Mapping[str, Any],
) -> str | None:
    """Return an aggregate-safe failure category, or ``None`` when valid."""

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:  # Corrupt/special tensor payloads fail as an aggregate category.
        return "unreadable_payload"
    embedded_provenance: Any = None
    embedded_source_hash: Any = None
    embedded_spatial_provenance: Any = None
    if isinstance(payload, Mapping):
        feature = payload.get("volume")
        embedded_provenance = payload.get("feature_provenance")
        embedded_source_hash = payload.get("source_content_sha256")
        embedded_spatial_provenance = payload.get("spatial_provenance")
    else:
        feature = payload
    if not isinstance(feature, Tensor) or feature.ndim != 4:
        return "invalid_tensor_rank"
    if not feature.is_floating_point() or not bool(torch.isfinite(feature).all()):
        return "invalid_tensor_values"
    expected_shape = (
        int(expected_provenance["feature_dim"]),
        *(int(item) for item in expected_provenance["feature_grid_shape"]),
    )
    if tuple(int(item) for item in feature.shape) != expected_shape:
        return "tensor_shape_mismatch"
    if embedded_source_hash is None:
        return "embedded_source_hash_missing"
    try:
        observed_source_hash = _validate_sha256(
            embedded_source_hash, field="cached feature source hash"
        )
    except ManifestBuildError:
        return "embedded_source_hash_invalid"
    if observed_source_hash != expected_source_hash:
        return "embedded_source_hash_mismatch"
    if embedded_provenance is None:
        return "embedded_provenance_missing"
    try:
        observed_provenance = validate_feature_provenance(embedded_provenance)
    except ManifestBuildError:
        return "embedded_provenance_invalid"
    if observed_provenance != expected_provenance:
        return "embedded_provenance_mismatch"
    if embedded_spatial_provenance is None:
        return "embedded_spatial_provenance_missing"
    try:
        observed_spatial = validate_spatial_provenance(embedded_spatial_provenance)
    except ManifestBuildError:
        return "embedded_spatial_provenance_invalid"
    if observed_spatial != expected_spatial_provenance:
        return "embedded_spatial_provenance_mismatch"
    return None


def _json_object(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not value:
        raise ManifestBuildError(f"{field} must be a non-empty JSON object")
    result = {str(key): deepcopy(item) for key, item in value.items()}
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ManifestBuildError(f"{field} must be finite and JSON serializable") from error
    return result


def validate_feature_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate shared encoder/cache provenance without accepting spatial aliases."""

    result = _json_object(value, field="feature_provenance")
    overlap = _SPATIAL_ONLY_FIELDS.intersection(result)
    if overlap:
        raise ManifestBuildError(
            "feature_provenance must not contain spatial provenance fields"
        )
    if not _REQUIRED_FEATURE_PROVENANCE.issubset(result):
        raise ManifestBuildError("feature_provenance lacks required extraction fields")
    required_text = (
        "extractor",
        "source_revision",
        "layer",
        "normalization",
    )
    for key in required_text:
        if _is_missing(result.get(key)):
            raise ManifestBuildError(f"feature_provenance must declare {key}")
        result[key] = str(result[key]).strip()
    if not _SOURCE_REVISION.fullmatch(result["source_revision"]):
        raise ManifestBuildError(
            "feature_provenance.source_revision must be a full 40- or 64-character "
            "hex revision"
        )
    result["source_revision"] = result["source_revision"].lower()
    result["checkpoint_sha256"] = _validate_sha256(
        result.get("checkpoint_sha256"), field="feature_provenance.checkpoint_sha256"
    )
    result["roi_index_sha256"] = _validate_sha256(
        result.get("roi_index_sha256"), field="feature_provenance.roi_index_sha256"
    )
    feature_dim = result.get("feature_dim")
    if isinstance(feature_dim, bool) or not isinstance(feature_dim, int) or feature_dim < 1:
        raise ManifestBuildError("feature_provenance.feature_dim must be a positive integer")
    result["feature_dim"] = int(feature_dim)
    result["feature_grid_shape"] = _positive_triplet(
        result["feature_grid_shape"],
        field="feature_grid_shape",
        integer=True,
        owner="feature_provenance",
    )
    result["input_shape"] = _positive_triplet(
        result["input_shape"],
        field="input_shape",
        integer=True,
        owner="feature_provenance",
    )
    return result


def _positive_triplet(
    value: Any, *, field: str, integer: bool, owner: str = "spatial_provenance"
) -> list[int] | list[float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ManifestBuildError(f"{owner}.{field} must have length three")
    parsed: list[int] | list[float]
    if integer:
        if any(isinstance(item, bool) or not isinstance(item, int) for item in value):
            raise ManifestBuildError(
                f"{owner}.{field} must contain positive integers"
            )
        parsed = [int(item) for item in value]
    else:
        if any(
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(float(item))
            for item in value
        ):
            raise ManifestBuildError(
                f"{owner}.{field} must contain positive finite numbers"
            )
        parsed = [float(item) for item in value]
    if any(item <= 0 for item in parsed):
        raise ManifestBuildError(f"{owner}.{field} must be positive")
    return parsed


def validate_spatial_provenance(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate ROI geometry and cached-grid geometry as distinct declarations."""

    result = _json_object(value, field="spatial_provenance")
    if "crop_shape" in result or "feature_grid_shape" in result:
        raise ManifestBuildError(
            "spatial_provenance must not contain crop_shape or feature_grid_shape; "
            "declare cached-grid geometry in feature_provenance"
        )
    required = {
        "reference_visit",
        "roi_source_visits",
        "spacing_mm",
        "orientation",
        "transform_id",
        "roi_voxel_shape",
        "registration_type",
        "deformable_registration",
    }
    if not required.issubset(result):
        raise ManifestBuildError("spatial_provenance lacks required geometry fields")
    if str(result["reference_visit"]).strip().upper() != "T0":
        raise ManifestBuildError("spatial_provenance.reference_visit must be T0")
    sources = result["roi_source_visits"]
    if (
        not isinstance(sources, list)
        or not sources
        or any(str(item).strip().upper() not in CONTEXT_TIMEPOINTS for item in sources)
    ):
        raise ManifestBuildError(
            "spatial_provenance.roi_source_visits must be non-empty context visits"
        )
    normalized_sources = [str(item).strip().upper() for item in sources]
    if len(set(normalized_sources)) != len(normalized_sources):
        raise ManifestBuildError(
            "spatial_provenance.roi_source_visits contains duplicates"
        )
    if _is_missing(result["orientation"]) or _is_missing(result["transform_id"]):
        raise ManifestBuildError(
            "spatial_provenance must declare orientation and transform_id"
        )
    result["reference_visit"] = "T0"
    result["roi_source_visits"] = normalized_sources
    result["spacing_mm"] = _positive_triplet(
        result["spacing_mm"], field="spacing_mm", integer=False
    )
    result["roi_voxel_shape"] = _positive_triplet(
        result["roi_voxel_shape"], field="roi_voxel_shape", integer=True
    )
    result["orientation"] = str(result["orientation"]).strip()
    result["transform_id"] = str(result["transform_id"]).strip()
    registration_type = str(result["registration_type"]).strip().casefold()
    if registration_type not in {"identity", "rigid", "affine", "rigid_then_affine"}:
        raise ManifestBuildError(
            "spatial_provenance.registration_type must be identity, rigid, affine, "
            "or rigid_then_affine"
        )
    if result["deformable_registration"] is not False:
        raise ManifestBuildError(
            "spatial_provenance.deformable_registration must be false"
        )
    result["registration_type"] = registration_type
    result["deformable_registration"] = False
    return result


def _split_index(
    fields: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> tuple[
    dict[str, str],
    dict[str, Mapping[str, str]],
    dict[str, str | None],
]:
    patient_column = _column(
        fields, ("PatientID", "patient_id", "patient"), label="split"
    )
    split_column = _column(fields, ("split", "partition", "set"), label="split")
    assert patient_column is not None and split_column is not None
    columns: dict[str, str | None] = {
        "patient": patient_column,
        "split": split_column,
        "hr": _column(
            fields,
            ("HR", "hr_status", "hr_positive", "HR Pos"),
            label="split",
            required=False,
        ),
        "her2": _column(
            fields,
            ("HER2", "her2_status", "her2_positive", "HER2 Pos"),
            label="split",
            required=False,
        ),
        "pcr": _column(
            fields,
            ("pCR", "pathologic_complete_response"),
            label="split",
            required=False,
        ),
    }
    result: dict[str, str] = {}
    patient_rows: dict[str, Mapping[str, str]] = {}
    duplicates = 0
    invalid_splits = 0
    for row in rows:
        patient = _patient_id(row.get(patient_column), label="split CSV")
        split = str(row.get(split_column, "")).strip().casefold()
        if split == "validation":
            split = "val"
        if split not in ALLOWED_SPLITS:
            invalid_splits += 1
            continue
        if patient in result:
            duplicates += 1
        else:
            result[patient] = split
            patient_rows[patient] = row
    if invalid_splits:
        raise ManifestBuildError(
            f"split CSV contains unsupported split assignments (count={invalid_splits})"
        )
    if duplicates:
        raise ManifestBuildError(
            f"split CSV contains duplicate patient rows (count={duplicates})"
        )
    return result, patient_rows, columns


def _trajectory_index(
    fields: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> tuple[dict[tuple[str, str], Mapping[str, str]], dict[str, str | None]]:
    patient_column = _column(
        fields, ("PatientID", "patient_id", "patient"), label="trajectory"
    )
    timepoint_column = _column(
        fields,
        ("tp_label", "timepoint", "time_point", "visit", "visit_id"),
        label="trajectory",
    )
    time_column = _column(
        fields,
        ("delta_t_days", "time_days", "days_from_t0", "delta_days"),
        label="trajectory",
    )
    assert patient_column is not None and timepoint_column is not None
    assert time_column is not None
    columns: dict[str, str | None] = {
        "patient": patient_column,
        "timepoint": timepoint_column,
        "time": time_column,
        "hr": _column(
            fields,
            ("HR", "hr_status", "hr_positive", "HR Pos"),
            label="trajectory",
            required=False,
        ),
        "her2": _column(
            fields,
            ("HER2", "her2_status", "her2_positive", "HER2 Pos"),
            label="trajectory",
            required=False,
        ),
        "pcr": _column(
            fields, ("pCR", "pathologic_complete_response"), label="trajectory", required=False
        ),
    }
    result: dict[tuple[str, str], Mapping[str, str]] = {}
    duplicates = 0
    for row in rows:
        patient = _patient_id(row.get(patient_column), label="trajectory CSV")
        point = _timepoint(row.get(timepoint_column), label="trajectory CSV")
        key = (patient, point)
        if key in result:
            duplicates += 1
        else:
            result[key] = row
    if duplicates:
        raise ManifestBuildError(
            f"trajectory CSV contains duplicate patient/timepoint rows (count={duplicates})"
        )
    return result, columns


def _ftv_index(
    fields: Sequence[str], rows: Sequence[Mapping[str, str]]
) -> tuple[dict[str, Mapping[str, str]], dict[str, str | None]]:
    patient_column = _column(
        fields, ("PatientID", "patient_id", "patient"), label="FTV"
    )
    assert patient_column is not None
    columns: dict[str, str | None] = {
        "patient": patient_column,
        "ftv_t0": _column(fields, ("cFTV_T0",), label="FTV"),
        "ftv_t1": _column(fields, ("cFTV_T1",), label="FTV"),
        "ftv_t3": _column(fields, ("cFTV_T3",), label="FTV"),
        "hr": _column(
            fields,
            ("HR", "hr_status", "hr_positive", "HR Pos"),
            label="FTV",
            required=False,
        ),
        "her2": _column(
            fields,
            ("HER2", "her2_status", "her2_positive", "HER2 Pos"),
            label="FTV",
            required=False,
        ),
        "pcr": _column(
            fields, ("pCR", "pathologic_complete_response"), label="FTV", required=False
        ),
    }
    result: dict[str, Mapping[str, str]] = {}
    duplicates = 0
    for row in rows:
        patient = _patient_id(row.get(patient_column), label="FTV CSV")
        if patient in result:
            duplicates += 1
        else:
            result[patient] = row
    if duplicates:
        raise ManifestBuildError(
            f"FTV CSV contains duplicate patient rows (count={duplicates})"
        )
    return result, columns


def _feature_index(
    fields: Sequence[str],
    rows: Sequence[Mapping[str, str]],
    *,
    index_path: Path,
    allowed_feature_root: Path,
) -> dict[tuple[str, str], dict[str, str | Path]]:
    columns = _require_exact_feature_columns(fields)
    result: dict[tuple[str, str], dict[str, str | Path]] = {}
    duplicate_keys = 0
    invalid_paths = 0
    missing_paths = 0
    for row in rows:
        patient = _patient_id(row.get(columns["PatientID"]), label="feature index")
        point = _timepoint(row.get(columns["tp_label"]), label="feature index")
        scan_id = str(row.get(columns["scan_id"], "")).strip()
        if not scan_id:
            raise ManifestBuildError("feature index contains missing scan identifiers")
        content_hash = _validate_sha256(
            row.get(columns["content_sha256"]), field="feature index content_sha256"
        )
        source_hash = _validate_sha256(
            row.get(columns["source_content_sha256"]),
            field="feature index source_content_sha256",
        )
        raw_path = str(row.get(columns["feature_path"], "")).strip()
        candidate = Path(raw_path).expanduser()
        if not raw_path or ".." in candidate.parts or candidate.suffix.casefold() != ".pt":
            invalid_paths += 1
            continue
        if not candidate.is_absolute():
            candidate = index_path.parent / candidate
        try:
            candidate = candidate.resolve(strict=False)
            candidate.relative_to(allowed_feature_root)
            exists = candidate.is_file()
        except ValueError:
            invalid_paths += 1
            continue
        except (OSError, RuntimeError):
            exists = False
        if not exists:
            missing_paths += 1
            continue
        key = (patient, point)
        if key in result:
            duplicate_keys += 1
        else:
            result[key] = {
                "scan_id": scan_id,
                "feature_path": candidate,
                "content_sha256": content_hash,
                "source_content_sha256": source_hash,
            }
    if invalid_paths:
        raise ManifestBuildError(
            f"feature index contains unsafe or non-.pt paths (count={invalid_paths})"
        )
    if missing_paths:
        raise ManifestBuildError(
            f"feature index references missing feature files (count={missing_paths})"
        )
    if duplicate_keys:
        raise ManifestBuildError(
            "feature index contains duplicate patient/timepoint rows "
            f"(count={duplicate_keys})"
        )
    return result


def _scalar_sources(
    patient: str,
    *,
    trajectory: Mapping[tuple[str, str], Mapping[str, str]],
    trajectory_column: str | None,
    split_row: Mapping[str, str],
    split_column: str | None,
    ftv_row: Mapping[str, str],
    ftv_column: str | None,
) -> list[str]:
    values: list[str] = []
    if split_column is not None and not _is_missing(split_row.get(split_column)):
        values.append(str(split_row[split_column]))
    if ftv_column is not None and not _is_missing(ftv_row.get(ftv_column)):
        values.append(str(ftv_row[ftv_column]))
    if trajectory_column is not None:
        for point in ALLOWED_TIMEPOINTS:
            row = trajectory.get((patient, point))
            if row is not None and not _is_missing(row.get(trajectory_column)):
                values.append(str(row[trajectory_column]))
    return values


def _patient_binary(
    patient: str,
    *,
    field: str,
    trajectory: Mapping[tuple[str, str], Mapping[str, str]],
    trajectory_column: str | None,
    split_row: Mapping[str, str],
    split_column: str | None,
    ftv_row: Mapping[str, str],
    ftv_column: str | None,
    allow_missing: bool,
) -> int | None:
    raw = _scalar_sources(
        patient,
        trajectory=trajectory,
        trajectory_column=trajectory_column,
        split_row=split_row,
        split_column=split_column,
        ftv_row=ftv_row,
        ftv_column=ftv_column,
    )
    if not raw:
        if allow_missing:
            return None
        raise ManifestBuildError(f"{field} is missing for one or more split patients")
    values = {_binary(value, field=field) for value in raw}
    if len(values) != 1:
        raise ManifestBuildError(f"{field} is inconsistent across source rows")
    return values.pop()


def _validate_expected_split_counts(value: Mapping[str, int]) -> dict[str, int]:
    if set(value) != set(ALLOWED_SPLITS):
        raise ManifestBuildError(
            "expected split counts must declare exactly train, val and test"
        )
    result: dict[str, int] = {}
    for split in ALLOWED_SPLITS:
        count = value[split]
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ManifestBuildError("expected split counts must be positive integers")
        result[split] = count
    return result


def _validate_global_feature_identity(
    entries: Iterable[Mapping[str, str | Path]],
) -> None:
    for field in ("scan_id", "feature_path", "content_sha256", "source_content_sha256"):
        values = [str(entry[field]) for entry in entries]
        duplicates = len(values) - len(set(values))
        if duplicates:
            raise ManifestBuildError(
                f"selected feature rows reuse {field} values (count={duplicates})"
            )


def build_manifest_records(
    *,
    root: str | Path = DEFAULT_ISPY2_ROOT,
    trajectory_manifest_path: str | Path,
    ftv_path: str | Path | None = None,
    ftv_audit_path: str | Path | None = None,
    roi_report_path: str | Path,
    qc_exclusion_path: str | Path | None = None,
    feature_index_path: str | Path,
    roi_index_path: str | Path,
    allowed_feature_root: str | Path | None = None,
    feature_provenance: Mapping[str, Any],
    spatial_provenance: Mapping[str, Any],
    expected_split_counts: Mapping[str, int] = DEFAULT_EXPECTED_SPLIT_COUNTS,
    modality: str = DEFAULT_MODALITY,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build validated records in memory without writing any files.

    Errors intentionally report only categories and aggregate counts.  Patient IDs and
    scan paths appear solely inside the returned governed manifest records.
    """

    root_path = Path(root).expanduser()
    if ftv_path is None or ftv_audit_path is None:
        raise ManifestBuildError(
            "governed FTV CSV and its audit report must both be provided"
        )
    ftv_source = Path(ftv_path).expanduser()
    ftv_audit_source = Path(ftv_audit_path).expanduser()
    try:
        validated_ftv_release = validate_ftv_release(ftv_source, ftv_audit_source)
    except FTVReleaseError as error:
        raise ManifestBuildError(str(error)) from error
    # These rows were parsed from the exact byte snapshot whose digest and release
    # report were validated above.  Never reopen the mutable CSV after validation.
    ftv_fields = list(validated_ftv_release["ftv_csv_fields"])
    ftv_rows = list(validated_ftv_release["ftv_csv_rows"])
    ftv_release = {
        key: value
        for key, value in validated_ftv_release.items()
        if key not in {"ftv_csv_fields", "ftv_csv_rows"}
    }
    index_path = Path(feature_index_path).expanduser()
    try:
        feature_root = (
            Path(allowed_feature_root).expanduser()
            if allowed_feature_root is not None
            else index_path.parent
        ).resolve(strict=True)
    except (OSError, RuntimeError) as error:
        raise ManifestBuildError("allowed feature root is unavailable") from error
    if not feature_root.is_dir():
        raise ManifestBuildError("allowed feature root is not a directory")
    if str(modality).strip().upper() != DEFAULT_MODALITY:
        raise ManifestBuildError("the confirmatory manifest modality must be SER")
    expected = _validate_expected_split_counts(expected_split_counts)
    feature_meta = validate_feature_provenance(feature_provenance)
    spatial_meta = validate_spatial_provenance(spatial_provenance)

    roi_index_source = Path(roi_index_path).expanduser()
    try:
        roi_index_sha256 = _sha256_file(roi_index_source)
    except OSError as error:
        # Never echo a governed path (which may contain a release identifier).
        raise ManifestBuildError("ROI index is unreadable") from error
    if feature_meta["roi_index_sha256"] != roi_index_sha256:
        raise ManifestBuildError(
            "feature provenance ROI-index digest does not match the selected ROI index"
        )

    split_path = root_path / SPLIT_RELATIVE_PATH
    split_fields, split_rows, split_sha256 = _read_csv(split_path, label="split")
    trajectory_fields, trajectory_rows, trajectory_sha256 = _read_csv(
        Path(trajectory_manifest_path).expanduser(), label="trajectory"
    )
    feature_fields, feature_rows, feature_index_sha256 = _read_csv(
        index_path, label="feature index"
    )

    full_splits, split_source_rows, split_columns = _split_index(
        split_fields, split_rows
    )
    full_observed_counts = Counter(full_splits.values())
    full_observed = {
        split: full_observed_counts.get(split, 0) for split in ALLOWED_SPLITS
    }
    if full_observed != expected:
        raise ManifestBuildError(
            f"patient split counts differ from the preregistered counts: "
            f"expected={expected}, observed={full_observed}"
        )
    roi_release = validate_roi_release_cohort(
        roi_report_path,
        qc_exclusion_path=qc_exclusion_path,
        expected_full_split_counts=expected,
        expected_evidence_sha256={
            "trajectory_manifest_sha256": trajectory_sha256,
            "split_sha256": split_sha256,
            "ftv_csv_sha256": str(validated_ftv_release["ftv_csv_sha256"]),
            "ftv_audit_report_sha256": str(
                validated_ftv_release["ftv_audit_report_sha256"]
            ),
        },
    )
    unknown_exclusions = roi_release.excluded_patient_ids.difference(full_splits)
    if unknown_exclusions:
        raise ManifestBuildError(
            "QC exclusion roster contains patients outside the preregistered split "
            f"(count={len(unknown_exclusions)})"
        )
    splits = {
        patient: split
        for patient, split in full_splits.items()
        if patient not in roi_release.excluded_patient_ids
    }
    if len(splits) != roi_release.patients_released:
        raise ManifestBuildError(
            "filtered split count disagrees with the ROI release report"
        )
    released_counts = Counter(splits.values())
    observed = {split: released_counts.get(split, 0) for split in ALLOWED_SPLITS}
    excluded_counts_raw = Counter(
        full_splits[patient] for patient in roi_release.excluded_patient_ids
    )
    excluded_counts = {
        split: excluded_counts_raw.get(split, 0) for split in ALLOWED_SPLITS
    }

    trajectory, trajectory_columns = _trajectory_index(
        trajectory_fields, trajectory_rows
    )
    ftv, ftv_columns = _ftv_index(ftv_fields, ftv_rows)
    features = _feature_index(
        feature_fields,
        feature_rows,
        index_path=index_path,
        allowed_feature_root=feature_root,
    )
    expected_feature_keys = {
        (patient, point) for patient in splits for point in REQUIRED_TIMEPOINTS
    }
    feature_keys = set(features)
    extra_features = len(feature_keys - expected_feature_keys)
    if all(
        columns["hr"] is None
        for columns in (split_columns, trajectory_columns, ftv_columns)
    ):
        raise ManifestBuildError("HR is absent from split, trajectory and FTV CSVs")
    if all(
        columns["her2"] is None
        for columns in (split_columns, trajectory_columns, ftv_columns)
    ):
        raise ManifestBuildError("HER2 is absent from split, trajectory and FTV CSVs")
    if all(
        columns["pcr"] is None
        for columns in (split_columns, trajectory_columns, ftv_columns)
    ):
        raise ManifestBuildError("pCR is absent from split, trajectory and FTV CSVs")

    missing_trajectory = sum(
        (patient, point) not in trajectory
        for patient in splits
        for point in REQUIRED_TIMEPOINTS
    )
    missing_ftv = sum(patient not in ftv for patient in splits)
    missing_features = sum(
        (patient, point) not in features
        for patient in splits
        for point in REQUIRED_TIMEPOINTS
    )
    if missing_trajectory or missing_ftv or missing_features or extra_features:
        raise ManifestBuildError(
            "required source rows are incomplete or outside the released cohort: "
            f"trajectory={missing_trajectory}, ftv={missing_ftv}, "
            f"features={missing_features}, extra_features={extra_features}"
        )

    selected_features = [
        features[(patient, point)]
        for patient in splits
        for point in REQUIRED_TIMEPOINTS
    ]
    _validate_global_feature_identity(selected_features)
    mismatches = 0
    unreadable = 0
    payload_failures: Counter[str] = Counter()
    for entry in selected_features:
        try:
            observed_hash = _sha256_file(Path(entry["feature_path"]))
        except OSError:
            unreadable += 1
            continue
        if observed_hash != entry["content_sha256"]:
            mismatches += 1
            continue
        payload_failure = _verify_cached_feature_payload(
            Path(entry["feature_path"]),
            expected_provenance=feature_meta,
            expected_source_hash=str(entry["source_content_sha256"]),
            expected_spatial_provenance=spatial_meta,
        )
        if payload_failure is not None:
            payload_failures[payload_failure] += 1
    if unreadable or mismatches or payload_failures:
        raise ManifestBuildError(
            "feature-file verification failed: "
            f"unreadable={unreadable}, hash_mismatch={mismatches}, "
            f"payload_failures={dict(sorted(payload_failures.items()))}"
        )

    records: list[dict[str, Any]] = []
    split_rank = {name: index for index, name in enumerate(ALLOWED_SPLITS)}
    ordered_patients = sorted(splits, key=lambda patient: (split_rank[splits[patient]], patient))
    for patient in ordered_patients:
        ftv_row = ftv[patient]
        times: dict[str, float] = {}
        for point in REQUIRED_TIMEPOINTS:
            trajectory_row = trajectory[(patient, point)]
            parsed_time = _finite_float(
                trajectory_row[trajectory_columns["time"]],
                field="trajectory delta_t_days",
            )
            assert parsed_time is not None
            times[point] = parsed_time
        if abs(times["T0"]) > 1e-6 or not (
            times["T0"] < times["T1"] < times["T3"]
        ):
            raise ManifestBuildError(
                "required visit times violate T0=0 and T0<T1<T3 chronology"
            )

        ftv_values: dict[str, float] = {}
        for point, column_key in (
            ("T0", "ftv_t0"),
            ("T1", "ftv_t1"),
            ("T3", "ftv_t3"),
        ):
            column = ftv_columns[column_key]
            assert column is not None
            parsed_ftv = _finite_float(ftv_row[column], field=f"cFTV_{point}")
            assert parsed_ftv is not None
            if parsed_ftv < 0:
                raise ManifestBuildError(f"cFTV_{point} contains negative values")
            ftv_values[point] = parsed_ftv

        hr = _patient_binary(
            patient,
            field="HR",
            trajectory=trajectory,
            trajectory_column=trajectory_columns["hr"],
            split_row=split_source_rows[patient],
            split_column=split_columns["hr"],
            ftv_row=ftv_row,
            ftv_column=ftv_columns["hr"],
            allow_missing=False,
        )
        her2 = _patient_binary(
            patient,
            field="HER2",
            trajectory=trajectory,
            trajectory_column=trajectory_columns["her2"],
            split_row=split_source_rows[patient],
            split_column=split_columns["her2"],
            ftv_row=ftv_row,
            ftv_column=ftv_columns["her2"],
            allow_missing=False,
        )
        pcr = _patient_binary(
            patient,
            field="pCR",
            trajectory=trajectory,
            trajectory_column=trajectory_columns["pcr"],
            split_row=split_source_rows[patient],
            split_column=split_columns["pcr"],
            ftv_row=ftv_row,
            ftv_column=ftv_columns["pcr"],
            allow_missing=True,
        )

        visits: dict[str, dict[str, Any]] = {}
        for point in REQUIRED_TIMEPOINTS:
            feature = features[(patient, point)]
            visits[point] = {
                "visit_id": point,
                "scan_id": feature["scan_id"],
                "content_sha256": feature["content_sha256"],
                "source_content_sha256": feature["source_content_sha256"],
                "volume": str(feature["feature_path"]),
                "time_days": times[point],
                "ftv": ftv_values[point],
            }
        visits["T0"]["state_action"] = [0, 0]
        visits["T1"]["state_action"] = [1, 0]
        target = visits["T3"]
        target["pcr"] = pcr
        records.append(
            {
                "schema_version": SCHEMA_VERSION,
                "modality": DEFAULT_MODALITY,
                "tensor_kind": DEFAULT_TENSOR_KIND,
                "feature_provenance": deepcopy(feature_meta),
                "patient_id": patient,
                "split": splits[patient],
                "clinical_features": list(CLINICAL_FEATURES),
                "action_features": list(ACTION_FEATURES),
                "clinical": [hr, her2],
                "context": [visits["T0"], visits["T1"]],
                "target": target,
                "spatial_provenance": deepcopy(spatial_meta),
                "next_action": [1, 1],
            }
        )

    summary: dict[str, Any] = {
        "schema_version": MANIFEST_BUILD_REPORT_SCHEMA_VERSION,
        "status": "ok",
        "builder_source_sha256": _sha256_file(Path(__file__).resolve()),
        "records": len(records),
        "split_counts": observed,
        "split_counts_full_cohort": full_observed,
        "split_counts_excluded_qc": excluded_counts,
        "context_visits": list(CONTEXT_TIMEPOINTS),
        "target_visit": TARGET_TIMEPOINT,
        "verified_feature_files": len(selected_features),
        "trajectory_manifest_sha256": trajectory_sha256,
        "roi_index_sha256": roi_index_sha256,
        "feature_index_sha256": feature_index_sha256,
        "ftv_release": ftv_release,
        "roi_release": roi_release.public_summary(),
    }
    release_bundle = {
        "roi_report_sha256": summary["roi_release"]["roi_report_sha256"],
        "qc_exclusion_roster_sha256": summary["roi_release"][
            "qc_exclusion_roster_sha256"
        ],
        "roi_index_sha256": roi_index_sha256,
    }
    summary["roi_release_bundle_sha256"] = hashlib.sha256(
        json.dumps(
            release_bundle,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return records, summary


def write_manifest_records(
    records: Sequence[Mapping[str, Any]], output_path: str | Path
) -> None:
    """Write complete records once, refusing to replace an existing artifact."""

    output = Path(output_path).expanduser()
    if output.exists():
        raise ManifestBuildError("output already exists; overwrite is forbidden")
    if not output.parent.is_dir():
        raise ManifestBuildError("output parent directory does not exist")
    lines: list[str] = []
    try:
        for record in records:
            lines.append(
                json.dumps(
                    record,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
    except (TypeError, ValueError) as error:
        raise ManifestBuildError("manifest records are not finite JSON objects") from error
    try:
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as error:
        raise ManifestBuildError("output already exists; overwrite is forbidden") from error
    except OSError as error:
        raise ManifestBuildError("output could not be written") from error


def build_and_write_manifest(
    *,
    output_path: str | Path,
    root: str | Path = DEFAULT_ISPY2_ROOT,
    trajectory_manifest_path: str | Path,
    ftv_path: str | Path | None = None,
    ftv_audit_path: str | Path | None = None,
    roi_report_path: str | Path,
    qc_exclusion_path: str | Path | None = None,
    feature_index_path: str | Path,
    roi_index_path: str | Path,
    allowed_feature_root: str | Path | None = None,
    feature_provenance: Mapping[str, Any],
    spatial_provenance: Mapping[str, Any],
    expected_split_counts: Mapping[str, int] = DEFAULT_EXPECTED_SPLIT_COUNTS,
    modality: str = DEFAULT_MODALITY,
) -> dict[str, Any]:
    """Validate all inputs, then create the immutable JSONL artifact."""

    output = Path(output_path).expanduser()
    if output.exists():
        raise ManifestBuildError("output already exists; overwrite is forbidden")
    records, summary = build_manifest_records(
        root=root,
        trajectory_manifest_path=trajectory_manifest_path,
        ftv_path=ftv_path,
        ftv_audit_path=ftv_audit_path,
        roi_report_path=roi_report_path,
        qc_exclusion_path=qc_exclusion_path,
        feature_index_path=feature_index_path,
        roi_index_path=roi_index_path,
        allowed_feature_root=allowed_feature_root,
        feature_provenance=feature_provenance,
        spatial_provenance=spatial_provenance,
        expected_split_counts=expected_split_counts,
        modality=modality,
    )
    write_manifest_records(records, output)
    try:
        summary["manifest_sha256"] = _sha256_file(output)
    except OSError as error:
        raise ManifestBuildError(
            "written manifest could not be hashed for release provenance"
        ) from error
    return summary


def _load_json_argument(inline: str | None, path: str | None, *, field: str) -> dict[str, Any]:
    try:
        if inline is not None:
            value = json.loads(inline)
        elif path is not None:
            with Path(path).expanduser().open("r", encoding="utf-8") as handle:
                value = json.load(handle)
        else:  # argparse makes this unreachable; kept fail-closed for direct calls.
            raise ManifestBuildError(f"{field} JSON is required")
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ManifestBuildError(f"{field} JSON could not be read") from error
    return _json_object(value, field=field)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a strict T0,T1-to-T3 ISPY2 cached-feature JSONL manifest; "
            "prints aggregate counts only."
        )
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ISPY2_ROOT)
    parser.add_argument(
        "--trajectory-manifest",
        type=Path,
        required=True,
        help=(
            "Governed trajectory CSV whose byte snapshot must match the ROI "
            "release evidence digest."
        ),
    )
    parser.add_argument(
        "--ftv-csv",
        type=Path,
        required=True,
        help="Governed computed-FTV CSV released by the full-cohort audit.",
    )
    parser.add_argument(
        "--ftv-audit-report",
        type=Path,
        required=True,
        help="Matching aggregate audit JSON that approves and hashes the FTV CSV.",
    )
    parser.add_argument(
        "--roi-release-report",
        type=Path,
        required=True,
        help="Aggregate converter-v2 report for the promoted full ROI release.",
    )
    parser.add_argument(
        "--qc-exclusion-roster",
        type=Path,
        help="Private governed roster; required only when the ROI report declares QC exclusions.",
    )
    parser.add_argument("--feature-index", type=Path, required=True)
    parser.add_argument(
        "--roi-index",
        type=Path,
        required=True,
        help="Governed ROI index hashed by the feature-cache provenance.",
    )
    parser.add_argument(
        "--allowed-feature-root",
        type=Path,
        help="Approved cache root; defaults to the feature-index directory.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modality", default=DEFAULT_MODALITY)
    feature_group = parser.add_mutually_exclusive_group(required=True)
    feature_group.add_argument("--feature-provenance-json")
    feature_group.add_argument("--feature-provenance-file")
    spatial_group = parser.add_mutually_exclusive_group(required=True)
    spatial_group.add_argument("--spatial-provenance-json")
    spatial_group.add_argument("--spatial-provenance-file")
    parser.add_argument(
        "--expected-split-counts-json",
        default=json.dumps(DEFAULT_EXPECTED_SPLIT_COUNTS, separators=(",", ":")),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        feature_provenance = _load_json_argument(
            args.feature_provenance_json,
            args.feature_provenance_file,
            field="feature_provenance",
        )
        spatial_provenance = _load_json_argument(
            args.spatial_provenance_json,
            args.spatial_provenance_file,
            field="spatial_provenance",
        )
        try:
            split_counts_value = json.loads(args.expected_split_counts_json)
        except json.JSONDecodeError as error:
            raise ManifestBuildError("expected split counts JSON is invalid") from error
        if not isinstance(split_counts_value, Mapping):
            raise ManifestBuildError("expected split counts JSON must be an object")
        summary = build_and_write_manifest(
            output_path=args.output,
            root=args.root,
            trajectory_manifest_path=args.trajectory_manifest,
            ftv_path=args.ftv_csv,
            ftv_audit_path=args.ftv_audit_report,
            roi_report_path=args.roi_release_report,
            qc_exclusion_path=args.qc_exclusion_roster,
            feature_index_path=args.feature_index,
            roi_index_path=args.roi_index,
            allowed_feature_root=args.allowed_feature_root,
            feature_provenance=feature_provenance,
            spatial_provenance=spatial_provenance,
            expected_split_counts=dict(split_counts_value),
            modality=args.modality,
        )
    except ManifestBuildError as error:
        print(json.dumps({"status": "error", "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
