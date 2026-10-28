"""Validation contract for a governed FTV CSV and its aggregate audit report."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
from pathlib import Path
from typing import Any, Mapping


class FTVReleaseError(ValueError):
    """Raised when an FTV artifact pair is incomplete or not approved."""


EXPECTED_CONSUMER_CONTRACT = (
    "require_report_matching_csv_sha256_and_zero_only_ftv_policy_v2"
)
APPROVED_CLINICAL_WORKBOOK_SHA256 = (
    "f714c7784b1e57daa74d7cfb20db71cd432b4e4596b9b4eacdd5a76b7f8a58dc"
)
APPROVED_TRAJECTORY_SHA256 = (
    "11ffa254b5dc9c4f5edc3132c089ec352ee979a4b30098a1f0c7a577ec45ff7a"
)
APPROVED_SPLIT_SHA256 = (
    "b34636e5166ba7187711887afb6087b672898463ef79a02ec489fe730cb05626"
)
APPROVED_MASK_DOCUMENT_SHA256 = (
    "0115d28ce5d11cc134f663ac111731a6df85b08945c9be17a9ac209b36cecb47"
)
APPROVED_MASK_SEMANTICS = "inverse_bit_encoded_ftv_analysis_mask_zero_is_included"
ZERO_ONLY_FTV_RELEASE_POLICY_ID = "zero_only_ftv_taxonomy_separation_v1"
DOCUMENTED_EXCLUSION_BIT_UNION = 107
DOCUMENTED_EXCLUSION_BITS = {
    "pe_threshold": 1,
    "mnc_filter": 2,
    "background_mask": 8,
    "manual_voi": 32,
    "omit_regions": 64,
}
POLICY_PERMITTED_UNDOCUMENTED_BITS = (16,)
REQUIRED_TIMEPOINTS = ("T0", "T1", "T3")
EXPECTED_COMPLETE_PATIENTS = 595
EXPECTED_MASKS = EXPECTED_COMPLETE_PATIENTS * len(REQUIRED_TIMEPOINTS)
EXPECTED_SPLIT_COUNTS = {"train": 416, "val": 89, "test": 90}
EXPECTED_CLINICAL_MATCHED_PATIENTS = 278
CLINICAL_VALIDATION_THRESHOLDS = {
    "minimum_pairs_per_timepoint": 270,
    "minimum_pearson_correlation": 0.995,
    "minimum_median_ratio": 0.98,
    "maximum_median_ratio": 1.02,
    "maximum_mape_percent": 5.0,
}
FTV_CSV_FIELDS = (
    "PatientID",
    "cFTV_T0",
    "cFTV_T1",
    "cFTV_T3",
    "included_voxels_T0",
    "included_voxels_T1",
    "included_voxels_T3",
    "voxel_volume_mm3_T0",
    "voxel_volume_mm3_T1",
    "voxel_volume_mm3_T3",
)
_PATIENT_ID = re.compile(r"^ISPY2-[0-9]+$")


def _read_bytes(path: Path, *, label: str) -> bytes:
    try:
        with path.open("rb") as handle:
            return handle.read()
    except FileNotFoundError as error:
        raise FTVReleaseError(f"required {label} is missing") from error
    except OSError as error:
        raise FTVReleaseError(f"required {label} is unreadable") from error


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _mapping(value: Any, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise FTVReleaseError(f"FTV audit report lacks valid {field}")
    return value


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise FTVReleaseError(f"FTV audit report has invalid {field}")
    return value


def _raw_code_count_table(value: Any, *, field: str) -> dict[int, int]:
    table = _mapping(value, field=field)
    result: dict[int, int] = {}
    for raw_code, raw_count in table.items():
        try:
            code = int(str(raw_code))
        except ValueError as error:
            raise FTVReleaseError(f"FTV audit report has invalid {field}") from error
        if str(code) != str(raw_code) or code < 0 or code > 255 or code in result:
            raise FTVReleaseError(f"FTV audit report has invalid {field}")
        count = _nonnegative_integer(raw_count, field=f"{field} voxel count")
        if count == 0:
            raise FTVReleaseError(f"FTV audit report has zero count in {field}")
        result[code] = count
    return result


def _finite_number(value: Any, *, field: str, minimum: float | None = None) -> float:
    if isinstance(value, bool):
        raise FTVReleaseError(f"FTV release has invalid {field}")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise FTVReleaseError(f"FTV release has invalid {field}") from error
    if not math.isfinite(result) or (minimum is not None and result < minimum):
        raise FTVReleaseError(f"FTV release has invalid {field}")
    return result


def _parse_governed_ftv_csv(content: bytes) -> tuple[list[str], list[dict[str, str]]]:
    try:
        text = content.decode("utf-8-sig")
        reader = csv.DictReader(io.StringIO(text, newline=""))
        if reader.fieldnames is None or tuple(reader.fieldnames) != FTV_CSV_FIELDS:
            raise FTVReleaseError("governed FTV CSV schema is not approved")
        rows: list[dict[str, str]] = []
        for row in reader:
            if None in row or set(row) != set(FTV_CSV_FIELDS):
                raise FTVReleaseError("governed FTV CSV contains a malformed row")
            normalized = {
                field: "" if row[field] is None else str(row[field]).strip()
                for field in FTV_CSV_FIELDS
            }
            rows.append(normalized)
    except UnicodeError as error:
        raise FTVReleaseError("governed FTV CSV is not valid UTF-8") from error
    except csv.Error as error:
        raise FTVReleaseError("governed FTV CSV is unreadable") from error
    if len(rows) != EXPECTED_COMPLETE_PATIENTS:
        raise FTVReleaseError("governed FTV CSV does not contain 595 patient rows")

    patient_ids: set[str] = set()
    for row in rows:
        patient_id = row["PatientID"]
        if not _PATIENT_ID.fullmatch(patient_id) or patient_id in patient_ids:
            raise FTVReleaseError(
                "governed FTV CSV contains invalid or duplicate patient identifiers"
            )
        patient_ids.add(patient_id)
        for timepoint in REQUIRED_TIMEPOINTS:
            ftv = _finite_number(
                row[f"cFTV_{timepoint}"], field=f"cFTV_{timepoint}", minimum=0.0
            )
            raw_included = row[f"included_voxels_{timepoint}"]
            try:
                included = int(raw_included)
            except ValueError as error:
                raise FTVReleaseError(
                    "governed FTV CSV has invalid included-voxel count"
                ) from error
            if str(included) != raw_included or included < 0:
                raise FTVReleaseError(
                    "governed FTV CSV has invalid included-voxel count"
                )
            voxel_volume = _finite_number(
                row[f"voxel_volume_mm3_{timepoint}"],
                field=f"voxel_volume_mm3_{timepoint}",
            )
            if voxel_volume <= 0:
                raise FTVReleaseError(
                    "governed FTV CSV has non-positive voxel volume"
                )
            expected_ftv = included * voxel_volume / 1000.0
            if not math.isclose(ftv, expected_ftv, rel_tol=1e-8, abs_tol=1e-10):
                raise FTVReleaseError(
                    "governed FTV CSV FTV disagrees with voxel accounting"
                )
    return list(FTV_CSV_FIELDS), rows


def _actual_undocumented_bit_audit(report: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute aggregate unknown-bit facts from the per-timepoint raw-code tables."""

    groups = _mapping(report.get("groups"), field="groups")
    observed_bits: set[int] = set()
    aggregate_bit_counts: dict[int, int] = {4: 0, 16: 0, 128: 0}
    total_unknown_masks = 0
    total_undocumented_only_voxels = 0
    total_undocumented_only_masks = 0
    for timepoint in REQUIRED_TIMEPOINTS:
        group = _mapping(groups.get(timepoint), field=f"groups.{timepoint}")
        if _nonnegative_integer(
            group.get("masks_audited"), field=f"groups.{timepoint}.masks_audited"
        ) != EXPECTED_COMPLETE_PATIENTS:
            raise FTVReleaseError("FTV audit group does not cover the full cohort")
        raw_counts = _raw_code_count_table(
            group.get("raw_code_voxels"),
            field=f"groups.{timepoint}.raw_code_voxels",
        )
        if not raw_counts or sum(raw_counts.values()) != _nonnegative_integer(
            group.get("total_voxels"), field=f"groups.{timepoint}.total_voxels"
        ):
            raise FTVReleaseError("FTV audit raw-code total is inconsistent")
        derived_undocumented_raw_counts = {
            code: count
            for code, count in raw_counts.items()
            if code & ~DOCUMENTED_EXCLUSION_BIT_UNION
        }
        reported_undocumented_raw_counts = _raw_code_count_table(
            group.get("undocumented_raw_code_voxels"),
            field=f"groups.{timepoint}.undocumented_raw_code_voxels",
        )
        if reported_undocumented_raw_counts != derived_undocumented_raw_counts:
            raise FTVReleaseError(
                "FTV audit undocumented raw-code table disagrees with raw-code truth"
            )
        derived_bit_counts: dict[int, int] = {}
        derived_only_voxels = 0
        derived_only_raw_counts: dict[int, int] = {}
        for code, count in derived_undocumented_raw_counts.items():
            unknown_component = code & ~DOCUMENTED_EXCLUSION_BIT_UNION
            for index in range(8):
                bit = 1 << index
                if unknown_component & bit:
                    observed_bits.add(bit)
                    derived_bit_counts[bit] = derived_bit_counts.get(bit, 0) + count
            if code & DOCUMENTED_EXCLUSION_BIT_UNION == 0:
                derived_only_voxels += count
                derived_only_raw_counts[code] = count

        reported_only_raw_counts = _raw_code_count_table(
            group.get("undocumented_only_raw_code_voxels"),
            field=f"groups.{timepoint}.undocumented_only_raw_code_voxels",
        )
        if reported_only_raw_counts != derived_only_raw_counts:
            raise FTVReleaseError(
                "FTV audit undocumented-only raw-code table is inconsistent"
            )

        reported_bits = _mapping(
            group.get("undocumented_single_bit_voxels"),
            field=f"groups.{timepoint}.undocumented_single_bit_voxels",
        )
        normalized_reported_bits: dict[int, int] = {}
        for raw_bit, raw_count in reported_bits.items():
            try:
                bit = int(str(raw_bit))
            except ValueError as error:
                raise FTVReleaseError("FTV audit has invalid undocumented bit key") from error
            if bit < 1 or bit > 128 or bit & (bit - 1):
                raise FTVReleaseError("FTV audit has invalid undocumented bit key")
            normalized_reported_bits[bit] = _nonnegative_integer(
                raw_count, field="undocumented single-bit voxel count"
            )
        if set(normalized_reported_bits) != {4, 16, 128}:
            raise FTVReleaseError("FTV audit undocumented bit table is incomplete")
        all_unknown_bits = set(normalized_reported_bits).union(derived_bit_counts)
        if any(
            normalized_reported_bits.get(bit, 0) != derived_bit_counts.get(bit, 0)
            for bit in all_unknown_bits
        ):
            raise FTVReleaseError(
                "FTV audit undocumented bit counts disagree with raw-code counts"
            )
        for bit in aggregate_bit_counts:
            aggregate_bit_counts[bit] += derived_bit_counts.get(bit, 0)

        reported_only_voxels = _nonnegative_integer(
            group.get("undocumented_only_voxels"),
            field="undocumented-only voxel count",
        )
        if reported_only_voxels != derived_only_voxels:
            raise FTVReleaseError(
                "FTV audit undocumented-only count disagrees with raw-code counts"
            )
        only_masks = _nonnegative_integer(
            group.get("masks_with_undocumented_only_voxels"),
            field="masks-with-undocumented-only count",
        )
        unknown_masks = _nonnegative_integer(
            group.get("masks_with_undocumented_bits"),
            field="masks-with-undocumented-bits count",
        )
        if unknown_masks > EXPECTED_COMPLETE_PATIENTS:
            raise FTVReleaseError("FTV audit undocumented mask accounting is invalid")
        if bool(derived_undocumented_raw_counts) != bool(unknown_masks):
            raise FTVReleaseError("FTV audit undocumented mask accounting is invalid")
        if (reported_only_voxels == 0) != (only_masks == 0):
            raise FTVReleaseError("FTV audit undocumented-only mask accounting is invalid")
        if only_masks > unknown_masks:
            raise FTVReleaseError("FTV audit undocumented mask accounting is invalid")
        total_unknown_masks += unknown_masks
        total_undocumented_only_voxels += reported_only_voxels
        total_undocumented_only_masks += only_masks

    return {
        "observed_bits": sorted(observed_bits),
        "observed_bit_union": sum(observed_bits),
        "single_bit_voxels": aggregate_bit_counts,
        "masks_with_undocumented_bits": total_unknown_masks,
        "undocumented_only_voxels": total_undocumented_only_voxels,
        "masks_with_undocumented_only_voxels": total_undocumented_only_masks,
    }


def _validate_clinical_ftv_validation(value: Any) -> None:
    clinical = _mapping(value, field="clinical_ftv_validation")
    if clinical.get("status") != "ok":
        raise FTVReleaseError("FTV clinical validation is not approved")
    if clinical.get("matched_patients_any_timepoint") != EXPECTED_CLINICAL_MATCHED_PATIENTS:
        raise FTVReleaseError("FTV clinical validation cohort is not approved")
    metrics = _mapping(
        clinical.get("metrics"), field="clinical_ftv_validation.metrics"
    )
    gates = _mapping(clinical.get("gates"), field="clinical_ftv_validation.gates")
    if gates.get("thresholds") != CLINICAL_VALIDATION_THRESHOLDS:
        raise FTVReleaseError("FTV clinical validation thresholds are not approved")
    if gates.get("all_passed") is not True:
        raise FTVReleaseError("FTV clinical validation aggregate gate did not pass")
    per_timepoint = _mapping(
        gates.get("per_timepoint"),
        field="clinical_ftv_validation.gates.per_timepoint",
    )
    if set(per_timepoint) != set(REQUIRED_TIMEPOINTS) or any(
        per_timepoint.get(timepoint) is not True for timepoint in REQUIRED_TIMEPOINTS
    ):
        raise FTVReleaseError("FTV clinical validation timepoint gate did not pass")
    if set(metrics) != set(REQUIRED_TIMEPOINTS):
        raise FTVReleaseError("FTV clinical validation metrics are incomplete")
    for timepoint in REQUIRED_TIMEPOINTS:
        metric = _mapping(
            metrics.get(timepoint),
            field=f"clinical_ftv_validation.metrics.{timepoint}",
        )
        pairs = _nonnegative_integer(metric.get("pairs"), field="clinical pairs")
        correlation = _finite_number(
            metric.get("pearson_correlation"), field="clinical Pearson correlation"
        )
        ratio = _finite_number(
            metric.get("median_computed_to_reference_ratio"),
            field="clinical median ratio",
        )
        mape = _finite_number(
            metric.get("mape_percent"), field="clinical MAPE", minimum=0.0
        )
        mae = _finite_number(
            metric.get("mae_ml"), field="clinical MAE", minimum=0.0
        )
        if (
            pairs != EXPECTED_CLINICAL_MATCHED_PATIENTS
            or correlation < CLINICAL_VALIDATION_THRESHOLDS[
                "minimum_pearson_correlation"
            ]
            or correlation > 1.0
            or not CLINICAL_VALIDATION_THRESHOLDS["minimum_median_ratio"]
            <= ratio
            <= CLINICAL_VALIDATION_THRESHOLDS["maximum_median_ratio"]
            or mape > CLINICAL_VALIDATION_THRESHOLDS["maximum_mape_percent"]
        ):
            raise FTVReleaseError(
                "FTV clinical validation metric does not satisfy its gate"
            )


def validate_ftv_release(
    ftv_csv_path: str | Path,
    audit_report_path: str | Path,
) -> dict[str, Any]:
    """Verify that an approved audit report cryptographically releases one CSV.

    The function never searches the dataset tree and has no fallback path.  Both
    artifacts must be explicitly supplied by the caller.
    """

    csv_path = Path(ftv_csv_path).expanduser()
    report_path = Path(audit_report_path).expanduser()
    report_content = _read_bytes(report_path, label="FTV audit report")
    try:
        report = json.loads(report_content.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise FTVReleaseError("required FTV audit report is unreadable") from error
    if not isinstance(report, Mapping):
        raise FTVReleaseError("FTV audit report must be a JSON object")
    if report.get("schema_version") != "med-jepa-ispy2-ftv-audit-v3":
        raise FTVReleaseError("FTV audit report schema is not approved")

    cohort = _mapping(report.get("cohort"), field="cohort")
    expected_cohort = {
        "complete_patients": EXPECTED_COMPLETE_PATIENTS,
        "required_timepoints": list(REQUIRED_TIMEPOINTS),
        "masks_audited": EXPECTED_MASKS,
        "unique_seg_objects": EXPECTED_MASKS,
        "split_counts": EXPECTED_SPLIT_COUNTS,
    }
    if any(cohort.get(field) != expected for field, expected in expected_cohort.items()):
        raise FTVReleaseError("FTV audit report does not cover the approved cohort")

    mask_contract = _mapping(report.get("mask_contract"), field="mask_contract")
    expected_mask_contract = {
        "semantics": APPROVED_MASK_SEMANTICS,
        "included_condition": "raw_value_equals_zero",
        "allowed_exclusion_bits": DOCUMENTED_EXCLUSION_BITS,
        "allowed_bit_union": DOCUMENTED_EXCLUSION_BIT_UNION,
        "undocumented_single_bits": [4, 16, 128],
        "document_sha256": APPROVED_MASK_DOCUMENT_SHA256,
    }
    if any(
        mask_contract.get(field) != expected
        for field, expected in expected_mask_contract.items()
    ):
        raise FTVReleaseError("FTV documented exclusion-bit contract is invalid")

    status = report.get("status")
    if status not in {"ok", "ok_ftv_pass_taxonomy_blocked"}:
        raise FTVReleaseError("FTV audit report status is not releasable")

    actual_unknown = _actual_undocumented_bit_audit(report)
    observed_bits = actual_unknown["observed_bits"]
    taxonomy_blocked_release = status == "ok_ftv_pass_taxonomy_blocked"
    if taxonomy_blocked_release:
        if observed_bits != list(POLICY_PERMITTED_UNDOCUMENTED_BITS):
            raise FTVReleaseError("taxonomy-blocked release has unsupported unknown bits")
        if actual_unknown["masks_with_undocumented_bits"] < 1:
            raise FTVReleaseError("taxonomy-blocked release has no unknown-bit masks")
        if (
            actual_unknown["undocumented_only_voxels"] != 0
            or actual_unknown["masks_with_undocumented_only_voxels"] != 0
        ):
            raise FTVReleaseError(
                "taxonomy-blocked release contains undocumented-only exclusions"
            )
    elif observed_bits or actual_unknown["masks_with_undocumented_bits"]:
        raise FTVReleaseError("taxonomy-pass release contains undocumented bits")

    bit_audit = _mapping(
        report.get("undocumented_bit_audit"), field="undocumented_bit_audit"
    )
    expected_bit_audit = {
        "present": bool(observed_bits),
        "observed_undocumented_single_bits": observed_bits,
        "observed_undocumented_bit_union": actual_unknown["observed_bit_union"],
        "undocumented_single_bit_voxels": {
            str(bit): count
            for bit, count in actual_unknown["single_bit_voxels"].items()
        },
        "masks_with_undocumented_bits": actual_unknown[
            "masks_with_undocumented_bits"
        ],
        "undocumented_only_voxels": actual_unknown[
            "undocumented_only_voxels"
        ],
        "masks_with_undocumented_only_voxels": actual_unknown[
            "masks_with_undocumented_only_voxels"
        ],
        "only_reviewed_bit16": observed_bits == [16],
        "unknown_bits_added_to_allowlist": False,
        "candidate_ftv_rule": "raw_value_equals_zero",
    }
    if any(
        bit_audit.get(field) != expected
        for field, expected in expected_bit_audit.items()
    ):
        raise FTVReleaseError("FTV aggregate undocumented-bit audit is inconsistent")

    gate = _mapping(report.get("gate_decision"), field="gate_decision")
    expected_gate = {
        "status": status,
        "candidate_ftv": "pass",
        "clinical_workbook_provenance": "pass",
        "exclusion_taxonomy": "blocked" if taxonomy_blocked_release else "pass",
        "zero_only_ftv_release_policy": "pass",
        "patient_csv_publication": "approved",
        "release_scope": (
            "zero_rule_ftv_only"
            if taxonomy_blocked_release
            else "ftv_and_documented_taxonomy"
        ),
        "taxonomy_claim_publication": (
            "withheld" if taxonomy_blocked_release else "approved"
        ),
    }
    if any(gate.get(field) != expected for field, expected in expected_gate.items()):
        raise FTVReleaseError("FTV audit gate decision is not fully approved")
    if gate.get("publication_blockers") != []:
        raise FTVReleaseError("FTV audit gate decision contains publication blockers")
    if gate.get("observed_undocumented_single_bits") != observed_bits:
        raise FTVReleaseError("FTV audit gate unknown-bit declaration is inconsistent")

    policy = _mapping(
        report.get("zero_only_ftv_release_policy"),
        field="zero_only_ftv_release_policy",
    )
    expected_policy = {
        "policy_id": ZERO_ONLY_FTV_RELEASE_POLICY_ID,
        "decision": "pass",
        "candidate_ftv_inclusion_rule": "raw_value_equals_zero",
        "permitted_undocumented_single_bits": list(
            POLICY_PERMITTED_UNDOCUMENTED_BITS
        ),
        "observed_undocumented_single_bits": observed_bits,
        "requires_undocumented_only_voxels_zero": True,
        "undocumented_only_voxels": 0,
        "masks_with_undocumented_only_voxels": 0,
        "assigns_semantics_to_undocumented_bits": False,
    }
    if any(policy.get(field) != expected for field, expected in expected_policy.items()):
        raise FTVReleaseError("FTV zero-only release policy assertion is invalid")
    input_contract = _mapping(report.get("input_contract"), field="input_contract")
    if input_contract.get("trajectory_manifest_sha256") != APPROVED_TRAJECTORY_SHA256:
        raise FTVReleaseError("FTV trajectory manifest digest is not approved")
    if input_contract.get("split_sha256") != APPROVED_SPLIT_SHA256:
        raise FTVReleaseError("FTV split digest is not approved")
    if (
        input_contract.get("clinical_workbook_sha256")
        != APPROVED_CLINICAL_WORKBOOK_SHA256
    ):
        raise FTVReleaseError("FTV clinical workbook digest is not approved")
    if input_contract.get("clinical_workbook_sha256_pinned_in_source") is not True:
        raise FTVReleaseError("FTV clinical workbook digest is not pinned in source")
    if input_contract.get("clinical_workbook_sha256_matches_pinned") is not True:
        raise FTVReleaseError("FTV clinical workbook digest does not match its pin")
    _validate_clinical_ftv_validation(report.get("clinical_ftv_validation"))
    patient_csv = _mapping(report.get("patient_csv"), field="patient_csv")
    if patient_csv.get("written") is not True:
        raise FTVReleaseError("FTV audit report does not confirm a written patient CSV")
    if patient_csv.get("governed_output") is not True:
        raise FTVReleaseError("FTV audit report does not confirm governed CSV output")
    if patient_csv.get("rows") != EXPECTED_COMPLETE_PATIENTS:
        raise FTVReleaseError("FTV audit report has invalid patient CSV row count")
    if patient_csv.get("contains_identifiers") is not True:
        raise FTVReleaseError("FTV audit report lacks patient CSV identifier warning")
    if patient_csv.get("consumer_contract") != EXPECTED_CONSUMER_CONTRACT:
        raise FTVReleaseError("FTV patient CSV consumer contract is missing or invalid")
    expected_sha256 = patient_csv.get("sha256")
    if (
        not isinstance(expected_sha256, str)
        or len(expected_sha256) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in expected_sha256)
    ):
        raise FTVReleaseError("FTV audit report has an invalid patient CSV SHA-256")

    csv_content = _read_bytes(csv_path, label="governed FTV CSV")
    observed_sha256 = _sha256_bytes(csv_content)
    if observed_sha256 != expected_sha256.casefold():
        raise FTVReleaseError("governed FTV CSV SHA-256 does not match audit report")
    ftv_fields, ftv_rows = _parse_governed_ftv_csv(csv_content)
    audit_sha256 = _sha256_bytes(report_content)
    return {
        "validated": True,
        "ftv_csv_sha256": observed_sha256,
        "ftv_audit_report_sha256": audit_sha256,
        "ftv_csv_fields": ftv_fields,
        "ftv_csv_rows": ftv_rows,
    }
