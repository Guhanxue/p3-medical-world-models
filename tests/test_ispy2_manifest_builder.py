from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from med_jepa.ftv_release import (
    APPROVED_CLINICAL_WORKBOOK_SHA256,
    APPROVED_MASK_DOCUMENT_SHA256,
    APPROVED_MASK_SEMANTICS,
    APPROVED_SPLIT_SHA256,
    APPROVED_TRAJECTORY_SHA256,
    CLINICAL_VALIDATION_THRESHOLDS,
    DOCUMENTED_EXCLUSION_BITS,
    DOCUMENTED_EXCLUSION_BIT_UNION,
    EXPECTED_CLINICAL_MATCHED_PATIENTS,
    EXPECTED_CONSUMER_CONTRACT,
    FTV_CSV_FIELDS,
    POLICY_PERMITTED_UNDOCUMENTED_BITS,
    ZERO_ONLY_FTV_RELEASE_POLICY_ID,
    validate_ftv_release,
)
from med_jepa.ispy2_manifest_builder import (
    ACTION_FEATURES,
    CLINICAL_FEATURES,
    DEFAULT_ISPY2_ROOT,
    MANIFEST_BUILD_REPORT_SCHEMA_VERSION,
    SCHEMA_VERSION,
    ManifestBuildError,
    build_and_write_manifest,
    build_manifest_records,
    main,
    validate_roi_release_cohort,
    validate_feature_provenance,
    validate_spatial_provenance,
)


SMALL_SPLITS = {"train": 2, "val": 1, "test": 1}
FEATURE_PROVENANCE = {
    "extractor": "AICONSlab/3DINO-ViT",
    "source_revision": "c" * 40,
    "checkpoint_sha256": "a" * 64,
    "roi_index_sha256": "b" * 64,
    "layer": "last_normalized_patch_tokens",
    "feature_dim": 4,
    "feature_grid_shape": [2, 2, 2],
    "input_shape": [4, 4, 4],
    "normalization": "fixed_ser_ratio_clip_0_3_to_minus1_plus1_v1",
}
SPATIAL_PROVENANCE = {
    "reference_visit": "T0",
    "roi_source_visits": ["T0"],
    "spacing_mm": [1.0, 1.0, 2.0],
    "orientation": "RAS",
    "transform_id": "context-only-t0-anchor-v1",
    "roi_voxel_shape": [96, 96, 64],
    "registration_type": "rigid",
    "deformable_registration": False,
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames is not None
        return list(reader.fieldnames), list(reader)


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.feature_index = root / "feature_cache" / "feature_index.csv"
        self.trajectory_manifest = root / "manifests" / "ispy2_trajectory_manifest.csv"
        self.roi_index = root / "governed" / "roi" / "release" / "roi_index.csv"
        self.ftv_csv = root / "governed" / "ftv" / "ispy2_ftv_ds3_v1.csv"
        self.ftv_audit = root / "results" / "ftv_audit" / "audit.json"
        self.roi_report = root / "results" / "roi_conversion" / "report.json"
        self.qc_roster = root / "governed" / "roi" / "release" / "qc_excluded_patients.csv"
        self.output = root / "output.jsonl"
        self.feature_provenance: dict[str, object] = {}
        self.patients = [
            ("ISPY2-100001", "train", 1, 0, 1),
            ("ISPY2-100002", "train", 0, 0, 0),
            ("ISPY2-100003", "val", 1, 1, 1),
            ("ISPY2-100004", "test", 0, 1, None),
        ]
        self._make()

    def _make(self) -> None:
        split_rows: list[dict[str, object]] = []
        trajectory_rows: list[dict[str, object]] = []
        ftv_rows: list[dict[str, object]] = []
        feature_rows: list[dict[str, object]] = []
        roi_index_rows: list[dict[str, object]] = []
        times = {"T0": 0.0, "T1": 40.0, "T2": 105.0, "T3": 170.0}
        for patient_number, (patient, *_rest) in enumerate(self.patients, start=1):
            for point in ("T0", "T1", "T3"):
                roi_index_rows.append(
                    {
                        "PatientID": patient,
                        "tp_label": point,
                        "scan_id": f"scan-{patient_number}-{point}",
                        "roi_path": f"roi/{patient_number}-{point}.pt",
                        "source_content_sha256": hashlib.sha256(
                            f"source:{patient}:{point}".encode()
                        ).hexdigest(),
                    }
                )
        _write_csv(
            self.roi_index,
            [
                "PatientID",
                "tp_label",
                "scan_id",
                "roi_path",
                "source_content_sha256",
            ],
            roi_index_rows,
        )
        self.feature_provenance = {
            **FEATURE_PROVENANCE,
            "roi_index_sha256": _sha256(self.roi_index),
        }
        for patient_number, (patient, split, hr, her2, pcr) in enumerate(
            self.patients, start=1
        ):
            split_rows.append({"PatientID": patient, "split": split})
            for point in ("T0", "T1", "T2", "T3"):
                trajectory_rows.append(
                    {
                        "PatientID": patient,
                        "tp_label": point,
                        "delta_t_days": times[point] + patient_number,
                        "HR": hr,
                        "HER2": her2,
                        "pCR": "" if pcr is None else pcr,
                    }
                )
            # T0 is required to be exactly zero for every patient.
            trajectory_rows[-4]["delta_t_days"] = 0.0
            ftv_rows.append(
                {
                    "PatientID": patient,
                    "cFTV_T0": 10.0 + patient_number,
                    "cFTV_T1": 6.0 + patient_number,
                    "cFTV_T3": 1.0 + patient_number,
                    "included_voxels_T0": int((10.0 + patient_number) * 500),
                    "included_voxels_T1": int((6.0 + patient_number) * 500),
                    "included_voxels_T3": int((1.0 + patient_number) * 500),
                    "voxel_volume_mm3_T0": 2.0,
                    "voxel_volume_mm3_T1": 2.0,
                    "voxel_volume_mm3_T3": 2.0,
                }
            )
            for point in ("T0", "T1", "T3"):
                feature_path = self.root / "feature_cache" / f"{patient}_{point}.pt"
                feature_path.parent.mkdir(parents=True, exist_ok=True)
                source_hash = hashlib.sha256(
                    f"source:{patient}:{point}".encode()
                ).hexdigest()
                torch.save(
                    {
                        "volume": torch.full(
                            (4, 2, 2, 2),
                            float(patient_number * 10 + int(point[1])),
                            dtype=torch.float16,
                        ),
                        "feature_provenance": self.feature_provenance,
                        "source_content_sha256": source_hash,
                        "spatial_provenance": SPATIAL_PROVENANCE,
                    },
                    feature_path,
                )
                feature_rows.append(
                    {
                        "PatientID": patient,
                        "tp_label": point,
                        "scan_id": f"scan-{patient_number}-{point}",
                        "feature_path": feature_path,
                        "content_sha256": _sha256(feature_path),
                        "source_content_sha256": source_hash,
                    }
                )
        for index in range(591):
            ftv_rows.append(
                {
                    "PatientID": f"ISPY2-{200000 + index}",
                    "cFTV_T0": 2.0,
                    "cFTV_T1": 1.0,
                    "cFTV_T3": 0.5,
                    "included_voxels_T0": 1000,
                    "included_voxels_T1": 500,
                    "included_voxels_T3": 250,
                    "voxel_volume_mm3_T0": 2.0,
                    "voxel_volume_mm3_T1": 2.0,
                    "voxel_volume_mm3_T3": 2.0,
                }
            )
        _write_csv(
            self.root / "manifests/ispy2_split.csv",
            ["PatientID", "split"],
            split_rows,
        )
        _write_csv(
            self.trajectory_manifest,
            ["PatientID", "tp_label", "delta_t_days", "HR", "HER2", "pCR"],
            trajectory_rows,
        )
        _write_csv(
            self.ftv_csv,
            list(FTV_CSV_FIELDS),
            ftv_rows,
        )
        self._write_ftv_audit()
        _write_csv(
            self.feature_index,
            [
                "PatientID",
                "tp_label",
                "scan_id",
                "feature_path",
                "content_sha256",
                "source_content_sha256",
            ],
            feature_rows,
        )
        self._write_roi_report(())

    def _write_roi_report(self, excluded_patients: tuple[str, ...]) -> None:
        excluded = len(excluded_patients)
        selected = len(self.patients)
        self.roi_report.parent.mkdir(parents=True, exist_ok=True)
        self.roi_report.write_text(
            json.dumps(
                {
                    "schema_version": "med-jepa-ispy2-ser-roi-converter-v2",
                    "status": (
                        "ok_roi_conversion_full_qc_exclusions"
                        if excluded
                        else "ok_roi_conversion_full"
                    ),
                    "aggregate_only": True,
                    "pixel_data_read": True,
                    "identifiers_paths_uids_dates_or_per_patient_measurements_emitted": False,
                    "cohort": {
                        "patients_total": selected,
                        "patients_selected": selected,
                        "patients_excluded_qc": excluded,
                        "patients_released": selected - excluded,
                        "visits_expected": selected * 3,
                        "split_counts_full_cohort": SMALL_SPLITS,
                    },
                    "roi_contract": {
                        "roi_schema_version": "med-jepa-ispy2-roi-v2",
                    },
                    "conversion": {
                        "failure_reasons": {},
                        "failure_diagnostics": {"metadata_rows": excluded},
                        "qc_exclusion_policy": (
                            "future_mask_outside_fixed_crop_patient_exclusion_v1"
                            if excluded
                            else None
                        ),
                        "release_promoted": True,
                    },
                    "evidence": {
                        "trajectory_manifest_sha256": _sha256(
                            self.root / "manifests/ispy2_trajectory_manifest.csv"
                        ),
                        "split_sha256": _sha256(
                            self.root / "manifests/ispy2_split.csv"
                        ),
                        "ftv_csv_sha256": _sha256(self.ftv_csv),
                        "ftv_audit_report_sha256": _sha256(self.ftv_audit),
                        "roi_spec_sha256": "5" * 64,
                        "roi_anchor_report_sha256": "6" * 64,
                        "converter_source_sha256": "7" * 64,
                    },
                    "next_gate": {
                        "can_cache_3dino": True,
                        "pilot_only": False,
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def set_qc_exclusions(self, *patients: str) -> None:
        excluded = tuple(sorted(set(patients)))
        known = {patient for patient, *_rest in self.patients}
        if not excluded or not set(excluded).issubset(known):
            raise ValueError("fixture exclusions must be non-empty known patients")
        _write_csv(
            self.qc_roster,
            list(("PatientID", "exclusion_reason", "failed_timepoints")),
            [
                {
                    "PatientID": patient,
                    "exclusion_reason": "future_mask_outside_fixed_crop",
                    "failed_timepoints": "T3",
                }
                for patient in excluded
            ],
        )
        roi_fields, roi_rows = _read_csv(self.roi_index)
        _write_csv(
            self.roi_index,
            roi_fields,
            [row for row in roi_rows if row["PatientID"] not in set(excluded)],
        )
        self.feature_provenance = {
            **FEATURE_PROVENANCE,
            "roi_index_sha256": _sha256(self.roi_index),
        }
        fields, rows = _read_csv(self.feature_index)
        released_rows = [
            row for row in rows if row["PatientID"] not in set(excluded)
        ]
        for row in released_rows:
            feature_path = Path(row["feature_path"])
            payload = torch.load(feature_path, map_location="cpu", weights_only=True)
            payload["feature_provenance"] = self.feature_provenance
            torch.save(payload, feature_path)
            row["content_sha256"] = _sha256(feature_path)
        _write_csv(
            self.feature_index,
            fields,
            released_rows,
        )
        self._write_roi_report(excluded)

    def rows(self, relative: str | Path) -> tuple[list[str], list[dict[str, str]]]:
        return _read_csv(self.root / relative)

    def _write_ftv_audit(self) -> None:
        self.ftv_audit.parent.mkdir(parents=True, exist_ok=True)
        empty_group = {
            "masks_audited": 595,
            "total_voxels": 892500,
            "raw_code_voxels": {"0": 595000, "1": 297500},
            "undocumented_raw_code_voxels": {},
            "undocumented_only_raw_code_voxels": {},
            "undocumented_single_bit_voxels": {
                "4": 0,
                "16": 0,
                "128": 0,
            },
            "undocumented_only_voxels": 0,
            "masks_with_undocumented_only_voxels": 0,
            "masks_with_undocumented_bits": 0,
        }
        self.ftv_audit.write_text(
            json.dumps(
                {
                    "schema_version": "med-jepa-ispy2-ftv-audit-v3",
                    "status": "ok",
                    "cohort": {
                        "complete_patients": 595,
                        "required_timepoints": ["T0", "T1", "T3"],
                        "masks_audited": 1785,
                        "unique_seg_objects": 1785,
                        "split_counts": {
                            "train": 416,
                            "val": 89,
                            "test": 90,
                        },
                    },
                    "mask_contract": {
                        "semantics": APPROVED_MASK_SEMANTICS,
                        "included_condition": "raw_value_equals_zero",
                        "allowed_exclusion_bits": DOCUMENTED_EXCLUSION_BITS,
                        "allowed_bit_union": DOCUMENTED_EXCLUSION_BIT_UNION,
                        "undocumented_single_bits": [4, 16, 128],
                        "document_sha256": APPROVED_MASK_DOCUMENT_SHA256,
                    },
                    "groups": {
                        "T0": dict(empty_group),
                        "T1": dict(empty_group),
                        "T3": dict(empty_group),
                    },
                    "undocumented_bit_audit": {
                        "present": False,
                        "observed_undocumented_single_bits": [],
                        "observed_undocumented_bit_union": 0,
                        "undocumented_single_bit_voxels": {
                            "4": 0,
                            "16": 0,
                            "128": 0,
                        },
                        "only_reviewed_bit16": False,
                        "masks_with_undocumented_bits": 0,
                        "undocumented_only_voxels": 0,
                        "masks_with_undocumented_only_voxels": 0,
                        "unknown_bits_added_to_allowlist": False,
                        "candidate_ftv_rule": "raw_value_equals_zero",
                    },
                    "gate_decision": {
                        "status": "ok",
                        "candidate_ftv": "pass",
                        "clinical_workbook_provenance": "pass",
                        "exclusion_taxonomy": "pass",
                        "zero_only_ftv_release_policy": "pass",
                        "patient_csv_publication": "approved",
                        "release_scope": "ftv_and_documented_taxonomy",
                        "taxonomy_claim_publication": "approved",
                        "publication_blockers": [],
                        "observed_undocumented_single_bits": [],
                    },
                    "zero_only_ftv_release_policy": {
                        "policy_id": ZERO_ONLY_FTV_RELEASE_POLICY_ID,
                        "decision": "pass",
                        "candidate_ftv_inclusion_rule": "raw_value_equals_zero",
                        "permitted_undocumented_single_bits": list(
                            POLICY_PERMITTED_UNDOCUMENTED_BITS
                        ),
                        "observed_undocumented_single_bits": [],
                        "requires_undocumented_only_voxels_zero": True,
                        "undocumented_only_voxels": 0,
                        "masks_with_undocumented_only_voxels": 0,
                        "assigns_semantics_to_undocumented_bits": False,
                    },
                    "input_contract": {
                        "trajectory_manifest_sha256": APPROVED_TRAJECTORY_SHA256,
                        "split_sha256": APPROVED_SPLIT_SHA256,
                        "clinical_workbook_sha256": (
                            APPROVED_CLINICAL_WORKBOOK_SHA256
                        ),
                        "clinical_workbook_sha256_pinned_in_source": True,
                        "clinical_workbook_sha256_matches_pinned": True,
                    },
                    "clinical_ftv_validation": {
                        "status": "ok",
                        "matched_patients_any_timepoint": (
                            EXPECTED_CLINICAL_MATCHED_PATIENTS
                        ),
                        "metrics": {
                            point: {
                                "pairs": EXPECTED_CLINICAL_MATCHED_PATIENTS,
                                "pearson_correlation": 0.999,
                                "mae_ml": 0.03,
                                "median_computed_to_reference_ratio": 1.0,
                                "mape_percent": 1.4,
                            }
                            for point in ("T0", "T1", "T3")
                        },
                        "gates": {
                            "thresholds": CLINICAL_VALIDATION_THRESHOLDS,
                            "per_timepoint": {
                                "T0": True,
                                "T1": True,
                                "T3": True,
                            },
                            "all_passed": True,
                        },
                    },
                    "patient_csv": {
                        "written": True,
                        "rows": 595,
                        "sha256": _sha256(self.ftv_csv),
                        "contains_identifiers": True,
                        "governed_output": True,
                        "consumer_contract": EXPECTED_CONSUMER_CONTRACT,
                    },
                }
            ),
            encoding="utf-8",
        )

    def rewrite(
        self,
        relative: str | Path,
        fields: list[str],
        rows: list[dict[str, object]],
    ) -> None:
        destination = self.root / relative
        _write_csv(destination, fields, rows)
        if destination.resolve() == self.ftv_csv.resolve():
            self._write_ftv_audit()
        if destination.resolve() == self.trajectory_manifest.resolve():
            self._write_roi_report(
                tuple(
                    sorted(
                        row["PatientID"]
                        for row in (
                            _read_csv(self.qc_roster)[1]
                            if self.qc_roster.exists()
                            else []
                        )
                    )
                )
            )

    def build(self, **overrides: object):
        options: dict[str, object] = {
            "root": self.root,
            "trajectory_manifest_path": self.trajectory_manifest,
            "ftv_path": self.ftv_csv,
            "ftv_audit_path": self.ftv_audit,
            "roi_report_path": self.roi_report,
            "qc_exclusion_path": self.qc_roster if self.qc_roster.exists() else None,
            "feature_index_path": self.feature_index,
            "roi_index_path": self.roi_index,
            "feature_provenance": self.feature_provenance,
            "spatial_provenance": SPATIAL_PROVENANCE,
            "expected_split_counts": SMALL_SPLITS,
        }
        options.update(overrides)
        return build_manifest_records(**options)  # type: ignore[arg-type]


class ISPY2ManifestBuilderTest(unittest.TestCase):
    def test_default_root_targets_governed_ds3(self) -> None:
        self.assertEqual(
            DEFAULT_ISPY2_ROOT,
            Path("data/ispy2"),
        )

    def test_builds_only_t0_t1_to_t3_with_exact_covariate_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            summary = build_and_write_manifest(
                root=fixture.root,
                trajectory_manifest_path=fixture.trajectory_manifest,
                ftv_path=fixture.ftv_csv,
                ftv_audit_path=fixture.ftv_audit,
                roi_report_path=fixture.roi_report,
                feature_index_path=fixture.feature_index,
                roi_index_path=fixture.roi_index,
                output_path=fixture.output,
                feature_provenance=fixture.feature_provenance,
                spatial_provenance=SPATIAL_PROVENANCE,
                expected_split_counts=SMALL_SPLITS,
            )
            records = [
                json.loads(line)
                for line in fixture.output.read_text(encoding="utf-8").splitlines()
            ]

            self.assertEqual(summary["records"], 4)
            self.assertEqual(
                summary["schema_version"], MANIFEST_BUILD_REPORT_SCHEMA_VERSION
            )
            self.assertEqual(len(summary["builder_source_sha256"]), 64)
            self.assertEqual(len(summary["roi_release_bundle_sha256"]), 64)
            self.assertEqual(summary["split_counts"], SMALL_SPLITS)
            self.assertEqual(summary["split_counts_full_cohort"], SMALL_SPLITS)
            self.assertEqual(
                summary["split_counts_excluded_qc"],
                {"train": 0, "val": 0, "test": 0},
            )
            self.assertEqual(summary["verified_feature_files"], 12)
            self.assertEqual(
                summary["trajectory_manifest_sha256"],
                _sha256(fixture.trajectory_manifest),
            )
            self.assertEqual(
                summary["roi_index_sha256"], _sha256(fixture.roi_index)
            )
            self.assertEqual(
                summary["feature_index_sha256"], _sha256(fixture.feature_index)
            )
            self.assertEqual(summary["manifest_sha256"], _sha256(fixture.output))
            self.assertTrue(summary["ftv_release"]["validated"])
            self.assertTrue(summary["roi_release"]["validated"])
            self.assertEqual(summary["roi_release"]["patients_released"], 4)
            self.assertIsNone(
                summary["roi_release"]["qc_exclusion_roster_sha256"]
            )
            self.assertEqual(
                summary["ftv_release"]["ftv_csv_sha256"], _sha256(fixture.ftv_csv)
            )
            self.assertEqual([record["split"] for record in records], ["train", "train", "val", "test"])
            first = records[0]
            self.assertEqual(first["schema_version"], SCHEMA_VERSION)
            self.assertEqual(first["modality"], "SER")
            self.assertEqual(first["tensor_kind"], "cached_features")
            self.assertEqual(
                first["feature_provenance"], fixture.feature_provenance
            )
            self.assertEqual(first["clinical_features"], list(CLINICAL_FEATURES))
            self.assertEqual(first["action_features"], list(ACTION_FEATURES))

            self.assertEqual(first["clinical"], [1, 0])
            self.assertEqual(
                [visit["visit_id"] for visit in first["context"]], ["T0", "T1"]
            )
            self.assertEqual(first["target"]["visit_id"], "T3")
            self.assertNotIn("T2", json.dumps(first))
            self.assertEqual(first["context"][0]["state_action"], [0, 0])
            self.assertEqual(first["context"][1]["state_action"], [1, 0])
            self.assertEqual(first["next_action"], [1, 1])
            self.assertEqual(first["context"][0]["ftv"], 11.0)
            self.assertEqual(first["context"][1]["ftv"], 7.0)
            self.assertEqual(first["target"]["ftv"], 2.0)
            self.assertEqual(first["target"]["pcr"], 1)
            self.assertIn("source_content_sha256", first["target"])
            self.assertTrue(Path(first["target"]["volume"]).is_absolute())
            spatial = first["spatial_provenance"]
            self.assertEqual(spatial["roi_voxel_shape"], [96, 96, 64])
            self.assertNotIn("feature_grid_shape", spatial)
            self.assertNotIn("crop_shape", spatial)
            self.assertEqual(
                first["feature_provenance"]["feature_grid_shape"], [2, 2, 2]
            )
            self.assertIsNone(records[-1]["target"]["pcr"])

    def test_qc_release_filters_split_and_reports_actual_counts_without_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            excluded_patient = fixture.patients[-1][0]
            fixture.set_qc_exclusions(excluded_patient)

            records, summary = fixture.build()

            self.assertEqual(len(records), 3)
            self.assertEqual(
                [record["split"] for record in records], ["train", "train", "val"]
            )
            self.assertNotIn(
                excluded_patient, {record["patient_id"] for record in records}
            )
            self.assertEqual(
                summary["split_counts"], {"train": 2, "val": 1, "test": 0}
            )
            self.assertEqual(summary["split_counts_full_cohort"], SMALL_SPLITS)
            self.assertEqual(
                summary["split_counts_excluded_qc"],
                {"train": 0, "val": 0, "test": 1},
            )
            self.assertEqual(summary["verified_feature_files"], 9)
            self.assertEqual(summary["roi_release"]["patients_selected"], 4)
            self.assertEqual(summary["roi_release"]["patients_excluded_qc"], 1)
            self.assertEqual(summary["roi_release"]["patients_released"], 3)
            self.assertEqual(
                summary["roi_release"]["roi_report_sha256"],
                _sha256(fixture.roi_report),
            )
            self.assertEqual(
                summary["roi_release"]["qc_exclusion_roster_sha256"],
                _sha256(fixture.qc_roster),
            )
            self.assertNotIn(excluded_patient, json.dumps(summary, sort_keys=True))

    def test_qc_release_requires_exact_private_roster(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.set_qc_exclusions(fixture.patients[-1][0])
            with self.assertRaisesRegex(ManifestBuildError, "roster is required"):
                fixture.build(qc_exclusion_path=None)

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.set_qc_exclusions(fixture.patients[-1][0])
            fields, rows = _read_csv(fixture.qc_roster)
            rows[0]["exclusion_reason"] = "unreviewed_reason"
            _write_csv(fixture.qc_roster, fields, rows)
            with self.assertRaisesRegex(ManifestBuildError, "reviewed contract"):
                fixture.build()

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.set_qc_exclusions(fixture.patients[-1][0])
            fields, rows = _read_csv(fixture.qc_roster)
            rows[0]["PatientID"] = "ISPY2-999999"
            _write_csv(fixture.qc_roster, fields, rows)
            with self.assertRaises(ManifestBuildError) as captured:
                fixture.build()
            self.assertIn("outside the preregistered split", str(captured.exception))
            self.assertNotIn("ISPY2-999999", str(captured.exception))

    def test_roi_release_report_is_aggregate_only_and_evidence_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            report = json.loads(fixture.roi_report.read_text(encoding="utf-8"))
            report["private_patient"] = fixture.patients[0][0]
            fixture.roi_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "aggregate-only"):
                fixture.build()

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            report = json.loads(fixture.roi_report.read_text(encoding="utf-8"))
            report["evidence"]["split_sha256"] = "f" * 64
            fixture.roi_report.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "does not match"):
                fixture.build()

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.trajectory_manifest.write_bytes(
                fixture.trajectory_manifest.read_bytes() + b"\n"
            )
            with self.assertRaises(ManifestBuildError) as captured:
                fixture.build()
            message = str(captured.exception)
            self.assertIn("does not match", message)
            self.assertNotIn(str(fixture.trajectory_manifest), message)
            self.assertNotIn(fixture.patients[0][0], message)

    def test_feature_provenance_is_bound_to_the_selected_roi_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.roi_index.write_bytes(fixture.roi_index.read_bytes() + b"\n")
            with self.assertRaises(ManifestBuildError) as captured:
                fixture.build()
            message = str(captured.exception)
            self.assertIn("ROI-index digest does not match", message)
            self.assertNotIn(str(fixture.roi_index), message)
            self.assertNotIn(fixture.patients[0][0], message)

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            private_missing_path = fixture.root / fixture.patients[0][0] / "roi_index.csv"
            with self.assertRaises(ManifestBuildError) as captured:
                fixture.build(roi_index_path=private_missing_path)
            message = str(captured.exception)
            self.assertEqual(message, "ROI index is unreadable")
            self.assertNotIn(fixture.patients[0][0], message)

    def test_feature_index_must_equal_the_released_cohort_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fields, original_rows = _read_csv(fixture.feature_index)
            excluded_patient = fixture.patients[-1][0]
            excluded_rows = [
                row for row in original_rows if row["PatientID"] == excluded_patient
            ]
            fixture.set_qc_exclusions(excluded_patient)
            _current_fields, released_rows = _read_csv(fixture.feature_index)
            _write_csv(
                fixture.feature_index,
                fields,
                [*released_rows, *excluded_rows],
            )
            with self.assertRaises(ManifestBuildError) as captured:
                fixture.build()
            self.assertIn("extra_features=3", str(captured.exception))
            self.assertNotIn(excluded_patient, str(captured.exception))

    def test_requires_explicit_ftv_csv_and_matching_audit_without_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            legacy = fixture.root / "clinical_codes/ispy2_ftv_computed.csv"
            legacy.parent.mkdir(parents=True)
            legacy.write_bytes(fixture.ftv_csv.read_bytes())
            for overrides in (
                {"ftv_path": None},
                {"ftv_audit_path": None},
                {"ftv_path": legacy, "ftv_audit_path": None},
            ):
                with self.subTest(overrides=overrides), self.assertRaisesRegex(
                    ManifestBuildError, "must both be provided"
                ):
                    fixture.build(**overrides)

    def test_builder_rejects_blocked_audit_and_ftv_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            report = json.loads(fixture.ftv_audit.read_text(encoding="utf-8"))
            report["status"] = "blocked_undocumented_ftv_exclusion_bits"
            fixture.ftv_audit.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "status is not releasable"):
                fixture.build()

        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.ftv_csv.write_text("tampered\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "does not match"):
                fixture.build()

    def test_builder_consumes_validated_ftv_snapshot_without_reopening_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            validated_snapshot = validate_ftv_release(
                fixture.ftv_csv, fixture.ftv_audit
            )
            fixture.ftv_csv.write_text(
                "tampered-after-validation\n", encoding="utf-8"
            )
            with patch(
                "med_jepa.ispy2_manifest_builder.validate_ftv_release",
                return_value=validated_snapshot,
            ):
                records, summary = fixture.build()
            self.assertEqual(len(records), 4)
            self.assertTrue(summary["ftv_release"]["validated"])
            self.assertNotIn("ftv_csv_rows", summary["ftv_release"])
            self.assertNotIn("ftv_csv_fields", summary["ftv_release"])

    def test_feature_index_relative_paths_are_resolved_against_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fields, rows = fixture.rows("feature_cache/feature_index.csv")
            for row in rows:
                row["feature_path"] = Path(row["feature_path"]).name
            fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
            records, _ = fixture.build()
            self.assertTrue(Path(records[0]["context"][0]["volume"]).is_absolute())

    def test_clinical_and_pcr_may_come_from_the_split_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            split_fields, split_rows = fixture.rows("manifests/ispy2_split.csv")
            lookup = {patient: (hr, her2, pcr) for patient, _, hr, her2, pcr in fixture.patients}
            split_fields.extend(["HR", "HER2", "pCR"])
            for row in split_rows:
                hr, her2, pcr = lookup[row["PatientID"]]
                row.update({"HR": str(hr), "HER2": str(her2), "pCR": "" if pcr is None else str(pcr)})
            fixture.rewrite("manifests/ispy2_split.csv", split_fields, split_rows)
            fixture._write_roi_report(())
            for relative in (
                "manifests/ispy2_trajectory_manifest.csv",
                "governed/ftv/ispy2_ftv_ds3_v1.csv",
            ):
                fields, rows = fixture.rows(relative)
                kept_fields = [field for field in fields if field not in {"HR", "HER2", "pCR"}]
                fixture.rewrite(
                    relative,
                    kept_fields,
                    [{field: row[field] for field in kept_fields} for row in rows],
                )
            records, _ = fixture.build()
            self.assertEqual(records[0]["clinical"], [1, 0])
            self.assertEqual(records[0]["target"]["pcr"], 1)
            self.assertIsNone(records[-1]["target"]["pcr"])

    def test_missing_required_rows_fail_without_identifiers(self) -> None:
        cases = (
            ("feature_cache/feature_index.csv", lambda row: not (row["PatientID"].endswith("004") and row["tp_label"] == "T3"), "incomplete"),
            ("manifests/ispy2_trajectory_manifest.csv", lambda row: not (row["PatientID"].endswith("004") and row["tp_label"] == "T1"), "incomplete"),
            ("governed/ftv/ispy2_ftv_ds3_v1.csv", lambda row: not row["PatientID"].endswith("004"), "595 patient rows"),
        )
        for relative, keep, expected in cases:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows(relative)
                fixture.rewrite(relative, fields, [row for row in rows if keep(row)])
                with self.assertRaises(ManifestBuildError) as captured:
                    fixture.build()
                message = str(captured.exception)
                self.assertIn(expected, message)
                self.assertNotIn("ISPY2-SECRET", message)

    def test_duplicate_rows_in_every_join_source_are_rejected(self) -> None:
        relatives = (
            ("manifests/ispy2_split.csv", "duplicate"),
            ("manifests/ispy2_trajectory_manifest.csv", "duplicate"),
            ("governed/ftv/ispy2_ftv_ds3_v1.csv", "595 patient rows"),
            ("feature_cache/feature_index.csv", "duplicate"),
        )
        for relative, expected in relatives:
            with self.subTest(relative=relative), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows(relative)
                fixture.rewrite(relative, fields, [*rows, rows[0]])
                with self.assertRaisesRegex(ManifestBuildError, expected):
                    fixture.build()

    def test_split_counts_are_preregistered_not_inferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            with self.assertRaisesRegex(ManifestBuildError, "preregistered counts"):
                fixture.build(expected_split_counts={"train": 1, "val": 1, "test": 2})

    def test_confirmatory_modality_is_locked_to_ser(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            with self.assertRaisesRegex(ManifestBuildError, "must be SER"):
                fixture.build(modality="DCE")

    def test_invalid_and_mismatched_feature_hashes_fail_closed(self) -> None:
        for replacement, expected in (("bad", "64-character"), ("0" * 64, "hash_mismatch")):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows("feature_cache/feature_index.csv")
                rows[0]["content_sha256"] = replacement
                fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
                with self.assertRaisesRegex(ManifestBuildError, expected):
                    fixture.build()

    def test_cached_tensor_shape_and_embedded_provenance_are_verified(self) -> None:
        for mutation, expected in (("shape", "tensor_shape_mismatch"), ("provenance", "embedded_provenance_mismatch")):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows("feature_cache/feature_index.csv")
                feature_path = Path(rows[0]["feature_path"])
                payload = torch.load(feature_path, map_location="cpu", weights_only=True)
                if mutation == "shape":
                    payload["volume"] = torch.zeros(4, 1, 2, 2)
                else:
                    payload["feature_provenance"] = {
                        **FEATURE_PROVENANCE,
                        "layer": "wrong-layer",
                    }
                torch.save(payload, feature_path)
                rows[0]["content_sha256"] = _sha256(feature_path)
                fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
                with self.assertRaisesRegex(ManifestBuildError, expected):
                    fixture.build()

    def test_scan_path_content_and_source_identifiers_must_be_globally_unique(self) -> None:
        for field in ("scan_id", "feature_path", "content_sha256", "source_content_sha256"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows("feature_cache/feature_index.csv")
                rows[1][field] = rows[0][field]
                if field == "feature_path":
                    rows[1]["content_sha256"] = rows[0]["content_sha256"]
                elif field == "content_sha256":
                    Path(rows[1]["feature_path"]).write_bytes(
                        Path(rows[0]["feature_path"]).read_bytes()
                    )
                fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
                with self.assertRaisesRegex(ManifestBuildError, f"reuse {field}"):
                    fixture.build()

    def test_feature_paths_must_be_safe_existing_pt_files(self) -> None:
        for replacement, expected in (
            ("../outside.pt", "unsafe"),
            ("missing.pt", "missing"),
            ("feature.npy", "non-.pt"),
        ):
            with self.subTest(replacement=replacement), tempfile.TemporaryDirectory() as directory:
                fixture = _Fixture(Path(directory))
                fields, rows = fixture.rows("feature_cache/feature_index.csv")
                rows[0]["feature_path"] = replacement
                fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
                with self.assertRaisesRegex(ManifestBuildError, expected):
                    fixture.build()

    def test_absolute_feature_path_cannot_escape_approved_cache_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fields, rows = fixture.rows("feature_cache/feature_index.csv")
            outside = fixture.root / "outside.pt"
            outside.write_bytes(Path(rows[0]["feature_path"]).read_bytes())
            rows[0]["feature_path"] = str(outside)
            rows[0]["content_sha256"] = _sha256(outside)
            fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
            with self.assertRaisesRegex(ManifestBuildError, "unsafe"):
                fixture.build()

    def test_feature_and_spatial_provenance_are_separate(self) -> None:
        with self.assertRaisesRegex(ManifestBuildError, "must not contain spatial"):
            validate_feature_provenance(
                {**FEATURE_PROVENANCE, "roi_voxel_shape": [96, 96, 64]}
            )
        with self.assertRaisesRegex(ManifestBuildError, "must not contain"):
            validate_spatial_provenance(
                {**SPATIAL_PROVENANCE, "crop_shape": [96, 96, 64]}
            )
        with self.assertRaisesRegex(ManifestBuildError, "must not contain"):
            validate_spatial_provenance(
                {**SPATIAL_PROVENANCE, "feature_grid_shape": [2, 2, 2]}
            )
        with self.assertRaisesRegex(ManifestBuildError, "deformable_registration"):
            validate_spatial_provenance(
                {**SPATIAL_PROVENANCE, "deformable_registration": True}
            )
        missing_grid = dict(FEATURE_PROVENANCE)
        missing_grid.pop("feature_grid_shape")
        with self.assertRaisesRegex(ManifestBuildError, "lacks required"):
            validate_feature_provenance(missing_grid)
        with self.assertRaisesRegex(ManifestBuildError, "40- or 64-character"):
            validate_feature_provenance(
                {**FEATURE_PROVENANCE, "source_revision": "short-tag"}
            )

    def test_refuses_overwrite_and_preserves_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fixture.output.write_text("sentinel\n", encoding="utf-8")
            with self.assertRaisesRegex(ManifestBuildError, "overwrite is forbidden"):
                build_and_write_manifest(
                    root=fixture.root,
                    trajectory_manifest_path=fixture.trajectory_manifest,
                    ftv_path=fixture.ftv_csv,
                    ftv_audit_path=fixture.ftv_audit,
                    roi_report_path=fixture.roi_report,
                    feature_index_path=fixture.feature_index,
                    roi_index_path=fixture.roi_index,
                    output_path=fixture.output,
                    feature_provenance=fixture.feature_provenance,
                    spatial_provenance=SPATIAL_PROVENANCE,
                    expected_split_counts=SMALL_SPLITS,
                )
            self.assertEqual(fixture.output.read_text(encoding="utf-8"), "sentinel\n")

    def test_cli_accepts_json_argument_or_file_and_logs_aggregates_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            feature_file = fixture.root / "feature_provenance.json"
            feature_file.write_text(
                json.dumps(fixture.feature_provenance), encoding="utf-8"
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return_code = main(
                    [
                        "--root",
                        str(fixture.root),
                        "--trajectory-manifest",
                        str(fixture.trajectory_manifest),
                        "--ftv-csv",
                        str(fixture.ftv_csv),
                        "--ftv-audit-report",
                        str(fixture.ftv_audit),
                        "--roi-release-report",
                        str(fixture.roi_report),
                        "--feature-index",
                        str(fixture.feature_index),
                        "--roi-index",
                        str(fixture.roi_index),
                        "--output",
                        str(fixture.output),
                        "--feature-provenance-file",
                        str(feature_file),
                        "--spatial-provenance-json",
                        json.dumps(SPATIAL_PROVENANCE),
                        "--expected-split-counts-json",
                        json.dumps(SMALL_SPLITS),
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertEqual(stderr.getvalue(), "")
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["records"], 4)
            self.assertNotIn("ISPY2-SECRET", stdout.getvalue())

    def test_cli_error_log_does_not_emit_patient_or_scan_identifiers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = _Fixture(Path(directory))
            fields, rows = fixture.rows("feature_cache/feature_index.csv")
            rows[0]["content_sha256"] = "invalid"
            fixture.rewrite("feature_cache/feature_index.csv", fields, rows)
            stdout = io.StringIO()
            stderr = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                return_code = main(
                    [
                        "--root",
                        str(fixture.root),
                        "--trajectory-manifest",
                        str(fixture.trajectory_manifest),
                        "--ftv-csv",
                        str(fixture.ftv_csv),
                        "--ftv-audit-report",
                        str(fixture.ftv_audit),
                        "--roi-release-report",
                        str(fixture.roi_report),
                        "--feature-index",
                        str(fixture.feature_index),
                        "--roi-index",
                        str(fixture.roi_index),
                        "--output",
                        str(fixture.output),
                        "--feature-provenance-json",
                        json.dumps(fixture.feature_provenance),
                        "--spatial-provenance-json",
                        json.dumps(SPATIAL_PROVENANCE),
                        "--expected-split-counts-json",
                        json.dumps(SMALL_SPLITS),
                    ]
                )
            self.assertEqual(return_code, 2)
            self.assertEqual(stdout.getvalue(), "")
            self.assertNotIn("ISPY2-SECRET", stderr.getvalue())
            self.assertNotIn("scan-", stderr.getvalue())
            self.assertFalse(fixture.output.exists())


if __name__ == "__main__":
    unittest.main()
