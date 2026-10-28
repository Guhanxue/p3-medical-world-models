from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from med_jepa import ispy2_ser_roi_converter as converter
from med_jepa.cache_3dino import load_roi_payload, sha256_file


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class _IdentityAffine:
    def TransformPoint(self, point):
        return tuple(float(value) for value in point)


class _Reference:
    def __init__(self, *, size=(4, 4, 4), spacing=(1.0, 1.0, 1.0)) -> None:
        self._size = size
        self._spacing = spacing

    def GetDirection(self):
        return (-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0)

    def GetOrigin(self):
        return (1.5, 1.5, -1.5)

    def GetSpacing(self):
        return self._spacing

    def GetSize(self):
        return self._size


class _PhysicalCenterImage:
    def __init__(self) -> None:
        self.seen_index = None

    def GetSize(self):
        return (4, 6, 8)

    def TransformContinuousIndexToPhysicalPoint(self, index):
        self.seen_index = tuple(index)
        return (10.0 + index[0], 20.0 + 2.0 * index[1], 30.0 + 3.0 * index[2])


class ISPY2SERRoiConverterTests(unittest.TestCase):
    def test_locked_spec_is_the_audited_canonical_ras_contract(self) -> None:
        spec = converter.load_locked_roi_spec(
            REPOSITORY_ROOT / "configs/ispy2_ser_roi_112_2p1mm_v1.json"
        )
        self.assertEqual(spec["shape_dhw"], (112, 112, 112))
        self.assertEqual(spec["spacing_dhw_mm"], (2.1, 2.1, 2.1))
        self.assertEqual(spec["spatial_provenance"]["orientation"], "RAS")
        self.assertEqual(spec["spatial_provenance"]["roi_source_visits"], ["T0"])
        self.assertEqual(spec["spatial_provenance"]["registration_type"], "rigid")

    def test_even_canonical_grid_centers_voxel_edges_on_t0_anchor(self) -> None:
        center = (10.0, 20.0, 30.0)
        geometry = converter.canonical_ras_grid_geometry(
            center,
            shape_dhw=(4, 4, 4),
            spacing_dhw_mm=(1.0, 1.0, 1.0),
        )
        self.assertEqual(geometry["size_xyz"], (4, 4, 4))
        self.assertEqual(geometry["origin_lps_mm"], (11.5, 21.5, 28.5))
        # The physical midpoint of the two central voxel centers is exactly the
        # prescribed T0 midpoint; pixel-support edges are +/- 2 mm from it.
        direction = np.asarray(geometry["direction_lps"]).reshape(3, 3)
        origin = np.asarray(geometry["origin_lps_mm"])
        spacing = np.asarray(geometry["spacing_xyz_mm"])
        size = np.asarray(geometry["size_xyz"])
        midpoint = origin + direction @ (((size - 1) / 2.0) * spacing)
        np.testing.assert_allclose(midpoint, center)
        self.assertEqual(tuple(size * spacing), (4.0, 4.0, 4.0))

    def test_registration_qc_uses_simpleitk_continuous_index_api(self) -> None:
        image = _PhysicalCenterImage()
        center = converter._image_physical_center(image)
        self.assertEqual(image.seen_index, (1.5, 2.5, 3.5))
        self.assertEqual(center, (11.5, 25.0, 40.5))

    def test_inverse_mask_anchor_center_uses_only_raw_zero_voxels(self) -> None:
        pixels = np.array(
            [
                [[0, 1], [1, 1]],
                [[1, 1], [1, 0]],
            ],
            dtype=np.uint8,
        )
        center = converter.inverse_mask_bbox_center(
            pixels,
            frame_to_ser=(1, 0),
            ser={
                "order": (0, 1),
                "spacing": (1.0, 2.0),  # DICOM (row, column)
                "slice_spacing_mm": 2.0,
                "orientation": (1.0, 0.0, 0.0, 0.0, 1.0, 0.0),
                "normal": (0.0, 0.0, 1.0),
                "positions": ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)),
            },
        )
        np.testing.assert_allclose(center, (1.0, 0.5, 1.0))

    def test_rescale_is_applied_before_padding_or_range_qc(self) -> None:
        converted = converter._rescale_pixel_array(
            np.array([[0, 100], [200, 300]], dtype=np.uint16),
            rows=2,
            columns=2,
            slope=0.01,
            intercept=0.5,
        )
        np.testing.assert_allclose(
            converted,
            np.array([[0.5, 1.5], [2.5, 3.5]], dtype=np.float32),
        )
        # A crop's outside padding is later injected into an already-rescaled image
        # as physical zero, not converted from an arbitrary stored-pixel zero.
        self.assertNotEqual(float(converted[0, 0]), 0.0)

    def test_future_mask_coverage_is_physical_and_does_not_recenter_grid(self) -> None:
        support = {
            "included": np.array([[[True, False], [False, False]]]),
            "positions": [(0.0, 0.0, 0.0)],
            "spacings": [(1.0, 1.0)],
            "orientations": [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
        }
        coverage = converter.future_mask_coverage(
            support,
            moving_to_fixed=_IdentityAffine(),
            reference=_Reference(),
        )
        self.assertTrue(coverage["inside_fixed_crop"])
        self.assertEqual(coverage["included_voxels"], 1)
        outside = dict(support)
        outside["positions"] = [(100.0, 0.0, 0.0)]
        outside_coverage = converter.future_mask_coverage(
            outside,
            moving_to_fixed=_IdentityAffine(),
            reference=_Reference(),
        )
        self.assertFalse(outside_coverage["inside_fixed_crop"])

    def test_empty_future_mask_has_no_fabricated_extent(self) -> None:
        support = {
            "included": np.zeros((1, 2, 2), dtype=bool),
            "positions": [(0.0, 0.0, 0.0)],
            "spacings": [(1.0, 1.0)],
            "orientations": [(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)],
        }
        coverage = converter.future_mask_coverage(
            support,
            moving_to_fixed=_IdentityAffine(),
            reference=_Reference(),
        )
        self.assertEqual(coverage["included_voxels"], 0)
        self.assertIsNone(coverage["inside_fixed_crop"])
        self.assertIsNone(coverage["minimum_edge_margin_mm"])

    def test_ftv_counts_accepts_zero_but_rejects_negative_counts(self) -> None:
        release = {
            "ftv_csv_rows": [
                {
                    "PatientID": "ISPY2-100001",
                    "included_voxels_T0": "12",
                    "included_voxels_T1": "3",
                    "included_voxels_T3": "0",
                }
            ]
        }
        self.assertEqual(
            converter._ftv_counts(release)["ISPY2-100001"],
            {"T0": 12, "T1": 3, "T3": 0},
        )

        release["ftv_csv_rows"][0]["included_voxels_T3"] = "-1"
        with self.assertRaises(converter.ROIConversionError) as captured:
            converter._ftv_counts(release)
        self.assertEqual(captured.exception.code, "governed_ftv_release_invalid")

    def test_lineage_csv_translation_renames_timepoint_and_preserves_na(self) -> None:
        payload_item = {"timepoint": "T3", "scan_id": "scan-1"}
        lineage_item = {
            "timepoint": "T3",
            "scan_id": "scan-1",
            "source_dicom_content_sha256": "a" * 64,
            "registration_status": "rigid",
            "registration_metric_value": -0.5,
            "registration_sampling_seed": converter.REGISTRATION_SAMPLING_SEED,
            "registration_iterations": 10,
            "registration_center_displacement_mm": 1.0,
            "registration_transform_matrix": "[1,0,0,0,1,0,0,0,1]",
            "registration_transform_offset_mm": "[0,0,0]",
            "registration_stop_condition": "done",
            "future_mask_mapping_status": "geometry_exact_empty_support",
            "future_mask_inside_fixed_crop": None,
            "future_mask_minimum_edge_margin_mm": None,
            "future_mask_included_voxels": 0,
        }
        row = converter._lineage_csv_row(
            patient_id="ISPY2-100001",
            payload_item=payload_item,
            lineage_item=lineage_item,
            tensor_hash="b" * 64,
        )
        self.assertNotIn("timepoint", row)
        self.assertEqual(row["tp_label"], "T3")
        self.assertEqual(row["future_mask_inside_fixed_crop"], "")
        self.assertEqual(row["future_mask_minimum_edge_margin_mm"], "")
        self.assertEqual(set(row), set(converter.CONVERSION_LINEAGE_FIELDS))

    def test_registration_sampling_seed_is_fixed_and_recorded(self) -> None:
        self.assertEqual(converter.REGISTRATION_SAMPLING_SEED, 20260728)
        source = (REPOSITORY_ROOT / "src/med_jepa/ispy2_ser_roi_converter.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "SetMetricSamplingPercentage(0.20, REGISTRATION_SAMPLING_SEED)",
            source,
        )

    def test_deterministic_pilot_selection_is_bounded_and_reproducible(self) -> None:
        patients = {"ISPY2-100001", "ISPY2-100002", "ISPY2-100003"}
        first = converter.deterministic_pilot_selection(patients, maximum=2)
        second = converter.deterministic_pilot_selection(reversed(tuple(patients)), maximum=2)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 2)
        with self.assertRaises(converter.ROIConversionError) as captured:
            converter.deterministic_pilot_selection(patients, maximum=4)
        self.assertEqual(captured.exception.code, "invalid_pilot_patient_count")

    def test_qc_exclusion_roster_is_private_deterministic_and_deduplicated(self) -> None:
        rows = converter._qc_exclusion_rows(
            [
                {
                    "code": "future_mask_outside_fixed_crop",
                    "patient_id": "ISPY2-100002",
                    "timepoint": "T3",
                },
                {
                    "code": "future_mask_outside_fixed_crop",
                    "patient_id": "ISPY2-100001",
                    "timepoint": "T1",
                },
                {
                    "code": "future_mask_outside_fixed_crop",
                    "patient_id": "ISPY2-100001",
                    "timepoint": "T3",
                },
            ]
        )
        self.assertEqual(
            rows,
            [
                {
                    "PatientID": "ISPY2-100001",
                    "exclusion_reason": "future_mask_outside_fixed_crop",
                    "failed_timepoints": "T1,T3",
                },
                {
                    "PatientID": "ISPY2-100002",
                    "exclusion_reason": "future_mask_outside_fixed_crop",
                    "failed_timepoints": "T3",
                },
            ],
        )

    def test_payload_index_contract_separates_roi_and_raw_dicom_hashes(self) -> None:
        spatial = {
            "reference_visit": "T0",
            "roi_source_visits": ["T0"],
            "spacing_mm": [1.0, 1.0, 1.0],
            "orientation": "RAS",
            "transform_id": "test-rigid-v1",
            "roi_voxel_shape": [4, 4, 4],
            "registration_type": "rigid",
            "deformable_registration": False,
        }
        source_digest = "b" * 64
        volume = torch.linspace(0.0, 3.0, 64).reshape(1, 4, 4, 4)
        payload = converter._payload(
            volume,
            spatial_provenance=spatial,
            ser_volume=converter.SERVolume(
                image=None,
                ser_geometry={},
                source_dicom_content_sha256=source_digest,
                rescale_slope=0.001,
                rescale_intercept=0.0,
            ),
            converter_source_sha256="a" * 64,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            filename = converter._opaque_roi_name(source_digest)
            converter._write_torch_exclusive(payload, root / "tensors" / filename)
            roi_path = root / "tensors" / filename
            tensor_digest = sha256_file(roi_path)
            with (root / "roi_index.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("PatientID", "tp_label", "scan_id", "roi_path", "source_content_sha256"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "PatientID": "P1",
                        "tp_label": "T0",
                        "scan_id": converter._scan_id(source_digest),
                        "roi_path": str(Path("tensors") / filename),
                        "source_content_sha256": tensor_digest,
                    }
                )
            record = read_records = converter.read_roi_index(
                root / "roi_index.csv",
                allowed_input_root=root,
                expected_timepoints=("T0",),
                expected_patients=1,
            )[0]
            self.assertEqual(record.source_content_sha256, tensor_digest)
            self.assertNotEqual(record.source_content_sha256, source_digest)
            output, _spatial, intensity = load_roi_payload(roi_path, (4, 4, 4))
            self.assertEqual(tuple(output.shape), (1, 4, 4, 4))
            self.assertEqual(intensity["source_dicom_content_sha256"], source_digest)
            self.assertNotIn("P1", filename)

    def test_anchor_release_rejects_wrong_grid_before_pixel_io(self) -> None:
        spec = converter.load_locked_roi_spec(
            REPOSITORY_ROOT / "configs/ispy2_ser_roi_112_2p1mm_v1.json"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "anchor.json"
            report = {
                "schema_version": converter.EXPECTED_ANCHOR_SCHEMA_VERSION,
                "status": "ok_anchor_geometry",
                "cohort": {"patients_expected": 595},
                "summary": {
                    "canonical_ras_grid_feasibility": {
                        "orientation": "RAS",
                        "roi_shape": [112, 112, 112],
                        "spacing_mm": [2.0, 2.0, 2.0],  # intentionally wrong
                        "pairs_with_support_inside_grid": 595,
                        "pairs_outside_grid": 0,
                        "pairs_with_required_margin": 595,
                        "pairs_below_required_margin": 0,
                        "required_margin_mm": 20.0,
                    }
                },
                "evidence": {
                    "trajectory_manifest_sha256": "1" * 64,
                    "ftv_csv_sha256": "2" * 64,
                    "ftv_audit_report_sha256": "3" * 64,
                },
            }
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaises(converter.ROIConversionError) as captured:
                converter.validate_anchor_release(
                    path,
                    roi_spec=spec,
                    manifest_sha256="1" * 64,
                    ftv_csv_sha256="2" * 64,
                    ftv_audit_report_sha256="3" * 64,
                )
            self.assertEqual(captured.exception.code, "roi_anchor_report_geometry_mismatch")


if __name__ == "__main__":
    unittest.main()
