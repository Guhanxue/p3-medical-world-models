from __future__ import annotations

import json
import unittest
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class ISPY2SERRoiSpecificationTests(unittest.TestCase):
    def test_first_roi_spec_preserves_the_native_3dino_grid_and_full_cohort_margin(self) -> None:
        path = REPOSITORY_ROOT / "configs/ispy2_ser_roi_112_2p1mm_v1.json"
        with path.open("r", encoding="utf-8") as handle:
            spec = json.load(handle)

        self.assertEqual(spec["schema_version"], "med-jepa-ispy2-ser-roi-spec-v1")
        self.assertEqual(spec["roi_id"], "ser_roi_112_2p1mm_t0bbox_v1")
        self.assertEqual(spec["tensor_shape"], [112, 112, 112])
        self.assertEqual(spec["spacing_mm"], [2.1, 2.1, 2.1])
        self.assertEqual(spec["orientation"], "RAS")
        self.assertEqual(spec["anchor"]["reference_visit"], "T0")
        self.assertEqual(spec["anchor"]["roi_source_visits"], ["T0"])
        self.assertEqual(
            spec["anchor"]["center_rule"],
            "t0_inverse_mask_bbox_midpoint_patient_space_v1",
        )
        self.assertEqual(spec["longitudinal_registration"]["registration_type"], "rigid")
        self.assertFalse(spec["longitudinal_registration"]["deformable_registration"])
        self.assertEqual(
            spec["longitudinal_registration"]["future_mask_role"],
            "coverage_qc_only_never_crop_centering",
        )
        evidence = spec["grid_selection_evidence"]
        self.assertEqual(evidence["patients"], 595)
        self.assertGreaterEqual(
            evidence["chosen_fov_mm"],
            evidence["required_fov_mm_with_20mm_margin"],
        )
        self.assertAlmostEqual(evidence["chosen_fov_mm"], 112 * 2.1)
        canonical = spec["canonical_ras_validation"]
        self.assertEqual(canonical["status"], "ok_anchor_geometry")
        self.assertEqual(canonical["patients_with_support_inside_grid"], 595)
        self.assertEqual(canonical["patients_with_required_20mm_margin"], 595)
        self.assertEqual(canonical["patients_outside_grid"], 0)
        self.assertEqual(canonical["patients_below_required_margin"], 0)
        self.assertGreaterEqual(canonical["minimum_observed_edge_margin_mm"], 20.0)


if __name__ == "__main__":
    unittest.main()
