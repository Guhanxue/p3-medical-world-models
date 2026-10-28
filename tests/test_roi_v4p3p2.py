from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p3p1.freeze_design import SCHEMA as V431_DESIGN_SCHEMA
from med_jepa.roi_v4p3p1.gate import evaluate_v431_train_oof_gate
from med_jepa.roi_v4p3p1.train import (
    PRIVATE_SCHEMA as V431_PRIVATE_SCHEMA,
    SCHEMA as V431_TRAINING_SCHEMA,
)
from med_jepa.roi_v4p3p2.config import V432RecoveryConfig
from med_jepa.roi_v4p3p2.freeze_recovery import STATUS, freeze
from med_jepa.roi_v4p3p2.gate import (
    evaluate_v432_recovery_gate,
    recovery_comparisons,
)


ROOT = Path(__file__).resolve().parents[1]
V431_CONFIG = ROOT / "configs/ispy2_roi_v4p3p1_seed2026.dev.json"
RECOVERY_CONFIG = ROOT / "configs/ispy2_roi_v4p3p2_seed2026.recovery.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _measurements():
    patient_ids = [f"private-patient-{index:03d}" for index in range(409)]
    baseline = torch.full((409, len(REGIONS)), 0.4, dtype=torch.float64)
    candidate = baseline.clone()
    candidate[:, REGIONS.index("t0_core")] -= 0.01
    candidate[:, REGIONS.index("t0_20mm_ring")] -= 0.002
    candidate[:, REGIONS.index("t0_disease_bed")] -= 0.004
    candidate[:, REGIONS.index("full_shared_valid_grid")] -= 0.001
    candidate[:, REGIONS.index("valid_non_disease_bed")] += 1e-8
    shuffled = candidate.clone()
    shuffled[:, REGIONS.index("t0_core")] += 0.005
    return patient_ids, candidate, baseline, shuffled


class V432GateTests(unittest.TestCase):
    def test_validation_comparisons_allow_empty_patient_regions_as_nan(self) -> None:
        patient_ids = [f"validation-{index:03d}" for index in range(88)]
        reference = torch.full((88, len(REGIONS)), 0.4, dtype=torch.float64)
        candidate = reference - 0.01
        shuffled = candidate + 0.005
        core = REGIONS.index("t0_core")
        candidate[0, core] = float("nan")
        reference[0, core] = float("nan")
        shuffled[0, core] = float("nan")

        comparisons, alignment = recovery_comparisons(
            candidate,
            reference,
            shuffled,
            patient_ids,
            seed=20260816,
            replicates=10_000,
        )

        self.assertEqual(comparisons["t0_core"]["n_observations"], 87)
        self.assertEqual(alignment["n_observations"], 87)
        self.assertEqual(
            comparisons["full_shared_valid_grid"]["n_observations"], 88
        )

    def test_validation_comparisons_still_reject_infinity(self) -> None:
        patient_ids = [f"validation-{index:03d}" for index in range(88)]
        reference = torch.full((88, len(REGIONS)), 0.4, dtype=torch.float64)
        candidate = reference.clone()
        shuffled = reference.clone()
        candidate[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "frozen contract"):
            recovery_comparisons(
                candidate,
                reference,
                shuffled,
                patient_ids,
                seed=20260816,
                replicates=10_000,
            )

    def test_fixed_numerical_margin_recovers_only_background_roundoff(self) -> None:
        ids, candidate, baseline, shuffled = _measurements()
        original = evaluate_v431_train_oof_gate(
            candidate,
            baseline,
            shuffled,
            ids,
            maximum_outside_core_support_absolute=0.0,
            seed=20260816,
        )
        self.assertFalse(original["development_gate_passed"])
        self.assertEqual(
            [name for name, passed in original["criterion_results"].items() if not passed],
            ["background_mean_not_worse_than_coarse_RRR"],
        )
        recovered = evaluate_v432_recovery_gate(
            candidate,
            baseline,
            shuffled,
            ids,
            maximum_outside_core_support_absolute=0.0,
            numerical_noninferiority_margin=1e-6,
            seed=20260816,
        )
        self.assertTrue(recovered["recovery_gate_passed"])
        self.assertTrue(all(recovered["criterion_results"].values()))
        self.assertFalse(recovered["OOF_models_refit_by_recovery"])

    def test_margin_is_not_tunable(self) -> None:
        ids, candidate, baseline, shuffled = _measurements()
        with self.assertRaisesRegex(ValueError, "constants changed"):
            evaluate_v432_recovery_gate(
                candidate,
                baseline,
                shuffled,
                ids,
                maximum_outside_core_support_absolute=0.0,
                numerical_noninferiority_margin=1e-5,
                seed=20260816,
            )

    def test_recovery_config_is_fixed(self) -> None:
        config = V432RecoveryConfig.load(RECOVERY_CONFIG)
        self.assertEqual(config.seed, 2026)
        self.assertEqual(config.final_epoch_count, 18)
        self.assertEqual(config.background_numerical_noninferiority_margin, 1e-6)


class V432FreezeTests(unittest.TestCase):
    def test_freeze_binds_failed_report_and_private_oof_without_refit(self) -> None:
        ids, candidate, baseline, shuffled = _measurements()
        original_gate = evaluate_v431_train_oof_gate(
            candidate,
            baseline,
            shuffled,
            ids,
            maximum_outside_core_support_absolute=0.0,
            seed=20260816,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            design = root / "v431_design.json"
            private = root / "v431_private.pt"
            report = root / "v431_report.json"
            output = root / "v432_lock.json"
            design.write_text(
                json.dumps({"schema_version": V431_DESIGN_SCHEMA}),
                encoding="utf-8",
            )
            torch.save(
                {
                    "schema_version": V431_PRIVATE_SCHEMA,
                    "patient_ids": ids,
                    "error_matrices": {
                        "candidate": candidate,
                        "coarse_rrr": baseline,
                        "occupancy_shuffled": shuffled,
                    },
                    "selected_candidate": None,
                },
                private,
            )
            report.write_text(
                json.dumps(
                    {
                        "schema_version": V431_TRAINING_SCHEMA,
                        "status": "completed_train409_core_residual_gate_fail_no_final_model",
                        "selected_candidate": None,
                        "final_checkpoint_created": False,
                        "validation_88_loaded": False,
                        "test_split_loaded": False,
                        "train409_cross_validation": {
                            "oof_gate": original_gate,
                            "candidate_summary": {
                                "best_epoch_counts": [24, 18, 1, 20, 12]
                            },
                        },
                        "evidence": {
                            "frozen_design_sha256": _sha(design),
                            "config_sha256": _sha(V431_CONFIG),
                            "private_train409_oof_measurements_sha256": _sha(private),
                            "manifest_sha256": "a" * 64,
                            "feature_index_sha256": "b" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            value = freeze(
                argparse.Namespace(
                    recovery_config=str(RECOVERY_CONFIG),
                    v4p3p1_config=str(V431_CONFIG),
                    v4p3p1_frozen_design=str(design),
                    v4p3p1_report=str(report),
                    v4p3p1_private_measurements=str(private),
                    output=str(output),
                )
            )
            self.assertEqual(value["status"], STATUS)
            self.assertTrue(value["recovery_gate"]["recovery_gate_passed"])
            self.assertFalse(value["protocol_change"]["OOF_models_refit"])
            self.assertFalse(value["training_started"])
            self.assertFalse(value["validation_88_loaded"])
            rendered = output.read_text(encoding="utf-8")
            self.assertNotIn("private-patient", rendered)


if __name__ == "__main__":
    unittest.main()
