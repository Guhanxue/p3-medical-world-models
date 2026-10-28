from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import torch

from med_jepa.roi_v4p3.train import SCHEMA as V43_TRAINING_SCHEMA
from med_jepa.roi_v4p3p1.anchor import anchor_bank
from med_jepa.roi_v4p3p1.config import V431TrainingConfig
from med_jepa.roi_v4p3p1.freeze_design import SCHEMA as DESIGN_SCHEMA, freeze
from med_jepa.roi_v4p3p1.gate import evaluate_v431_train_oof_gate
from med_jepa.roi_v4p3p1.losses import masked_fractional_core_objective
from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA
from med_jepa.roi_v4p3p1.train import _derangement, _inner_split


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/ispy2_roi_v4p3p1_seed2026.dev.json"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class V431ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(11)
        self.model = CoarseRRRAnchoredCoreResidualJEPA(8, 1)
        self.source = torch.randn(2, 1024, 7, 7, 7)
        self.anchor = torch.randn_like(self.source)
        self.core = torch.zeros(2, 1, 7, 7, 7)
        self.ring = torch.zeros_like(self.core)
        self.core[:, :, 3, 3, 3] = 0.6
        self.ring[:, :, 2:5, 2:5, 2:5] = 0.2
        self.ring[:, :, 3, 3, 3] = 0.1
        self.source_time = torch.tensor([30.0, 60.0])
        self.target_time = torch.tensor([90.0, 120.0])
        self.action = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def _forward(self):
        return self.model.step(
            self.source,
            self.anchor,
            self.source_time,
            self.target_time,
            self.action,
            self.core,
            self.ring,
            history_delta=torch.randn_like(self.source),
        )

    def test_zero_initialized_model_is_exact_coarse_rrr_anchor(self) -> None:
        output = self._forward()
        self.assertTrue(torch.equal(output.predicted, self.anchor))
        self.assertEqual(
            float(output.gated_core_residual.detach().abs().max()), 0.0
        )

    def test_residual_is_exactly_zero_where_fractional_core_is_zero(self) -> None:
        with torch.no_grad():
            self.model.head.bias.fill_(1.0)
        output = self._forward()
        outside = output.core_gate == 0
        self.assertEqual(
            float(
                output.gated_core_residual.detach()[
                    outside.expand_as(output.gated_core_residual)
                ].abs().max()
            ),
            0.0,
        )
        self.assertGreater(
            float(
                output.gated_core_residual.detach()[
                    ~outside.expand_as(output.gated_core_residual)
                ].abs().mean()
            ),
            0.0,
        )

    def test_ring_never_opens_a_zero_core_gate(self) -> None:
        with torch.no_grad():
            self.model.head.bias.fill_(1.0)
        self.core.zero_()
        self.ring.fill_(0.5)
        output = self._forward()
        self.assertEqual(
            float(output.gated_core_residual.detach().abs().max()),
            0.0,
        )

    def test_fractional_core_objective_is_finite(self) -> None:
        with torch.no_grad():
            self.model.head.bias.fill_(0.1)
        output = self._forward()
        result = masked_fractional_core_objective(
            output,
            torch.randn_like(self.source),
            torch.ones(2, 1, 7, 7, 7, dtype=torch.bool),
            self.core,
        )
        self.assertEqual(set(result), {"total", "masked_core", "eligible_patients"})
        self.assertTrue(torch.isfinite(result["total"]))

    def test_empty_mask_core_intersection_returns_differentiable_zero(self) -> None:
        output = self._forward()
        result = masked_fractional_core_objective(
            output,
            torch.randn_like(self.source),
            torch.zeros(2, 1, 7, 7, 7, dtype=torch.bool),
            self.core,
        )
        self.assertEqual(float(result["total"].detach()), 0.0)
        result["total"].backward()


class V431ProtocolTests(unittest.TestCase):
    def test_config_is_one_candidate_train409_only(self) -> None:
        config = V431TrainingConfig.load(CONFIG)
        self.assertEqual(config.seed, 2026)
        self.assertEqual(config.grouped_cv_folds, 5)
        self.assertEqual(config.bootstrap_replicates, 10_000)
        self.assertFalse(hasattr(config, "ring_gate_beta_candidates"))

    def test_gate_requires_core_gain_and_region_safety(self) -> None:
        ids = [f"patient-{index:03d}" for index in range(409)]
        baseline = torch.ones(409, 5)
        candidate = baseline.clone()
        candidate[:, 0] = 0.80
        candidate[:, 1] = 0.95
        candidate[:, 2] = 0.90
        candidate[:, 3] = 0.98
        candidate[:, 4] = 0.95
        shuffled = candidate.clone()
        shuffled[:, 0] = 0.90
        result = evaluate_v431_train_oof_gate(
            candidate,
            baseline,
            shuffled,
            ids,
            maximum_outside_core_support_absolute=0.0,
            seed=123,
        )
        self.assertTrue(result["development_gate_passed"])
        self.assertTrue(all(result["criterion_results"].values()))
        candidate[:, 3] = 1.01
        failed = evaluate_v431_train_oof_gate(
            candidate,
            baseline,
            shuffled,
            ids,
            maximum_outside_core_support_absolute=0.0,
            seed=123,
        )
        self.assertFalse(failed["development_gate_passed"])
        self.assertFalse(
            failed["criterion_results"][
                "background_mean_not_worse_than_coarse_RRR"
            ]
        )

    def test_anchor_bank_is_index_aligned_and_fail_closed_elsewhere(self) -> None:
        indices = torch.tensor([3, 1])
        values = torch.randn(2, 1024, 7, 7, 7)
        bank = anchor_bank(5, indices, values)
        self.assertTrue(torch.equal(bank[3], values[0]))
        self.assertTrue(torch.equal(bank[1], values[1]))
        self.assertTrue(torch.isnan(bank[0]).all())

    def test_nested_splits_and_shuffle_are_patient_disjoint(self) -> None:
        outer = torch.arange(327)
        fit, calibration = _inner_split(outer, seed=2026)
        self.assertFalse(set(fit.tolist()) & set(calibration.tolist()))
        shuffled = _derangement(torch.arange(82), seed=2026)
        self.assertFalse(bool(torch.any(shuffled == torch.arange(82))))

    def test_freeze_binds_the_v43_negative_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent_path = root / "v43.json"
            names = [
                f"beta-{beta:.2f}_outside-{outside:.2f}"
                for beta in (0.25, 0.50, 0.75)
                for outside in (0.01, 0.05)
            ]
            gates = {}
            summaries = {}
            for index, name in enumerate(names):
                gates[name] = {
                    "candidate_vs_strongest_information_matched_baseline": {
                        "t0_core": {"ci_low": 0.01 if index >= 2 else -0.01}
                    }
                }
                summaries[name] = {
                    "fold_evidence": [
                        {
                            "strongest_baseline": {
                                "selected_family": (
                                    "occupancy_conditioned_coarse_RRR"
                                )
                            }
                        }
                        for _ in range(5)
                    ]
                }
            parent_path.write_text(
                json.dumps(
                    {
                        "schema_version": V43_TRAINING_SCHEMA,
                        "status": "completed_train409_oof_gate_fail_no_final_model",
                        "patient_counts": {
                            "train_loaded": 409,
                            "validation_loaded": 0,
                            "test_loaded": 0,
                        },
                        "selected_feature_variant": "local64",
                        "selected_candidate": None,
                        "final_checkpoint_created": False,
                        "validation_88_loaded": False,
                        "test_split_loaded": False,
                        "train409_cross_validation": {
                            "passing_candidates": [],
                            "outer_held_fold_used_for_early_stopping": False,
                            "baseline_selected_inside_outer_training_fold": True,
                            "candidate_oof_gates": gates,
                            "candidate_summaries": summaries,
                        },
                        "evidence": {
                            "manifest_sha256": "1" * 64,
                            "feature_index_sha256": "2" * 64,
                        },
                    }
                ),
                encoding="utf-8",
            )
            output = root / "design.json"
            report = freeze(
                argparse.Namespace(
                    config=str(CONFIG),
                    v4p3_report=str(parent_path),
                    output=str(output),
                )
            )
            self.assertEqual(report["schema_version"], DESIGN_SCHEMA)
            self.assertTrue(
                report["adaptive_development_disclosure"][
                    "derived_after_reviewing_v4p3_train409_OOF"
                ]
            )
            self.assertFalse(report["training_started"])
            self.assertFalse(report["validation_88_loaded"])
            self.assertFalse(report["test_split_loaded"])
            self.assertTrue(output.is_file())

    def test_source_contract_has_no_pixel_decoder_or_global_neural_branch(self) -> None:
        source = "\n".join(
            (ROOT / "src/med_jepa/roi_v4p3p1" / name).read_text(
                encoding="utf-8"
            )
            for name in (
                "anchor.py",
                "freeze_design.py",
                "gate.py",
                "losses.py",
                "model.py",
                "train.py",
            )
        ).lower()
        self.assertNotIn("pixel_decoder", source)
        self.assertNotIn("output_norm", source)
        self.assertNotIn("global_head", source)


if __name__ == "__main__":
    unittest.main()
