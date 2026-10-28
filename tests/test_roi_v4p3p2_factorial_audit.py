from __future__ import annotations

import unittest

import torch

from med_jepa.roi_v4.audit_mechanistic import REGIONS
from med_jepa.roi_v4p3p2.audit_factorial import (
    CC,
    CS,
    DERANGEMENTS,
    POP,
    SC,
    SS,
    _arm_inputs,
    _derangements,
    _factorial_contrasts,
    _population_prediction,
)


class V432FactorialAuditTests(unittest.TestCase):
    def test_derangements_are_unique_and_exclude_self_assignment(self) -> None:
        permutations, digests = _derangements(
            88,
            seed=20260817,
        )

        self.assertEqual(permutations.shape, (DERANGEMENTS, 88))
        self.assertEqual(len(digests), DERANGEMENTS)
        self.assertEqual(len(set(digests)), DERANGEMENTS)
        expected = torch.arange(88)
        for permutation in permutations:
            self.assertFalse(bool(torch.any(permutation == expected)))
            self.assertTrue(torch.equal(permutation.sort().values, expected))

    def test_arm_inputs_change_only_the_named_factor(self) -> None:
        recipient_source = torch.full((2, 3, 1, 1, 1), 1.0)
        recipient_history = torch.full_like(recipient_source, 2.0)
        recipient_core = torch.full((2, 1, 1, 1, 1), 3.0)
        recipient_ring = torch.full_like(recipient_core, 4.0)
        donor_source = torch.full_like(recipient_source, 5.0)
        donor_history = torch.full_like(recipient_source, 6.0)
        donor_core = torch.full_like(recipient_core, 7.0)
        donor_ring = torch.full_like(recipient_core, 8.0)

        arms = _arm_inputs(
            recipient_source,
            recipient_history,
            recipient_core,
            recipient_ring,
            donor_source,
            donor_history,
            donor_core,
            donor_ring,
        )

        self.assertTrue(torch.equal(arms[CS][0], recipient_source))
        self.assertTrue(torch.equal(arms[CS][1], recipient_history))
        self.assertTrue(torch.equal(arms[CS][2], donor_core))
        self.assertTrue(torch.equal(arms[CS][3], donor_ring))
        self.assertTrue(torch.equal(arms[SC][0], donor_source))
        self.assertTrue(torch.equal(arms[SC][1], donor_history))
        self.assertTrue(torch.equal(arms[SC][2], recipient_core))
        self.assertTrue(torch.equal(arms[SC][3], recipient_ring))
        self.assertTrue(torch.equal(arms[SS][0], donor_source))
        self.assertTrue(torch.equal(arms[SS][2], donor_core))

    def test_population_template_uses_recipient_core_as_exact_gate(self) -> None:
        anchor = torch.ones(2, 3, 7, 7, 7)
        template = torch.full((3, 7, 7, 7), 2.0)
        core = torch.zeros(2, 1, 7, 7, 7)
        core[0, :, 1:3] = 0.5
        core[1, :, 4:6] = 1.0

        prediction = _population_prediction(anchor, template, core)
        residual = prediction - anchor

        self.assertTrue(
            torch.equal(
                residual[core.expand_as(residual) == 0],
                torch.zeros_like(residual[core.expand_as(residual) == 0]),
            )
        )
        self.assertTrue(
            torch.allclose(residual[0, :, 1:3], torch.ones_like(residual[0, :, 1:3]))
        )
        self.assertTrue(
            torch.allclose(
                residual[1, :, 4:6], torch.full_like(residual[1, :, 4:6], 2.0)
            )
        )

    def test_factorial_contrasts_have_declared_sign_and_interaction(self) -> None:
        patients = 8
        shape = (patients, len(REGIONS))
        arm_errors = {
            CC: torch.zeros(shape),
            CS: torch.full(shape, 2.0),
            SC: torch.full(shape, 3.0),
            SS: torch.full(shape, 4.0),
            POP: torch.full(shape, 5.0),
        }
        patient_ids = [f"patient-{index}" for index in range(patients)]

        report = _factorial_contrasts(
            arm_errors,
            patient_ids,
            seed=20260817,
            replicates=200,
        )

        for region in REGIONS:
            self.assertAlmostEqual(
                report["support_penalty_at_correct_content_CS_minus_CC"][region][
                    "mean"
                ],
                2.0,
            )
            self.assertAlmostEqual(
                report["content_penalty_at_correct_support_SC_minus_CC"][region][
                    "mean"
                ],
                3.0,
            )
            self.assertAlmostEqual(
                report["joint_penalty_SS_minus_CC"][region]["mean"],
                4.0,
            )
            self.assertAlmostEqual(
                report["population_penalty_POP_minus_CC"][region]["mean"],
                5.0,
            )
            self.assertAlmostEqual(
                report["support_main_effect"][region]["mean"],
                1.5,
            )
            self.assertAlmostEqual(
                report["content_main_effect"][region]["mean"],
                2.5,
            )
            self.assertAlmostEqual(
                report["content_by_support_interaction"][region]["mean"],
                1.0,
            )


if __name__ == "__main__":
    unittest.main()
