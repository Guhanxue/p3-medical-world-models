from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch

from med_jepa.roi_v4p3p2.audit_matched_prior import (
    _maximum_outside_core,
    _recipient_matched_population_prediction,
)


class _FakeMatchedPriorModel:
    def eval(self) -> "_FakeMatchedPriorModel":
        return self

    def step(
        self,
        source: torch.Tensor,
        coarse_rrr_anchor: torch.Tensor,
        source_time: torch.Tensor,
        target_time: torch.Tensor,
        action: torch.Tensor,
        core_occupancy: torch.Tensor,
        ring_occupancy: torch.Tensor,
        *,
        history_delta: torch.Tensor,
        context_mask: torch.Tensor,
    ) -> SimpleNamespace:
        del coarse_rrr_anchor, source_time, target_time, ring_occupancy, context_mask
        raw = (
            source
            + history_delta
            + 10.0 * core_occupancy
            + action[:, 0, None, None, None, None]
        )
        return SimpleNamespace(raw_core_residual=raw)


class V432MatchedPriorAuditTests(unittest.TestCase):
    def test_marginalizes_donor_content_under_recipient_context(self) -> None:
        donors = 3
        grid = (7, 7, 7)
        donor_source = torch.stack(
            [torch.full((1, *grid), value) for value in (1.0, 3.0, 5.0)]
        )
        donor_history = torch.stack(
            [torch.full((1, *grid), value) for value in (2.0, 4.0, 6.0)]
        )
        recipient_times = torch.tensor([[0.0, 10.0, 20.0], [0.0, 12.0, 24.0]])
        recipient_action = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        recipient_core = torch.full((2, 1, *grid), 0.5)
        recipient_core[1] = 0.0
        recipient_core[1, :, 2:5, 2:5, 2:5] = 1.0
        recipient_ring = torch.zeros_like(recipient_core)
        anchors = torch.full((2, 1, *grid), 100.0)
        config = SimpleNamespace(
            mask_fractions=(0.5,),
            evaluation_mask_draws_per_fraction=1,
            batch_size=2,
        )

        prediction, mean_raw, outside = _recipient_matched_population_prediction(
            _FakeMatchedPriorModel(),
            donor_source,
            donor_history,
            recipient_times,
            recipient_action,
            recipient_core,
            recipient_ring,
            anchors,
            config,
            seed=20260818,
            device=torch.device("cpu"),
            donor_chunk_size=2,
        )

        # Mean donor source + history is (1+2 + 3+4 + 5+6) / 3 = 7.
        self.assertTrue(torch.allclose(mean_raw[0], torch.full_like(mean_raw[0], 13.0)))
        self.assertTrue(
            torch.allclose(prediction[0], torch.full_like(prediction[0], 106.5))
        )
        inside = recipient_core[1].bool().expand_as(prediction[1])
        self.assertTrue(
            torch.allclose(
                mean_raw[1][inside], torch.full_like(mean_raw[1][inside], 19.0)
            )
        )
        self.assertTrue(
            torch.allclose(
                prediction[1][inside], torch.full_like(prediction[1][inside], 119.0)
            )
        )
        self.assertTrue(torch.equal(prediction[1][~inside], anchors[1][~inside]))
        self.assertEqual(outside, 0.0)
        self.assertEqual(donors, donor_source.shape[0])

    def test_outside_core_check_rejects_leaked_residual(self) -> None:
        anchor = torch.zeros(1, 2, 7, 7, 7)
        prediction = anchor.clone()
        prediction[:, :, 0, 0, 0] = 0.25
        core = torch.zeros(1, 1, 7, 7, 7)

        self.assertEqual(_maximum_outside_core(prediction, anchor, core), 0.25)

    def test_invalid_donor_chunk_is_rejected(self) -> None:
        grid = (7, 7, 7)
        donor_source = torch.ones(2, 1, *grid)
        config = SimpleNamespace(
            mask_fractions=(0.5,),
            evaluation_mask_draws_per_fraction=1,
            batch_size=1,
        )
        with self.assertRaises(ValueError):
            _recipient_matched_population_prediction(
                _FakeMatchedPriorModel(),
                donor_source,
                donor_source,
                torch.tensor([[0.0, 1.0, 2.0]]),
                torch.zeros(1, 2),
                torch.ones(1, 1, *grid),
                torch.zeros(1, 1, *grid),
                torch.zeros(1, 1, *grid),
                config,
                seed=1,
                device=torch.device("cpu"),
                donor_chunk_size=0,
            )


if __name__ == "__main__":
    unittest.main()
