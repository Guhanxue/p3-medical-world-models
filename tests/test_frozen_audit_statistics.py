from __future__ import annotations

import unittest

import torch

from med_jepa.frozen_audit.statistics import (
    TorchReducedRankRegressor,
    TorchRidgeRegressor,
    TorchSlowFeatureAnalysis,
    TrainPCA,
    TrainStandardizer,
    classification_metrics,
    grouped_ridge_cv,
    paired_patient_bootstrap,
    patient_group_kfold,
    patient_level_permutation_null,
    regression_metrics,
)


class MetricTests(unittest.TestCase):
    def test_regression_metrics_support_multioutput_and_validate_r2(self) -> None:
        target = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        prediction = target + torch.tensor([[1.0, -1.0], [0.0, 0.0], [-1.0, 1.0]])

        metrics = regression_metrics(prediction, target)

        self.assertAlmostEqual(metrics.mae, 4.0 / 6.0)
        self.assertAlmostEqual(metrics.rmse, (4.0 / 6.0) ** 0.5)
        self.assertAlmostEqual(metrics.r2, 0.75)
        self.assertEqual(metrics.n_observations, 6)
        with self.assertRaisesRegex(ValueError, "constant target"):
            regression_metrics(torch.ones(4), torch.ones(4))
        tiny = torch.tensor([0.0, 1e-12, 2e-12, 3e-12], dtype=torch.float64)
        tiny_metrics = regression_metrics(tiny.clone(), tiny)
        self.assertAlmostEqual(tiny_metrics.r2, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            regression_metrics(torch.tensor([0.0, float("nan")]), torch.zeros(2))

    def test_binary_metrics_are_exact_and_tie_invariant(self) -> None:
        perfect = classification_metrics(
            torch.tensor([0.9, 0.8, 0.2, 0.1]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
        )
        self.assertEqual(perfect.auroc, 1.0)
        self.assertEqual(perfect.auprc, 1.0)
        self.assertAlmostEqual(perfect.brier, 0.025)
        self.assertEqual((perfect.n_positive, perfect.n_negative), (2, 2))

        tied_first = classification_metrics(
            torch.full((4,), 0.5), torch.tensor([1.0, 0.0, 1.0, 0.0])
        )
        tied_second = classification_metrics(
            torch.full((4,), 0.5), torch.tensor([0.0, 1.0, 0.0, 1.0])
        )
        self.assertEqual(tied_first.auroc, 0.5)
        self.assertEqual(tied_first.auprc, 0.5)
        self.assertEqual(tied_first, tied_second)
        with self.assertRaisesRegex(ValueError, "both label classes"):
            classification_metrics(torch.full((3,), 0.2), torch.zeros(3))


class PatientResamplingTests(unittest.TestCase):
    def test_paired_bootstrap_is_patient_weighted_and_deterministic(self) -> None:
        # The first patient has three rows but receives the same total weight as
        # either one-row patient.
        candidate = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0])
        reference = torch.tensor([1.0, 1.0, 1.0, 2.0, 3.0])
        patient_ids = ["many", "many", "many", "one", "two"]

        first = paired_patient_bootstrap(
            candidate,
            reference,
            patient_ids,
            higher_is_better=False,
            n_bootstrap=512,
            seed=77,
        )
        second = paired_patient_bootstrap(
            candidate,
            reference,
            patient_ids,
            higher_is_better=False,
            n_bootstrap=512,
            seed=77,
        )

        self.assertEqual(first, second)
        self.assertEqual(first.n_patients, 3)
        self.assertEqual(first.n_observations, 5)
        self.assertAlmostEqual(first.candidate_mean, 1.0)
        self.assertAlmostEqual(first.reference_mean, 2.0)
        self.assertAlmostEqual(first.improvement, 1.0)
        self.assertEqual(first.bootstrap_positive_fraction, 1.0)
        with self.assertRaisesRegex(ValueError, "entirely finite"):
            paired_patient_bootstrap(
                torch.tensor([0.0, float("inf")]),
                torch.ones(2),
                ["a", "b"],
                higher_is_better=False,
            )

    def test_group_folds_never_split_patients(self) -> None:
        patient_ids = ["a", "a", "b", "c", "c", "d", "e", "f"]
        first = patient_group_kfold(patient_ids, n_splits=3, seed=41)
        second = patient_group_kfold(patient_ids, n_splits=3, seed=41)

        self.assertEqual(
            [fold.validation_indices.tolist() for fold in first],
            [fold.validation_indices.tolist() for fold in second],
        )
        held_out_rows: list[int] = []
        for fold in first:
            self.assertTrue(set(fold.train_patients).isdisjoint(fold.validation_patients))
            self.assertTrue(
                set(fold.train_indices.tolist()).isdisjoint(
                    fold.validation_indices.tolist()
                )
            )
            held_out_rows.extend(fold.validation_indices.tolist())
        self.assertEqual(sorted(held_out_rows), list(range(len(patient_ids))))


class TrainTransformTests(unittest.TestCase):
    def test_standardizer_uses_only_explicit_training_state(self) -> None:
        train = torch.tensor([[1.0, 5.0], [3.0, 5.0], [5.0, 5.0]])
        validation = torch.tensor([[101.0, 5.0], [103.0, 5.0]])
        scaler = TrainStandardizer().fit(train)

        transformed_train = scaler.transform(train)
        transformed_validation = scaler.transform(validation)

        torch.testing.assert_close(
            transformed_train.mean(dim=0), torch.zeros(2, dtype=torch.float64)
        )
        self.assertGreater(float(transformed_validation[:, 0].mean()), 50.0)
        torch.testing.assert_close(
            scaler.inverse_transform(transformed_validation), validation.double()
        )

    def test_pca_recovers_known_train_subspace_without_refitting_validation(self) -> None:
        train = torch.tensor(
            [[-2.0, -4.0, 1.0], [-1.0, -2.0, 1.0], [1.0, 2.0, 1.0], [2.0, 4.0, 1.0]]
        )
        validation = torch.tensor([[3.0, 6.0, 1.0], [-3.0, -6.0, 1.0]])
        pca = TrainPCA(1).fit(train)

        reconstructed = pca.inverse_transform(pca.transform(validation))

        torch.testing.assert_close(reconstructed, validation.double(), atol=1e-10, rtol=0.0)
        self.assertAlmostEqual(float(pca.explained_variance_ratio_.sum()), 1.0)
        with self.assertRaisesRegex(ValueError, "rank bound"):
            TrainPCA(3).fit(torch.randn(3, 5))


class LinearModelTests(unittest.TestCase):
    def test_ridge_recovers_single_and_multioutput_linear_maps(self) -> None:
        generator = torch.Generator().manual_seed(5)
        x = torch.randn(80, 5, generator=generator)
        coefficient = torch.tensor(
            [[1.0, 0.0], [0.5, -1.0], [0.0, 2.0], [-0.5, 0.2], [0.3, 0.7]]
        )
        y = x @ coefficient + torch.tensor([0.2, -0.4])

        multi = TorchRidgeRegressor(alpha=1e-10).fit(x[:60], y[:60])
        single = TorchRidgeRegressor(alpha=1e-10).fit(x[:60], y[:60, 0])

        self.assertLess(regression_metrics(multi.predict(x[60:]), y[60:]).rmse, 1e-6)
        self.assertLess(
            regression_metrics(single.predict(x[60:]), y[60:, 0]).rmse, 1e-6
        )

    def test_reduced_rank_regression_enforces_requested_rank(self) -> None:
        generator = torch.Generator().manual_seed(13)
        x = torch.randn(100, 6, generator=generator)
        latent = x @ torch.tensor([1.0, -0.5, 0.0, 0.2, 0.4, -0.1])
        y = torch.stack((latent, 2.0 * latent, -0.5 * latent), dim=1)
        model = TorchReducedRankRegressor(rank=1, alpha=1e-8).fit(x[:75], y[:75])

        assert model.coef_ is not None
        self.assertLessEqual(int(torch.linalg.matrix_rank(model.coef_)), 1)
        self.assertLess(regression_metrics(model.predict(x[75:]), y[75:]).rmse, 1e-6)

    def test_grouped_cv_fits_fold_local_pipeline_and_selects_alpha(self) -> None:
        generator = torch.Generator().manual_seed(23)
        x = torch.randn(60, 4, generator=generator)
        y = x @ torch.tensor([1.0, -0.5, 0.25, 0.1]) + 0.3
        patient_ids = [f"p{row // 2}" for row in range(60)]

        result = grouped_ridge_cv(
            x,
            y,
            patient_ids,
            (0.0, 0.1, 1000.0),
            n_splits=5,
            seed=19,
            pca_components=4,
        )

        self.assertIn(result.best_alpha, (0.0, 0.1))
        self.assertEqual(result.n_patients, 30)
        self.assertEqual(result.n_splits, 5)
        self.assertEqual(len(result.fold_mae_by_alpha[0.0]), 5)


class SlowFeatureTests(unittest.TestCase):
    def test_sfa_selects_the_slowest_unit_variance_direction(self) -> None:
        patient = torch.arange(12, dtype=torch.float64)[:, None]
        visit = torch.arange(4, dtype=torch.float64)[None, :]
        slow = patient + 0.02 * visit
        fast = 0.2 * patient + torch.where(
            (visit.long() % 2) == 0,
            torch.ones_like(visit),
            -torch.ones_like(visit),
        )
        trajectories = torch.stack((slow, fast), dim=-1)
        times = torch.tensor([0.0, 20.0, 50.0, 90.0]).expand(12, -1)

        model = TorchSlowFeatureAnalysis(2).fit(trajectories, times)
        transformed = model.transform(trajectories)
        measured = model.score_slowness(trajectories, times)

        self.assertEqual(tuple(transformed.shape), (12, 4, 2))
        assert model.slowness_ is not None
        self.assertLessEqual(float(model.slowness_[0]), float(model.slowness_[1]))
        torch.testing.assert_close(measured, model.slowness_, atol=1e-8, rtol=1e-8)
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            model.fit(trajectories, torch.zeros(12, 4))


class PermutationTests(unittest.TestCase):
    def test_patient_level_permutation_null_is_deterministic_and_detects_signal(self) -> None:
        patient_values = torch.arange(24, dtype=torch.float64)
        features = patient_values.repeat_interleave(2)[:, None]
        targets = (2.0 * patient_values).repeat_interleave(2)
        patient_ids = [f"p{index}" for index in range(24) for _ in range(2)]

        def correlation(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            left = x[:, 0] - x[:, 0].mean()
            right = y - y.mean()
            return (left * right).sum() / (left.norm() * right.norm())

        first = patient_level_permutation_null(
            features,
            targets,
            patient_ids,
            correlation,
            n_permutations=199,
            seed=31,
        )
        second = patient_level_permutation_null(
            features,
            targets,
            patient_ids,
            correlation,
            n_permutations=199,
            seed=31,
        )

        self.assertEqual(first, second)
        self.assertAlmostEqual(first.observed, 1.0)
        self.assertEqual(first.n_patients, 24)
        self.assertLessEqual(first.p_value, 0.01)

    def test_two_sided_permutation_does_not_assume_zero_centered_null(self) -> None:
        patient_values = torch.arange(24, dtype=torch.float64)
        features = patient_values[:, None]
        targets = 2.0 * patient_values
        patient_ids = [f"p{index}" for index in range(24)]

        def shifted_negative_correlation(
            x: torch.Tensor, y: torch.Tensor
        ) -> torch.Tensor:
            left = x[:, 0] - x[:, 0].mean()
            right = y - y.mean()
            correlation = (left * right).sum() / (left.norm() * right.norm())
            return 10.0 - correlation

        result = patient_level_permutation_null(
            features,
            targets,
            patient_ids,
            shifted_negative_correlation,
            n_permutations=199,
            seed=31,
            alternative="two-sided",
        )

        self.assertAlmostEqual(result.observed, 9.0)
        self.assertGreater(result.null_mean, result.observed)
        self.assertLessEqual(result.p_value, 0.02)


if __name__ == "__main__":
    unittest.main()
