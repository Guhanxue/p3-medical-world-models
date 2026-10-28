"""Leakage-resistant statistical primitives for frozen-feature audits.

This module intentionally depends only on PyTorch and the Python standard
library.  It is designed for small longitudinal cohorts in which several rows
may belong to one patient.  Resampling and cross-validation therefore operate
on patient groups, never on rows.

The estimators do not impute or silently discard non-finite values.  Callers
must construct a governed, finite analysis matrix before fitting them.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from dataclasses import dataclass
import math
from typing import TypeVar

import torch
from torch import Tensor


@dataclass(frozen=True)
class RegressionMetrics:
    """Element-weighted regression metrics over an aligned tensor."""

    mae: float
    rmse: float
    r2: float
    n_observations: int


@dataclass(frozen=True)
class ClassificationMetrics:
    """Binary probability metrics.

    ``auprc`` uses the average-precision (step-integral) convention and handles
    equal scores as one threshold group, so its value cannot depend on an
    arbitrary ordering of tied patients.
    """

    auroc: float
    auprc: float
    brier: float
    n_observations: int
    n_positive: int
    n_negative: int


@dataclass(frozen=True)
class PairedPatientBootstrapResult:
    """Paired patient-bootstrap comparison of decomposable row scores.

    Rows are averaged within patient before any point estimate or resampling.
    ``improvement`` is oriented so that positive values always favor the
    candidate.
    """

    candidate_mean: float
    reference_mean: float
    improvement: float
    ci_low: float
    ci_high: float
    bootstrap_positive_fraction: float
    patient_win_fraction: float
    confidence_level: float
    n_patients: int
    n_observations: int
    n_bootstrap: int
    higher_is_better: bool


@dataclass(frozen=True)
class GroupFold:
    """One patient-disjoint cross-validation fold."""

    train_indices: Tensor
    validation_indices: Tensor
    train_patients: tuple[Hashable, ...]
    validation_patients: tuple[Hashable, ...]


@dataclass(frozen=True)
class RidgeCVResult:
    """Patient-grouped ridge selection using validation-fold MAE."""

    best_alpha: float
    mean_mae_by_alpha: dict[float, float]
    fold_mae_by_alpha: dict[float, tuple[float, ...]]
    n_splits: int
    n_patients: int


@dataclass(frozen=True)
class PermutationNullResult:
    """Patient-level target-permutation null with a plus-one p-value."""

    observed: float
    null_mean: float
    null_std: float
    p_value: float
    alternative: str
    n_patients: int
    n_permutations: int
    null_values: tuple[float, ...]


def _positive_integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _seed(value: object) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value >= 2**63
    ):
        raise ValueError("seed must be an integer in [0, 2**63)")
    return value


def _finite_number(value: object, *, name: str, nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    number = float(value)
    if not math.isfinite(number) or (nonnegative and number < 0.0):
        raise ValueError(f"{name} must be a finite{' non-negative' if nonnegative else ''} number")
    return number


def _finite_tensor(
    values: Tensor,
    *,
    name: str,
    dimensions: tuple[int, ...] | None = None,
    minimum_rows: int = 1,
) -> Tensor:
    if not isinstance(values, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dimensions is not None and values.ndim not in dimensions:
        rendered = " or ".join(str(value) for value in dimensions)
        raise ValueError(f"{name} must have {rendered} dimensions")
    if values.ndim < 1 or values.shape[0] < minimum_rows:
        raise ValueError(f"{name} must contain at least {minimum_rows} rows")
    if not values.is_floating_point():
        raise TypeError(f"{name} must be floating point")
    if not bool(torch.isfinite(values).all()):
        raise ValueError(f"{name} must be entirely finite")
    return values.detach().to(device="cpu", dtype=torch.float64)


def _matrix(values: Tensor, *, name: str, minimum_rows: int = 1) -> Tensor:
    result = _finite_tensor(
        values, name=name, dimensions=(2,), minimum_rows=minimum_rows
    )
    if result.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    return result


def _target_matrix(
    values: Tensor, *, name: str = "targets", minimum_rows: int = 1
) -> tuple[Tensor, bool]:
    result = _finite_tensor(
        values, name=name, dimensions=(1, 2), minimum_rows=minimum_rows
    )
    was_vector = result.ndim == 1
    if was_vector:
        result = result[:, None]
    if result.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one outcome")
    return result, was_vector


def _validate_patient_ids(
    patient_ids: Sequence[Hashable], *, rows: int
) -> tuple[tuple[Hashable, ...], dict[Hashable, list[int]]]:
    if isinstance(patient_ids, (str, bytes)) or len(patient_ids) != rows:
        raise ValueError("patient_ids must align one-for-one with rows")
    ordered: list[Hashable] = []
    groups: dict[Hashable, list[int]] = {}
    for row, patient_id in enumerate(patient_ids):
        try:
            hash(patient_id)
        except TypeError as error:
            raise TypeError("patient IDs must be hashable") from error
        if patient_id not in groups:
            ordered.append(patient_id)
            groups[patient_id] = []
        groups[patient_id].append(row)
    if not ordered:
        raise ValueError("at least one patient is required")
    return tuple(ordered), groups


def regression_metrics(predictions: Tensor, targets: Tensor) -> RegressionMetrics:
    """Return MAE, RMSE and variance-weighted R2 for finite aligned tensors.

    Multi-output tensors are flattened for MAE/RMSE.  R2 uses the total target
    sum of squares across outputs, which is equivalent to a variance-weighted
    multi-output score.  A constant target has undefined R2 and is rejected.
    """

    prediction = _finite_tensor(
        predictions, name="predictions", dimensions=(1, 2)
    )
    target = _finite_tensor(targets, name="targets", dimensions=(1, 2))
    if prediction.shape != target.shape:
        raise ValueError("predictions and targets must have identical shape")
    residual = prediction - target
    target_centered = target - target.mean(dim=0, keepdim=True)
    total = target_centered.square().sum()
    if float(total) == 0.0:
        raise ValueError("R2 is undefined for a constant target")
    return RegressionMetrics(
        mae=float(residual.abs().mean()),
        rmse=float(residual.square().mean().sqrt()),
        r2=float(1.0 - residual.square().sum() / total),
        n_observations=target.numel(),
    )


def _binary_labels(values: Tensor) -> Tensor:
    labels = _finite_tensor(values, name="labels", dimensions=(1,))
    if not bool(((labels == 0.0) | (labels == 1.0)).all()):
        raise ValueError("labels must contain only zero and one")
    positive = int(labels.sum().item())
    if positive == 0 or positive == labels.numel():
        raise ValueError("binary metrics require both label classes")
    return labels


def classification_metrics(
    probabilities: Tensor, labels: Tensor
) -> ClassificationMetrics:
    """Return exact tied-score AUROC, average precision and Brier score."""

    scores = _finite_tensor(
        probabilities, name="probabilities", dimensions=(1,)
    )
    target = _binary_labels(labels)
    if scores.shape != target.shape:
        raise ValueError("probabilities and labels must have identical shape")
    if not bool(((scores >= 0.0) & (scores <= 1.0)).all()):
        raise ValueError("probabilities must lie in [0, 1]")

    positive_scores = scores[target == 1.0]
    negative_scores = scores[target == 0.0]
    differences = positive_scores[:, None] - negative_scores[None, :]
    auroc = (
        (differences > 0.0).to(torch.float64)
        + 0.5 * (differences == 0.0).to(torch.float64)
    ).mean()

    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    sorted_labels = target[order]
    cumulative_positive = sorted_labels.cumsum(dim=0)
    row_count = torch.arange(1, target.numel() + 1, dtype=torch.float64)
    # Evaluate precision/recall only after the final member of each tied-score
    # group.  This makes average precision invariant to within-tie ordering.
    group_ends = torch.ones(target.numel(), dtype=torch.bool)
    if target.numel() > 1:
        group_ends[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    true_positive = cumulative_positive[group_ends]
    precision = true_positive / row_count[group_ends]
    recall = true_positive / target.sum()
    previous_recall = torch.cat((recall.new_zeros(1), recall[:-1]))
    auprc = ((recall - previous_recall) * precision).sum()

    return ClassificationMetrics(
        auroc=float(auroc),
        auprc=float(auprc),
        brier=float((scores - target).square().mean()),
        n_observations=target.numel(),
        n_positive=int(target.sum().item()),
        n_negative=int((1.0 - target).sum().item()),
    )


def _patient_scalar_means(
    values: Tensor, patient_ids: Sequence[Hashable], *, name: str
) -> tuple[tuple[Hashable, ...], Tensor]:
    working = _finite_tensor(values, name=name, dimensions=(1,))
    ordered, groups = _validate_patient_ids(patient_ids, rows=working.shape[0])
    means = torch.stack(
        [working[torch.tensor(groups[patient], dtype=torch.long)].mean() for patient in ordered]
    )
    return ordered, means


def paired_patient_bootstrap(
    candidate_values: Tensor,
    reference_values: Tensor,
    patient_ids: Sequence[Hashable],
    *,
    higher_is_better: bool,
    n_bootstrap: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 20_260_728,
) -> PairedPatientBootstrapResult:
    """Compare aligned, decomposable row scores by resampling patients.

    Examples of valid inputs are patient/transition absolute errors (set
    ``higher_is_better=False``) or per-patient similarity scores (set it to
    ``True``).  Non-decomposable statistics such as AUROC should instead be
    recomputed inside every patient bootstrap and are intentionally not accepted
    by this scalar-score helper.
    """

    _positive_integer(n_bootstrap, name="n_bootstrap")
    _seed(seed)
    confidence = _finite_number(confidence_level, name="confidence_level")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one")
    if not isinstance(higher_is_better, bool):
        raise TypeError("higher_is_better must be boolean")
    candidate_ids, candidate = _patient_scalar_means(
        candidate_values, patient_ids, name="candidate_values"
    )
    reference_ids, reference = _patient_scalar_means(
        reference_values, patient_ids, name="reference_values"
    )
    if candidate_values.shape != reference_values.shape:
        raise ValueError("candidate_values and reference_values must have identical shape")
    if candidate_ids != reference_ids:
        raise RuntimeError("paired patient aggregation produced inconsistent groups")
    if candidate.numel() < 2:
        raise ValueError("paired bootstrap requires at least two patients")

    signed = candidate - reference if higher_is_better else reference - candidate
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draws = torch.randint(
        candidate.numel(),
        (n_bootstrap, candidate.numel()),
        generator=generator,
        dtype=torch.long,
    )
    replicate_effects = signed[draws].mean(dim=1)
    tail = (1.0 - confidence) / 2.0
    interval = torch.quantile(
        replicate_effects,
        torch.tensor((tail, 1.0 - tail), dtype=torch.float64),
    )
    return PairedPatientBootstrapResult(
        candidate_mean=float(candidate.mean()),
        reference_mean=float(reference.mean()),
        improvement=float(signed.mean()),
        ci_low=float(interval[0]),
        ci_high=float(interval[1]),
        bootstrap_positive_fraction=float((replicate_effects > 0.0).double().mean()),
        patient_win_fraction=float((signed > 0.0).double().mean()),
        confidence_level=confidence,
        n_patients=candidate.numel(),
        n_observations=candidate_values.numel(),
        n_bootstrap=n_bootstrap,
        higher_is_better=higher_is_better,
    )


class TrainStandardizer:
    """Feature standardizer whose state is fitted from one explicit matrix."""

    def __init__(self, *, epsilon: float = 1e-12) -> None:
        self.epsilon = _finite_number(epsilon, name="epsilon", nonnegative=True)
        if self.epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        self.mean_: Tensor | None = None
        self.scale_: Tensor | None = None

    def fit(self, train_features: Tensor) -> "TrainStandardizer":
        values = _matrix(train_features, name="train_features", minimum_rows=2)
        scale = values.std(dim=0, unbiased=False)
        self.mean_ = values.mean(dim=0)
        self.scale_ = torch.where(scale > self.epsilon, scale, torch.ones_like(scale))
        return self

    def _state(self) -> tuple[Tensor, Tensor]:
        if self.mean_ is None or self.scale_ is None:
            raise RuntimeError("fit must be called before transform")
        return self.mean_, self.scale_

    def transform(self, features: Tensor) -> Tensor:
        mean, scale = self._state()
        values = _matrix(features, name="features")
        if values.shape[1] != mean.numel():
            raise ValueError("features do not match the fitted feature dimension")
        return (values - mean) / scale

    def inverse_transform(self, standardized: Tensor) -> Tensor:
        mean, scale = self._state()
        values = _matrix(standardized, name="standardized")
        if values.shape[1] != mean.numel():
            raise ValueError("standardized data do not match the fitted feature dimension")
        return values * scale + mean

    def fit_transform(self, train_features: Tensor) -> Tensor:
        return self.fit(train_features).transform(train_features)


class TrainPCA:
    """SVD PCA fitted only from an explicit training matrix."""

    def __init__(
        self,
        n_components: int,
        *,
        whiten: bool = False,
        epsilon: float = 1e-12,
    ) -> None:
        self.n_components = _positive_integer(n_components, name="n_components")
        if not isinstance(whiten, bool):
            raise TypeError("whiten must be boolean")
        self.whiten = whiten
        self.epsilon = _finite_number(epsilon, name="epsilon", nonnegative=True)
        if self.epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        self.mean_: Tensor | None = None
        self.components_: Tensor | None = None
        self.explained_variance_: Tensor | None = None
        self.explained_variance_ratio_: Tensor | None = None

    def fit(self, train_features: Tensor) -> "TrainPCA":
        values = _matrix(train_features, name="train_features", minimum_rows=2)
        maximum_rank = min(values.shape[0] - 1, values.shape[1])
        if self.n_components > maximum_rank:
            raise ValueError(
                "n_components exceeds the centered training-matrix rank bound"
            )
        mean = values.mean(dim=0)
        centered = values - mean
        _, singular_values, right_vectors = torch.linalg.svd(
            centered, full_matrices=False
        )
        all_variance = singular_values.square() / float(values.shape[0] - 1)
        total_variance = all_variance.sum()
        if float(total_variance) <= self.epsilon:
            raise ValueError("PCA requires non-constant training features")
        self.mean_ = mean
        self.components_ = right_vectors[: self.n_components]
        self.explained_variance_ = all_variance[: self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        return self

    def _state(self) -> tuple[Tensor, Tensor, Tensor]:
        if (
            self.mean_ is None
            or self.components_ is None
            or self.explained_variance_ is None
        ):
            raise RuntimeError("fit must be called before transform")
        return self.mean_, self.components_, self.explained_variance_

    def transform(self, features: Tensor) -> Tensor:
        mean, components, variance = self._state()
        values = _matrix(features, name="features")
        if values.shape[1] != mean.numel():
            raise ValueError("features do not match the fitted PCA dimension")
        scores = (values - mean) @ components.T
        if self.whiten:
            scores = scores / variance.clamp_min(self.epsilon).sqrt()
        return scores

    def inverse_transform(self, scores: Tensor) -> Tensor:
        mean, components, variance = self._state()
        values = _matrix(scores, name="scores")
        if values.shape[1] != self.n_components:
            raise ValueError("scores do not match n_components")
        if self.whiten:
            values = values * variance.clamp_min(self.epsilon).sqrt()
        return values @ components + mean

    def fit_transform(self, train_features: Tensor) -> Tensor:
        return self.fit(train_features).transform(train_features)


class TorchRidgeRegressor:
    """Deterministic single- or multi-output ridge regression."""

    def __init__(self, alpha: float = 1.0, *, fit_intercept: bool = True) -> None:
        self.alpha = _finite_number(alpha, name="alpha", nonnegative=True)
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be boolean")
        self.fit_intercept = fit_intercept
        self.coef_: Tensor | None = None
        self.intercept_: Tensor | None = None
        self.target_was_vector_: bool | None = None

    def fit(self, features: Tensor, targets: Tensor) -> "TorchRidgeRegressor":
        x = _matrix(features, name="features", minimum_rows=2)
        y, was_vector = _target_matrix(targets, minimum_rows=2)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and targets must contain the same rows")
        x_mean = x.mean(dim=0) if self.fit_intercept else x.new_zeros(x.shape[1])
        y_mean = y.mean(dim=0) if self.fit_intercept else y.new_zeros(y.shape[1])
        centered_x = x - x_mean
        centered_y = y - y_mean
        gram = centered_x.T @ centered_x
        right = centered_x.T @ centered_y
        if self.alpha > 0.0:
            gram = gram + self.alpha * torch.eye(gram.shape[0], dtype=gram.dtype)
            try:
                coefficient = torch.linalg.solve(gram, right)
            except torch.linalg.LinAlgError:
                coefficient = torch.linalg.pinv(gram) @ right
        else:
            coefficient = torch.linalg.pinv(centered_x) @ centered_y
        self.coef_ = coefficient
        self.intercept_ = y_mean - x_mean @ coefficient
        self.target_was_vector_ = was_vector
        return self

    def _state(self) -> tuple[Tensor, Tensor, bool]:
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.target_was_vector_ is None
        ):
            raise RuntimeError("fit must be called before predict")
        return self.coef_, self.intercept_, self.target_was_vector_

    def predict(self, features: Tensor) -> Tensor:
        coefficient, intercept, was_vector = self._state()
        x = _matrix(features, name="features")
        if x.shape[1] != coefficient.shape[0]:
            raise ValueError("features do not match the fitted feature dimension")
        prediction = x @ coefficient + intercept
        return prediction[:, 0] if was_vector else prediction


class TorchReducedRankRegressor:
    """Ridge-stabilized reduced-rank multi-output regression.

    The ridge fitted-value subspace is projected onto its leading right singular
    vectors.  This is the classical reduced-rank construction when ``alpha=0``
    and a stable low-dimensional analogue when predictors outnumber patients.
    """

    def __init__(
        self,
        rank: int,
        *,
        alpha: float = 1.0,
        fit_intercept: bool = True,
    ) -> None:
        self.rank = _positive_integer(rank, name="rank")
        self.alpha = _finite_number(alpha, name="alpha", nonnegative=True)
        if not isinstance(fit_intercept, bool):
            raise TypeError("fit_intercept must be boolean")
        self.fit_intercept = fit_intercept
        self.coef_: Tensor | None = None
        self.intercept_: Tensor | None = None
        self.target_was_vector_: bool | None = None

    def fit(
        self, features: Tensor, targets: Tensor
    ) -> "TorchReducedRankRegressor":
        x = _matrix(features, name="features", minimum_rows=2)
        y, was_vector = _target_matrix(targets, minimum_rows=2)
        if x.shape[0] != y.shape[0]:
            raise ValueError("features and targets must contain the same rows")
        maximum_rank = min(x.shape[1], y.shape[1])
        if self.rank > maximum_rank:
            raise ValueError("rank exceeds the coefficient-matrix rank bound")
        x_mean = x.mean(dim=0) if self.fit_intercept else x.new_zeros(x.shape[1])
        y_mean = y.mean(dim=0) if self.fit_intercept else y.new_zeros(y.shape[1])
        centered_x = x - x_mean
        centered_y = y - y_mean
        ridge = TorchRidgeRegressor(alpha=self.alpha, fit_intercept=False).fit(
            centered_x, centered_y
        )
        assert ridge.coef_ is not None
        fitted = centered_x @ ridge.coef_
        _, singular_values, right_vectors = torch.linalg.svd(
            fitted, full_matrices=False
        )
        numerical_rank = int(
            (singular_values > torch.finfo(torch.float64).eps).sum().item()
        )
        retained = min(self.rank, max(1, numerical_rank))
        projection = right_vectors[:retained].T @ right_vectors[:retained]
        coefficient = ridge.coef_ @ projection
        self.coef_ = coefficient
        self.intercept_ = y_mean - x_mean @ coefficient
        self.target_was_vector_ = was_vector
        return self

    def predict(self, features: Tensor) -> Tensor:
        if (
            self.coef_ is None
            or self.intercept_ is None
            or self.target_was_vector_ is None
        ):
            raise RuntimeError("fit must be called before predict")
        x = _matrix(features, name="features")
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError("features do not match the fitted feature dimension")
        prediction = x @ self.coef_ + self.intercept_
        return prediction[:, 0] if self.target_was_vector_ else prediction


class TorchSlowFeatureAnalysis:
    """Compact classical slow-feature analysis for fixed-length trajectories.

    ``fit`` accepts ``[patients, visits, features]`` and learns directions with
    unit training variance that minimize mean squared consecutive velocity.  If
    irregular visit times are supplied, velocities are divided by elapsed time
    after normalizing time units by the median training interval.  SFA is an
    invariance/persistence diagnostic, not evidence of treatment-response
    prediction; the class deliberately exposes only transforms and slowness.
    """

    def __init__(self, n_components: int, *, epsilon: float = 1e-10) -> None:
        self.n_components = _positive_integer(n_components, name="n_components")
        self.epsilon = _finite_number(epsilon, name="epsilon", nonnegative=True)
        if self.epsilon == 0.0:
            raise ValueError("epsilon must be positive")
        self.mean_: Tensor | None = None
        self.components_: Tensor | None = None
        self.slowness_: Tensor | None = None
        self.input_rank_: int | None = None
        self.time_scale_: float | None = None

    @staticmethod
    def _trajectory_features(features: Tensor, *, name: str) -> Tensor:
        values = _finite_tensor(
            features, name=name, dimensions=(3,), minimum_rows=2
        )
        if values.shape[1] < 2 or values.shape[2] < 1:
            raise ValueError(
                f"{name} must have at least two visits and one feature"
            )
        return values

    @staticmethod
    def _time_intervals(times: Tensor | None, shape: tuple[int, int]) -> tuple[Tensor, float]:
        patients, visits = shape
        if times is None:
            return torch.ones(patients, visits - 1, dtype=torch.float64), 1.0
        working = _finite_tensor(times, name="times", dimensions=(2,), minimum_rows=2)
        if tuple(working.shape) != (patients, visits):
            raise ValueError("times must have shape [patients, visits]")
        intervals = working[:, 1:] - working[:, :-1]
        if not bool((intervals > 0.0).all()):
            raise ValueError("visit times must be strictly increasing within patient")
        median = float(intervals.median())
        if not math.isfinite(median) or median <= 0.0:
            raise ValueError("training time intervals have an invalid median")
        return intervals / median, median

    def fit(
        self, features: Tensor, times: Tensor | None = None
    ) -> "TorchSlowFeatureAnalysis":
        values = self._trajectory_features(features, name="features")
        patients, visits, dimensions = values.shape
        intervals, time_scale = self._time_intervals(times, (patients, visits))
        flattened = values.reshape(patients * visits, dimensions)
        mean = flattened.mean(dim=0)
        centered = flattened - mean
        _, singular_values, right_vectors = torch.linalg.svd(
            centered, full_matrices=False
        )
        tolerance = max(self.epsilon, float(singular_values.max()) * self.epsilon)
        retained = singular_values > tolerance
        rank = int(retained.sum().item())
        if rank < self.n_components:
            raise ValueError("n_components exceeds the numerical training-feature rank")
        singular_values = singular_values[retained]
        right_vectors = right_vectors[retained]
        # With covariance normalized by M-1, this map gives each retained
        # whitened coordinate unit sample variance.
        whitener = right_vectors.T * (
            math.sqrt(flattened.shape[0] - 1) / singular_values
        )
        whitened = (centered @ whitener).reshape(patients, visits, rank)
        velocity = (whitened[:, 1:] - whitened[:, :-1]) / intervals[..., None]
        velocity = velocity.reshape(-1, rank)
        velocity_covariance = velocity.T @ velocity / float(velocity.shape[0])
        eigenvalues, eigenvectors = torch.linalg.eigh(velocity_covariance)
        directions = eigenvectors[:, : self.n_components]
        self.mean_ = mean
        self.components_ = whitener @ directions
        self.slowness_ = eigenvalues[: self.n_components].clamp_min(0.0)
        self.input_rank_ = rank
        self.time_scale_ = time_scale
        return self

    def _state(self) -> tuple[Tensor, Tensor]:
        if self.mean_ is None or self.components_ is None:
            raise RuntimeError("fit must be called before transform")
        return self.mean_, self.components_

    def transform(self, features: Tensor) -> Tensor:
        """Transform ``[N,D]`` or ``[N,T,D]`` while preserving leading axes."""

        mean, components = self._state()
        values = _finite_tensor(features, name="features", dimensions=(2, 3))
        if values.shape[-1] != mean.numel():
            raise ValueError("features do not match the fitted SFA dimension")
        shape = values.shape[:-1]
        transformed = (values.reshape(-1, values.shape[-1]) - mean) @ components
        return transformed.reshape(*shape, self.n_components)

    def score_slowness(
        self, features: Tensor, times: Tensor | None = None
    ) -> Tensor:
        """Return mean squared normalized velocity for each learned component."""

        values = self._trajectory_features(features, name="features")
        if self.time_scale_ is None:
            raise RuntimeError("fit must be called before score_slowness")
        if times is None:
            intervals = torch.ones(
                values.shape[0], values.shape[1] - 1, dtype=torch.float64
            )
        else:
            working_times = _finite_tensor(
                times, name="times", dimensions=(2,), minimum_rows=2
            )
            if tuple(working_times.shape) != tuple(values.shape[:2]):
                raise ValueError("times must have shape [patients, visits]")
            intervals = working_times[:, 1:] - working_times[:, :-1]
            if not bool((intervals > 0.0).all()):
                raise ValueError(
                    "visit times must be strictly increasing within patient"
                )
            intervals = intervals / self.time_scale_
        transformed = self.transform(values)
        velocity = (transformed[:, 1:] - transformed[:, :-1]) / intervals[..., None]
        return velocity.square().mean(dim=(0, 1))


def patient_group_kfold(
    patient_ids: Sequence[Hashable],
    *,
    n_splits: int = 5,
    seed: int = 0,
    shuffle: bool = True,
) -> tuple[GroupFold, ...]:
    """Return deterministic folds that never split one patient across sets."""

    folds = _positive_integer(n_splits, name="n_splits")
    _seed(seed)
    if not isinstance(shuffle, bool):
        raise TypeError("shuffle must be boolean")
    ordered, groups = _validate_patient_ids(patient_ids, rows=len(patient_ids))
    if len(ordered) < folds:
        raise ValueError("n_splits may not exceed the number of patients")
    positions = torch.arange(len(ordered), dtype=torch.long)
    if shuffle:
        positions = positions[
            torch.randperm(len(ordered), generator=torch.Generator().manual_seed(seed))
        ]
    fold_patients: list[list[Hashable]] = [[] for _ in range(folds)]
    for sequence_index, patient_position in enumerate(positions.tolist()):
        fold_patients[sequence_index % folds].append(ordered[patient_position])

    result: list[GroupFold] = []
    all_rows = set(range(len(patient_ids)))
    for held_out in fold_patients:
        validation_rows = sorted(row for patient in held_out for row in groups[patient])
        validation_set = set(validation_rows)
        train_rows = sorted(all_rows - validation_set)
        train_patients = tuple(patient for patient in ordered if patient not in held_out)
        result.append(
            GroupFold(
                train_indices=torch.tensor(train_rows, dtype=torch.long),
                validation_indices=torch.tensor(validation_rows, dtype=torch.long),
                train_patients=train_patients,
                validation_patients=tuple(held_out),
            )
        )
    return tuple(result)


def grouped_ridge_cv(
    features: Tensor,
    targets: Tensor,
    patient_ids: Sequence[Hashable],
    alphas: Sequence[float],
    *,
    n_splits: int = 5,
    seed: int = 0,
    standardize: bool = True,
    pca_components: int | None = None,
) -> RidgeCVResult:
    """Select ridge alpha with patient-disjoint folds and fold-local transforms."""

    x = _matrix(features, name="features", minimum_rows=2)
    y, was_vector = _target_matrix(targets, minimum_rows=2)
    if x.shape[0] != y.shape[0] or len(patient_ids) != x.shape[0]:
        raise ValueError("features, targets and patient_ids must align")
    if not isinstance(standardize, bool):
        raise TypeError("standardize must be boolean")
    if not alphas:
        raise ValueError("alphas must contain at least one value")
    checked_alphas = tuple(
        _finite_number(value, name="alpha", nonnegative=True) for value in alphas
    )
    if len(set(checked_alphas)) != len(checked_alphas):
        raise ValueError("alphas must be unique")
    if pca_components is not None:
        _positive_integer(pca_components, name="pca_components")
    folds = patient_group_kfold(
        patient_ids, n_splits=n_splits, seed=seed, shuffle=True
    )
    scores: dict[float, list[float]] = {alpha: [] for alpha in checked_alphas}
    for fold in folds:
        train_x = x[fold.train_indices]
        validation_x = x[fold.validation_indices]
        if standardize:
            scaler = TrainStandardizer().fit(train_x)
            train_x = scaler.transform(train_x)
            validation_x = scaler.transform(validation_x)
        if pca_components is not None:
            pca = TrainPCA(pca_components).fit(train_x)
            train_x = pca.transform(train_x)
            validation_x = pca.transform(validation_x)
        train_y = y[fold.train_indices]
        validation_y = y[fold.validation_indices]
        target_for_fit = train_y[:, 0] if was_vector else train_y
        target_for_metric = validation_y[:, 0] if was_vector else validation_y
        for alpha in checked_alphas:
            prediction = TorchRidgeRegressor(alpha=alpha).fit(
                train_x, target_for_fit
            ).predict(validation_x)
            # Fold selection needs only decomposable MAE.  Calling the public
            # metric bundle here would make an otherwise valid fold fail when
            # its held-out target happens to be constant (R2 alone is then
            # undefined, while MAE remains perfectly well-defined).
            scores[alpha].append(
                float((prediction - target_for_metric).abs().mean())
            )
    mean_scores = {
        alpha: float(sum(values) / len(values)) for alpha, values in scores.items()
    }
    best = min(checked_alphas, key=lambda alpha: (mean_scores[alpha], alpha))
    unique_patients, _ = _validate_patient_ids(patient_ids, rows=len(patient_ids))
    return RidgeCVResult(
        best_alpha=best,
        mean_mae_by_alpha=mean_scores,
        fold_mae_by_alpha={alpha: tuple(values) for alpha, values in scores.items()},
        n_splits=len(folds),
        n_patients=len(unique_patients),
    )


Statistic = Callable[[Tensor, Tensor], float | Tensor]


def _patient_matrix_means(
    values: Tensor,
    ordered: tuple[Hashable, ...],
    groups: dict[Hashable, list[int]],
) -> Tensor:
    return torch.stack(
        [
            values[torch.tensor(groups[patient], dtype=torch.long)].mean(dim=0)
            for patient in ordered
        ]
    )


def patient_level_permutation_null(
    features: Tensor,
    targets: Tensor,
    patient_ids: Sequence[Hashable],
    statistic: Statistic,
    *,
    n_permutations: int = 1_000,
    seed: int = 0,
    alternative: str = "greater",
) -> PermutationNullResult:
    """Permute patient-aggregated targets against patient-aggregated features.

    Repeated transition rows are averaged within patient before permutation.  A
    statistic that needs cross-validation must perform that fitting internally;
    this function only guarantees that the permutation unit is the patient.
    """

    x = _matrix(features, name="features", minimum_rows=2)
    y, was_vector = _target_matrix(targets, minimum_rows=2)
    if x.shape[0] != y.shape[0] or len(patient_ids) != x.shape[0]:
        raise ValueError("features, targets and patient_ids must align")
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    permutations = _positive_integer(n_permutations, name="n_permutations")
    _seed(seed)
    if alternative not in {"greater", "less", "two-sided"}:
        raise ValueError("alternative must be greater, less, or two-sided")
    ordered, groups = _validate_patient_ids(patient_ids, rows=x.shape[0])
    if len(ordered) < 2:
        raise ValueError("permutation null requires at least two patients")
    patient_x = _patient_matrix_means(x, ordered, groups)
    patient_y = _patient_matrix_means(y, ordered, groups)
    if was_vector:
        patient_y = patient_y[:, 0]

    def evaluate(left: Tensor, right: Tensor) -> float:
        value = statistic(left, right)
        if isinstance(value, Tensor):
            if value.numel() != 1:
                raise ValueError("statistic must return one scalar")
            value = float(value.detach().cpu())
        return _finite_number(value, name="statistic result")

    observed = evaluate(patient_x, patient_y)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    null_values = [
        evaluate(
            patient_x,
            patient_y[
                torch.randperm(len(ordered), generator=generator, dtype=torch.long)
            ],
        )
        for _ in range(permutations)
    ]
    null = torch.tensor(null_values, dtype=torch.float64)
    if alternative == "greater":
        extreme = int((null >= observed).sum().item())
        p_value = (extreme + 1.0) / (permutations + 1.0)
    elif alternative == "less":
        extreme = int((null <= observed).sum().item())
        p_value = (extreme + 1.0) / (permutations + 1.0)
    else:
        # Do not assume that the permutation null is centred at zero.  A
        # doubled empirical tail probability remains valid for shifted
        # statistics (for example, a correlation plus a fixed offset).
        upper = (int((null >= observed).sum().item()) + 1.0) / (
            permutations + 1.0
        )
        lower = (int((null <= observed).sum().item()) + 1.0) / (
            permutations + 1.0
        )
        p_value = min(1.0, 2.0 * min(upper, lower))
    return PermutationNullResult(
        observed=observed,
        null_mean=float(null.mean()),
        null_std=float(null.std(unbiased=False)),
        p_value=p_value,
        alternative=alternative,
        n_patients=len(ordered),
        n_permutations=permutations,
        null_values=tuple(null_values),
    )


__all__ = [
    "ClassificationMetrics",
    "GroupFold",
    "PairedPatientBootstrapResult",
    "PermutationNullResult",
    "RegressionMetrics",
    "RidgeCVResult",
    "TorchReducedRankRegressor",
    "TorchRidgeRegressor",
    "TorchSlowFeatureAnalysis",
    "TrainPCA",
    "TrainStandardizer",
    "classification_metrics",
    "grouped_ridge_cv",
    "paired_patient_bootstrap",
    "patient_group_kfold",
    "patient_level_permutation_null",
    "regression_metrics",
]
