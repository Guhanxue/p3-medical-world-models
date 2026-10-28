"""Leakage-resistant patient trajectory contracts.

The predictor only receives context fields. Target images and target outcomes are kept
in separate fields and are used exclusively by target branches and losses.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Sequence

import torch
from torch import Tensor


@dataclass
class TrajectorySample:
    patient_id: str
    context_volumes: Tensor  # [S, C, D, H, W]
    context_times_days: Tensor  # [S]
    context_valid: Tensor  # [S], bool
    state_actions: Tensor  # [S, A], exposure before/at each observed state
    clinical: Tensor  # [K], static prognostic covariates (not actions)
    context_ftv: Tensor  # [S]
    target_volume: Tensor  # [C, D, H, W]
    target_time_days: Tensor  # scalar
    target_ftv: Tensor  # scalar
    next_action: Tensor  # [A], treatment over last-context -> target interval
    target_pcr: Tensor = field(
        default_factory=lambda: torch.tensor(float("nan"), dtype=torch.float32)
    )  # scalar; binary or NaN when unavailable

    def validate(self) -> None:
        if not self.patient_id:
            raise ValueError("patient_id must be non-empty")
        if self.context_volumes.ndim != 5:
            raise ValueError("context_volumes must have shape [S,C,D,H,W]")
        if not torch.isfinite(self.context_volumes).all():
            raise ValueError("context_volumes must contain only finite values")
        steps = self.context_volumes.shape[0]
        if self.context_times_days.shape != (steps,):
            raise ValueError("context_times_days must have shape [S]")
        if self.context_valid.shape != (steps,):
            raise ValueError("context_valid must have shape [S]")
        if self.context_valid.dtype != torch.bool:
            raise ValueError("context_valid must be boolean")
        if self.state_actions.ndim != 2 or self.state_actions.shape[0] != steps:
            raise ValueError("state_actions must have shape [S,A]")
        if self.next_action.shape != (self.state_actions.shape[1],):
            raise ValueError("next_action must have shape [A]")
        if self.context_ftv.shape != (steps,):
            raise ValueError("context_ftv must have shape [S]")
        if self.target_volume.shape != self.context_volumes.shape[1:]:
            raise ValueError("target_volume must match context spatial/channel shape")
        if not torch.isfinite(self.target_volume).all():
            raise ValueError("target_volume must contain only finite values")
        if (
            self.target_time_days.ndim != 0
            or self.target_ftv.ndim != 0
            or self.target_pcr.ndim != 0
        ):
            raise ValueError("target_time_days, target_ftv, and target_pcr must be scalars")
        if not torch.isfinite(self.context_times_days).all() or not torch.isfinite(
            self.target_time_days
        ).all():
            raise ValueError("visit times must contain only finite values")
        if not torch.isfinite(self.state_actions).all() or not torch.isfinite(
            self.next_action
        ).all():
            raise ValueError("action features must contain only finite values")
        if not torch.isfinite(self.clinical).all():
            raise ValueError(
                "clinical features must be finite; encode missingness explicitly"
            )
        all_ftv = torch.cat((self.context_ftv, self.target_ftv.reshape(1)))
        if torch.isinf(all_ftv).any() or torch.any(all_ftv[torch.isfinite(all_ftv)] < 0):
            raise ValueError("FTV values must be non-negative or NaN when missing")
        if torch.isinf(self.target_pcr):
            raise ValueError("target_pcr must be binary or NaN when missing")
        if torch.isfinite(self.target_pcr) and not bool(
            (self.target_pcr == 0) | (self.target_pcr == 1)
        ):
            raise ValueError("target_pcr must be binary or NaN when missing")
        valid_idx = torch.nonzero(self.context_valid, as_tuple=False).flatten()
        if valid_idx.numel() == 0:
            raise ValueError("each trajectory needs at least one valid context state")
        if not torch.equal(valid_idx, torch.arange(valid_idx.numel())):
            raise ValueError("valid context states must be left-aligned")
        valid_times = self.context_times_days[valid_idx]
        if valid_times.numel() > 1 and torch.any(valid_times[1:] <= valid_times[:-1]):
            raise ValueError("valid context times must be strictly increasing")
        if self.target_time_days <= valid_times[-1]:
            raise ValueError("target time must be after the last context time")


@dataclass
class TrajectoryBatch:
    patient_ids: list[str]
    context_volumes: Tensor  # [B, S, C, D, H, W]
    context_times_days: Tensor  # [B, S]
    context_valid: Tensor  # [B, S]
    state_actions: Tensor  # [B, S, A]
    clinical: Tensor  # [B, K]
    context_ftv: Tensor  # [B, S]
    target_volume: Tensor  # [B, C, D, H, W]
    target_time_days: Tensor  # [B]
    target_ftv: Tensor  # [B]
    next_action: Tensor  # [B, A]
    target_pcr: Tensor = field(default_factory=lambda: torch.empty(0))  # [B]

    def __post_init__(self) -> None:
        # Preserve compatibility with callers created before target_pcr existed.
        if self.target_pcr.numel() == 0:
            self.target_pcr = torch.full(
                (self.context_volumes.shape[0],),
                float("nan"),
                dtype=torch.float32,
                device=self.context_volumes.device,
            )

    def to(self, device: torch.device | str) -> "TrajectoryBatch":
        values: dict[str, object] = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.to(device) if isinstance(value, Tensor) else value
        return TrajectoryBatch(**values)  # type: ignore[arg-type]

    @property
    def batch_size(self) -> int:
        return self.context_volumes.shape[0]

    def validate(self) -> None:
        batch, steps = self.context_volumes.shape[:2]
        if len(self.patient_ids) != batch:
            raise ValueError("patient_ids length does not match batch size")
        expected = {
            "context_times_days": (batch, steps),
            "context_valid": (batch, steps),
            "context_ftv": (batch, steps),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if self.state_actions.shape[:2] != (batch, steps):
            raise ValueError("state_actions must have shape [B,S,A]")
        if self.next_action.shape != (batch, self.state_actions.shape[-1]):
            raise ValueError("next_action must have shape [B,A]")
        if self.clinical.ndim != 2 or self.clinical.shape[0] != batch:
            raise ValueError("clinical must have shape [B,K]")
        if self.target_volume.shape != (batch, *self.context_volumes.shape[2:]):
            raise ValueError("target_volume must have shape [B,C,D,H,W]")
        if (
            self.target_time_days.shape != (batch,)
            or self.target_ftv.shape != (batch,)
            or self.target_pcr.shape != (batch,)
        ):
            raise ValueError("target time, FTV, and pCR must have shape [B]")
        if self.context_valid.dtype != torch.bool:
            raise ValueError("context_valid must be boolean")
        if not torch.isfinite(self.context_times_days[self.context_valid]).all():
            raise ValueError("valid context times must be finite")
        if not torch.isfinite(self.target_time_days).all():
            raise ValueError("target times must be finite")
        if not torch.isfinite(self.state_actions).all() or not torch.isfinite(
            self.next_action
        ).all():
            raise ValueError("action features must be finite")
        if not torch.isfinite(self.clinical).all():
            raise ValueError("clinical features must be finite")
        all_ftv = torch.cat((self.context_ftv[self.context_valid], self.target_ftv))
        if torch.isinf(all_ftv).any() or torch.any(all_ftv[torch.isfinite(all_ftv)] < 0):
            raise ValueError("FTV values must be non-negative or NaN when missing")
        if torch.isinf(self.target_pcr).any():
            raise ValueError("target_pcr must be binary or NaN when missing")
        finite_pcr = self.target_pcr[torch.isfinite(self.target_pcr)]
        if torch.any((finite_pcr != 0) & (finite_pcr != 1)):
            raise ValueError("target_pcr must be binary or NaN when missing")
        lengths = self.context_valid.sum(dim=1)
        if torch.any(lengths < 1):
            raise ValueError("every patient needs at least one valid context state")
        expected_valid = (
            torch.arange(steps, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)
        )
        if not torch.equal(self.context_valid, expected_valid):
            raise ValueError("valid context states must be left-aligned")
        valid_pairs = self.context_valid[:, 1:] & self.context_valid[:, :-1]
        time_deltas = self.context_times_days[:, 1:] - self.context_times_days[:, :-1]
        if torch.any(time_deltas[valid_pairs] <= 0):
            raise ValueError("valid context times must be strictly increasing")
        last_idx = lengths - 1
        last_times = self.context_times_days.gather(1, last_idx[:, None]).squeeze(1)
        if torch.any(self.target_time_days <= last_times):
            raise ValueError("target times must be after the last valid context times")


def collate_trajectories(samples: Sequence[TrajectorySample]) -> TrajectoryBatch:
    """Pad variable-length histories without mixing target information into context."""

    if not samples:
        raise ValueError("cannot collate an empty sample list")
    for sample in samples:
        sample.validate()

    max_steps = max(sample.context_volumes.shape[0] for sample in samples)
    action_dim = samples[0].state_actions.shape[1]
    clinical_dim = samples[0].clinical.numel()
    volume_shape = samples[0].context_volumes.shape[1:]
    dtype = samples[0].context_volumes.dtype

    for sample in samples:
        if sample.context_volumes.shape[1:] != volume_shape:
            raise ValueError("all volumes in a batch must have the same shape")
        if sample.state_actions.shape[1] != action_dim:
            raise ValueError("all samples must use the same action dimension")
        if sample.clinical.numel() != clinical_dim:
            raise ValueError("all samples must use the same clinical dimension")

    batch_size = len(samples)
    context_volumes = torch.zeros(
        (batch_size, max_steps, *volume_shape), dtype=dtype
    )
    context_times = torch.zeros((batch_size, max_steps), dtype=torch.float32)
    context_valid = torch.zeros((batch_size, max_steps), dtype=torch.bool)
    state_actions = torch.zeros(
        (batch_size, max_steps, action_dim), dtype=torch.float32
    )
    context_ftv = torch.full((batch_size, max_steps), float("nan"))

    for row, sample in enumerate(samples):
        steps = sample.context_volumes.shape[0]
        context_volumes[row, :steps] = sample.context_volumes
        context_times[row, :steps] = sample.context_times_days.float()
        context_valid[row, :steps] = sample.context_valid
        state_actions[row, :steps] = sample.state_actions.float()
        context_ftv[row, :steps] = sample.context_ftv.float()

    batch = TrajectoryBatch(
        patient_ids=[sample.patient_id for sample in samples],
        context_volumes=context_volumes,
        context_times_days=context_times,
        context_valid=context_valid,
        state_actions=state_actions,
        clinical=torch.stack([sample.clinical.float() for sample in samples]),
        context_ftv=context_ftv,
        target_volume=torch.stack([sample.target_volume for sample in samples]),
        target_time_days=torch.stack(
            [sample.target_time_days.float() for sample in samples]
        ),
        target_ftv=torch.stack([sample.target_ftv.float() for sample in samples]),
        next_action=torch.stack([sample.next_action.float() for sample in samples]),
        target_pcr=torch.stack([sample.target_pcr.float() for sample in samples]),
    )
    batch.validate()
    return batch
