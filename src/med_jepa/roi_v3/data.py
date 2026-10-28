"""Hash-bound development-only dataset adapter for ROI-aware v3 experiments."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from med_jepa.data.contracts import (
    TrajectoryBatch,
    TrajectorySample,
    collate_trajectories,
)
from med_jepa.data.manifest import ManifestTrajectoryDataset
from med_jepa.roi_v3.context_weights import (
    CONTEXT_WEIGHT_INDEX_COLUMNS,
    FEATURE_GRID_SHAPE_DHW,
    ContextWeightError,
    load_context_weight_payload,
    sha256_file,
)


DEVELOPMENT_SPLITS = ("train", "val")
DEFAULT_DEVELOPMENT_COUNTS = {"train": 409, "val": 88}


class ROIContextDataError(ValueError):
    """Raised when the context-weight/data join violates its release contract."""


class DevelopmentManifestTrajectoryDataset(ManifestTrajectoryDataset):
    """Parse only the governed train/validation prefix of the frozen manifest.

    The historical loader validates a complete manifest before filtering a split.
    That behavior is appropriate for its original experiment, but would materialize
    held-out test metadata in this development-only arm.  This adapter copies exactly
    the first 409+88 physical JSONL records to a private temporary view, validates
    them with the unchanged historical loader, and never reads the next/test line.

    Formal ISPY2 feature paths are absolute.  We enforce that invariant before using
    the temporary view, so deleting the view cannot change tensor resolution.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        *,
        expected_development_counts: Mapping[str, int] = DEFAULT_DEVELOPMENT_COUNTS,
        **manifest_kwargs: Any,
    ) -> None:
        if split not in DEVELOPMENT_SPLITS:
            raise ROIContextDataError(
                "development manifest loader accepts only train or val"
            )
        counts = ROIAwareManifestDataset._expected_counts(
            expected_development_counts
        )
        source = Path(manifest_path).expanduser().resolve()
        expected_rows = sum(counts.values())
        raw_lines: list[bytes] = []
        seen_patients: set[str] = set()
        observed_counts = {name: 0 for name in DEVELOPMENT_SPLITS}
        try:
            with source.open("rb") as handle:
                for row_index in range(expected_rows):
                    line = handle.readline()
                    if not line or not line.strip():
                        raise ROIContextDataError(
                            "development manifest prefix has wrong cardinality"
                        )
                    try:
                        record = json.loads(line)
                    except (UnicodeError, json.JSONDecodeError) as error:
                        raise ROIContextDataError(
                            "development manifest record is invalid JSON"
                        ) from error
                    if not isinstance(record, dict):
                        raise ROIContextDataError(
                            "development manifest record must be an object"
                        )
                    patient = str(record.get("patient_id") or "").strip()
                    record_split = str(record.get("split") or "").strip().casefold()
                    expected_split = (
                        "train" if row_index < counts["train"] else "val"
                    )
                    if (
                        not patient
                        or patient in seen_patients
                        or record_split != expected_split
                    ):
                        raise ROIContextDataError(
                            "development manifest prefix order is invalid"
                        )
                    visits = [*record.get("context", []), record.get("target", {})]
                    if not visits or any(not isinstance(visit, dict) for visit in visits):
                        raise ROIContextDataError(
                            "development manifest visits are malformed"
                        )
                    for visit in visits:
                        volume = Path(str(visit.get("volume") or "")).expanduser()
                        if not volume.is_absolute():
                            raise ROIContextDataError(
                                "formal development feature paths must be absolute"
                            )
                    seen_patients.add(patient)
                    observed_counts[record_split] += 1
                    raw_lines.append(line)
        except OSError as error:
            raise ROIContextDataError("development manifest is unreadable") from error
        if observed_counts != counts:
            raise ROIContextDataError(
                "development manifest patient counts do not match"
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".med-jepa-roi-v3-development-",
                suffix=".jsonl",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                os.chmod(temporary_path, 0o600)
                handle.writelines(raw_lines)
                handle.flush()
                os.fsync(handle.fileno())
            super().__init__(
                temporary_path,
                split,
                expected_patient_counts=counts,
                **manifest_kwargs,
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
        # All accepted tensor paths are absolute, but restoring the true source root
        # also keeps diagnostics and provenance semantically accurate.
        self.path = source
        self.root = source.parent
        self.test_records_parsed = False
        self.test_tensor_files_loaded = False


def _digest(value: Any, *, field: str) -> str:
    text = str(value).strip().lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise ROIContextDataError(f"{field} must be a 64-character hex digest")
    return text


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


@dataclass(frozen=True)
class ROIAwareTrajectorySample:
    """One existing trajectory plus fixed T0-derived core/ring patch weights."""

    trajectory: TrajectorySample
    roi_weights: Tensor  # [2,7,7,7], ordered core then ring

    def validate(self) -> None:
        self.trajectory.validate()
        if not isinstance(self.roi_weights, Tensor) or not self.roi_weights.is_floating_point():
            raise ROIContextDataError("roi_weights must be a floating-point tensor")
        if tuple(self.roi_weights.shape) != (2, *FEATURE_GRID_SHAPE_DHW):
            raise ROIContextDataError("roi_weights must have shape [2,7,7,7]")
        if not bool(torch.isfinite(self.roi_weights).all()) or torch.any(
            (self.roi_weights < 0) | (self.roi_weights > 1)
        ):
            raise ROIContextDataError("roi_weights must contain finite occupancies in [0,1]")
        if torch.any(self.roi_weights.reshape(2, -1).sum(dim=1) <= 0):
            raise ROIContextDataError("both ROI weight regions must be non-empty")


@dataclass(frozen=True)
class ROIAwareTrajectoryBatch:
    """A normal trajectory batch with one shared spatial weight tensor per patient."""

    trajectory: TrajectoryBatch
    roi_weights: Tensor  # [B,2,7,7,7]

    @property
    def batch_size(self) -> int:
        return self.trajectory.batch_size

    def validate(self) -> None:
        self.trajectory.validate()
        expected = (self.trajectory.batch_size, 2, *FEATURE_GRID_SHAPE_DHW)
        if not isinstance(self.roi_weights, Tensor) or not self.roi_weights.is_floating_point():
            raise ROIContextDataError("roi_weights must be a floating-point tensor")
        if tuple(self.roi_weights.shape) != expected:
            raise ROIContextDataError(f"roi_weights must have shape {expected}")
        if not bool(torch.isfinite(self.roi_weights).all()) or torch.any(
            (self.roi_weights < 0) | (self.roi_weights > 1)
        ):
            raise ROIContextDataError("roi_weights must contain finite occupancies in [0,1]")
        if torch.any(self.roi_weights.reshape(self.batch_size, 2, -1).sum(dim=2) <= 0):
            raise ROIContextDataError("both ROI regions must be non-empty for every patient")

    def to(self, device: torch.device | str) -> "ROIAwareTrajectoryBatch":
        result = ROIAwareTrajectoryBatch(
            trajectory=self.trajectory.to(device),
            roi_weights=self.roi_weights.to(device),
        )
        result.validate()
        return result


def collate_roi_trajectories(
    samples: Sequence[ROIAwareTrajectorySample],
) -> ROIAwareTrajectoryBatch:
    if not samples:
        raise ROIContextDataError("cannot collate an empty ROI-aware sample list")
    for sample in samples:
        sample.validate()
    result = ROIAwareTrajectoryBatch(
        trajectory=collate_trajectories([sample.trajectory for sample in samples]),
        roi_weights=torch.stack([sample.roi_weights.float() for sample in samples]),
    )
    result.validate()
    return result


@dataclass(frozen=True)
class _WeightRecord:
    patient_id: str
    split: str
    path: Path
    content_sha256: str


class ROIAwareManifestDataset(Dataset[ROIAwareTrajectorySample]):
    """Attach immutable context weights to an existing manifest dataset.

    The supplied index must cover exactly the frozen development cohort and contain
    only ``train`` and ``val`` rows.  A caller-provided SHA-256 binds the index before
    any patient join occurs.  Test records and future-mask fields have no accepted
    code path in this adapter.
    """

    def __init__(
        self,
        trajectory_dataset: ManifestTrajectoryDataset,
        context_weight_index_path: str | Path,
        *,
        context_weight_root: str | Path,
        expected_context_weight_index_sha256: str,
        expected_development_counts: Mapping[str, int] = DEFAULT_DEVELOPMENT_COUNTS,
    ) -> None:
        if not isinstance(trajectory_dataset, ManifestTrajectoryDataset):
            raise TypeError("trajectory_dataset must be a ManifestTrajectoryDataset")
        self.trajectory_dataset = trajectory_dataset
        self.index_path = Path(context_weight_index_path).expanduser().resolve()
        self.context_weight_root = Path(context_weight_root).expanduser().resolve()
        if not self.context_weight_root.is_dir():
            raise ROIContextDataError("context_weight_root is unavailable")
        expected_index = _digest(
            expected_context_weight_index_sha256,
            field="expected_context_weight_index_sha256",
        )
        try:
            observed_index = sha256_file(self.index_path)
        except OSError as error:
            raise ROIContextDataError("context-weight index is unreadable") from error
        if observed_index != expected_index:
            raise ROIContextDataError("context-weight index SHA-256 mismatch")

        expected_counts = self._expected_counts(expected_development_counts)
        records = self._read_index()
        observed_counts = {
            split: sum(record.split == split for record in records.values())
            for split in DEVELOPMENT_SPLITS
        }
        if observed_counts != expected_counts:
            raise ROIContextDataError(
                "context-weight development counts do not match the locked contract"
            )

        selected_splits = {str(record.get("split")) for record in trajectory_dataset.records}
        if len(selected_splits) != 1 or not selected_splits.issubset(DEVELOPMENT_SPLITS):
            raise ROIContextDataError("ROI-aware datasets are restricted to train or val")
        self.split = next(iter(selected_splits))
        base_patients = [str(record.get("patient_id")) for record in trajectory_dataset.records]
        if len(set(base_patients)) != len(base_patients):
            raise ROIContextDataError("trajectory split contains duplicate patients")
        indexed_patients = {
            patient for patient, record in records.items() if record.split == self.split
        }
        if set(base_patients) != indexed_patients:
            raise ROIContextDataError(
                "context-weight index does not exactly cover the selected trajectory split"
            )
        self._records = {patient: records[patient] for patient in base_patients}
        self._verified_payloads: set[Path] = set()
        self._weight_cache: dict[Path, Tensor] = {}

    @staticmethod
    def _expected_counts(value: Mapping[str, int]) -> dict[str, int]:
        if set(value) != set(DEVELOPMENT_SPLITS):
            raise ROIContextDataError("expected development counts must contain train and val")
        result: dict[str, int] = {}
        for split in DEVELOPMENT_SPLITS:
            count = value[split]
            if isinstance(count, bool) or not isinstance(count, int) or count < 1:
                raise ROIContextDataError("expected development counts must be positive integers")
            result[split] = int(count)
        return result

    def _read_index(self) -> dict[str, _WeightRecord]:
        try:
            with self.index_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if reader.fieldnames is None or tuple(reader.fieldnames) != CONTEXT_WEIGHT_INDEX_COLUMNS:
                    raise ROIContextDataError("context-weight index columns do not match")
                rows = list(reader)
        except (OSError, csv.Error, UnicodeError) as error:
            raise ROIContextDataError("context-weight index is unreadable") from error
        if not rows:
            raise ROIContextDataError("context-weight index is empty")

        result: dict[str, _WeightRecord] = {}
        paths: set[Path] = set()
        hashes: set[str] = set()
        for row in rows:
            patient = str(row.get("PatientID") or "").strip()
            split = str(row.get("split") or "").strip().casefold()
            if not patient or patient in result:
                raise ROIContextDataError("context-weight index contains duplicate/blank patients")
            if split not in DEVELOPMENT_SPLITS:
                raise ROIContextDataError("context-weight index must not contain test rows")
            digest = _digest(row.get("content_sha256"), field="content_sha256")
            raw_path = Path(str(row.get("weights_path") or "").strip()).expanduser()
            if not str(raw_path) or str(raw_path) == ".":
                raise ROIContextDataError("context-weight index contains a blank path")
            path = (
                raw_path.resolve()
                if raw_path.is_absolute()
                else (self.index_path.parent / raw_path).resolve()
            )
            if not _inside(path, self.context_weight_root):
                raise ROIContextDataError("context-weight path escapes its allowed root")
            if path in paths or digest in hashes:
                raise ROIContextDataError("context-weight path or content is reused")
            if not path.is_file():
                raise ROIContextDataError("context-weight payload is unavailable")
            paths.add(path)
            hashes.add(digest)
            result[patient] = _WeightRecord(patient, split, path, digest)
        return result

    def __len__(self) -> int:
        return len(self.trajectory_dataset)

    def _load_weights(self, record: _WeightRecord) -> Tensor:
        cached = self._weight_cache.get(record.path)
        if cached is not None:
            return cached
        if record.path not in self._verified_payloads:
            try:
                observed = sha256_file(record.path)
            except OSError as error:
                raise ROIContextDataError("context-weight payload is unreadable") from error
            if observed != record.content_sha256:
                raise ROIContextDataError("context-weight payload SHA-256 mismatch")
            self._verified_payloads.add(record.path)
        try:
            payload = load_context_weight_payload(record.path)
        except ContextWeightError as error:
            raise ROIContextDataError("context-weight payload violates its schema") from error
        weights = payload["roi_weights"].float().contiguous()
        self._weight_cache[record.path] = weights
        return weights

    def __getitem__(self, index: int) -> ROIAwareTrajectorySample:
        trajectory = self.trajectory_dataset[index]
        record = self._records.get(trajectory.patient_id)
        if record is None:
            raise ROIContextDataError("trajectory patient is absent from the weight index")
        result = ROIAwareTrajectorySample(
            trajectory=trajectory,
            roi_weights=self._load_weights(record),
        )
        result.validate()
        return result


__all__ = [
    "DEFAULT_DEVELOPMENT_COUNTS",
    "DEVELOPMENT_SPLITS",
    "DevelopmentManifestTrajectoryDataset",
    "ROIAwareManifestDataset",
    "ROIAwareTrajectoryBatch",
    "ROIAwareTrajectorySample",
    "ROIContextDataError",
    "collate_roi_trajectories",
]
