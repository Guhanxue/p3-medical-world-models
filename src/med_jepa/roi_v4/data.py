"""Development-only patient states expanded into all governed v4 transitions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from med_jepa.roi_v3.data import DevelopmentManifestTrajectoryDataset


DEVELOPMENT_COUNTS = {"train": 409, "val": 88}
VISITS = ("T0", "T1", "T3")


@dataclass(frozen=True)
class PatientStates:
    patient_id: str
    split: str
    features: Tensor  # [3,1024,7,7,7]
    times_days: Tensor  # [3]
    action_01: Tensor  # [2]
    action_13: Tensor  # [2]
    action_03: Tensor  # [2], cumulative logical OR
    ftv: Tensor  # [3]
    clinical: Tensor  # [2], static HR/HER2 context covariates
    pcr: Tensor  # scalar, evaluation-only outcome or NaN

    def validate(self) -> None:
        if self.split not in DEVELOPMENT_COUNTS or not self.patient_id:
            raise ValueError("invalid development patient metadata")
        if self.features.shape != (3, 1024, 7, 7, 7):
            raise ValueError("patient features violate the governed grid")
        if self.times_days.shape != (3,) or torch.any(self.times_days[1:] <= self.times_days[:-1]):
            raise ValueError("patient visit times must be strictly increasing")
        for action in (self.action_01, self.action_13, self.action_03):
            if action.shape != (2,) or torch.any((action != 0) & (action != 1)):
                raise ValueError("v4 actions must be binary length-two vectors")
        if not torch.equal(self.action_03, torch.maximum(self.action_01, self.action_13)):
            raise ValueError("direct action must be the cumulative interval action")
        if self.ftv.shape != (3,) or torch.isinf(self.ftv).any():
            raise ValueError("FTV must be finite-or-NaN and aligned to visits")
        if self.clinical.shape != (2,) or not bool(torch.isfinite(self.clinical).all()):
            raise ValueError("clinical context must be a finite length-two vector")
        if self.pcr.ndim != 0 or torch.isinf(self.pcr):
            raise ValueError("pCR must be a scalar binary value or NaN")
        if torch.isfinite(self.pcr) and not bool((self.pcr == 0) | (self.pcr == 1)):
            raise ValueError("pCR must be binary when observed")


def load_development_states(manifest: str | Path, split: str) -> list[PatientStates]:
    if split not in DEVELOPMENT_COUNTS:
        raise ValueError("v4 development loader accepts train or val only")
    dataset = DevelopmentManifestTrajectoryDataset(
        manifest,
        split,
        expected_development_counts=DEVELOPMENT_COUNTS,
        expected_history_length=2,
        expected_context_visits=["T0", "T1"],
        expected_target_visit="T3",
        expected_clinical_features=["hr_positive", "her2_positive"],
        expected_action_features=["phase_1_exposed", "phase_2_planned"],
        expected_schema_version="med-jepa.ispy2.t0-t1-to-t3.v1",
        expected_modality="SER",
        expected_tensor_kind="cached_features",
        expected_extractor="AICONSlab/3DINO-ViT",
        expected_source_revision="85bd4435c1b2ada41cd34cd15cad17c4d3c88d89",
        expected_checkpoint_sha256="b68dde2e435648bdf72edae70f970b07e4a903a8122c5be9e360bdb076cf695c",
        expected_checkpoint_key="teacher",
        expected_volume_shape=[7, 7, 7],
        require_spatial_provenance=True,
        require_feature_provenance=True,
        require_content_hashes=True,
    )
    result: list[PatientStates] = []
    for index in range(len(dataset)):
        sample = dataset[index]
        features = torch.cat((sample.context_volumes, sample.target_volume[None]), dim=0).float()
        times = torch.cat((sample.context_times_days, sample.target_time_days[None])).float()
        ftv = torch.cat((sample.context_ftv, sample.target_ftv[None])).float()
        action_01 = sample.state_actions[1].float()
        action_13 = sample.next_action.float()
        item = PatientStates(
            patient_id=sample.patient_id,
            split=split,
            features=features,
            times_days=times,
            action_01=action_01,
            action_13=action_13,
            action_03=torch.maximum(action_01, action_13),
            ftv=ftv,
            clinical=sample.clinical.float(),
            pcr=sample.target_pcr.float(),
        )
        item.validate()
        result.append(item)
    if len(result) != DEVELOPMENT_COUNTS[split]:
        raise ValueError("development split cardinality mismatch")
    return result


def stack_batch(records: list[PatientStates], indices: Tensor) -> dict[str, Tensor | list[str]]:
    selected = [records[int(index)] for index in indices.tolist()]
    return {
        "patient_ids": [item.patient_id for item in selected],
        "features": torch.stack([item.features for item in selected]),
        "times": torch.stack([item.times_days for item in selected]),
        "action_01": torch.stack([item.action_01 for item in selected]),
        "action_13": torch.stack([item.action_13 for item in selected]),
        "action_03": torch.stack([item.action_03 for item in selected]),
        "ftv": torch.stack([item.ftv for item in selected]),
        "clinical": torch.stack([item.clinical for item in selected]),
        "pcr": torch.stack([item.pcr for item in selected]),
    }


__all__ = ["DEVELOPMENT_COUNTS", "PatientStates", "VISITS", "load_development_states", "stack_batch"]
