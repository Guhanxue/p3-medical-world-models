"""Isolated ROI-aware v3 experiment components.

The package is deliberately separate from the locked v1/v2 data and model paths.
Nothing in :mod:`med_jepa` imports it implicitly.
"""

from med_jepa.roi_v3.context_weights import (
    CONTEXT_WEIGHT_INDEX_COLUMNS,
    CONTEXT_WEIGHT_RELEASE_SCHEMA_VERSION,
    CONTEXT_WEIGHT_SCHEMA_VERSION,
    build_context_weight_payload,
    build_development_context_weight_release,
    load_context_weight_payload,
    read_development_split_prefix,
)
from med_jepa.roi_v3.data import (
    DevelopmentManifestTrajectoryDataset,
    ROIAwareManifestDataset,
    ROIAwareTrajectoryBatch,
    ROIAwareTrajectorySample,
    collate_roi_trajectories,
)

__all__ = [
    "CONTEXT_WEIGHT_INDEX_COLUMNS",
    "CONTEXT_WEIGHT_RELEASE_SCHEMA_VERSION",
    "CONTEXT_WEIGHT_SCHEMA_VERSION",
    "DevelopmentManifestTrajectoryDataset",
    "ROIAwareManifestDataset",
    "ROIAwareTrajectoryBatch",
    "ROIAwareTrajectorySample",
    "build_context_weight_payload",
    "build_development_context_weight_release",
    "collate_roi_trajectories",
    "load_context_weight_payload",
    "read_development_split_prefix",
]
