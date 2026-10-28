"""Feature views and mask-derived outcomes for the frozen-feature audit.

The functions in this module are deliberately model-free.  They summarize the
already-governed final 3DINO patch grid and the development-only canonical mask
sidecars; no DICOM pixels are read and no neural-network parameter is updated.
"""

from __future__ import annotations

import hashlib
import math
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

from med_jepa.ispy2_manifest_builder import (
    ManifestBuildError,
    validate_feature_provenance,
    validate_spatial_provenance,
)


FEATURE_GRID_SHAPE_DHW = (7, 7, 7)
ROI_SHAPE_DHW = (112, 112, 112)
EXPECTED_SPACING_DHW_MM = (2.1, 2.1, 2.1)
FEATURE_PAYLOAD_SCHEMA_FIELDS = {
    "volume",
    "feature_provenance",
    "source_content_sha256",
    "spatial_provenance",
    "source_intensity_provenance",
}
MASK_SCHEMA_VERSION = "spatial-med-jepa-ispy2-mask-v1"


class FrozenFeatureContractError(ValueError):
    """Raised when a governed frozen artifact violates the audit contract."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_sha256(value: Any, *, field: str = "sha256") -> str:
    digest = str(value or "").strip().lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise FrozenFeatureContractError(f"{field} must be a 64-character hex digest")
    return digest


def load_feature_grid(
    path: str | Path,
    *,
    expected_sha256: str,
    expected_channels: int = 1024,
) -> tuple[Tensor, dict[str, Any], dict[str, Any]]:
    """Hash-check and load one cached final-layer 3DINO grid.

    Only weights-only ``torch.load`` is permitted.  The returned tensor is
    float32 CPU data with shape ``[F,7,7,7]``.
    """

    source = Path(path).expanduser().resolve()
    expected = valid_sha256(expected_sha256, field="feature content_sha256")
    try:
        observed = sha256_file(source)
    except OSError as error:
        raise FrozenFeatureContractError("feature payload is unreadable") from error
    if observed != expected:
        raise FrozenFeatureContractError("feature payload SHA-256 mismatch")
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001 - normalize unsafe path details
        raise FrozenFeatureContractError("feature payload is unreadable") from error
    if not isinstance(value, Mapping) or set(value) != FEATURE_PAYLOAD_SCHEMA_FIELDS:
        raise FrozenFeatureContractError("feature payload fields do not match")
    volume = value.get("volume")
    if not isinstance(volume, Tensor) or not volume.is_floating_point():
        raise FrozenFeatureContractError("feature volume must be floating point")
    volume = volume.to(device="cpu", dtype=torch.float32).contiguous()
    if tuple(volume.shape) != (expected_channels, *FEATURE_GRID_SHAPE_DHW):
        raise FrozenFeatureContractError(
            "feature volume must have the governed [F,7,7,7] shape"
        )
    if not bool(torch.isfinite(volume).all()):
        raise FrozenFeatureContractError("feature volume contains non-finite values")
    try:
        provenance = validate_feature_provenance(value.get("feature_provenance"))
        spatial = validate_spatial_provenance(value.get("spatial_provenance"))
    except (ManifestBuildError, TypeError) as error:
        raise FrozenFeatureContractError("feature provenance is invalid") from error
    if (
        int(provenance.get("feature_dim", -1)) != expected_channels
        or tuple(provenance.get("feature_grid_shape", ())) != FEATURE_GRID_SHAPE_DHW
        or provenance.get("layer") != "last_normalized_patch_tokens"
    ):
        raise FrozenFeatureContractError("feature provenance does not match the audit")
    if (
        spatial.get("orientation") != "RAS"
        or tuple(spatial.get("spacing_mm", ())) != EXPECTED_SPACING_DHW_MM
        or tuple(spatial.get("roi_voxel_shape", ())) != ROI_SHAPE_DHW
        or spatial.get("reference_visit") != "T0"
        or spatial.get("roi_source_visits") != ["T0"]
        or spatial.get("registration_type") != "rigid"
        or spatial.get("deformable_registration") is not False
    ):
        raise FrozenFeatureContractError("feature spatial provenance is incompatible")
    return volume, provenance, spatial


def load_mask_sidecar(
    path: str | Path,
    *,
    expected_sha256: str,
) -> dict[str, Any]:
    """Load the exact development-only Spatial Med-JEPA mask sidecar schema."""

    source = Path(path).expanduser().resolve()
    expected = valid_sha256(expected_sha256, field="mask content_sha256")
    try:
        observed = sha256_file(source)
    except OSError as error:
        raise FrozenFeatureContractError("mask sidecar is unreadable") from error
    if observed != expected:
        raise FrozenFeatureContractError("mask sidecar SHA-256 mismatch")
    try:
        value = torch.load(source, map_location="cpu", weights_only=True)
    except Exception as error:  # noqa: BLE001
        raise FrozenFeatureContractError("mask sidecar is unreadable") from error
    if not isinstance(value, Mapping) or value.get("schema_version") != MASK_SCHEMA_VERSION:
        raise FrozenFeatureContractError("mask sidecar schema is unsupported")
    required = {
        "schema_version",
        "spatial_provenance",
        "valid_support",
        "t0_ftv_inclusion_support",
        "t0_disease_bed_20mm",
        "optional_evaluation_only",
        "source_hashes",
    }
    if set(value) != required:
        raise FrozenFeatureContractError("mask sidecar fields do not match")
    valid = value.get("valid_support")
    optional = value.get("optional_evaluation_only")
    if not isinstance(valid, Mapping) or set(valid) != {"T0", "T1", "T3"}:
        raise FrozenFeatureContractError("mask valid-support contract does not match")
    if not isinstance(optional, Mapping) or set(optional) - {"t3_ftv_inclusion_support"}:
        raise FrozenFeatureContractError("mask optional-evaluation contract does not match")

    def mask(raw: Any, *, name: str, allow_empty: bool = False) -> Tensor:
        if not isinstance(raw, Tensor) or raw.dtype != torch.bool:
            raise FrozenFeatureContractError(f"{name} must be a boolean tensor")
        result = raw.to(device="cpu").contiguous()
        if tuple(result.shape) != (1, *ROI_SHAPE_DHW):
            raise FrozenFeatureContractError(f"{name} has the wrong shape")
        if not allow_empty and not bool(result.any()):
            raise FrozenFeatureContractError(f"{name} is empty")
        return result

    valid_support = {visit: mask(valid[visit], name=f"valid_support.{visit}") for visit in valid}
    t0 = mask(value.get("t0_ftv_inclusion_support"), name="t0_ftv_inclusion_support")
    bed = mask(value.get("t0_disease_bed_20mm"), name="t0_disease_bed_20mm")
    if torch.any(t0 & ~bed) or torch.any(t0 & ~valid_support["T0"]):
        raise FrozenFeatureContractError("T0 inclusion support violates its geometry")
    raw_t3 = optional.get("t3_ftv_inclusion_support")
    t3 = None if raw_t3 is None else mask(
        raw_t3,
        name="t3_ftv_inclusion_support",
        allow_empty=True,
    )
    if t3 is not None and torch.any(t3 & ~valid_support["T3"]):
        raise FrozenFeatureContractError("T3 inclusion support violates its geometry")
    try:
        spatial = validate_spatial_provenance(value.get("spatial_provenance"))
    except (ManifestBuildError, TypeError) as error:
        raise FrozenFeatureContractError("mask spatial provenance is invalid") from error
    spacing = tuple(spatial.get("spacing_mm", ()))
    if (
        spatial.get("orientation") != "RAS"
        or spacing != EXPECTED_SPACING_DHW_MM
        or tuple(spatial.get("roi_voxel_shape", ())) != ROI_SHAPE_DHW
        or spatial.get("reference_visit") != "T0"
        or spatial.get("roi_source_visits") != ["T0"]
        or spatial.get("registration_type") != "rigid"
        or spatial.get("deformable_registration") is not False
    ):
        raise FrozenFeatureContractError(
            "mask spatial provenance is incompatible with the governed ROI grid"
        )
    return {
        "t0": t0,
        "t3": t3,
        "t0_bed": bed,
        "valid_support": valid_support,
        "spacing_dhw_mm": spacing,
        "spatial_provenance": spatial,
    }


def weighted_patch_mean(features: Tensor, weights: Tensor) -> Tensor:
    """Occupancy-weighted mean of a ``[F,7,7,7]`` grid."""

    if features.ndim != 4 or tuple(features.shape[-3:]) != FEATURE_GRID_SHAPE_DHW:
        raise FrozenFeatureContractError("features must have shape [F,7,7,7]")
    if tuple(weights.shape) != FEATURE_GRID_SHAPE_DHW:
        raise FrozenFeatureContractError("weights must have shape [7,7,7]")
    if not features.is_floating_point() or not weights.is_floating_point():
        raise FrozenFeatureContractError("features and weights must be floating point")
    weights = weights.to(dtype=torch.float32, device=features.device)
    if not bool(torch.isfinite(weights).all()) or torch.any((weights < 0) | (weights > 1)):
        raise FrozenFeatureContractError("weights must be finite occupancies in [0,1]")
    denominator = weights.sum()
    if float(denominator) <= 0:
        raise FrozenFeatureContractError("weighted patch region is empty")
    return (features.float() * weights.unsqueeze(0)).sum(dim=(1, 2, 3)) / denominator


def feature_views(
    features: Tensor, roi_weights: Tensor | None = None
) -> dict[str, Tensor]:
    """Return global/spatial summaries and optional T0-defined local summaries.

    A patient can legitimately lack a non-empty core or ring after intersecting
    T0/T1 acquisition-valid support.  Global and coarse-spatial frozen views
    remain usable in that case; callers must treat the local view as a
    separately eligible cohort rather than replacing it with a zero vector.
    """

    if roi_weights is not None and tuple(roi_weights.shape) != (
        2,
        *FEATURE_GRID_SHAPE_DHW,
    ):
        raise FrozenFeatureContractError("roi_weights must have shape [2,7,7,7]")
    flattened = features.float().reshape(features.shape[0], -1)
    global_mean = flattened.mean(dim=1)
    global_std = flattened.std(dim=1, unbiased=False)
    # A deterministic coarse pyramid retains token arrangement while remaining
    # small enough for patient-level classical probes.  This is explicitly not
    # claimed to be equivalent to the complete 7-cube grid.
    pyramid = F.adaptive_avg_pool3d(
        features.float().unsqueeze(0), output_size=(3, 3, 3)
    )[0].permute(1, 2, 3, 0).reshape(27, features.shape[0])
    result = {
        "global_mean": global_mean,
        "global_mean_std": torch.cat((global_mean, global_std)),
        "spatial_pyramid_3x3x3": pyramid,
    }
    if roi_weights is not None:
        core = weighted_patch_mean(features, roi_weights[0])
        ring = weighted_patch_mean(features, roi_weights[1])
        result.update(
            {
                "core_mean": core,
                "ring_mean": ring,
                "core_ring_mean": torch.cat((core, ring)),
            }
        )
    return result


def occupancy_geometry(weights: Tensor) -> Tensor:
    """Small mask-only negative-control vector on the 7-cube feature grid."""

    if tuple(weights.shape) != (2, *FEATURE_GRID_SHAPE_DHW):
        raise FrozenFeatureContractError("roi_weights must have shape [2,7,7,7]")
    coordinates = torch.stack(
        torch.meshgrid(
            *(torch.arange(size, dtype=torch.float32) for size in FEATURE_GRID_SHAPE_DHW),
            indexing="ij",
        ),
        dim=-1,
    ).reshape(-1, 3)
    values: list[Tensor] = []
    for region in weights.float():
        flat = region.reshape(-1)
        total = flat.sum()
        if float(total) <= 0:
            raise FrozenFeatureContractError("occupancy region is empty")
        centroid = (flat.unsqueeze(1) * coordinates).sum(dim=0) / total
        spread = torch.sqrt(
            (flat.unsqueeze(1) * (coordinates - centroid).square()).sum(dim=0)
            / total
        )
        values.extend((total.unsqueeze(0), centroid, spread, region.max().unsqueeze(0)))
    return torch.cat(values)


def _component_count(mask: Tensor) -> int:
    """Count 6-connected components, using SciPy when present and a sparse fallback."""

    array = mask.detach().to(device="cpu").numpy().astype(bool, copy=False)
    try:
        import numpy as np
        from scipy.ndimage import label

        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, :] = 1
        structure[1, :, 1] = 1
        structure[:, 1, 1] = 1
        return int(label(array, structure=structure)[1])
    except ImportError:
        occupied = {tuple(int(item) for item in row) for row in torch.nonzero(mask, as_tuple=False)}
        components = 0
        while occupied:
            components += 1
            start = occupied.pop()
            queue: deque[tuple[int, int, int]] = deque((start,))
            while queue:
                d, h, w = queue.popleft()
                for neighbor in (
                    (d - 1, h, w), (d + 1, h, w),
                    (d, h - 1, w), (d, h + 1, w),
                    (d, h, w - 1), (d, h, w + 1),
                ):
                    if neighbor in occupied:
                        occupied.remove(neighbor)
                        queue.append(neighbor)
        return components


def morphology_descriptors(
    mask: Tensor,
    *,
    spacing_dhw_mm: Sequence[float] = EXPECTED_SPACING_DHW_MM,
) -> dict[str, float]:
    """Describe FTV-inclusion support; this is not a pathology segmentation label.

    ``maximum_axis_extent_mm`` is the maximum axis-aligned bounding-box extent, not
    a true Feret diameter.  Empty support is represented by zeros so pCR/complete
    response cases remain valid outcomes.
    """

    if mask.dtype != torch.bool:
        raise FrozenFeatureContractError("morphology mask must be boolean")
    if tuple(mask.shape) == (1, *ROI_SHAPE_DHW):
        mask = mask[0]
    if tuple(mask.shape) != ROI_SHAPE_DHW:
        raise FrozenFeatureContractError("morphology mask has the wrong shape")
    spacing = tuple(float(item) for item in spacing_dhw_mm)
    if len(spacing) != 3 or any(not math.isfinite(item) or item <= 0 for item in spacing):
        raise FrozenFeatureContractError("spacing must contain three positive values")
    count = int(mask.sum())
    if count == 0:
        return {
            "inclusion_volume_ml": 0.0,
            "surface_area_mm2": 0.0,
            "sphericity": 0.0,
            "maximum_axis_extent_mm": 0.0,
            "extent_d_mm": 0.0,
            "extent_h_mm": 0.0,
            "extent_w_mm": 0.0,
            "connected_components_6": 0.0,
        }
    volume_mm3 = count * math.prod(spacing)
    surface = 0.0
    face_areas = (
        spacing[1] * spacing[2],
        spacing[0] * spacing[2],
        spacing[0] * spacing[1],
    )
    for axis, face_area in enumerate(face_areas):
        boundary = int(mask.select(axis, 0).sum()) + int(mask.select(axis, mask.shape[axis] - 1).sum())
        left = mask.narrow(axis, 0, mask.shape[axis] - 1)
        right = mask.narrow(axis, 1, mask.shape[axis] - 1)
        transitions = int(torch.logical_xor(left, right).sum())
        surface += (boundary + transitions) * face_area
    coordinates = torch.nonzero(mask, as_tuple=False)
    extents = [
        (int(coordinates[:, axis].max()) - int(coordinates[:, axis].min()) + 1)
        * spacing[axis]
        for axis in range(3)
    ]
    sphericity = (
        math.pi ** (1.0 / 3.0) * (6.0 * volume_mm3) ** (2.0 / 3.0) / surface
        if surface > 0
        else 0.0
    )
    return {
        "inclusion_volume_ml": volume_mm3 / 1000.0,
        "surface_area_mm2": float(surface),
        "sphericity": float(sphericity),
        "maximum_axis_extent_mm": float(max(extents)),
        "extent_d_mm": float(extents[0]),
        "extent_h_mm": float(extents[1]),
        "extent_w_mm": float(extents[2]),
        "connected_components_6": float(_component_count(mask)),
    }


__all__ = [
    "EXPECTED_SPACING_DHW_MM",
    "FEATURE_GRID_SHAPE_DHW",
    "FrozenFeatureContractError",
    "feature_views",
    "load_feature_grid",
    "load_mask_sidecar",
    "morphology_descriptors",
    "occupancy_geometry",
    "sha256_file",
    "valid_sha256",
    "weighted_patch_mean",
]
