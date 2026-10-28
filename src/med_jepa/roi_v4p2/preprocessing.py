"""Frozen input interventions evaluated before any v4.2 training."""

from __future__ import annotations

import itertools
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor

from med_jepa.roi_v4.audit_mechanistic import REGIONS


INPUT_SHAPE = (112, 112, 112)
PATCH_SIZE = 16
SHIFT_VOXELS = 8
SYMMETRIC_PAD = SHIFT_VOXELS // 2
LOCAL_CROP_SIZES = (80, 64)


def voxel_regions(mask: Mapping[str, Any]) -> Tensor:
    valid = (
        mask["valid_support"]["T0"]
        & mask["valid_support"]["T1"]
        & mask["valid_support"]["T3"]
    )
    core = mask["t0"] & valid
    ring = mask["t0_bed"] & ~mask["t0"] & valid
    disease = core | ring
    background = ~mask["t0_bed"] & valid
    regions = torch.cat((core, ring, disease, background, valid), dim=0).float()
    if regions.shape != (len(REGIONS), *INPUT_SHAPE):
        raise ValueError("context mask does not match the canonical ROI")
    return regions


def token_weights(regions: Tensor, patch_size: int = PATCH_SIZE) -> Tensor:
    if regions.ndim != 4 or any(
        size % int(patch_size) for size in regions.shape[-3:]
    ):
        raise ValueError("voxel regions are incompatible with patch pooling")
    return F.avg_pool3d(
        regions.float()[None], kernel_size=patch_size, stride=patch_size
    )[0].contiguous()


def symmetric_shifted_views(volume: Tensor) -> list[Tensor]:
    """Return eight half-patch-separated views around a symmetric center."""

    if volume.ndim != 4 or tuple(volume.shape[-3:]) != INPUT_SHAPE:
        raise ValueError("shifted-window input must be [C,112,112,112]")
    padded = F.pad(
        volume,
        (
            SYMMETRIC_PAD,
            SYMMETRIC_PAD,
            SYMMETRIC_PAD,
            SYMMETRIC_PAD,
            SYMMETRIC_PAD,
            SYMMETRIC_PAD,
        ),
    )
    result: list[Tensor] = []
    for parity in itertools.product((0, 1), repeat=3):
        starts = [SHIFT_VOXELS * value for value in parity]
        z, y, x = starts
        result.append(
            padded[:, z : z + 112, y : y + 112, x : x + 112].contiguous()
        )
    return result


def interleave_shifted_feature_views(features: Sequence[Tensor]) -> Tensor:
    if len(features) != 8 or any(
        value.ndim != 4 or tuple(value.shape[-3:]) != (7, 7, 7)
        for value in features
    ):
        raise ValueError("eight [C,7,7,7] shifted feature maps are required")
    channels = int(features[0].shape[0])
    if any(int(value.shape[0]) != channels for value in features):
        raise ValueError("shifted feature channels do not agree")
    dense = torch.empty(channels, 14, 14, 14, dtype=features[0].dtype)
    for feature, parity in zip(
        features, itertools.product((0, 1), repeat=3), strict=True
    ):
        dense[:, parity[0] :: 2, parity[1] :: 2, parity[2] :: 2] = feature
    return dense.contiguous()


def shifted_token_weights(regions: Tensor) -> Tensor:
    views = symmetric_shifted_views(regions)
    pooled = [token_weights(view) for view in views]
    return interleave_shifted_feature_views(pooled)


def mask_center(mask: Tensor) -> tuple[float, float, float]:
    if mask.shape != (1, *INPUT_SHAPE) or mask.dtype != torch.bool:
        raise ValueError("T0 anchor mask must be boolean [1,112,112,112]")
    coordinates = torch.nonzero(mask[0], as_tuple=False)
    if len(coordinates) == 0:
        raise ValueError("T0 anchor mask is empty")
    center = (coordinates.float().amin(0) + coordinates.float().amax(0)) / 2.0
    return tuple(float(value) for value in center)


def crop_resize(
    volume: Tensor,
    center: tuple[float, float, float],
    *,
    crop_size: int,
    output_shape: tuple[int, int, int] = INPUT_SHAPE,
    padding_value: float = 0.0,
) -> Tensor:
    if volume.ndim != 4 or tuple(volume.shape[-3:]) != INPUT_SHAPE:
        raise ValueError("local crop input must be [C,112,112,112]")
    if crop_size < 16 or crop_size > 112:
        raise ValueError("local crop size must be between 16 and 112 voxels")
    start = [int(round(value - (crop_size - 1) / 2.0)) for value in center]
    crop = torch.full(
        (volume.shape[0], crop_size, crop_size, crop_size),
        float(padding_value),
        dtype=volume.dtype,
    )
    source_slices = []
    target_slices = []
    for begin, extent in zip(start, INPUT_SHAPE, strict=True):
        source_begin = max(0, begin)
        source_end = min(extent, begin + crop_size)
        target_begin = source_begin - begin
        target_end = target_begin + (source_end - source_begin)
        source_slices.append(slice(source_begin, source_end))
        target_slices.append(slice(target_begin, target_end))
    crop[
        :,
        target_slices[0],
        target_slices[1],
        target_slices[2],
    ] = volume[
        :,
        source_slices[0],
        source_slices[1],
        source_slices[2],
    ]
    return F.interpolate(
        crop.float()[None],
        size=output_shape,
        mode="trilinear",
        align_corners=False,
    )[0].to(volume.dtype).contiguous()


def local_magnified_view(
    normalized_volume: Tensor,
    center: tuple[float, float, float],
    *,
    crop_size: int,
) -> Tensor:
    # Add one before resampling so physical zero padding remains normalized -1.
    return crop_resize(
        normalized_volume + 1.0,
        center,
        crop_size=crop_size,
        padding_value=0.0,
    ).sub(1.0).clamp(-1.0, 1.0)


def local_token_weights(
    regions: Tensor,
    center: tuple[float, float, float],
    *,
    crop_size: int,
) -> Tensor:
    transformed = crop_resize(
        regions,
        center,
        crop_size=crop_size,
        padding_value=0.0,
    ).clamp(0.0, 1.0)
    return token_weights(transformed)


def patient_visit_reference_align_ser(
    volumes: Tensor,
    valid_support: Tensor,
    t0_disease_bed: Tensor,
) -> tuple[Tensor, dict[str, Any]]:
    """Align visits with one patient-level robust scale-estimator hierarchy.

    IQR is used only when it is non-degenerate at every visit.  Otherwise the
    wider P90-P10 span is used at every visit.  If that also degenerates, all
    visits receive median-only alignment.  Mixing scale estimators within one
    trajectory is deliberately prohibited.
    """

    if volumes.shape != (3, 1, *INPUT_SHAPE):
        raise ValueError("SER trajectories must be [3,1,112,112,112]")
    if valid_support.shape != (3, 1, *INPUT_SHAPE):
        raise ValueError("visit-valid support must align to SER trajectories")
    if t0_disease_bed.shape != (1, *INPUT_SHAPE):
        raise ValueError("T0 disease bed must be [1,112,112,112]")
    valid_support = valid_support.bool()
    reference = valid_support & ~t0_disease_bed.bool()[None]
    medians: list[Tensor] = []
    iqrs: list[Tensor] = []
    p10_p90_spans: list[Tensor] = []
    for visit in range(3):
        values = volumes[visit][reference[visit]]
        if values.numel() < 512:
            raise ValueError("too few non-disease reference voxels for SER alignment")
        quantiles = torch.quantile(
            values.float(), torch.tensor([0.10, 0.25, 0.50, 0.75, 0.90])
        )
        iqr = quantiles[3] - quantiles[1]
        p10_p90 = quantiles[4] - quantiles[0]
        medians.append(quantiles[2])
        iqrs.append(iqr)
        p10_p90_spans.append(p10_p90)
    threshold = 1e-4
    if all(float(value) > threshold for value in iqrs):
        scales = iqrs
        estimator = "IQR_all_visits"
    elif all(float(value) > threshold for value in p10_p90_spans):
        scales = p10_p90_spans
        estimator = "P90_minus_P10_all_visits"
    else:
        scales = [torch.ones_like(medians[0]) for _ in range(3)]
        estimator = "median_only_due_to_degenerate_scale"
    aligned = torch.zeros_like(volumes, dtype=torch.float32)
    for visit in range(3):
        value = (
            (volumes[visit].float() - medians[visit])
            * (scales[0] / scales[visit])
            + medians[0]
        ).clamp(0.0, 3.0)
        aligned[visit] = torch.where(valid_support[visit], value, 0.0)
    return aligned, {
        "visit_reference_medians": [float(value) for value in medians],
        "visit_reference_iqrs": [float(value) for value in iqrs],
        "visit_reference_p10_p90_spans": [
            float(value) for value in p10_p90_spans
        ],
        "scale_estimator": estimator,
        "reference": "visit_valid_support_outside_t0_disease_bed",
    }


__all__ = [
    "LOCAL_CROP_SIZES",
    "crop_resize",
    "interleave_shifted_feature_views",
    "local_magnified_view",
    "local_token_weights",
    "mask_center",
    "patient_visit_reference_align_ser",
    "shifted_token_weights",
    "symmetric_shifted_views",
    "token_weights",
    "voxel_regions",
]
