"""Append-only, non-pixel Med-JEPA v4 research design."""

from med_jepa.roi_v4.contracts import (
    CURRENT_RELEASE_VISITS,
    FEATURE_SHAPE,
    ordered_pairs_and_triplets,
)
from med_jepa.roi_v4.model import DetachedFTVProbe, TokenDynamicsJEPA

__all__ = [
    "CURRENT_RELEASE_VISITS",
    "DetachedFTVProbe",
    "FEATURE_SHAPE",
    "TokenDynamicsJEPA",
    "ordered_pairs_and_triplets",
]
