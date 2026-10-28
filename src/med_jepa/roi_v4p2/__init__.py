"""Med-JEPA v4.2 shrinkage-anchored residual dynamics research program."""

from med_jepa.roi_v4p2.config import V42TrainingConfig
from med_jepa.roi_v4p2.model import ShrinkageAnchoredTokenDynamicsJEPA

__all__ = ["ShrinkageAnchoredTokenDynamicsJEPA", "V42TrainingConfig"]
