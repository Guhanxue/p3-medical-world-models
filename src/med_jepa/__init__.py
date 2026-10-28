"""P³ and Cancer JEPA research code for longitudinal breast MRI forecasting."""

__version__ = "0.1.0"
__all__ = ["CancerJEPA"]


def __getattr__(name: str):
    if name == "CancerJEPA":
        from med_jepa.roi_v4p3p1.model import CoarseRRRAnchoredCoreResidualJEPA

        return CoarseRRRAnchoredCoreResidualJEPA
    raise AttributeError(name)
