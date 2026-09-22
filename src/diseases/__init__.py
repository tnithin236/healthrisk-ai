"""Registry of supported diseases. To add one, create a module with a
`DiseaseConfig` (see heart.py) and register it here."""
from ..config import DiseaseConfig
from .diabetes import DIABETES
from .heart import HEART
from .kidney import KIDNEY
from .liver import LIVER
from .lung import LUNG
from .stroke import STROKE

REGISTRY = {cfg.key: cfg for cfg in (HEART, DIABETES, STROKE, LUNG, KIDNEY, LIVER)}


def get_disease(key: str) -> DiseaseConfig:
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown disease '{key}'. Available: {sorted(REGISTRY)}") from None
