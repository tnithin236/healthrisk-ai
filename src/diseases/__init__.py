"""Registry of supported diseases. To add one, create a module with a
`DiseaseConfig` (see heart.py) and register it here."""
from ..config import DiseaseConfig
from .heart import HEART

REGISTRY = {cfg.key: cfg for cfg in (HEART,)}


def get_disease(key: str) -> DiseaseConfig:
    try:
        return REGISTRY[key]
    except KeyError:
        raise ValueError(f"Unknown disease '{key}'. Available: {sorted(REGISTRY)}") from None
