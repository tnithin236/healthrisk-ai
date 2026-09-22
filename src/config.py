"""Dataclasses that describe one disease-prediction task.

Everything disease-specific (features, valid ranges, UI widgets, synthetic data,
feature engineering) lives in a `DiseaseConfig`. The rest of the code base is
generic, so adding a new disease means adding one file under `src/diseases/`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    name: str                     # canonical column name
    label: str                    # human-readable label for the UI
    kind: str                     # "number" | "binary" | "category"
    default: float = 0
    min: Optional[float] = None   # UI bounds (numbers only)
    max: Optional[float] = None
    step: Optional[float] = None  # int step -> integer widget, float step -> float widget
    unit: str = ""
    options: Optional[Mapping[int, str]] = None  # code -> label (binary / category)
    help: str = ""
    valid_range: Optional[tuple] = None  # physiologically plausible range, used in cleaning
    decimals: int = 1                    # display precision for float features

    def format_value(self, value) -> str:
        """Pretty-print a raw value, e.g. `145 mmHg` or `Smoker`."""
        if self.options is not None:
            return str(self.options.get(int(value), value))
        if isinstance(self.step, float):
            text = f"{float(value):.{self.decimals}f}"
        else:
            text = f"{float(value):.0f}"
        return f"{text} {self.unit}".strip()


@dataclass(frozen=True)
class DiseaseConfig:
    key: str
    title: str
    icon: str
    target: str                                  # 1 = disease present
    features: tuple                              # tuple[FeatureSpec, ...]
    aliases: Mapping[str, str]                   # external column name -> canonical name
    synthesize: Callable[..., pd.DataFrame]      # (n, seed) -> DataFrame
    engineer: Callable[[pd.DataFrame], pd.DataFrame]
    engineered_binary: tuple = ()                # engineered 0/1 columns
    thresholds: tuple = (0.33, 0.66)             # Low < t0 <= Medium < t1 <= High
    description: str = ""                        # one-liner shown under the page title
    engineered_desc: str = ""                    # text for the About tab
    dataset_hint: str = ""                       # which public dataset to use for real training
    synthetic_rows: int = 2500                   # rows of demo data (rarer diseases need more)
    target_map: Optional[Callable] = None        # Series -> 0/1 Series (default: value > 0)
    dedupe: bool = True                          # drop exact duplicate rows (off for mostly-binary data)

    @property
    def feature_names(self) -> list:
        return [f.name for f in self.features]

    def feature(self, name: str) -> FeatureSpec:
        return next(f for f in self.features if f.name == name)
