"""Loading and cleaning. Works for the synthetic generator or a real CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DiseaseConfig

_TEXT_MAP = {
    "m": 1, "male": 1, "f": 0, "female": 0,
    "yes": 1, "y": 1, "true": 1, "no": 0, "n": 0, "false": 0,
    "presence": 1, "absence": 0, "disease": 1, "positive": 1, "negative": 0,
}


def _text_to_numeric(s: pd.Series) -> pd.Series:
    if s.dtype == object or str(s.dtype).startswith("string"):
        mapped = s.astype(str).str.strip().str.lower().map(_TEXT_MAP)
        s = mapped.where(mapped.notna(), pd.to_numeric(s, errors="coerce"))
    return pd.to_numeric(s, errors="coerce")


def normalize_columns(cfg: DiseaseConfig, df: pd.DataFrame) -> pd.DataFrame:
    """Rename known aliases, convert text codes, binarise the target."""
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
    rename = {k: v for k, v in cfg.aliases.items() if k in df.columns and v not in df.columns}
    df = df.rename(columns=rename)
    if "target" in df.columns and cfg.target != "target":
        df = df.rename(columns={"target": cfg.target})

    for spec in cfg.features:
        if spec.name in df.columns:
            df[spec.name] = _text_to_numeric(df[spec.name])
            # Some sources code categories 1..K instead of 0..K-1 (e.g. UCI chest pain 1-4).
            if spec.kind == "category" and spec.options:
                vals, keys = set(df[spec.name].dropna().unique()), set(spec.options)
                if vals - keys and vals <= {k + 1 for k in keys}:
                    df[spec.name] = df[spec.name] - 1

    if cfg.target in df.columns:
        t = _text_to_numeric(df[cfg.target])
        df[cfg.target] = (t > 0).astype(float).where(t.notna())   # UCI 'num' 0-4 -> 0/1
    return df


def load_data(cfg: DiseaseConfig, path: str | None = None, n_synthetic: int = 2500,
              seed: int = 42) -> tuple[pd.DataFrame, str]:
    """Return (raw dataframe, source label)."""
    if path:
        df = pd.read_csv(path)
        return normalize_columns(cfg, df), f"file:{Path(path).name}"
    return normalize_columns(cfg, cfg.synthesize(n_synthetic, seed)), "synthetic"


def clean(cfg: DiseaseConfig, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Keep known columns, coerce types, drop bad rows, null out impossible values."""
    if cfg.target not in df.columns:
        raise ValueError(f"Target column '{cfg.target}' not found. Columns: {list(df.columns)}")

    report: dict = {"rows_in": int(len(df))}
    features = [f for f in cfg.feature_names if f in df.columns]
    report["features_used"] = features
    report["features_missing_from_data"] = [f for f in cfg.feature_names if f not in df.columns]
    if not features:
        raise ValueError("None of the expected feature columns were found in the data.")

    df = df[features + [cfg.target]].copy()
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    n = len(df)
    df = df.dropna(subset=[cfg.target])
    report["dropped_missing_target"] = int(n - len(df))

    n = len(df)
    df = df.drop_duplicates()
    report["dropped_duplicates"] = int(n - len(df))

    out_of_range = {}
    for spec in cfg.features:
        if spec.name in df.columns and spec.valid_range:
            lo, hi = spec.valid_range
            bad = df[spec.name].notna() & ((df[spec.name] < lo) | (df[spec.name] > hi))
            if bad.any():
                out_of_range[spec.name] = int(bad.sum())
                df.loc[bad, spec.name] = np.nan
    report["impossible_values_set_to_nan"] = out_of_range

    df[cfg.target] = df[cfg.target].astype(int)
    report["missing_values_remaining"] = {k: int(v) for k, v in df.isna().sum().items() if v}
    report["rows_out"] = int(len(df))
    report["prevalence"] = round(float(df[cfg.target].mean()), 4)
    return df.reset_index(drop=True), report
