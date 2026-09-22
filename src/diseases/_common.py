"""Small helpers shared by the disease modules."""
from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def solve_intercept(z: np.ndarray, prevalence: float) -> float:
    """Bisection for b such that mean(sigmoid(z + b)) == prevalence."""
    lo, hi = -15.0, 15.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if sigmoid(z + mid).mean() < prevalence else (lo, mid)
    return (lo + hi) / 2


def ordinal(s: pd.Series, bins) -> pd.Series:
    """Bucket a numeric series into 0..len(bins), keeping NaN as NaN."""
    codes = np.digitize(s.to_numpy(dtype=float), bins)
    return pd.Series(codes, index=s.index, dtype=float).where(s.notna())


def flag(cond: pd.Series, source: pd.Series) -> pd.Series:
    """0/1 flag that stays NaN wherever the source value is missing."""
    return cond.astype(float).where(source.notna())


def add_messiness(df: pd.DataFrame, rng, *, missing: dict | None = None,
                  impossible: dict | None = None, dup_frac: float = 0.006, seed: int = 0):
    """Make synthetic data behave like real data so cleaning has work to do.

    missing:    {col: rate}            -> that fraction becomes NaN
    impossible: {col: (value, rate)}   -> that fraction gets an impossible value (e.g. 0)
    """
    df = df.copy()
    n = len(df)
    for col, rate in (missing or {}).items():
        df.loc[rng.random(n) < rate, col] = np.nan
    for col, (value, rate) in (impossible or {}).items():
        df.loc[rng.random(n) < rate, col] = value
    if dup_frac <= 0:
        return df.reset_index(drop=True)
    dupes = df.sample(n=max(5, int(n * dup_frac)), random_state=seed)
    return pd.concat([df, dupes], ignore_index=True)


def choose(rng, probs: np.ndarray) -> np.ndarray:
    """Vectorised categorical draw. probs: (n, k) rows sum to 1 -> ints in 0..k-1."""
    return np.minimum((rng.random(len(probs))[:, None] > probs.cumsum(axis=1)).sum(axis=1),
                      probs.shape[1] - 1)
