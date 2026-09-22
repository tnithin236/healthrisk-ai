"""Model-agnostic explanations in terms of the RAW inputs a user types in.

Primary backend: SHAP's permutation explainer (Shapley values).
Fallback (if `shap` isn't installed or errors): our own Monte-Carlo Shapley
estimator - same idea, so the dashboard keeps working either way.

Explaining the whole pipeline's `predict_proba` (rather than the inner tree/linear
model) means contributions are attributed to features like "Cholesterol" or
"Age", not to one-hot columns like "chest_pain_3", and it works for every model
in the zoo including SVM and KNN.

Contributions are in probability units: base + sum(contributions) = P(disease).
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Explanation:
    values: np.ndarray          # (n_rows, n_features) contribution to P(disease)
    base: np.ndarray            # (n_rows,) expected probability over the background
    feature_names: list
    backend: str


class RiskExplainer:
    def __init__(self, model, feature_names, background: pd.DataFrame, n_background: int = 30,
                 max_evals: int = 300, n_paths: int = 200, prefer_shap: bool = True, seed: int = 0):
        self.model = model
        self.features = list(feature_names)
        self.max_evals = max_evals
        self.n_paths = n_paths
        self.rng = np.random.default_rng(seed)

        # Reference ("background") rows. A small *random* subset can badly misrepresent the
        # population (its mean risk drifts, biasing every attribution), so pick rows evenly
        # spaced by predicted risk: a compact set whose mean risk matches the full pool.
        pool = background[self.features]
        pool = pool.fillna(pool.median(numeric_only=True))
        order = np.argsort(self._f(pool.to_numpy(dtype=float)))
        k = min(n_background, len(pool))
        keep = order[np.linspace(0, len(order) - 1, k).round().astype(int)]
        self.bg = pool.to_numpy(dtype=float)[keep]

        self.backend = "sampling"
        self._shap = None
        if prefer_shap:
            try:
                import shap  # noqa: WPS433 (optional dependency)
                masker = shap.maskers.Independent(self.bg, max_samples=len(self.bg))
                self._shap = shap.Explainer(self._f, masker, algorithm="permutation",
                                            feature_names=self.features, seed=seed)
                self.backend = "shap"
            except Exception as exc:  # ImportError or version mismatch
                warnings.warn(f"SHAP unavailable ({exc!r}); using built-in Shapley sampling.")

    # ------------------------------------------------------------------ #
    def _f(self, X) -> np.ndarray:
        """P(disease) for an array of raw feature rows."""
        frame = pd.DataFrame(np.asarray(X, dtype=float), columns=self.features)
        return self.model.predict_proba(frame)[:, 1]

    def explain(self, X: pd.DataFrame, max_evals: int | None = None) -> Explanation:
        arr = X[self.features].to_numpy(dtype=float)
        if self._shap is not None:
            try:
                sv = self._shap(arr, max_evals=max_evals or self.max_evals, silent=True)
                return Explanation(np.asarray(sv.values), np.asarray(sv.base_values).reshape(-1),
                                   self.features, "shap")
            except Exception as exc:
                warnings.warn(f"SHAP failed at explain time ({exc!r}); falling back to sampling.")
                self._shap, self.backend = None, "sampling"
        values, base = self._sampling(arr)
        return Explanation(values, base, self.features, "sampling")

    # ------------------------------------------------------------------ #
    def _sampling(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo Shapley: walk from a background row to x, switching
        features on in a random order, and credit each feature with the change in
        predicted probability it caused. Averaged over many (row, order) paths."""
        n_feat = len(self.features)
        # Use every background row equally often -> the baseline is EXACTLY mean(f(background)),
        # identical on every call (random starting rows made it jump by several points).
        reps = max(1, round(self.n_paths / len(self.bg)))
        starts = np.repeat(self.bg, reps, axis=0)
        P = len(starts)
        out = np.zeros((len(X), n_feat))
        base = np.zeros(len(X))
        idx = np.arange(P)
        for i, x in enumerate(X):
            perms = np.argsort(self.rng.random((P, n_feat)), axis=1)
            cur = starts.copy()
            path = np.empty((P, n_feat + 1, n_feat))
            path[:, 0, :] = cur
            for k in range(n_feat):
                cols = perms[:, k]
                cur[idx, cols] = x[cols]
                path[:, k + 1, :] = cur
            p = self._f(path.reshape(-1, n_feat)).reshape(P, n_feat + 1)
            contrib = np.zeros((P, n_feat))
            np.put_along_axis(contrib, perms, np.diff(p, axis=1), axis=1)
            out[i], base[i] = contrib.mean(axis=0), p[:, 0].mean()
        return out, base
