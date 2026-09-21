"""Feature engineering + preprocessing + feature selection + classifier, in ONE
sklearn Pipeline. Because every step lives inside the pipeline:
  * cross-validation never leaks information from validation folds, and
  * the app can feed raw user input straight into `model.predict_proba`.
"""
from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DiseaseConfig
from .diseases import get_disease


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Applies the disease-specific `engineer` function. Stateless."""

    def __init__(self, disease: str = "heart"):
        self.disease = disease

    def fit(self, X, y=None):
        self.columns_ = list(X.columns)
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X, columns=self.columns_)
        return get_disease(self.disease).engineer(X)


def column_groups(cfg: DiseaseConfig, engineered_columns: list) -> tuple[list, list, list]:
    """Split engineered columns into (numeric, categorical, binary)."""
    cols = set(engineered_columns)
    cat = [f.name for f in cfg.features if f.kind == "category" and f.name in cols]
    binary = [f.name for f in cfg.features if f.kind == "binary" and f.name in cols]
    binary += [c for c in cfg.engineered_binary if c in cols]
    numeric = [c for c in engineered_columns if c not in cat and c not in binary]
    return numeric, cat, binary


def build_pipeline(cfg: DiseaseConfig, X_sample: pd.DataFrame, classifier,
                   select: bool = True, seed: int = 42) -> Pipeline:
    """Assemble the full pipeline. `X_sample` is only used to discover column names."""
    engineered = list(FeatureEngineer(cfg.key).fit(X_sample).transform(X_sample.head(20)).columns)
    numeric, cat, binary = column_groups(cfg, engineered)

    prep = ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                       sparse_output=False))]), cat),
            ("bin", SimpleImputer(strategy="most_frequent"), binary),
        ],
        verbose_feature_names_out=False,
    )

    selector = (
        SelectFromModel(
            RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                   random_state=seed, n_jobs=-1),
            threshold="0.5*mean",     # drop features less than half as important as average
        )
        if select else "passthrough"
    )
    return Pipeline([
        ("engineer", FeatureEngineer(cfg.key)),
        ("prep", prep),
        ("select", selector),
        ("clf", classifier),
    ])


def selected_feature_names(pipe: Pipeline) -> list:
    """Names of the columns that actually reach the classifier."""
    names = pipe.named_steps["prep"].get_feature_names_out()
    sel = pipe.named_steps["select"]
    return list(names[sel.get_support()]) if hasattr(sel, "get_support") else list(names)
