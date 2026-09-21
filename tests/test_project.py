"""Fast tests (< 30 s). Run with:  pytest -q"""
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from src.data import clean, load_data, normalize_columns
from src.diseases import get_disease
from src.explain import RiskExplainer
from src.pipeline import build_pipeline
from src.predict import risk_level

warnings.filterwarnings("ignore")
CFG = get_disease("heart")


@pytest.fixture(scope="module")
def data():
    raw, _ = load_data(CFG, n_synthetic=800, seed=1)
    df, report = clean(CFG, raw)
    return df, report


def test_cleaning_removes_duplicates_and_impossible_values(data):
    df, report = data
    assert report["dropped_duplicates"] > 0
    assert df["cholesterol"].dropna().min() >= 80
    assert df["resting_bp"].dropna().min() >= 60
    assert not df.duplicated().any()
    assert set(df["target"].unique()) <= {0, 1}


def test_uci_style_columns_are_normalised():
    uci = pd.DataFrame({"Age": [63, 40], "Sex": [1, 0], "cp": [1, 4], "trestbps": [145, 120],
                        "chol": [233, 200], "fbs": [1, 0], "thalach": [150, 170],
                        "exang": [0, 1], "num": [0, 3]})
    out = normalize_columns(CFG, uci)
    assert out["chest_pain"].tolist() == [0, 3]          # 1-4 shifted to 0-3
    assert out["target"].tolist() == [0.0, 1.0]           # 0-4 severity -> binary
    assert {"resting_bp", "cholesterol", "max_hr", "exercise_angina"} <= set(out.columns)


def test_feature_engineering_tolerates_nan_and_missing_columns():
    X = pd.DataFrame({"age": [50, np.nan], "max_hr": [150, 140], "resting_bp": [np.nan, 150],
                      "cholesterol": [250, np.nan]})          # no bmi / smoking columns
    out = CFG.engineer(X)
    assert np.isnan(out.loc[0, "bp_stage"]) and out.loc[1, "bp_stage"] == 3
    assert out.loc[0, "chol_high"] == 1 and np.isnan(out.loc[1, "chol_high"])
    assert out.loc[0, "hr_reserve_pct"] == pytest.approx(150 / 170)


def test_pipeline_accepts_raw_rows_with_nans(data):
    df, _ = data
    X, y = df[CFG.feature_names], df["target"]
    pipe = build_pipeline(CFG, X, LogisticRegression(max_iter=1000)).fit(X, y)
    row = X.head(3).copy()
    row.iloc[0, 3] = np.nan                                   # missing value at inference time
    p = pipe.predict_proba(row)[:, 1]
    assert p.shape == (3,) and ((0 <= p) & (p <= 1)).all()


def test_explanations_add_up_to_the_prediction(data):
    """base + sum(contributions) must equal P(disease) - the defining Shapley property."""
    df, _ = data
    X, y = df[CFG.feature_names], df["target"]
    pipe = build_pipeline(CFG, X, LogisticRegression(max_iter=1000)).fit(X, y)
    ex = RiskExplainer(pipe, CFG.feature_names, X, n_background=20, n_paths=150, seed=0)
    rows = X.dropna().head(5)
    e = ex.explain(rows)
    pred = pipe.predict_proba(rows)[:, 1]
    assert np.abs(e.base + e.values.sum(axis=1) - pred).max() < 0.03


def test_explanation_direction_is_sensible(data):
    """A very high-risk profile should get positive contributions from the obvious drivers."""
    df, _ = data
    X, y = df[CFG.feature_names], df["target"]
    pipe = build_pipeline(CFG, X, LogisticRegression(max_iter=1000)).fit(X, y)
    ex = RiskExplainer(pipe, CFG.feature_names, X, n_background=30, n_paths=300, seed=0)
    risky = pd.DataFrame([dict(age=65, sex=1, chest_pain=3, resting_bp=170, cholesterol=300,
                               fasting_bs=1, max_hr=110, exercise_angina=1, bmi=34, smoking=1)])
    e = ex.explain(risky)
    contrib = dict(zip(e.feature_names, e.values[0]))
    assert contrib["exercise_angina"] > 0 and contrib["max_hr"] > 0 and contrib["chest_pain"] > 0
    assert pipe.predict_proba(risky)[0, 1] > 0.8


def test_risk_levels():
    th = CFG.thresholds
    assert [risk_level(p, th) for p in (0.18, 0.46, 0.82)] == ["Low", "Medium", "High"]
