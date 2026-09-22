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


# --------------------------------------------------------------------------- #
# Multi-disease tests
# --------------------------------------------------------------------------- #
from src.diseases import REGISTRY  # noqa: E402


def test_every_disease_generates_cleans_and_trains():
    for key, cfg in REGISTRY.items():
        raw, _ = load_data(cfg, n_synthetic=700, seed=3)
        df, rep = clean(cfg, raw)
        assert rep["rows_out"] > 500, key
        assert 0.02 < rep["prevalence"] < 0.8, (key, rep["prevalence"])
        assert rep["features_missing_from_data"] == [], key
        X, y = df[cfg.feature_names], df[cfg.target]
        pipe = build_pipeline(cfg, X, LogisticRegression(max_iter=2000)).fit(X, y)
        p = pipe.predict_proba(X.head(5))[:, 1]
        assert ((0 <= p) & (p <= 1)).all(), key
        # every widget default must be a valid raw value the pipeline can score
        defaults = pd.DataFrame([{f.name: f.default for f in cfg.features}])
        assert 0 <= pipe.predict_proba(defaults)[0, 1] <= 1, key


def test_every_disease_default_values_are_inside_ui_and_valid_ranges():
    for key, cfg in REGISTRY.items():
        for f in cfg.features:
            if f.kind == "number":
                assert f.min <= f.default <= f.max, (key, f.name)
                if f.valid_range:
                    assert f.valid_range[0] <= f.default <= f.valid_range[1], (key, f.name)
            else:
                assert int(f.default) in f.options, (key, f.name)


def test_every_disease_explanations_are_additive():
    for key, cfg in REGISTRY.items():
        raw, _ = load_data(cfg, n_synthetic=600, seed=5)
        df, _ = clean(cfg, raw)
        X, y = df[cfg.feature_names], df[cfg.target]
        pipe = build_pipeline(cfg, X, LogisticRegression(max_iter=2000)).fit(X, y)
        ex = RiskExplainer(pipe, cfg.feature_names, X, n_background=20, n_paths=150, seed=0)
        rows = X.dropna().head(3)
        e = ex.explain(rows)
        assert np.abs(e.base + e.values.sum(axis=1) - pipe.predict_proba(rows)[:, 1]).max() < 0.03, key


def test_liver_ilpd_label_is_1_disease_2_healthy():
    ilpd = pd.DataFrame({"Age": [65, 30, 50], "Gender": ["Female", "Male", "Male"],
                         "Total_Bilirubin": [0.7, 1.0, 5.0], "Direct_Bilirubin": [0.1, 0.2, 2.0],
                         "Alkaline_Phosphotase": [187, 150, 300], "Alamine_Aminotransferase": [16, 25, 80],
                         "Aspartate_Aminotransferase": [18, 30, 90], "Total_Protiens": [6.8, 7.0, 6.0],
                         "Albumin": [3.3, 3.8, 2.9], "Albumin_and_Globulin_Ratio": [0.9, 1.1, 0.8],
                         "Dataset": [1, 2, 1]})
    cfg = REGISTRY["liver"]
    out = normalize_columns(cfg, ilpd)
    assert out["target"].tolist() == [1.0, 0.0, 1.0]
    assert out["sex"].tolist() == [0, 1, 1]
    assert {"alk_phos", "alt", "ast", "total_protein", "ag_ratio"} <= set(out.columns)


def test_lung_survey_yes_no_coded_1_2_and_text_labels():
    survey = pd.DataFrame({"GENDER": ["M", "F"], "AGE": [69, 60], "SMOKING": [1, 2],
                           "YELLOW_FINGERS": [2, 1], "COUGHING": [2, 1], "LUNG_CANCER": ["YES", "NO"]})
    out = normalize_columns(REGISTRY["lung"], survey)
    assert out["smoking"].tolist() == [0, 1]           # 1 = NO, 2 = YES
    assert out["coughing"].tolist() == [1, 0]
    assert out["sex"].tolist() == [1, 0]
    assert out["target"].tolist() == [1.0, 0.0]


def test_kidney_uci_messy_text_values():
    ckd = pd.DataFrame({"age": [48, 62], "bp": [80, 80], "sg": [1.02, 1.01], "al": [1, 4],
                        "bgr": ["121", "?"], "sc": [1.2, "\t3.8"], "htn": ["yes", "\tno"],
                        "dm": ["\tyes", " yes"], "classification": ["ckd", "ckd\t"]})
    cfg = REGISTRY["kidney"]
    out = normalize_columns(cfg, ckd)
    assert out["hypertension"].tolist() == [1, 0] and out["diabetes"].tolist() == [1, 1]
    assert out["target"].tolist() == [1.0, 1.0]
    cleaned, _ = clean(cfg, out)
    assert cleaned["blood_glucose"].isna().sum() == 1     # '?' -> NaN


def test_stroke_kaggle_text_smoking_and_missing_bmi():
    kg = pd.DataFrame({"id": [1, 2, 3], "gender": ["Male", "Female", "Other"], "age": [67, 61, 80],
                       "hypertension": [0, 0, 1], "heart_disease": [1, 0, 0],
                       "avg_glucose_level": [228.7, 202.2, 105.9], "bmi": [36.6, "N/A", 32.5],
                       "smoking_status": ["formerly smoked", "never smoked", "Unknown"],
                       "stroke": [1, 1, 0]})
    out = normalize_columns(REGISTRY["stroke"], kg)
    assert out["smoking_status"].tolist()[:2] == [1, 0] and np.isnan(out["smoking_status"].iloc[2])
    assert out["target"].tolist() == [1.0, 1.0, 0.0]


def test_diabetes_pima_zeros_become_missing():
    pima = pd.DataFrame({"Pregnancies": [6, 1], "Glucose": [148, 0], "BloodPressure": [72, 66],
                         "SkinThickness": [35, 0], "Insulin": [0, 94], "BMI": [33.6, 26.6],
                         "DiabetesPedigreeFunction": [0.627, 0.351], "Age": [50, 31], "Outcome": [1, 0]})
    cfg = REGISTRY["diabetes"]
    cleaned, rep = clean(cfg, normalize_columns(cfg, pima))
    assert np.isnan(cleaned.loc[1, "glucose"]) and np.isnan(cleaned.loc[0, "insulin"])
    assert np.isnan(cleaned.loc[1, "skin_thickness"])


def test_class_weights_only_for_imbalanced_outcomes():
    from src.models import get_models
    assert get_models(0, 0.06)["Logistic Regression"].class_weight == "balanced"
    assert get_models(0, 0.45)["Logistic Regression"].class_weight is None


def test_explainer_baseline_is_deterministic_and_matches_background_mean():
    """The 'population average' shown to users must not jitter between clicks."""
    df, _ = clean(CFG, load_data(CFG, n_synthetic=600, seed=2)[0])
    X, y = df[CFG.feature_names], df["target"]
    pipe = build_pipeline(CFG, X, LogisticRegression(max_iter=1000)).fit(X, y)
    ex = RiskExplainer(pipe, CFG.feature_names, X, n_background=25, n_paths=200, seed=0)
    row = X.dropna().head(1)
    bases = [ex.explain(row).base[0] for _ in range(4)]
    assert max(bases) - min(bases) < 1e-9
    assert bases[0] == pytest.approx(ex._f(ex.bg).mean())
