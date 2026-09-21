"""Heart-disease risk task: feature schema, synthetic demo data, feature engineering.

Column conventions follow the classic UCI Heart Disease (Cleveland) dataset where
possible, so a real CSV can be dropped in via `python train.py --data heart.csv`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec

CHEST_PAIN = {0: "Typical angina", 1: "Atypical angina", 2: "Non-anginal pain", 3: "Asymptomatic"}

FEATURES = (
    FeatureSpec("age", "Age", "number", default=54, min=18, max=100, step=1,
                unit="years", valid_range=(18, 100)),
    FeatureSpec("sex", "Sex", "binary", default=1, options={0: "Female", 1: "Male"}),
    FeatureSpec("chest_pain", "Chest pain type", "category", default=2, options=CHEST_PAIN,
                help="Counter-intuitive but real: in the UCI Heart Disease data, patients with NO "
                     "chest pain ('Asymptomatic') were more often diagnosed with heart disease "
                     "(silent ischaemia), so the model treats it as a risk signal."),
    FeatureSpec("resting_bp", "Resting blood pressure", "number", default=130, min=80, max=220,
                step=1, unit="mmHg", valid_range=(60, 250), help="Systolic pressure at rest."),
    FeatureSpec("cholesterol", "Serum cholesterol", "number", default=215, min=100, max=600,
                step=1, unit="mg/dL", valid_range=(80, 600)),
    FeatureSpec("fasting_bs", "Fasting blood sugar", "binary", default=0,
                options={0: "≤ 120 mg/dL", 1: "> 120 mg/dL"}),
    FeatureSpec("max_hr", "Maximum heart rate", "number", default=150, min=60, max=220, step=1,
                unit="bpm", valid_range=(60, 220), help="Peak heart rate reached during exercise test."),
    FeatureSpec("exercise_angina", "Exercise-induced angina", "binary", default=0,
                options={0: "No", 1: "Yes"}),
    FeatureSpec("bmi", "BMI", "number", default=27.0, min=12.0, max=60.0, step=0.1,
                unit="kg/m²", valid_range=(12, 60)),
    FeatureSpec("smoking", "Smoking status", "binary", default=0,
                options={0: "Non-smoker", 1: "Current smoker"}),
)

# Column names seen in public datasets -> canonical names used here.
ALIASES = {
    "trestbps": "resting_bp", "restbp": "resting_bp", "blood_pressure": "resting_bp",
    "chol": "cholesterol",
    "cp": "chest_pain", "chest_pain_type": "chest_pain",
    "fbs": "fasting_bs", "fasting_blood_sugar": "fasting_bs",
    "thalach": "max_hr", "thalch": "max_hr", "max_heart_rate": "max_hr",
    "exang": "exercise_angina", "exercise_induced_angina": "exercise_angina",
    "gender": "sex", "smoker": "smoking",
    "num": "target", "output": "target", "heart_disease": "target", "hd": "target",
}


# --------------------------------------------------------------------------- #
# Feature engineering (runs inside the sklearn Pipeline, so it is applied
# identically at training time and at prediction time).
# --------------------------------------------------------------------------- #
def _ordinal(s: pd.Series, bins) -> pd.Series:
    """Bucket a numeric series into 0..len(bins), keeping NaN as NaN."""
    codes = np.digitize(s.to_numpy(dtype=float), bins)
    return pd.Series(codes, index=s.index, dtype=float).where(s.notna())


def _flag(cond: pd.Series, source: pd.Series) -> pd.Series:
    return cond.astype(float).where(source.notna())


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated features. Tolerates missing columns and NaNs."""
    df = df.copy()
    if {"max_hr", "age"} <= set(df.columns):
        predicted_max = (220 - df["age"]).clip(lower=1)
        df["hr_reserve_pct"] = df["max_hr"] / predicted_max        # % of age-predicted max HR
    if "resting_bp" in df.columns:
        df["bp_stage"] = _ordinal(df["resting_bp"], [120, 130, 140])  # normal/elevated/stage1/stage2
    if "bmi" in df.columns:
        df["bmi_class"] = _ordinal(df["bmi"], [18.5, 25, 30, 35])
    if "cholesterol" in df.columns:
        df["chol_high"] = _flag(df["cholesterol"] >= 240, df["cholesterol"])

    parts = [df[c].fillna(0) for c in ("fasting_bs", "smoking", "chol_high") if c in df.columns]
    if "bp_stage" in df.columns:
        parts.append((df["bp_stage"] >= 2).astype(float))
    if "bmi" in df.columns:
        parts.append((df["bmi"] >= 30).astype(float))
    if parts:
        df["risk_factor_count"] = sum(parts)
    return df


# --------------------------------------------------------------------------- #
# Synthetic demo data
# --------------------------------------------------------------------------- #
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def synthesize(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    """Generate a realistic-looking (but entirely fake) heart-disease dataset.

    Risk factors -> latent risk -> disease -> symptoms (chest pain, exercise
    angina, lower peak heart rate). Adds missing values, impossible values and a
    few duplicates so the cleaning step has real work to do.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(54, 9.5, n), 29, 80).round()
    sex = rng.binomial(1, 0.62, n)
    bmi = np.clip(rng.normal(26.5 + 0.03 * (age - 54), 4.3, n), 16, 48).round(1)
    smoking = rng.binomial(1, np.where(sex == 1, 0.30, 0.18))
    bp = np.clip(124 + 0.45 * (age - 50) + 0.7 * (bmi - 26) + rng.normal(0, 15, n), 90, 200).round()
    chol = np.clip(212 + 0.7 * (age - 50) + rng.normal(0, 42, n), 110, 420).round()
    fbs = rng.binomial(1, _sigmoid(-2.0 + 0.03 * (age - 50) + 0.08 * (bmi - 27)))

    z = (0.045 * (age - 54) + 0.6 * sex + 0.022 * (bp - 130) + 0.006 * (chol - 215)
         + 0.7 * fbs + 0.04 * (bmi - 26.5) + 0.6 * smoking + rng.normal(0, 0.6, n))
    lo, hi = -10.0, 10.0                       # bisection: hit ~45% prevalence
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _sigmoid(z + mid).mean() < 0.45 else (lo, mid)
    disease = rng.binomial(1, _sigmoid(z + (lo + hi) / 2))

    p_cp = np.where(disease[:, None] == 1, [[.15, .08, .17, .60]], [[.10, .27, .38, .25]])
    chest_pain = np.minimum((rng.random(n)[:, None] > p_cp.cumsum(axis=1)).sum(axis=1), 3)
    exang = rng.binomial(1, np.where(disease == 1, 0.55, 0.14))
    max_hr = np.clip(208 - 0.75 * age - 0.25 * (bmi - 26.5) - 14 * disease + rng.normal(0, 15, n),
                     72, 202).round()

    df = pd.DataFrame({
        "age": age, "sex": sex, "chest_pain": chest_pain, "resting_bp": bp,
        "cholesterol": chol, "fasting_bs": fbs, "max_hr": max_hr,
        "exercise_angina": exang, "bmi": bmi, "smoking": smoking, "target": disease,
    })

    # Real-world messiness -----------------------------------------------------
    for col in ("cholesterol", "resting_bp", "bmi", "max_hr"):
        df.loc[rng.random(n) < 0.02, col] = np.nan            # ~2% missing
    df.loc[rng.random(n) < 0.005, "cholesterol"] = 0           # impossible values
    df.loc[rng.random(n) < 0.003, "resting_bp"] = 0
    dupes = df.sample(n=max(5, n // 150), random_state=seed)   # a few exact duplicates
    return pd.concat([df, dupes], ignore_index=True)


HEART = DiseaseConfig(
    key="heart",
    title="Heart Disease",
    icon="❤️",
    target="target",
    features=FEATURES,
    aliases=ALIASES,
    synthesize=synthesize,
    engineer=engineer,
    engineered_binary=("chol_high",),
    thresholds=(0.33, 0.66),
)
