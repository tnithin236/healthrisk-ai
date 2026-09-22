"""Stroke risk. Schema follows the Kaggle 'Healthcare Stroke Prediction' dataset.

Strokes are rare (~5% of rows), so this task is heavily imbalanced: the training code
switches on class weighting automatically, and risk bands are set relative to the low base rate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec
from ._common import add_messiness, choose, flag, ordinal, sigmoid, solve_intercept

SMOKING = {0: "Never smoked", 1: "Formerly smoked", 2: "Currently smokes"}

FEATURES = (
    FeatureSpec("age", "Age", "number", default=55, min=18, max=100, step=1, unit="years",
                valid_range=(0, 110)),
    FeatureSpec("sex", "Sex", "binary", default=0, options={0: "Female", 1: "Male"}),
    FeatureSpec("hypertension", "Hypertension", "binary", default=0, options={0: "No", 1: "Yes"},
                help="Diagnosed high blood pressure."),
    FeatureSpec("heart_disease", "Heart disease", "binary", default=0, options={0: "No", 1: "Yes"}),
    FeatureSpec("avg_glucose_level", "Average glucose", "number", default=105, min=50, max=300,
                step=1, unit="mg/dL", valid_range=(40, 400)),
    FeatureSpec("bmi", "BMI", "number", default=27.0, min=12.0, max=60.0, step=0.1, unit="kg/m²",
                valid_range=(10, 80)),
    FeatureSpec("smoking_status", "Smoking status", "category", default=0, options=SMOKING),
)

ALIASES = {"gender": "sex", "stroke": "target", "avg_glucose": "avg_glucose_level"}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" in df.columns:
        df["age_over_65"] = (df["age"] >= 65).astype(float).where(df["age"].notna())
        df["age_decade"] = (df["age"] // 10).astype(float)
    if "avg_glucose_level" in df.columns:
        df["glucose_high"] = flag(df["avg_glucose_level"] >= 140, df["avg_glucose_level"])
    if "bmi" in df.columns:
        df["bmi_class"] = ordinal(df["bmi"], [18.5, 25, 30, 35])
    parts = [df[c].fillna(0) for c in ("hypertension", "heart_disease", "glucose_high", "age_over_65")
             if c in df.columns]
    if parts:
        df["vascular_burden"] = sum(parts)
    return df


def synthesize(n: int = 6000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(48, 20, n), 18, 90).round()
    sex = rng.binomial(1, 0.42, n)
    bmi = np.clip(rng.normal(28 + 0.02 * (age - 48), 6.5, n), 14, 60).round(1)
    hyp = rng.binomial(1, sigmoid(-3.6 + 0.045 * (age - 50) + 0.06 * (bmi - 28)))
    heart = rng.binomial(1, sigmoid(-4.6 + 0.05 * (age - 50) + 0.4 * sex))
    diabetic = rng.random(n) < sigmoid(-2.3 + 0.03 * (age - 50) + 0.05 * (bmi - 28))
    glucose = np.clip(84 + 0.25 * (age - 48) + 0.4 * (bmi - 28) + rng.gamma(2, 8, n)
                      + 85 * diabetic * rng.random(n), 55, 300).round()
    p_smoke = np.column_stack([0.50 - 0.002 * (age - 48), np.full(n, 0.25), np.full(n, 0.25)])
    p_smoke[:, 0] = 1 - p_smoke[:, 1:].sum(axis=1)
    smoking = choose(rng, p_smoke)

    z = (0.075 * (age - 50) + 0.7 * hyp + 0.8 * heart + 0.005 * (glucose - 105)
         + 0.10 * (smoking == 1) + 0.25 * (smoking == 2) + 0.01 * (bmi - 28) + rng.normal(0, 0.5, n))
    stroke = rng.binomial(1, sigmoid(z + solve_intercept(z, 0.06)))

    df = pd.DataFrame({"age": age, "sex": sex, "hypertension": hyp, "heart_disease": heart,
                       "avg_glucose_level": glucose, "bmi": bmi,
                       "smoking_status": smoking, "target": stroke})
    return add_messiness(df, rng, missing={"bmi": 0.04, "smoking_status": 0.15},
                         impossible={"bmi": (0, 0.002)}, seed=seed)


STROKE = DiseaseConfig(
    key="stroke", title="Stroke", icon="🧠", target="target", features=FEATURES,
    aliases=ALIASES, synthesize=synthesize, engineer=engineer,
    engineered_binary=("age_over_65", "glucose_high"),
    thresholds=(0.05, 0.15), synthetic_rows=6000,
    description="Estimates stroke risk from age, blood pressure history, heart disease, glucose, BMI and smoking.",
    engineered_desc="age ≥ 65 flag, age decade, high-glucose flag, BMI class, vascular-burden score",
    dataset_hint="Kaggle 'Stroke Prediction Dataset' (healthcare-dataset-stroke-data.csv). "
                 "Only ~5% of rows are strokes; risk bands here are relative to that low base rate.",
)
