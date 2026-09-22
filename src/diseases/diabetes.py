"""Type-2 diabetes risk. Schema follows the Pima Indians Diabetes dataset.

NB: in the real Pima data a value of 0 for glucose / blood pressure / skin thickness /
insulin / BMI really means "not measured". Those zeros fall outside `valid_range`, so the
cleaning step turns them into NaN automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec
from ._common import add_messiness, flag, ordinal, sigmoid, solve_intercept

FEATURES = (
    FeatureSpec("age", "Age", "number", default=35, min=18, max=100, step=1, unit="years",
                valid_range=(18, 100)),
    FeatureSpec("pregnancies", "Pregnancies", "number", default=1, min=0, max=17, step=1,
                valid_range=(0, 20), help="Number of times pregnant (0 if not applicable)."),
    FeatureSpec("glucose", "Plasma glucose", "number", default=110, min=40, max=300, step=1,
                unit="mg/dL", valid_range=(40, 300), help="2-hour oral glucose tolerance test."),
    FeatureSpec("blood_pressure", "Diastolic blood pressure", "number", default=72, min=30,
                max=140, step=1, unit="mmHg", valid_range=(30, 140)),
    FeatureSpec("skin_thickness", "Triceps skin-fold thickness", "number", default=25, min=3,
                max=100, step=1, unit="mm", valid_range=(3, 100)),
    FeatureSpec("insulin", "2-hour serum insulin", "number", default=90, min=5, max=900, step=1,
                unit="µU/mL", valid_range=(5, 900)),
    FeatureSpec("bmi", "BMI", "number", default=30.0, min=12.0, max=70.0, step=0.1,
                unit="kg/m²", valid_range=(12, 70)),
    FeatureSpec("pedigree", "Diabetes pedigree function", "number", default=0.4, min=0.05,
                max=3.0, step=0.01, valid_range=(0.05, 3.0), decimals=2,
                help="Score of genetic predisposition from family history (higher = stronger)."),
)

ALIASES = {
    "bloodpressure": "blood_pressure", "skinthickness": "skin_thickness",
    "diabetespedigreefunction": "pedigree", "outcome": "target", "diabetes": "target",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"glucose", "insulin"} <= set(df.columns):
        df["insulin_glucose_index"] = df["glucose"] * df["insulin"] / 405   # HOMA-IR-style proxy
    if "glucose" in df.columns:
        df["glucose_class"] = ordinal(df["glucose"], [100, 126, 200])       # normal/pre/diabetic/severe
    if "bmi" in df.columns:
        df["bmi_class"] = ordinal(df["bmi"], [18.5, 25, 30, 35])
    if "blood_pressure" in df.columns:
        df["bp_high"] = flag(df["blood_pressure"] >= 90, df["blood_pressure"])
    return df


def synthesize(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(21 + rng.gamma(2.2, 6.5, n), 21, 81).round()
    bmi = np.clip(rng.normal(32, 6.5, n), 18, 60).round(1)
    preg = rng.poisson(np.clip(0.2 * (age - 18), 0.3, 10))
    pedigree = np.clip(np.exp(rng.normal(-0.9, 0.65, n)), 0.08, 2.4).round(2)

    z = 0.03 * (age - 33) + 0.08 * (bmi - 32) + 0.10 * preg + 1.0 * pedigree + rng.normal(0, 0.8, n)
    disease = rng.binomial(1, sigmoid(z + solve_intercept(z, 0.35)))

    glucose = np.clip(105 + 8 * (bmi - 32) / 6.5 + 0.3 * (age - 33) + 30 * disease
                      + rng.normal(0, 22, n), 44, 199).round()
    insulin = np.clip(np.exp(4.4 + 0.02 * (bmi - 32) + 0.35 * disease + rng.normal(0, 0.6, n)),
                      14, 600).round()
    bp = np.clip(rng.normal(69 + 0.15 * (age - 33) + 0.4 * (bmi - 32), 11), 40, 122).round()
    skin = np.clip(20 + 0.9 * (bmi - 32) + rng.normal(0, 8, n), 7, 60).round()

    df = pd.DataFrame({"age": age, "pregnancies": preg, "glucose": glucose,
                       "blood_pressure": bp, "skin_thickness": skin, "insulin": insulin,
                       "bmi": bmi, "pedigree": pedigree, "target": disease})
    # Pima-style: "0" is used for 'not measured' - a lot of it for insulin and skin-fold.
    return add_messiness(df, rng,
                         impossible={"insulin": (0, 0.25), "skin_thickness": (0, 0.20),
                                     "blood_pressure": (0, 0.03), "glucose": (0, 0.006),
                                     "bmi": (0, 0.015)}, seed=seed)


DIABETES = DiseaseConfig(
    key="diabetes", title="Diabetes", icon="🩸", target="target", features=FEATURES,
    aliases=ALIASES, synthesize=synthesize, engineer=engineer, engineered_binary=("bp_high",),
    thresholds=(0.30, 0.60),
    description="Estimates the risk of diabetes from glucose, insulin, BMI, blood pressure and family history.",
    engineered_desc="insulin-glucose index (HOMA-IR-style), glucose class, BMI class, high-blood-pressure flag",
    dataset_hint="Pima Indians Diabetes - diabetes.csv (8 features + Outcome)",
)
