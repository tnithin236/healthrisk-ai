"""Lung-disease risk from smoking history, comorbidity and respiratory symptoms.

Schema follows the popular Kaggle 'Lung Cancer' survey dataset (YES/NO answers coded 1/2).
The data-loader converts that 1/2 coding to 0/1 automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec
from ._common import add_messiness, ordinal, sigmoid, solve_intercept

YN = {0: "No", 1: "Yes"}


def _yn(name, label, help=""):
    return FeatureSpec(name, label, "binary", default=0, options=YN, help=help)


FEATURES = (
    FeatureSpec("age", "Age", "number", default=55, min=18, max=100, step=1, unit="years",
                valid_range=(18, 100)),
    FeatureSpec("sex", "Sex", "binary", default=1, options={0: "Female", 1: "Male"}),
    _yn("smoking", "Smoker (current or former)"),
    _yn("yellow_fingers", "Yellow fingers", "Nicotine staining, a marker of heavy smoking."),
    _yn("chronic_disease", "Chronic disease", "Any diagnosed long-term condition (e.g. COPD, asthma)."),
    _yn("fatigue", "Persistent fatigue"),
    _yn("wheezing", "Wheezing"),
    _yn("alcohol", "Regular alcohol use"),
    _yn("coughing", "Persistent cough"),
    _yn("shortness_of_breath", "Shortness of breath"),
    _yn("swallowing_difficulty", "Difficulty swallowing"),
    _yn("chest_pain", "Chest pain"),
)

ALIASES = {"gender": "sex", "lung_cancer": "target", "alcohol_consuming": "alcohol",
           "allergy_": "allergy"}

SYMPTOMS = ("coughing", "wheezing", "shortness_of_breath", "chest_pain", "fatigue",
            "swallowing_difficulty")
RESPIRATORY = ("coughing", "wheezing", "shortness_of_breath")


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    sym = [c for c in SYMPTOMS if c in df.columns]
    resp = [c for c in RESPIRATORY if c in df.columns]
    if sym:
        df["symptom_count"] = df[sym].fillna(0).sum(axis=1)
    if resp:
        df["respiratory_count"] = df[resp].fillna(0).sum(axis=1)
        if "smoking" in df.columns:
            df["smoker_with_resp"] = ((df["smoking"].fillna(0) == 1) & (df["respiratory_count"] >= 2)).astype(float)
    if "age" in df.columns:
        df["age_band"] = ordinal(df["age"], [40, 50, 60, 70])
    return df


def synthesize(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(58, 12, n), 25, 88).round()
    sex = rng.binomial(1, 0.5, n)
    smoking = rng.binomial(1, np.where(sex == 1, 0.55, 0.42))
    alcohol = rng.binomial(1, np.where(sex == 1, 0.50, 0.30))
    chronic = rng.binomial(1, sigmoid(-0.6 + 0.02 * (age - 58) + 0.5 * smoking))

    z = 0.05 * (age - 58) + 1.1 * smoking + 0.5 * chronic + 0.3 * alcohol + 0.2 * sex + rng.normal(0, 0.6, n)
    d = rng.binomial(1, sigmoid(z + solve_intercept(z, 0.40)))

    b = lambda logit: rng.binomial(1, sigmoid(logit))
    df = pd.DataFrame({
        "age": age, "sex": sex, "smoking": smoking,
        "yellow_fingers": b(-1.5 + 1.8 * smoking + 0.5 * d),
        "chronic_disease": chronic,
        "fatigue": b(-0.9 + 1.0 * d),
        "wheezing": b(-1.6 + 1.6 * d + 0.5 * smoking),
        "alcohol": alcohol,
        "coughing": b(-1.4 + 2.0 * d + 0.6 * smoking),
        "shortness_of_breath": b(-1.3 + 1.6 * d),
        "swallowing_difficulty": b(-2.0 + 1.2 * d),
        "chest_pain": b(-1.4 + 1.3 * d),
        "target": d,
    })
    return add_messiness(df, rng, missing={"age": 0.01}, impossible={"age": (5, 0.004)}, dup_frac=0, seed=seed)


LUNG = DiseaseConfig(
    key="lung", title="Lung Disease", icon="🫁", target="target", features=FEATURES,
    aliases=ALIASES, synthesize=synthesize, engineer=engineer,
    engineered_binary=("smoker_with_resp",), thresholds=(0.33, 0.66),
    dedupe=False,   # 12 yes/no columns + integer age: identical rows are usually different patients
    description="Estimates lung-disease risk from smoking history, comorbidities and respiratory symptoms.",
    engineered_desc="symptom count, respiratory-symptom count, age band, smoker-with-respiratory-symptoms flag",
    dataset_hint="Kaggle 'Lung Cancer' survey dataset (survey_lung_cancer.csv). That sample is ~87% positive, "
                 "so its probabilities do not represent the general population.",
)
