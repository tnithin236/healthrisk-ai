"""Liver-disease risk. Schema follows the Indian Liver Patient Dataset (ILPD).

In ILPD the label column 'Dataset' is 1 = liver patient and 2 = NOT a liver patient,
so this module supplies its own `target_map`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec
from ._common import add_messiness, flag, sigmoid, solve_intercept

FEATURES = (
    FeatureSpec("age", "Age", "number", default=45, min=18, max=100, step=1, unit="years",
                valid_range=(0, 110)),
    FeatureSpec("sex", "Sex", "binary", default=1, options={0: "Female", 1: "Male"}),
    FeatureSpec("alcohol_use", "Regular alcohol use", "binary", default=0, options={0: "No", 1: "Yes"}),
    FeatureSpec("total_bilirubin", "Total bilirubin", "number", default=1.0, min=0.1, max=80.0,
                step=0.1, unit="mg/dL", valid_range=(0.1, 80)),
    FeatureSpec("direct_bilirubin", "Direct bilirubin", "number", default=0.3, min=0.05, max=30.0,
                step=0.1, unit="mg/dL", valid_range=(0.05, 30)),
    FeatureSpec("alk_phos", "Alkaline phosphatase (ALP)", "number", default=190, min=30, max=2500,
                step=1, unit="U/L", valid_range=(30, 2500)),
    FeatureSpec("alt", "ALT (SGPT)", "number", default=30, min=5, max=3000, step=1, unit="U/L",
                valid_range=(5, 3000)),
    FeatureSpec("ast", "AST (SGOT)", "number", default=32, min=5, max=5000, step=1, unit="U/L",
                valid_range=(5, 5000)),
    FeatureSpec("total_protein", "Total protein", "number", default=6.6, min=2.0, max=11.0,
                step=0.1, unit="g/dL", valid_range=(2, 11)),
    FeatureSpec("albumin", "Albumin", "number", default=3.5, min=0.5, max=6.0, step=0.1,
                unit="g/dL", valid_range=(0.5, 6)),
    FeatureSpec("ag_ratio", "Albumin / globulin ratio", "number", default=1.0, min=0.1, max=3.5,
                step=0.05, valid_range=(0.1, 3.5), decimals=2),
)

ALIASES = {
    "gender": "sex", "alcohol": "alcohol_use",
    "alkaline_phosphotase": "alk_phos", "alkaline_phosphatase": "alk_phos",
    "alamine_aminotransferase": "alt", "sgpt": "alt",
    "aspartate_aminotransferase": "ast", "sgot": "ast",
    "total_protiens": "total_protein", "albumin_and_globulin_ratio": "ag_ratio",
    "dataset": "target", "liver_disease": "target",
}


def _target_map(t: pd.Series) -> pd.Series:
    return (t == 1).astype(float).where(t.notna())      # ILPD: 1 = disease, 2 = healthy


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"ast", "alt"} <= set(df.columns):
        df["ast_alt_ratio"] = df["ast"] / df["alt"].clip(lower=1)          # De Ritis ratio
    if {"direct_bilirubin", "total_bilirubin"} <= set(df.columns):
        df["direct_fraction"] = df["direct_bilirubin"] / df["total_bilirubin"].clip(lower=0.1)
    for col in ("alt", "ast", "alk_phos", "total_bilirubin"):              # heavy right tails
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    if "albumin" in df.columns:
        df["albumin_low"] = flag(df["albumin"] < 3.5, df["albumin"])
    limits = {"alt": 40, "ast": 40, "alk_phos": 130, "total_bilirubin": 1.2}
    parts = [(df[c] > lim).astype(float) for c, lim in limits.items() if c in df.columns]
    if parts:
        df["abnormal_lab_count"] = sum(parts)
    return df


def synthesize(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(44, 16, n), 18, 85).round()
    sex = rng.binomial(1, 0.75, n)
    alcohol = rng.binomial(1, np.where(sex == 1, 0.38, 0.10))
    z = 0.015 * (age - 44) + 0.4 * alcohol + 0.3 * sex + rng.normal(0, 0.6, n)
    d = rng.binomial(1, sigmoid(z + solve_intercept(z, 0.45)))

    tb = np.clip(np.exp(rng.normal(-0.05 + 0.50 * d, 0.85)), 0.2, 75)
    db = np.minimum(np.clip(tb * (0.22 + 0.10 * d) * np.exp(rng.normal(0, 0.40, n)), 0.1, 30), tb)
    alk = np.clip(np.exp(rng.normal(5.25 + 0.16 * d, 0.48)), 60, 2100)
    alt = np.clip(np.exp(rng.normal(3.3 + 0.45 * d + 0.25 * alcohol, 0.80)), 10, 2000)
    ast = np.clip(np.exp(rng.normal(3.4 + 0.50 * d + 0.30 * alcohol, 0.80)), 10, 4900)
    tp = np.clip(rng.normal(6.6 - 0.15 * d, 0.85), 2.7, 9.6)
    alb = np.clip(rng.normal(3.5 - 0.25 * d, 0.65), 0.9, 5.5)
    ag = np.clip(alb / np.maximum(tp - alb, 0.5) * np.exp(rng.normal(0, 0.08, n)), 0.3, 2.8)

    df = pd.DataFrame({"age": age, "sex": sex, "alcohol_use": alcohol,
                       "total_bilirubin": tb.round(1), "direct_bilirubin": db.round(1),
                       "alk_phos": alk.round(), "alt": alt.round(), "ast": ast.round(),
                       "total_protein": tp.round(1), "albumin": alb.round(1),
                       "ag_ratio": ag.round(2), "target": d})
    return add_messiness(df, rng, missing={"ag_ratio": 0.01, "albumin": 0.01, "alt": 0.01},
                         impossible={"alt": (0, 0.003)}, seed=seed)


LIVER = DiseaseConfig(
    key="liver", title="Liver Disease", icon="🫀", target="target", features=FEATURES,
    aliases=ALIASES, synthesize=synthesize, engineer=engineer,
    engineered_binary=("albumin_low",), thresholds=(0.33, 0.66), target_map=_target_map,
    description="Estimates liver-disease risk from liver-function blood tests and alcohol use.",
    engineered_desc="AST/ALT ratio, direct-bilirubin fraction, log-transformed enzymes, low-albumin flag, abnormal-lab count",
    dataset_hint="Indian Liver Patient Dataset (ILPD, indian_liver_patient.csv). Its label is 1 = disease, "
                 "2 = healthy; this is handled automatically.",
)
