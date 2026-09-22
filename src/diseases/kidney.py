"""Chronic kidney disease (CKD) risk. Schema follows the UCI Chronic Kidney Disease dataset."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import DiseaseConfig, FeatureSpec
from ._common import add_messiness, flag, sigmoid, solve_intercept

YN = {0: "No", 1: "Yes"}

FEATURES = (
    FeatureSpec("age", "Age", "number", default=50, min=18, max=100, step=1, unit="years",
                valid_range=(2, 100)),
    FeatureSpec("blood_pressure", "Blood pressure (diastolic)", "number", default=78, min=40,
                max=180, step=1, unit="mmHg", valid_range=(30, 200)),
    FeatureSpec("specific_gravity", "Urine specific gravity", "number", default=1.020, min=1.000,
                max=1.040, step=0.005, valid_range=(1.0, 1.04), decimals=3,
                help="Urine concentration; low values suggest the kidneys are not concentrating urine."),
    FeatureSpec("albumin", "Urine albumin (0-5 scale)", "number", default=0, min=0, max=5, step=1,
                valid_range=(0, 5), help="Dipstick protein level: 0 = none ... 5 = very high."),
    FeatureSpec("blood_glucose", "Random blood glucose", "number", default=115, min=30, max=500,
                step=1, unit="mg/dL", valid_range=(30, 600)),
    FeatureSpec("blood_urea", "Blood urea", "number", default=35, min=5, max=300, step=1,
                unit="mg/dL", valid_range=(5, 400)),
    FeatureSpec("serum_creatinine", "Serum creatinine", "number", default=1.0, min=0.2, max=25.0,
                step=0.1, unit="mg/dL", valid_range=(0.2, 30), help="Key kidney-function marker."),
    FeatureSpec("hemoglobin", "Hemoglobin", "number", default=14.0, min=3.0, max=18.0, step=0.1,
                unit="g/dL", valid_range=(3, 20)),
    FeatureSpec("hypertension", "Hypertension", "binary", default=0, options=YN),
    FeatureSpec("diabetes", "Diabetes mellitus", "binary", default=0, options=YN),
    FeatureSpec("anemia", "Anemia", "binary", default=0, options=YN),
    FeatureSpec("pedal_edema", "Pedal edema (ankle swelling)", "binary", default=0, options=YN),
)

ALIASES = {
    "bp": "blood_pressure", "sg": "specific_gravity", "al": "albumin", "bgr": "blood_glucose",
    "bu": "blood_urea", "sc": "serum_creatinine", "hemo": "hemoglobin", "htn": "hypertension",
    "dm": "diabetes", "ane": "anemia", "pe": "pedal_edema", "classification": "target",
    "class": "target", "ckd": "target",
}


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if {"serum_creatinine", "age"} <= set(df.columns):
        scr = df["serum_creatinine"].clip(lower=0.2)
        df["egfr_est"] = 175 * scr ** -1.154 * df["age"].clip(lower=1) ** -0.203   # MDRD (male form)
    if "hemoglobin" in df.columns:
        df["low_hemoglobin"] = flag(df["hemoglobin"] < 12, df["hemoglobin"])
    if "albumin" in df.columns:
        df["proteinuria"] = flag(df["albumin"] >= 1, df["albumin"])
    parts = [df[c].fillna(0) for c in ("hypertension", "diabetes", "anemia", "pedal_edema") if c in df.columns]
    if parts:
        df["comorbidity_count"] = sum(parts)
    return df


def synthesize(n: int = 2500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(50, 17, n), 18, 90).round()
    hyp = rng.binomial(1, sigmoid(-1.8 + 0.03 * (age - 50)))
    dm = rng.binomial(1, sigmoid(-2.2 + 0.03 * (age - 50)))

    z = 0.03 * (age - 50) + 1.0 * hyp + 0.9 * dm + rng.normal(0, 0.8, n)
    d = rng.binomial(1, sigmoid(z + solve_intercept(z, 0.40)))

    creat = np.clip(np.exp(np.where(d == 1, rng.normal(0.65, 0.8, n), rng.normal(-0.10, 0.22, n))), 0.4, 20).round(1)
    urea = np.clip(np.exp(np.where(d == 1, rng.normal(4.2, 0.55, n), rng.normal(3.4, 0.28, n))), 10, 300).round()
    hemo = np.clip(np.where(d == 1, rng.normal(10.8, 2.1, n), rng.normal(14.4, 1.3, n)), 3.5, 17.5).round(1)
    alb_p = np.where(d[:, None] == 1, [[.15, .15, .22, .22, .18, .08]], [[.90, .06, .03, .01, 0, 0]])
    from ._common import choose
    albumin = choose(rng, alb_p)
    sg = np.clip(rng.normal(1.021 - 0.008 * d, 0.004, n), 1.005, 1.030)
    sg = (np.round(sg / 0.005) * 0.005).round(3)
    glucose = np.clip(rng.normal(108 + 55 * dm + 10 * d, 28, n), 60, 490).round()
    bp = np.clip(rng.normal(72 + 8 * hyp + 4 * d, 10, n), 50, 140).round()
    anemia = np.where(rng.random(n) < 0.05, 1 - (hemo < 11.5), (hemo < 11.5)).astype(int)
    edema = rng.binomial(1, sigmoid(-2.8 + 2.4 * d))

    df = pd.DataFrame({"age": age, "blood_pressure": bp, "specific_gravity": sg, "albumin": albumin,
                       "blood_glucose": glucose, "blood_urea": urea, "serum_creatinine": creat,
                       "hemoglobin": hemo, "hypertension": hyp, "diabetes": dm, "anemia": anemia,
                       "pedal_edema": edema, "target": d})
    return add_messiness(df, rng,
                         missing={"blood_glucose": 0.10, "blood_urea": 0.05, "hemoglobin": 0.10,
                                  "specific_gravity": 0.07, "albumin": 0.06, "blood_pressure": 0.03,
                                  "serum_creatinine": 0.04},
                         impossible={"serum_creatinine": (0, 0.004)}, seed=seed)


KIDNEY = DiseaseConfig(
    key="kidney", title="Kidney Disease", icon="🫘", target="target", features=FEATURES,
    aliases=ALIASES, synthesize=synthesize, engineer=engineer,
    engineered_binary=("low_hemoglobin", "proteinuria"), thresholds=(0.30, 0.65),
    description="Estimates chronic-kidney-disease risk from blood and urine tests plus comorbidities.",
    engineered_desc="estimated GFR (MDRD), low-hemoglobin flag, proteinuria flag, comorbidity count",
    dataset_hint="UCI Chronic Kidney Disease (kidney_disease.csv). Messy text values ('\\tyes', '?') are handled.",
)
