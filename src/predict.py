"""Inference helpers shared by the Streamlit app, tests and notebooks."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .diseases import REGISTRY, get_disease
from .explain import RiskExplainer

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"


def model_path(disease: str) -> Path:
    return MODEL_DIR / f"{disease}_model.joblib"


def report_dir(disease: str) -> Path:
    return REPORT_DIR / disease


def available_diseases() -> list:
    return [k for k in REGISTRY if model_path(k).exists()]


def load_bundle(disease: str) -> dict:
    return joblib.load(model_path(disease))


def build_local_explainer(bundle: dict) -> RiskExplainer:
    return RiskExplainer(bundle["model"], bundle["raw_features"], bundle["background"],
                         n_background=30, max_evals=300, n_paths=200)


def risk_level(prob: float, thresholds) -> str:
    t0, t1 = thresholds
    return "Low" if prob < t0 else "Medium" if prob < t1 else "High"


def predict_one(bundle: dict, explainer: RiskExplainer, values: dict) -> dict:
    """Predict risk for one patient given raw feature values, plus an explanation."""
    X = pd.DataFrame([values])[bundle["raw_features"]].astype(float)
    prob = float(bundle["model"].predict_proba(X)[0, 1])
    ex = explainer.explain(X)
    return {
        "probability": prob,
        "level": risk_level(prob, bundle["thresholds"]),
        "contributions": pd.Series(ex.values[0], index=ex.feature_names),
        "base": float(ex.base[0]),
        "backend": ex.backend,
    }


def top_factors(result: dict, cfg, values: dict, k: int = 3) -> list:
    """Top-k features by absolute contribution as (label, value_text, points, direction)."""
    c = result["contributions"]
    out = []
    for name in c.abs().sort_values(ascending=False).index[:k]:
        spec = cfg.feature(name)
        pts = c[name] * 100
        out.append((spec.label, spec.format_value(values[name]), pts,
                    "raises" if pts > 0 else "lowers"))
    return out
