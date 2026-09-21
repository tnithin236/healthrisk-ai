"""HealthRisk AI - interactive dashboard.   Run with:  streamlit run app.py"""
from __future__ import annotations

import json

import matplotlib
import pandas as pd
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.diseases import get_disease  # noqa: E402
from src.plots import contribution_figure, gauge_figure  # noqa: E402
from src.predict import (available_diseases, build_local_explainer, load_bundle,  # noqa: E402
                         predict_one, report_dir, top_factors)

st.set_page_config(page_title="HealthRisk AI", page_icon="🩺", layout="wide")

LEVEL_STYLE = {
    "Low": ("#2e9e5b", "#e6f5ec"),
    "Medium": ("#c77d00", "#fff3dc"),
    "High": ("#d64545", "#fde8e8"),
}


@st.cache_resource(show_spinner="Loading model …")
def load_assets(disease: str):
    bundle = load_bundle(disease)
    return bundle, build_local_explainer(bundle)


def input_widget(spec, key_prefix: str):
    """Render the right Streamlit widget for a FeatureSpec and return its value."""
    key = f"{key_prefix}_{spec.name}"
    label = f"{spec.label} ({spec.unit})" if spec.unit else spec.label
    if spec.kind == "number":
        if isinstance(spec.step, float):
            return st.number_input(label, float(spec.min), float(spec.max), float(spec.default),
                                   float(spec.step), help=spec.help or None, key=key)
        return st.number_input(label, int(spec.min), int(spec.max), int(spec.default),
                               int(spec.step or 1), help=spec.help or None, key=key)
    codes = list(spec.options)
    return st.selectbox(label, codes, index=codes.index(int(spec.default)),
                        format_func=lambda c: spec.options[c], help=spec.help or None, key=key)


def risk_card(level: str, prob: float):
    fg, bg = LEVEL_STYLE[level]
    st.markdown(
        f"""<div style="background:{bg};border-left:8px solid {fg};padding:14px 18px;border-radius:8px;">
        <div style="font-size:0.9rem;color:#555;">Estimated risk</div>
        <div style="font-size:2rem;font-weight:700;color:{fg};line-height:1.2;">{level.upper()} &nbsp;·&nbsp; {prob:.0%}</div>
        </div>""",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Sidebar: choose disease / model info
# --------------------------------------------------------------------------- #
diseases = available_diseases()
if not diseases:
    st.title("🩺 HealthRisk AI")
    st.error("No trained model found. Run `python train.py` first, then reload this page.")
    st.stop()

with st.sidebar:
    st.title("🩺 HealthRisk AI")
    disease = st.selectbox("Condition", diseases,
                           format_func=lambda k: f"{get_disease(k).icon} {get_disease(k).title}")
    cfg = get_disease(disease)
    bundle, explainer = load_assets(disease)
    st.caption("Model")
    st.write(f"**{bundle['best_model']}** (probability-calibrated)")
    m = bundle["test_metrics"]
    c1, c2 = st.columns(2)
    c1.metric("ROC-AUC", f"{m['roc_auc']:.3f}")
    c2.metric("Recall", f"{m['recall']:.2f}")
    st.caption(f"Trained on {bundle['n_rows']:,} rows · source: {bundle['data_source']}")
    st.caption(f"Explanations: {'SHAP' if explainer.backend == 'shap' else 'Shapley sampling'}")
    st.divider()
    st.caption("⚕️ Educational project. Not a medical device and not a substitute for professional "
               "medical advice, diagnosis or treatment.")

if bundle["data_source"] == "synthetic":
    st.warning("**Demo mode:** this model was trained on *synthetic* data, so the numbers below are "
               "illustrative only. Retrain on a real dataset with `python train.py --data your.csv`.")

st.title(f"{cfg.icon} {cfg.title} Risk Predictor")
tab_predict, tab_models, tab_explain, tab_data, tab_about = st.tabs(
    ["🩺 Predict", "📊 Model comparison", "🔍 Global explainability", "📈 Data explorer", "ℹ️ About"])

# --------------------------------------------------------------------------- #
# Predict
# --------------------------------------------------------------------------- #
with tab_predict:
    left, right = st.columns([1, 1.3], gap="large")
    with left:
        st.subheader("Patient details")
        with st.form("patient_form"):
            values = {}
            specs = [cfg.feature(f) for f in bundle["raw_features"]]
            grid = st.columns(2)
            for i, spec in enumerate(specs):
                with grid[i % 2]:
                    values[spec.name] = input_widget(spec, disease)
            submitted = st.form_submit_button("Predict risk", type="primary")

    with right:
        st.subheader("Result")
        if not submitted:
            st.info("Enter the patient's values and press **Predict risk**.")
        else:
            with st.spinner("Computing risk and explanation …"):
                result = predict_one(bundle, explainer, values)
            prob, level = result["probability"], result["level"]

            risk_card(level, prob)
            fig = gauge_figure(prob, bundle["thresholds"])
            st.pyplot(fig)
            plt.close(fig)
            t0, t1 = bundle["thresholds"]
            st.caption(f"Low < {t0:.0%} ≤ Medium < {t1:.0%} ≤ High · population average ≈ "
                       f"{result['base']:.0%}")

            st.markdown("**Top factors**")
            for label, value_text, pts, direction in top_factors(result, cfg, values, k=3):
                arrow = "🔺" if pts > 0 else "🔻"
                st.markdown(f"{arrow} **{label}** ({value_text}) {direction} risk by "
                            f"**{abs(pts):.1f} points**")

            labels = {s.name: f"{s.label} = {s.format_value(values[s.name])}" for s in specs}
            fig = contribution_figure(result["contributions"], labels,
                                      result["base"] * 100, prob * 100)
            st.pyplot(fig)
            plt.close(fig)
            st.caption("Red bars push risk up, green bars pull it down, relative to an average "
                       "patient. Effects add up to the difference between this patient's risk and "
                       "the average. They describe what the *model* relies on, not medical causation.")

# --------------------------------------------------------------------------- #
# Model comparison
# --------------------------------------------------------------------------- #
rep = report_dir(disease)
with tab_models:
    st.subheader("Model comparison")
    st.caption("Models are ranked by 5-fold cross-validated ROC-AUC on the training split. The "
               "hold-out test columns are computed once on data no model saw during selection.")
    comp = bundle["comparison"]
    show = comp[["model", "cv_roc_auc", "cv_roc_auc_std", "test_accuracy", "test_precision",
                 "test_recall", "test_f1", "test_roc_auc", "test_brier"]].copy()
    show.columns = ["Model", "CV ROC-AUC", "± std", "Accuracy", "Precision", "Recall", "F1",
                    "Test ROC-AUC", "Brier ↓"]
    st.dataframe(show.round(3), hide_index=True)
    st.bar_chart(comp.set_index("model")[["cv_roc_auc", "test_roc_auc"]]
                 .rename(columns={"cv_roc_auc": "CV ROC-AUC", "test_roc_auc": "Test ROC-AUC"}))
    a, b, c = st.columns(3)
    for col, name, cap in [(a, "roc_curves.png", "ROC curves"),
                           (b, "calibration.png", "Calibration"),
                           (c, "confusion_matrix.png", "Confusion matrix (best model)")]:
        if (rep / name).exists():
            col.image(str(rep / name), caption=cap)
    st.info("In a screening setting a missed case (false negative) is usually costlier than a false "
            "alarm, so the decision threshold would normally be tuned for recall rather than left at 0.5.")

# --------------------------------------------------------------------------- #
# Global explainability
# --------------------------------------------------------------------------- #
with tab_explain:
    st.subheader("What drives the model overall?")
    if (rep / "global_importance.png").exists():
        st.image(str(rep / "global_importance.png"))
    st.caption("Mean absolute effect of each feature on predicted risk across a sample of patients. "
               "Features removed by automatic feature selection show ~0 effect.")

# --------------------------------------------------------------------------- #
# Data explorer
# --------------------------------------------------------------------------- #
with tab_data:
    st.subheader("Exploratory data analysis")
    cr = bundle["cleaning_report"]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Rows (clean)", f"{cr['rows_out']:,}")
    k2.metric("Disease prevalence", f"{cr['prevalence']:.1%}")
    k3.metric("Duplicates removed", cr["dropped_duplicates"])
    k4.metric("Impossible values nulled", sum(cr["impossible_values_set_to_nan"].values()))
    with st.expander("Full cleaning report"):
        st.code(json.dumps(cr, indent=2), language="json")
    for name, cap in [("eda_distributions.png", "Distributions by outcome"),
                      ("eda_categorical.png", "Prevalence by category"),
                      ("eda_correlation.png", "Correlations")]:
        if (rep / name).exists():
            st.image(str(rep / name), caption=cap)

# --------------------------------------------------------------------------- #
# About
# --------------------------------------------------------------------------- #
with tab_about:
    st.subheader("How it works")
    st.code(
        "Patient data → Cleaning → EDA → Feature engineering → Feature selection\n"
        "            → Model training & comparison → Calibration → Risk prediction\n"
        "            → SHAP explanation → Streamlit dashboard", language="text")
    st.markdown(
        "- **Cleaning:** duplicates dropped, physiologically impossible values set to missing, then "
        "median / most-frequent imputation *inside* the pipeline (no leakage).\n"
        "- **Feature engineering:** heart-rate reserve, blood-pressure stage, BMI class, high-cholesterol "
        "flag, risk-factor count.\n"
        "- **Feature selection:** random-forest importance threshold.\n"
        "- **Probabilities:** sigmoid-calibrated, so 70% means roughly 70% of similar patients had the condition.\n"
        "- **Explainability:** Shapley values computed on the *raw inputs* through the full pipeline.")
    st.caption(f"Trained {bundle['trained_at']} · Python {bundle['versions']['python']} · "
               f"scikit-learn {bundle['versions']['sklearn']}")
    st.warning("This tool estimates statistical risk from a handful of variables. It cannot diagnose "
               "disease. Always consult a qualified clinician about health concerns.")
