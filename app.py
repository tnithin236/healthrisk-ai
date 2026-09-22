"""HealthRisk AI - interactive dashboard.   Run with:  streamlit run app.py"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from src.diseases import REGISTRY, get_disease
from src.predict import (available_diseases, build_local_explainer, load_bundle,
                         predict_one, report_dir, top_factors)

st.set_page_config(page_title="HealthRisk AI", page_icon="🩺", layout="wide")

# --------------------------------------------------------------------------- #
# Style constants
# --------------------------------------------------------------------------- #
LEVEL_STYLE = {
    "Low": dict(bg="#eafaf0", fg="#159457", chip_bg="#e2f8ea", chip_fg="#159457", icon="✅"),
    "Medium": dict(bg="#fff8e8", fg="#b9790a", chip_bg="#fff1d6", chip_fg="#b9790a", icon="⚠️"),
    "High": dict(bg="#fdeceb", fg="#d64545", chip_bg="#fbe0df", chip_fg="#c23b3b", icon="🚨"),
}
LEVEL_DESC = {
    "Low": "You are at a lower risk of {d}.",
    "Medium": "You have a moderate risk of {d}. It may help to discuss this with a doctor.",
    "High": "You are at a higher risk of {d}. Please consider consulting a healthcare professional.",
}
CARD_STYLE = ("background:#ffffff;border:1px solid #eef0f6;border-radius:16px;"
             "padding:20px 22px;margin-bottom:16px;box-shadow:0 2px 10px rgba(20,30,60,0.04);")

FEATURE_ICON = {
    "age": "🎂", "sex": "🧑", "chest_pain": "💢", "resting_bp": "🩺", "blood_pressure": "🩺",
    "cholesterol": "🧪", "fasting_bs": "🍬", "glucose": "🍬", "blood_glucose": "🍬",
    "max_hr": "❤️", "exercise_angina": "🏃", "bmi": "⚖️", "smoking": "🚬", "smoking_status": "🚬",
    "pregnancies": "🤰", "insulin": "💉", "skin_thickness": "📏", "pedigree": "🧬",
    "hypertension": "🩺", "heart_disease": "❤️", "avg_glucose_level": "🍬",
    "alcohol": "🍷", "alcohol_use": "🍷", "yellow_fingers": "🖐️", "chronic_disease": "🏥",
    "fatigue": "😴", "wheezing": "💨", "coughing": "😷", "shortness_of_breath": "🌬️",
    "swallowing_difficulty": "🥤", "specific_gravity": "🧪", "albumin": "🧪", "blood_urea": "🧪",
    "serum_creatinine": "🧪", "hemoglobin": "🩸", "diabetes": "🩸", "anemia": "🩸",
    "pedal_edema": "🦶", "total_bilirubin": "🧪", "direct_bilirubin": "🧪", "alk_phos": "🧪",
    "alt": "🧪", "ast": "🧪", "total_protein": "🧪", "ag_ratio": "🧪",
}


def ficon(name: str) -> str:
    return FEATURE_ICON.get(name, "🔎")


def h(s: str) -> str:
    """Flatten a multi-line HTML snippet before handing it to st.markdown().

    Streamlit's markdown parser follows CommonMark: a line indented 4+ spaces
    (which Python's own code indentation produces for free in every f-string
    built inside a function) is read as an *indented code block*, and a blank
    line splits one HTML block into several. Either turns "rendered HTML" into
    literal, escaped tag text on the page. Stripping each line and dropping
    blank lines removes both triggers, so nested/joined snippets stay safe too.
    """
    return "\n".join(line.strip() for line in s.strip().splitlines() if line.strip())


CUSTOM_CSS = """
<style>
#MainMenu, footer, header {visibility: hidden;}
.stApp { background: #f4f6fb; }
.block-container { padding-top: 1.6rem; padding-bottom: 2rem; }
h1, h2, h3 { color: #152047; }

/* ---- Sidebar ---- */
section[data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #edf0f7; }
section[data-testid="stSidebar"] div[data-testid="stButton"] button {
    width: 100%; text-align: left !important; justify-content: flex-start !important;
    font-size: 0.93rem; font-weight: 500; padding: 0.6rem 0.9rem; border-radius: 10px;
    margin-bottom: 3px; transition: all .12s ease;
}
section[data-testid="stSidebar"] div[data-testid="stButton"] button p { text-align: left; }
section[data-testid="stSidebar"] button[kind="secondary"] {
    background: transparent !important; color: #45506b !important;
    border: 1px solid transparent !important; box-shadow: none !important;
}
section[data-testid="stSidebar"] button[kind="secondary"]:hover {
    background: #eef3ff !important; color: #2f6fed !important;
}
section[data-testid="stSidebar"] button[kind="primary"] {
    background: #2f6fed !important; color: #ffffff !important; border: none !important;
    box-shadow: 0 4px 10px rgba(47,111,237,0.30) !important;
}

/* ---- Bordered containers used as cards for real widgets ---- */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #ffffff !important; border-radius: 16px !important;
    border: 1px solid #eef0f6 !important;
    box-shadow: 0 2px 10px rgba(20, 30, 60, 0.04) !important;
}

/* ---- Predict button ---- */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(135deg, #2f6fed, #1c4fd6) !important; color: white !important;
    border: none !important; border-radius: 12px !important; padding: 0.7rem !important;
    font-weight: 700 !important; font-size: 0.98rem !important;
    box-shadow: 0 6px 16px rgba(47,111,237,0.30) !important;
}

/* ---- Sub-nav radio (Predict / Model comparison / ...) ---- */
div[data-testid="stRadio"] > div { gap: 6px; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Cached data
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model …")
def load_assets(disease: str):
    bundle = load_bundle(disease)
    return bundle, build_local_explainer(bundle)


# --------------------------------------------------------------------------- #
# HTML builders (pure display -- no Streamlit widgets inside, so a plain
# st.markdown(..., unsafe_allow_html=True) call is enough to render a "card")
# --------------------------------------------------------------------------- #
def risk_panel_html(cfg, prob: float, level: str, base: float) -> str:
    sty = LEVEL_STYLE[level]
    t0, t1 = cfg.thresholds
    xmax = 1.0 if t1 > 0.4 else (0.5 if t1 > 0.1 else 0.25)

    def pct(v):
        return min(v, xmax) / xmax * 100

    seg1, seg2 = pct(t0), pct(t1) - pct(t0)
    seg3 = 100 - pct(t1)
    marker = pct(prob)
    tick_defs = [(0.0, "0%"), (t0, f"{t0:.0%}"), (t1, f"{t1:.0%}"),
                (xmax, "100%" if xmax >= 1 else f"{xmax:.0%}+")]
    ticks_html = "".join(
        f'<span style="position:absolute; left:{pct(tv):.1f}%; transform:translateX(-50%); '
        f'white-space:nowrap;">{tl}</span>' for tv, tl in tick_defs)
    desc = LEVEL_DESC[level].format(d=cfg.title.lower())
    return h(f"""
    <div style="{CARD_STYLE}">
      <div style="position:relative; background:{sty['bg']}; border-radius:14px; padding:18px 20px;">
        <div style="position:absolute; top:14px; right:16px; background:{sty['chip_bg']};
                    color:{sty['chip_fg']}; font-weight:700; font-size:0.72rem; padding:4px 11px;
                    border-radius:999px;">● {level} Risk</div>
        <div style="display:flex; align-items:center; gap:14px;">
          <div style="width:50px;height:50px;border-radius:50%;background:#ffffffb0;
                      display:flex;align-items:center;justify-content:center;font-size:1.5rem;">{sty['icon']}</div>
          <div>
            <div style="font-size:0.82rem;color:#5b6785;font-weight:600;">Estimated Risk</div>
            <div style="font-size:1.85rem;font-weight:800;color:{sty['fg']};line-height:1.15;">
              {level.upper()} · {prob:.0%}</div>
          </div>
        </div>
        <div style="margin-top:6px;color:#5b6785;font-size:0.86rem;">{desc}</div>
      </div>
      <div style="margin-top:20px; padding:0 2px;">
        <div style="display:flex; height:12px; border-radius:8px; overflow:hidden;">
          <div style="width:{seg1:.2f}%; background:#34c185;"></div>
          <div style="width:{seg2:.2f}%; background:#f2b134;"></div>
          <div style="width:{seg3:.2f}%; background:#f16063;"></div>
        </div>
        <div style="position:relative; height:18px;">
          <div style="position:absolute; left:calc({marker:.2f}% - 6px); top:-19px;
                      color:#152047; font-size:13px;">▼</div>
        </div>
        <div style="position:relative; height:16px; font-size:0.72rem; color:#5b6785;">{ticks_html}</div>
      </div>
      <div style="margin-top:12px; background:#eef4ff; color:#33507a; font-size:0.8rem;
                  padding:9px 13px; border-radius:10px; display:flex; align-items:center; gap:8px;">
        <span>ℹ️</span><span>Low &lt; {t0:.0%} ≤ Medium &lt; {t1:.0%} ≤ High &nbsp;·&nbsp;
        Population average ≈ {base:.0%}</span>
      </div>
    </div>
    """)


def factor_row_html(icon: str, name: str, detail: str, pts: float) -> str:
    up = pts > 0
    chip_bg, chip_fg = ("#fdeceb", "#d64545") if up else ("#eafaf0", "#159457")
    sign = "+" if up else "-"
    return h(f"""
    <div style="display:flex; align-items:center; justify-content:space-between; padding:12px 14px;
                border:1px solid #eef0f6; border-radius:12px; margin-bottom:10px;">
      <div style="display:flex; align-items:center; gap:12px;">
        <div style="width:38px;height:38px;border-radius:10px;background:{chip_bg};
                    display:flex;align-items:center;justify-content:center;font-size:1.05rem;">{icon}</div>
        <div>
          <div style="font-weight:600; color:#152047; font-size:0.92rem;">{name}</div>
          <div style="color:#5b6785; font-size:0.8rem;">{detail}</div>
        </div>
      </div>
      <div style="background:{chip_bg}; color:{chip_fg}; font-weight:700; font-size:0.82rem;
                  padding:5px 12px; border-radius:999px; white-space:nowrap;">{sign}{abs(pts):.1f} pts</div>
    </div>
    """)


def top_factors_html(rows) -> str:
    items = "".join(factor_row_html(icon, name, detail, pts) for icon, name, detail, pts, _ in rows)
    return h(f"""
    <div style="{CARD_STYLE}">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <div style="width:34px;height:34px;border-radius:10px;background:#eef4ff;display:flex;
                    align-items:center;justify-content:center;font-size:1rem;">📊</div>
        <div style="font-weight:700; color:#152047; font-size:1.02rem;">Top Factors</div>
      </div>
      {items}
    </div>
    """)


def diverging_bars_html(contrib_series: pd.Series, labels: dict) -> str:
    vals = contrib_series * 100
    order = vals.abs().sort_values(ascending=False).index
    maxabs = max(vals.abs().max(), 1.0)
    rows = []
    for name in order:
        v = vals[name]
        width = abs(v) / maxabs * 50
        color = "#f16063" if v >= 0 else "#34c185"
        bar_style = (f"left:50%; width:{width:.2f}%; background:{color}; border-radius:0 6px 6px 0;"
                    if v >= 0 else
                    f"right:50%; width:{width:.2f}%; background:{color}; border-radius:6px 0 0 6px;")
        sign = "+" if v >= 0 else ""
        rows.append(h(f"""
        <div style="margin-bottom:14px;">
          <div style="display:flex; justify-content:space-between; font-size:0.82rem;
                      color:#152047; margin-bottom:4px;">
            <span style="font-weight:600;">{labels[name]}</span>
            <span style="font-weight:700; color:{color};">{sign}{v:.1f}</span>
          </div>
          <div style="position:relative; height:10px; background:#f1f3f9; border-radius:6px;">
            <div style="position:absolute; top:0; bottom:0; {bar_style}"></div>
            <div style="position:absolute; left:50%; top:-2px; bottom:-2px; width:1px; background:#c7cde0;"></div>
          </div>
        </div>
        """))
    return "\n".join(rows)


def why_panel_html(contrib_series: pd.Series, labels: dict, base_pct: float, prob_pct: float) -> str:
    bars = diverging_bars_html(contrib_series, labels)
    return h(f"""
    <div style="{CARD_STYLE}">
      <div style="display:flex; align-items:center; gap:10px; margin-bottom:2px;">
        <div style="width:34px;height:34px;border-radius:10px;background:#eef4ff;display:flex;
                    align-items:center;justify-content:center;font-size:1rem;">🧮</div>
        <div style="font-weight:700; color:#152047; font-size:1.02rem;">Why {prob_pct:.0f}%?</div>
      </div>
      <div style="color:#5b6785; font-size:0.78rem; margin:0 0 14px 44px;">
        Population average ≈ {base_pct:.0f}%</div>
      {bars}
      <div style="color:#8791ab; font-size:0.74rem; margin-top:4px;">Red bars push risk up, green bars
      pull it down, relative to an average patient. They describe what the model relies on, not
      medical causation.</div>
    </div>
    """)


def placeholder_panel_html() -> str:
    return h(f"""
    <div style="{CARD_STYLE} display:flex; flex-direction:column; align-items:center;
                justify-content:center; text-align:center; min-height:320px; color:#5b6785;">
      <div style="font-size:2.4rem; margin-bottom:10px;">🩺</div>
      <div style="font-weight:600; color:#152047; margin-bottom:4px;">No prediction yet</div>
      <div style="font-size:0.88rem;">Fill in the patient details on the left and press
      <b>Predict Risk</b> to see the result.</div>
    </div>
    """)


# --------------------------------------------------------------------------- #
# Widgets
# --------------------------------------------------------------------------- #
def input_widget(spec, key_prefix: str):
    key = f"{key_prefix}_{spec.name}"
    label = f"{ficon(spec.name)} {spec.label} ({spec.unit})" if spec.unit else f"{ficon(spec.name)} {spec.label}"
    if spec.kind == "number":
        if isinstance(spec.step, float):
            return st.number_input(label, float(spec.min), float(spec.max), float(spec.default),
                                   float(spec.step), format=f"%.{spec.decimals}f",
                                   help=spec.help or None, key=key)
        return st.number_input(label, int(spec.min), int(spec.max), int(spec.default),
                               int(spec.step or 1), help=spec.help or None, key=key)
    codes = list(spec.options)
    return st.selectbox(label, codes, index=codes.index(int(spec.default)),
                        format_func=lambda c: spec.options[c], help=spec.help or None, key=key)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #
if "page" not in st.session_state:
    st.session_state.page = "home"

trained = set(available_diseases())

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown(h("""
    <div style="display:flex; align-items:center; gap:10px; padding:4px 2px 18px;">
      <div style="font-size:1.7rem;">🩺</div>
      <div>
        <div style="font-weight:800; font-size:1.15rem; color:#152047; line-height:1.1;">
          HealthRisk<span style="color:#2f6fed;"> AI</span></div>
        <div style="color:#9aa3bd; font-size:0.72rem;">Better Insights. Healthier Lives.</div>
      </div>
    </div>
    """), unsafe_allow_html=True)

    nav_items = [("home", "🏠", "Home")] + [(k, cfg.icon, cfg.title) for k, cfg in REGISTRY.items()]
    for key, icon, label in nav_items:
        active = st.session_state.page == key
        suffix = "" if key == "home" or key in trained else "  ⚪"
        if st.button(f"{icon}  {label}{suffix}", key=f"nav_{key}", use_container_width=True,
                    type="primary" if active else "secondary"):
            st.session_state.page = key
            st.rerun()

    st.markdown(h("""
    <div style="background:#eef4ff; border-radius:14px; padding:16px; margin-top:14px;">
      <div style="font-weight:700; color:#152047; font-size:0.88rem; margin-bottom:6px;">
        💡 AI for a Healthier Tomorrow</div>
      <div style="color:#5b6785; font-size:0.78rem; line-height:1.4;">
        Early detection. Better decisions. A healthier future.</div>
      <div style="color:#9aa3bd; font-size:0.7rem; margin-top:8px;">
        ⚕️ Educational project — not a medical device, and not a substitute for professional
        medical advice, diagnosis or treatment.</div>
    </div>
    """), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Home page
# --------------------------------------------------------------------------- #
def render_home():
    st.markdown(h("""
    <div style="margin-bottom:22px;">
      <div style="font-size:1.6rem; font-weight:800; color:#152047;">Welcome to HealthRisk AI 👋</div>
      <div style="color:#5b6785; font-size:0.95rem; margin-top:4px;">Pick a condition below (or
      from the sidebar) to estimate its risk from a patient's vitals and labs.</div>
    </div>
    """), unsafe_allow_html=True)

    cols = st.columns(3)
    for i, (key, cfg) in enumerate(REGISTRY.items()):
        with cols[i % 3]:
            with st.container(border=True):
                is_trained = key in trained
                status = "✅ Trained" if is_trained else "⚪ Not trained yet"
                status_color = "#159457" if is_trained else "#9aa3bd"
                st.markdown(h(f"""
                <div style="text-align:center; padding:6px 0 2px;">
                  <div style="font-size:2.2rem;">{cfg.icon}</div>
                  <div style="font-weight:700; color:#152047; margin-top:6px;">{cfg.title}</div>
                  <div style="color:#5b6785; font-size:0.8rem; margin:6px 0 10px; min-height:54px;">
                    {cfg.description}</div>
                  <div style="font-size:0.76rem; color:{status_color}; margin-bottom:10px;">{status}</div>
                </div>
                """), unsafe_allow_html=True)
                if st.button("Open", key=f"open_{key}", use_container_width=True):
                    st.session_state.page = key
                    st.rerun()

    st.markdown(h("""<div style="margin-top:10px; color:#9aa3bd; font-size:0.78rem;">⚕️ Educational
    project. Not a medical device and not a substitute for professional medical advice, diagnosis
    or treatment.</div>"""), unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Disease page: Predict tab
# --------------------------------------------------------------------------- #
def render_predict(cfg, bundle, explainer):
    left, right = st.columns([1, 1.05], gap="large")
    result_key = f"result_{cfg.key}"

    with left:
        with st.container(border=True):
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:14px;">
              <div style="width:52px;height:52px;border-radius:14px;background:#eef4ff;
                          display:flex;align-items:center;justify-content:center;font-size:1.5rem;">
                {cfg.icon}</div>
              <div>
                <div style="font-weight:800; font-size:1.25rem; color:#152047;">
                  {cfg.title} Prediction</div>
                <div style="color:#5b6785; font-size:0.86rem;">Enter the patient details to check
                the risk of {cfg.title.lower()}.</div>
              </div>
            </div>
            <hr style="border:none;border-top:1px solid #eef0f6;margin:16px 0 14px;">
            <div style="display:flex; align-items:center; gap:8px; font-weight:700; color:#152047;
                        margin-bottom:10px;"><span>🧑‍⚕️</span><span>Patient Details</span></div>
            """, unsafe_allow_html=True)

            with st.form(f"form_{cfg.key}"):
                values = {}
                specs = [cfg.feature(f) for f in bundle["raw_features"]]
                grid = st.columns(2)
                for i, spec in enumerate(specs):
                    with grid[i % 2]:
                        values[spec.name] = input_widget(spec, cfg.key)
                submitted = st.form_submit_button("🩺  Predict Risk  →", use_container_width=True)

            if submitted:
                result = predict_one(bundle, explainer, values)
                st.session_state[result_key] = (result, values)

    with right:
        stored = st.session_state.get(result_key)
        if not stored:
            st.markdown(placeholder_panel_html(), unsafe_allow_html=True)
        else:
            result, values = stored
            prob, level, base = result["probability"], result["level"], result["base"]
            st.markdown(risk_panel_html(cfg, prob, level, base), unsafe_allow_html=True)

            specs = [cfg.feature(f) for f in bundle["raw_features"]]
            rows = [(ficon(name), label, f"({value_text}) {direction} risk", pts, direction)
                   for label, value_text, pts, direction in top_factors(result, cfg, values, k=3)
                   for name in [next(s.name for s in specs if s.label == label)]]
            st.markdown(top_factors_html(rows), unsafe_allow_html=True)

            labels = {s.name: s.label for s in specs}
            st.markdown(why_panel_html(result["contributions"], labels, base * 100, prob * 100),
                       unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Disease page: other tabs (kept close to the original, restyled lightly)
# --------------------------------------------------------------------------- #
def render_models(cfg, bundle, rep):
    with st.container(border=True):
        st.markdown("#### 📊 Model comparison")
        st.caption("Models are ranked by 5-fold cross-validated ROC-AUC on the training split. "
                  "The hold-out test columns are computed once on data no model saw during selection.")
        comp = bundle["comparison"]
        cols = {"model": "Model", "cv_roc_auc": "CV ROC-AUC", "cv_roc_auc_std": "± std",
                "test_accuracy": "Accuracy", "test_precision": "Precision", "test_recall": "Recall",
                "test_f1": "F1", "test_roc_auc": "Test ROC-AUC", "test_pr_auc": "PR-AUC",
                "test_brier": "Brier ↓"}
        show = comp[[c for c in cols if c in comp.columns]].rename(columns=cols)
        st.dataframe(show.round(3), hide_index=True)
        st.bar_chart(comp.set_index("model")[["cv_roc_auc", "test_roc_auc"]]
                     .rename(columns={"cv_roc_auc": "CV ROC-AUC", "test_roc_auc": "Test ROC-AUC"}))

        a, b, c = st.columns(3)
        for col, name, cap in [(a, "roc_curves.png", "ROC curves"), (b, "calibration.png", "Calibration"),
                               (c, "confusion_matrix.png", "Confusion matrix (flagged at Medium+)")]:
            if (rep / name).exists():
                col.image(str(rep / name), caption=cap)

        prevalence = bundle["cleaning_report"]["prevalence"]
        if prevalence < 0.30:
            st.info(f"Only {prevalence:.1%} of cases are positive, so accuracy is misleading "
                    "(predicting 'no disease' for everyone would still score high). Look at "
                    "ROC-AUC, PR-AUC, recall and the Brier score instead.")
        st.info("In a screening setting a missed case (false negative) is usually costlier than a "
                "false alarm, so the decision threshold would normally be tuned for recall rather "
                "than left at 0.5.")


def render_explain(rep):
    with st.container(border=True):
        st.markdown("#### 🔍 Global explainability")
        st.markdown("What drives the model overall?")
        if (rep / "global_importance.png").exists():
            st.image(str(rep / "global_importance.png"))
        st.caption("Mean absolute effect of each feature on predicted risk across a sample of "
                  "patients. Features removed by automatic feature selection show ~0 effect.")


def render_data(bundle, rep):
    with st.container(border=True):
        st.markdown("#### 📈 Data explorer")
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


def render_about(cfg, bundle):
    with st.container(border=True):
        st.markdown("#### ℹ️ How it works")
        st.code(
            "Patient data → Cleaning → EDA → Feature engineering → Feature selection\n"
            "            → Model training & comparison → Calibration → Risk prediction\n"
            "            → SHAP explanation → Streamlit dashboard", language="text")
        st.markdown(
            f"- **Feature engineering:** {cfg.engineered_desc}.\n"
            "- **Feature selection:** random-forest importance threshold.\n"
            "- **Probabilities:** sigmoid-calibrated, so 70% means roughly 70% of similar "
            "patients had the condition.\n"
            "- **Explainability:** Shapley values computed on the *raw inputs* through the "
            "full pipeline.")
        if cfg.dataset_hint:
            st.markdown(f"**Suggested real dataset:** {cfg.dataset_hint}")
        st.caption(f"Trained {bundle['trained_at']} · Python {bundle['versions']['python']} · "
                  f"scikit-learn {bundle['versions']['sklearn']}")
        st.warning("This tool estimates statistical risk from a handful of variables. It cannot "
                  "diagnose disease. Always consult a qualified clinician about health concerns.")


# --------------------------------------------------------------------------- #
# Disease page router
# --------------------------------------------------------------------------- #
def render_disease_page(key: str):
    cfg = get_disease(key)
    if key not in trained:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:18px;">
          <div style="font-size:2rem;">{cfg.icon}</div>
          <div style="font-weight:800; font-size:1.5rem; color:#152047;">{cfg.title} Predictor</div>
        </div>
        """, unsafe_allow_html=True)
        with st.container(border=True):
            st.info(f"No trained model for **{cfg.title}** yet. Train it, then reload this page:")
            st.code(f"python train.py --disease {key}", language="bash")
            st.caption("Or train every condition at once with `python train.py --disease all`.")
        return

    bundle, explainer = load_assets(key)
    rep = report_dir(key)

    if bundle["data_source"] == "synthetic":
        st.warning("**Demo mode:** this model was trained on *synthetic* data, so the numbers "
                  "here are illustrative only. Retrain on a real dataset with "
                  f"`python train.py --disease {key} --data your.csv`.")

    m = bundle["test_metrics"]
    top = st.columns([3, 1, 1])
    with top[0]:
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:10px;">
          <span style="font-size:1.6rem;">{cfg.icon}</span>
          <span style="font-weight:800; font-size:1.35rem; color:#152047;">{cfg.title} Predictor</span>
        </div>
        """, unsafe_allow_html=True)
    top[1].metric("Model", bundle["best_model"])
    top[2].metric("Test ROC-AUC", f"{m['roc_auc']:.3f}")

    sub_key = f"subtab_{key}"
    sub_tabs = [("predict", "🩺 Predict"), ("models", "📊 Model Comparison"),
               ("explain", "🔍 Explainability"), ("data", "📈 Data Explorer"), ("about", "ℹ️ About")]
    choice = st.radio("View", [s[0] for s in sub_tabs], format_func=lambda k: dict(sub_tabs)[k],
                      horizontal=True, label_visibility="collapsed", key=sub_key)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    if choice == "predict":
        render_predict(cfg, bundle, explainer)
    elif choice == "models":
        render_models(cfg, bundle, rep)
    elif choice == "explain":
        render_explain(rep)
    elif choice == "data":
        render_data(bundle, rep)
    else:
        render_about(cfg, bundle)


# --------------------------------------------------------------------------- #
# Route
# --------------------------------------------------------------------------- #
if st.session_state.page == "home":
    render_home()
else:
    render_disease_page(st.session_state.page)
