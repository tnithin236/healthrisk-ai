# 🩺 HealthRisk AI – Disease Risk Prediction & Explainable ML System

An end-to-end machine-learning project that estimates a patient's **risk probability** for six conditions,
explains **why** (SHAP), compares **six models** per condition, and serves everything in an interactive
**Streamlit dashboard**.

| | Condition | Modelled on (public dataset) | Inputs |
|---|---|---|---|
| ❤️ | **Heart disease** | UCI Heart Disease (Cleveland) | age, sex, chest pain, BP, cholesterol, blood sugar, max heart rate, exercise angina, BMI, smoking |
| 🩸 | **Diabetes** | Pima Indians Diabetes | age, pregnancies, glucose, BP, skin-fold, insulin, BMI, pedigree function |
| 🧠 | **Stroke** | Kaggle Stroke Prediction | age, sex, hypertension, heart disease, glucose, BMI, smoking status |
| 🫁 | **Lung disease** | Kaggle Lung Cancer survey | age, sex, smoking, yellow fingers, chronic disease, alcohol + 6 respiratory symptoms |
| 🫘 | **Kidney disease** | UCI Chronic Kidney Disease | age, BP, urine gravity & albumin, glucose, urea, creatinine, hemoglobin + 4 comorbidities |
| 🫀 | **Liver disease** | Indian Liver Patient (ILPD) | age, sex, alcohol, bilirubin (total/direct), ALP, ALT, AST, protein, albumin, A/G ratio |

> ⚕️ **Educational project – not a medical device.** It estimates statistical risk from a handful of variables and
> cannot diagnose disease.

```
Patient Data → Cleaning → EDA → Feature Engineering → Feature Selection
            → Model Training & Comparison → Calibration → Risk Prediction
            → SHAP Explanation → Streamlit Dashboard
```

## Quick start

```bash
pip install -r requirements.txt

python train.py --disease all     # all six conditions on SYNTHETIC demo data (~6 min)
# or one at a time:  python train.py --disease heart   (heart | diabetes | stroke | lung | kidney | liver)

streamlit run app.py              # pick a condition in the sidebar
pytest -q                         # optional: 17 tests
```

`train.py` writes `models/<disease>_model.joblib` and all charts/tables to `reports/<disease>/`.
Conditions you haven't trained yet appear in the dropdown as *(not trained)* with the command to run.

### Use a real dataset (strongly recommended)

Download the matching public CSV, then train that condition on it:

```bash
python train.py --disease heart    --data data/heart.csv
python train.py --disease diabetes --data data/diabetes.csv
python train.py --disease stroke   --data data/healthcare-dataset-stroke-data.csv
python train.py --disease lung     --data data/survey_lung_cancer.csv
python train.py --disease kidney   --data data/kidney_disease.csv
python train.py --disease liver    --data data/indian_liver_patient.csv
```

Column names are auto-mapped, features your file lacks are dropped (the dashboard form adapts), and these real-data
quirks are handled for you:

| Dataset | Handled automatically |
|---|---|
| Heart (UCI) | `num` 0–4 → 0/1; chest pain coded 1–4 → 0–3 |
| Diabetes (Pima) | zeros in glucose / BP / skin-fold / insulin / BMI mean *not measured* → treated as missing |
| Stroke (Kaggle) | text smoking status; `"N/A"` BMI; `"Unknown"` smoking → missing; ~5% positives → class weighting |
| Lung (survey) | yes/no coded **1 = No, 2 = Yes**; `M/F`; `YES/NO` label; deduplication off (see below) |
| Kidney (UCI CKD) | tabs / `?` / stray spaces in text fields; `ckd` / `notckd` label |
| Liver (ILPD) | label coded **1 = disease, 2 = healthy**; misspelt column names (`Protiens`) |

Still check before you trust a result: the **target direction** (`1 = disease`) and category codings in *your* copy
of the file – public copies of these datasets are not all identical.

## Project layout

```
healthrisk-ai/
├── app.py                    # Streamlit dashboard (Predict · Model comparison · Explainability · EDA · About)
├── train.py                  # clean → EDA → compare → calibrate → explain → save   (--disease <name>|all)
├── src/
│   ├── config.py             # FeatureSpec / DiseaseConfig dataclasses
│   ├── diseases/             # one file per condition: schema, aliases, feature engineering, demo data
│   │   ├── heart.py  diabetes.py  stroke.py  lung.py  kidney.py  liver.py
│   │   ├── _common.py        # shared helpers (bucketing, synthetic-data utilities)
│   │   └── __init__.py       # REGISTRY
│   ├── data.py               # loading, column normalisation, cleaning
│   ├── pipeline.py           # feature engineering + preprocessing + selection + classifier (one Pipeline)
│   ├── models.py             # Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN
│   ├── evaluate.py           # stratified CV + hold-out metrics
│   ├── explain.py            # SHAP explainer (with built-in Shapley-sampling fallback)
│   ├── plots.py              # ROC, calibration, confusion matrix, EDA, risk gauge, contribution chart
│   └── predict.py            # inference helpers used by the app
└── tests/test_project.py     # cleaning, real-dataset quirks, explanation additivity, all six diseases
```

## What the pipeline does

| Stage | Details |
|---|---|
| **Cleaning** | Drops rows without a label; physiologically impossible values become missing; exact duplicates dropped (except lung – see below). A cleaning report is saved and shown in the app. |
| **EDA** | Class balance, distributions by outcome, correlation heatmap, prevalence by category. |
| **Feature engineering** | Clinically motivated per disease, e.g. heart-rate reserve (heart), HOMA-IR-style index (diabetes), vascular-burden score (stroke), estimated GFR (kidney), AST/ALT ratio (liver). |
| **Feature selection** | Random-forest importance threshold (`--no-select` to disable), fitted inside each CV fold. |
| **Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost (sklearn gradient boosting if not installed), SVM, KNN. |
| **Imbalance** | If the outcome is rare (<30% or >70%) models train with class weights; probabilities are re-calibrated afterwards. |
| **Evaluation** | 5-fold stratified CV → winner chosen on CV ROC-AUC → accuracy, precision, recall, F1, ROC-AUC, **PR-AUC**, Brier on a hold-out set that played no part in selection. |
| **Calibration** | Sigmoid calibration so "70%" behaves like ~70%. (On stroke it cut the Brier score from 0.168 to 0.049.) |
| **Risk levels** | Per-disease bands, e.g. heart Low < 33% ≤ Medium < 66% ≤ High; stroke Low < 5% ≤ Medium < 15% ≤ High (relative to its low base rate). Edit `thresholds` in the disease file. |
| **Explainability** | SHAP (permutation explainer) on the *whole pipeline*, so effects are attributed to raw inputs like "Cholesterol" or "Age" for any model type. Effects add up: `average risk + Σ effects = this patient's risk`. |

### Design choices worth knowing

- **No leakage:** imputation, scaling, selection and feature engineering all live inside the `Pipeline`; the app feeds raw values straight into `predict_proba`.
- **Model selection ≠ model reporting:** the winner is chosen on CV; the test set is only used to report.
- **Stable, representative explanations:** SHAP reference rows are spaced evenly by predicted risk and each used equally often, so the "population average" shown is exact and does not jitter between clicks.
- **Flag threshold = risk band:** headline recall / confusion matrix treat "Medium or High" as a positive flag, matching what the dashboard tells the user.
- **Deduplication is per-disease:** lung data is 12 yes/no columns plus an integer age, so identical rows are usually different patients; deduplicating would silently discard real data.

## Results on the bundled synthetic data

> ⚠️ These come from **synthetic** data generated in `src/diseases/*.py`. They test the machinery, **not** clinical
> performance. Logistic Regression often wins because the generators use (near-)linear risk functions; real data will
> rank models differently. Retrain on real datasets and report *those* numbers.

| Condition | Rows | Positive rate | Best model (by CV) | CV ROC-AUC | Test ROC-AUC | Test PR-AUC | Recall @ Medium+ |
|---|---:|---:|---|---:|---:|---:|---:|
| ❤️ Heart Disease | 2,500 | 45.6% | Logistic Regression | 0.881 ± 0.016 | 0.875 | 0.870 | 0.85 (≥33%) |
| 🩸 Diabetes | 2,500 | 35.0% | Logistic Regression | 0.873 ± 0.008 | 0.867 | 0.764 | 0.82 (≥30%) |
| 🧠 Stroke | 5,991 | 5.7% | Logistic Regression | 0.834 ± 0.019 | 0.822 | 0.228 | 0.80 (≥5%) |
| 🫁 Lung Disease | 2,500 | 39.5% | Logistic Regression | 0.894 ± 0.022 | 0.894 | 0.837 | 0.84 (≥33%) |
| 🫘 Kidney Disease | 2,500 | 39.2% | Gradient Boosting* | 0.998 ± 0.001 | 1.000 | 1.000 | 0.99 (≥30%) |
| 🫀 Liver Disease | 2,500 | 43.6% | Logistic Regression | 0.854 ± 0.018 | 0.852 | 0.807 | 0.88 (≥33%) |

\*sklearn gradient boosting stands in for XGBoost when `xgboost` isn't installed; on your machine it will say `XGBoost`.

Reading the table: **stroke** looks weak on PR-AUC (0.23) because strokes are rare – a 5.7% base rate makes PR-AUC
hard to raise, which is exactly why accuracy would be misleading there. **Kidney** is near-perfect because its
synthetic labs separate the classes cleanly; the real UCI CKD dataset is also famously easy (models routinely exceed
0.99 AUC), so treat a perfect score as a warning about the dataset, not proof of a great model.

## Limitations & responsible-use notes

- **Association, not causation.** Explanations describe what the *model* relies on, not what causes disease.
- **Dataset quirks become model quirks.** In the real UCI heart data, patients reporting *no* chest pain were more
  often diagnosed (silent ischaemia), so the model treats "Asymptomatic" as a risk signal; the dashboard says so.
- **Skewed public samples.** The Kaggle lung-cancer survey is ~87% positive and ILPD ~71%, so probabilities trained on
  them do not represent the general population; the bands (Low/Medium/High) will look very high.
- **Small real datasets** (Cleveland ~300 rows, CKD 400, lung survey ~300) give noisy metrics – read the CV std.
- **The lung model is a symptom/lifestyle screener, not an imaging or biopsy tool.** Symptoms like cough are partly
  *consequences* of disease, which flatters accuracy relative to a true pre-diagnosis risk model.
- **Threshold matters.** For screening you would tune for recall and accept more false alarms.
- **Fairness and generalisation are not evaluated.** Before any real use: subgroup analysis (sex, age, ethnicity),
  external validation, and clinical oversight.

## Adding another condition

1. Copy `src/diseases/liver.py` → `src/diseases/<name>.py`; define the `FeatureSpec`s (valid ranges + UI widgets),
   column aliases, an `engineer()` function, a `synthesize()` generator and a `DiseaseConfig`
   (set `target_map` if the label isn't "greater than 0 = disease", `dedupe=False` for mostly-binary data).
2. Register it in `src/diseases/__init__.py`.
3. `python train.py --disease <name>` – the dashboard picks it up automatically.

## Ideas to extend

Hyper-parameter search (`RandomizedSearchCV`), threshold optimisation for recall, SHAP dependence plots,
what-if sliders ("what if cholesterol dropped to 200?"), subgroup fairness report, multi-condition patient summary,
Docker + Streamlit Cloud deployment, MLflow experiment tracking.
