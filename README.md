# 🩺 HealthRisk AI – Disease Risk Prediction & Explainable ML System

An end-to-end machine-learning project that estimates a patient's **risk probability** for a disease,
explains **why** (SHAP), compares **six models**, and serves everything in an interactive **Streamlit dashboard**.

It is built around **one disease first (heart disease)**, with a config-driven design so that diabetes, stroke,
kidney, liver or lung models can be added by writing one small file.

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

python train.py              # trains on built-in SYNTHETIC demo data (~1 min)
streamlit run app.py         # opens the dashboard
pytest -q                    # optional: run the tests
```

`train.py` writes the model to `models/heart_model.joblib` and all charts/tables to `reports/heart/`.

### Use a real dataset (recommended)

Download a heart-disease CSV (e.g. the UCI *Heart Disease* / Cleveland dataset from UCI or Kaggle), then:

```bash
python train.py --data data/heart.csv
```

Column names are auto-mapped (`trestbps→resting_bp`, `chol→cholesterol`, `cp→chest_pain`, `thalach→max_hr`,
`exang→exercise_angina`, `num/target→target`, …). Features that your file doesn't have (the UCI data has no BMI or
smoking) are simply left out, and **the dashboard form adapts automatically**.

Check before you train:
- **Target direction.** The code expects `1 = disease`. UCI's `num` (0–4) is binarised to `>0`. Some Kaggle copies
  flip the target, so verify against the dataset's documentation.
- **Chest-pain coding.** UCI uses 1–4 (auto-shifted to 0–3 here: typical, atypical, non-anginal, asymptomatic);
  some Kaggle copies use different 0–3 meanings. Confirm the mapping matches your source.

## Project layout

```
healthrisk-ai/
├── app.py                  # Streamlit dashboard (Predict · Model comparison · Explainability · EDA · About)
├── train.py                # Full pipeline: clean → EDA → compare → calibrate → explain → save
├── src/
│   ├── config.py           # FeatureSpec / DiseaseConfig dataclasses
│   ├── diseases/heart.py   # Heart schema, UCI aliases, feature engineering, synthetic data
│   ├── data.py             # Loading, column normalisation, cleaning
│   ├── pipeline.py         # Feature engineering + preprocessing + selection + classifier (one sklearn Pipeline)
│   ├── models.py           # Logistic Regression, Decision Tree, Random Forest, XGBoost, SVM, KNN
│   ├── evaluate.py         # Stratified CV + hold-out metrics
│   ├── explain.py          # SHAP explainer (with built-in Shapley-sampling fallback)
│   ├── plots.py            # ROC, calibration, confusion matrix, EDA, risk gauge, contribution chart
│   └── predict.py          # Inference helpers used by the app
└── tests/test_project.py   # Cleaning, feature engineering, explanation-additivity tests
```

## What the pipeline does

| Stage | Details |
|---|---|
| **Cleaning** | Drops duplicates and rows without a label; physiologically impossible values (e.g. cholesterol = 0) become missing; a cleaning report is saved and shown in the app. |
| **EDA** | Class balance, distributions by outcome, correlation heatmap, prevalence by category. |
| **Feature engineering** | Heart-rate reserve (% of age-predicted max), blood-pressure stage, BMI class, high-cholesterol flag, risk-factor count. |
| **Feature selection** | Random-forest importance threshold (`--no-select` to disable), fitted inside each CV fold. |
| **Models** | Logistic Regression, Decision Tree, Random Forest, XGBoost (falls back to sklearn gradient boosting if not installed), SVM, KNN. |
| **Evaluation** | 5-fold stratified CV on the training split → pick the winner by CV ROC-AUC → report accuracy, precision, recall, F1, ROC-AUC, Brier on a hold-out set that played no part in selection. |
| **Calibration** | The winner is wrapped in sigmoid calibration so "70%" behaves like ~70%. |
| **Risk levels** | Low < 33% ≤ Medium < 66% ≤ High (edit `thresholds` in the disease config). |
| **Explainability** | SHAP (permutation explainer) run on the *whole pipeline*, so effects are attributed to raw inputs like "Cholesterol" or "Age", for any model type. Effects add up: `average risk + Σ effects = this patient's risk`. |

### Design choices worth knowing

- **No leakage:** imputation, scaling, feature selection and engineering all live inside the `Pipeline`, so cross-validation never sees validation-fold statistics, and the app can feed raw values straight into `predict_proba`.
- **Model selection ≠ model reporting:** the winner is chosen on CV, and the test set is only used to report.
- **Representative SHAP background:** reference rows are spaced evenly by predicted risk, so a small background set still matches the population's average risk (a random handful can be badly unrepresentative).
- **Deployment refit:** after reporting test metrics, the final model is refit on all labelled data.

## Results on the bundled synthetic data

> ⚠️ These come from **synthetic** data generated by `src/diseases/heart.py` (2,500 rows, ~46% prevalence). They test the
> machinery, **not** clinical performance. Logistic Regression wins here largely because the generator's risk
> function is close to linear; on real data the ranking will differ. Retrain on a real dataset and report *those* numbers.

| Model | CV ROC-AUC | Test accuracy | Precision | Recall | F1 | Test ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.881 ± 0.016 | 0.782 | 0.787 | 0.715 | 0.749 | 0.875 |
| Gradient Boosting (XGBoost n/a) | 0.871 ± 0.013 | 0.786 | 0.801 | 0.706 | 0.751 | 0.860 |
| SVM (RBF) | 0.869 ± 0.017 | 0.766 | 0.771 | 0.693 | 0.730 | 0.846 |
| Random Forest | 0.867 ± 0.017 | 0.778 | 0.791 | 0.697 | 0.741 | 0.851 |
| KNN | 0.856 ± 0.022 | 0.764 | 0.786 | 0.662 | 0.719 | 0.837 |
| Decision Tree | 0.849 ± 0.014 | 0.776 | 0.802 | 0.675 | 0.733 | 0.829 |

(Your run will show `XGBoost` in place of the sklearn fallback once `xgboost` is installed.)

## Limitations & responsible-use notes

- **Association, not causation.** Explanations describe what the *model* relies on, not what causes disease.
- **Dataset quirks become model quirks.** In the real UCI data, patients reporting *no* chest pain were more often
  diagnosed (silent ischaemia), so the model treats "Asymptomatic" as a risk signal. The dashboard says so.
- **Small real datasets** (Cleveland has ~300 rows) give noisy metrics – look at the CV standard deviations.
- **Threshold matters.** The 0.5 classification threshold is a default; for screening you would tune for recall.
- **Fairness and generalisation** are not evaluated here. Before any real use: subgroup analysis (sex, age, ethnicity),
  external validation, and clinical oversight.

## Adding another disease

1. Copy `src/diseases/heart.py` → `src/diseases/diabetes.py`; define the `FeatureSpec`s (with valid ranges and UI
   widgets), column aliases, an `engineer()` function and a `synthesize()` generator.
2. Register it in `src/diseases/__init__.py`.
3. `python train.py --disease diabetes --data data/diabetes.csv` – the dashboard picks it up automatically.

Suggested public datasets: Pima Indians Diabetes, Kaggle Stroke Prediction, Indian Liver Patient (ILPD),
Chronic Kidney Disease (UCI).

## Ideas to extend

Hyper-parameter search (`RandomizedSearchCV`), threshold optimisation for recall, SHAP dependence plots,
what-if sliders ("what if cholesterol dropped to 200?"), subgroup fairness report, Docker + Streamlit Cloud deployment,
MLflow experiment tracking.
