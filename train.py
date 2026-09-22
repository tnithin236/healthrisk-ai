"""Train, compare, calibrate, explain and save a disease-risk model.

    python train.py                       # heart disease on synthetic demo data
    python train.py --data data/heart.csv # your real dataset
"""
from __future__ import annotations

import argparse
import json
import platform
import warnings
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split

from src.data import clean, load_data
from src.diseases import REGISTRY, get_disease
from src.evaluate import classification_metrics, compare_models
from src.explain import RiskExplainer
from src.models import get_models, is_imbalanced
from src.pipeline import build_pipeline, selected_feature_names
from src.plots import (plot_calibration, plot_confusion, plot_eda, plot_global_importance,
                       plot_roc)
from src.predict import MODEL_DIR, model_path, report_dir

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--disease", default="heart", help=f"One of {sorted(REGISTRY)} or 'all'.")
    ap.add_argument("--data", default=None, help="CSV path. Omit to use synthetic demo data.")
    ap.add_argument("--n-synthetic", type=int, default=None,
                    help="Rows of synthetic data (default: per-disease).")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-select", action="store_true", help="Disable feature selection.")
    ap.add_argument("--skip-eda", action="store_true")
    args = ap.parse_args()

    if args.disease == "all":
        if args.data:
            ap.error("--data cannot be combined with --disease all (each disease needs its own file).")
        for key in REGISTRY:
            print(f"\n{'=' * 70}\n{REGISTRY[key].icon}  {REGISTRY[key].title}\n{'=' * 70}")
            train_one(get_disease(key), args)
    else:
        train_one(get_disease(args.disease), args)


def train_one(cfg, args):
    out = report_dir(cfg.key)
    out.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    # 1-2. Data + cleaning ------------------------------------------------------
    print(f"[1/7] Loading {cfg.title} data …")
    raw, source = load_data(cfg, args.data, args.n_synthetic, args.seed)
    df, report = clean(cfg, raw)
    print(f"      source={source}  rows {report['rows_in']} -> {report['rows_out']}  "
          f"prevalence={report['prevalence']:.1%}")
    print(f"      cleaning: {json.dumps({k: v for k, v in report.items() if k.startswith(('dropped', 'impossible'))})}")
    (out / "cleaning_report.json").write_text(json.dumps(report, indent=2))

    # 3. EDA ---------------------------------------------------------------------
    if not args.skip_eda:
        print("[2/7] Exploratory data analysis …")
        plot_eda(df, cfg, out)

    features = report["features_used"]
    X, y = df[features], df[cfg.target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed)

    # 4-6. Feature engineering/selection happen inside the pipeline; compare models
    if is_imbalanced(float(y_train.mean())):
        print(f"      note: imbalanced outcome ({y_train.mean():.1%} positive) -> class-weighted training")
    print(f"[3/7] Comparing models ({args.cv}-fold CV on {len(X_train)} rows, "
          f"then hold-out test on {len(X_test)}) …")
    table, fitted = compare_models(cfg, X_train, y_train, X_test, y_test, seed=args.seed,
                                   cv_folds=args.cv, select=not args.no_select)
    table.to_csv(out / "model_comparison.csv", index=False)
    plot_roc(fitted, X_test, y_test, out / "roc_curves.png")

    best_name = table.loc[0, "model"]            # chosen on CV AUC, NOT on the test set
    print(f"[4/7] Best model by CV ROC-AUC: {best_name}")
    print("      features reaching the classifier:", selected_feature_names(fitted[best_name]))

    # 7. Calibrate so that '70%' behaves like ~70% -----------------------------
    print("[5/7] Calibrating probabilities …")
    models_train = lambda: get_models(args.seed, float(y_train.mean()))[best_name]
    make = lambda: build_pipeline(cfg, X_train, models_train(), select=not args.no_select, seed=args.seed)
    calibrated = CalibratedClassifierCV(make(), method="sigmoid", cv=5).fit(X_train, y_train)
    p_raw = fitted[best_name].predict_proba(X_test)[:, 1]
    p_cal = calibrated.predict_proba(X_test)[:, 1]
    flag_t = cfg.thresholds[0]          # "Medium or High" = flagged; consistent with the app's risk bands
    final_metrics = classification_metrics(y_test, p_cal, threshold=flag_t)
    final_metrics["threshold"] = flag_t
    print(f"      Brier score: raw {classification_metrics(y_test, p_raw)['brier']:.4f} "
          f"-> calibrated {final_metrics['brier']:.4f}")
    plot_calibration({"Uncalibrated": (y_test, p_raw), "Calibrated": (y_test, p_cal)},
                     out / "calibration.png")
    plot_confusion(y_test, p_cal, out / "confusion_matrix.png", threshold=flag_t)

    # Deploy: refit on ALL labelled data (test metrics above come from the train-only model)
    print("[6/7] Refitting on all data for deployment …")
    deployed = CalibratedClassifierCV(
        build_pipeline(cfg, X, get_models(args.seed, float(y.mean()))[best_name],
                       select=not args.no_select, seed=args.seed),
        method="sigmoid", cv=5).fit(X, y)

    # Global explanation -------------------------------------------------------
    print("[7/7] Global explainability …")
    ge = RiskExplainer(deployed, features, X, n_background=10, max_evals=60, n_paths=40, seed=args.seed)
    sample = X.sample(n=min(100, len(X)), random_state=args.seed)
    ex = ge.explain(sample)
    importance = pd.Series(np.abs(ex.values).mean(axis=0) * 100,
                           index=[cfg.feature(f).label for f in features])
    plot_global_importance(importance, out / "global_importance.png", ex.backend)
    print(f"      backend={ex.backend}; top drivers:")
    for name, val in importance.sort_values(ascending=False).head(5).items():
        print(f"        {name:<26} {val:5.2f} pts")

    bundle = {
        "disease": cfg.key,
        "model": deployed,
        "best_model": best_name,
        "raw_features": features,
        "thresholds": cfg.thresholds,
        "background": X.sample(n=min(300, len(X)), random_state=args.seed),
        "comparison": table,
        "test_metrics": final_metrics,
        "global_importance": importance.sort_values(ascending=False),
        "data_source": source,
        "cleaning_report": report,
        "n_rows": int(len(df)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__},
    }
    joblib.dump(bundle, model_path(cfg.key))
    print(f"\nSaved model -> {model_path(cfg.key)}\nSaved reports -> {out}/")
    if source == "synthetic":
        print("\n⚠  Trained on SYNTHETIC data - metrics are illustrative only.")


if __name__ == "__main__":
    main()
