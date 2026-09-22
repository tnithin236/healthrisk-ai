"""Model comparison: stratified CV on the training split + hold-out test metrics."""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate

from .config import DiseaseConfig
from .models import get_models
from .pipeline import build_pipeline

CV_SCORING = {"accuracy": "accuracy", "precision": "precision", "recall": "recall",
              "f1": "f1", "roc_auc": "roc_auc", "pr_auc": "average_precision"}


def classification_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    pred = (np.asarray(proba) >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "brier": brier_score_loss(y_true, proba),
    }


def compare_models(cfg: DiseaseConfig, X_train, y_train, X_test, y_test, *, seed: int = 42,
                   cv_folds: int = 5, select: bool = True) -> tuple[pd.DataFrame, dict]:
    """Returns (comparison table, {model name: pipeline fitted on the training split})."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    rows, fitted = [], {}
    prevalence = float(np.mean(y_train))
    for name, clf in get_models(seed, prevalence).items():
        t0 = time.time()
        pipe = build_pipeline(cfg, X_train, clf, select=select, seed=seed)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=CV_SCORING, n_jobs=1)
        pipe.fit(X_train, y_train)
        fitted[name] = pipe
        test = classification_metrics(y_test, pipe.predict_proba(X_test)[:, 1])
        row = {"model": name}
        row.update({f"cv_{m}": scores[f"test_{m}"].mean() for m in CV_SCORING})
        row["cv_roc_auc_std"] = scores["test_roc_auc"].std()
        row.update({f"test_{m}": v for m, v in test.items()})
        row["seconds"] = time.time() - t0
        rows.append(row)
        print(f"  {name:<34} CV AUC {row['cv_roc_auc']:.3f} ± {row['cv_roc_auc_std']:.3f}"
              f" | test AUC {row['test_roc_auc']:.3f} | {row['seconds']:.0f}s")
    table = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False).reset_index(drop=True)
    return table, fitted
