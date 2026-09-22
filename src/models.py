"""The model zoo compared by `train.py`."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def is_imbalanced(prevalence: float | None, limit: float = 0.30) -> bool:
    return prevalence is not None and min(prevalence, 1 - prevalence) < limit


def get_models(seed: int = 42, prevalence: float | None = None) -> dict:
    """Model zoo. When the outcome is rare (or overwhelmingly common) the models that support
    it are trained with class weights; probabilities are re-calibrated afterwards."""
    cw = "balanced" if is_imbalanced(prevalence) else None
    spw = (1 - prevalence) / prevalence if cw and prevalence else 1.0   # XGBoost equivalent
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, class_weight=cw),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10,
                                                class_weight=cw, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                class_weight=cw, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", scale_pos_weight=spw,
            random_state=seed, n_jobs=-1,
        )
    except Exception:  # not installed, or missing native lib (e.g. libomp on macOS)
        models["Gradient Boosting (XGBoost n/a)"] = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=250, class_weight=cw, random_state=seed)
    models["SVM (RBF)"] = SVC(C=1.0, kernel="rbf", probability=True, class_weight=cw,
                              random_state=seed)
    models["KNN"] = KNeighborsClassifier(n_neighbors=15)
    return models
