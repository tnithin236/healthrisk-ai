"""The model zoo compared by `train.py`."""
from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def get_models(seed: int = 42) -> dict:
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                                random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=3, subsample=0.8,
            colsample_bytree=0.8, eval_metric="logloss", random_state=seed, n_jobs=-1,
        )
    except Exception:  # not installed, or missing native lib (e.g. libomp on macOS)
        models["Gradient Boosting (XGBoost n/a)"] = HistGradientBoostingClassifier(
            learning_rate=0.05, max_depth=3, max_iter=250, random_state=seed)
    models["SVM (RBF)"] = SVC(C=1.0, kernel="rbf", probability=True, random_state=seed)
    models["KNN"] = KNeighborsClassifier(n_neighbors=15)
    return models
