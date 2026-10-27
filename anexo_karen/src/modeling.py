"""
Modelado supervisado: pipelines, espacios de búsqueda y validación cruzada estratificada.

Todo el preprocesamiento va DENTRO del Pipeline, por lo que en cada fold el escalador y los
codificadores se ajustan solo con las filas de entrenamiento de ese fold (sin leakage).
"""
from scipy.stats import loguniform
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from preprocessing import build_preprocessor

RANDOM_STATE = 42
N_SPLITS = 5

# Métrica principal de selección: PR-AUC (average_precision). Es independiente del umbral y es
# la más informativa con clase minoritaria. Se reportan también ROC-AUC, F1, Recall y Precision.
SCORING = {"pr_auc": "average_precision", "roc_auc": "roc_auc", "f1": "f1",
           "recall": "recall", "precision": "precision"}


def get_cv():
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)


def make_pipe(model):
    return Pipeline([("pre", build_preprocessor()), ("model", model)])


def _search_kwargs():
    return dict(scoring=SCORING, refit="pr_auc", cv=get_cv(), return_train_score=True, n_jobs=-1)


def search_logreg():
    """Regresión logística regularizada: L1 vs L2 y fuerza de regularización C (menor C = más regularización)."""
    pipe = make_pipe(LogisticRegression(solver="liblinear", max_iter=5000, class_weight="balanced",
                                        random_state=RANDOM_STATE))
    grid = {"model__penalty": ["l1", "l2"], "model__C": [0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]}
    return GridSearchCV(pipe, grid, **_search_kwargs())


def search_rf(n_iter=12):
    """Random Forest con parámetros que limitan la complejidad (profundidad, hojas mínimas)."""
    pipe = make_pipe(RandomForestClassifier(class_weight="balanced", n_jobs=1, random_state=RANDOM_STATE))
    dist = {"model__n_estimators": [200, 300],
            "model__max_depth": [4, 6, 8, 10, 12],
            "model__min_samples_leaf": [5, 10, 20, 40],
            "model__max_features": ["sqrt", 0.3]}
    return RandomizedSearchCV(pipe, dist, n_iter=n_iter, random_state=RANDOM_STATE, **_search_kwargs())


def search_hgb(n_iter=25):
    """Gradient Boosting con early stopping, profundidad limitada y regularización L2."""
    pipe = make_pipe(HistGradientBoostingClassifier(
        max_iter=500, early_stopping=True, validation_fraction=0.15, n_iter_no_change=20,
        class_weight="balanced", random_state=RANDOM_STATE))
    dist = {"model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4, 6],
            "model__max_leaf_nodes": [8, 15, 31],
            "model__min_samples_leaf": [20, 50, 100],
            "model__l2_regularization": [0, 1, 10, 50]}
    return RandomizedSearchCV(pipe, dist, n_iter=n_iter, random_state=RANDOM_STATE, **_search_kwargs())


# ----------------------------------------------------------------------------------------------
# Selección parsimoniosa (regla del error estándar) para controlar el overfitting
# ----------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.base import clone


def pick_parsimonious(search):
    """
    Entre todas las configuraciones cuyo PR-AUC de validación está a menos de 1 error estándar
    de la mejor, elige la que tiene MENOR brecha train - validación (la más simple/estable).
    Así no se premia a un modelo que gana 0,001 de PR-AUC a costa de memorizar el train.
    """
    r = pd.DataFrame(search.cv_results_)
    i_best = r["mean_test_pr_auc"].idxmax()
    se = r.loc[i_best, "std_test_pr_auc"] / np.sqrt(N_SPLITS)
    cand = r[r["mean_test_pr_auc"] >= r.loc[i_best, "mean_test_pr_auc"] - se].copy()
    cand["gap"] = cand["mean_train_pr_auc"] - cand["mean_test_pr_auc"]
    return int(cand["gap"].idxmin())


def fit_selected(search, idx, X, y):
    """Reajusta con todo el train la configuración elegida (índice de cv_results_)."""
    params = pd.DataFrame(search.cv_results_).loc[idx, "params"]
    return clone(search.estimator).set_params(**params).fit(X, y)
