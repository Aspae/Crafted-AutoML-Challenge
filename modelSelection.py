"""
Model selection step of the AutoML pipeline.

Given a problem type and the (already preprocessed) train / validation sets, this module:
    1. builds a list of candidate models, each wrapped in a Pipeline,
    2. tunes the hyperparameters of each candidate with a cross-validated random search,
    3. compares the tuned candidates on the validation set,
    4. refits the winner on train + validation and returns it with a report.

Expected data format:
    - X: np.ndarray or SciPy sparse matrix
    - y: 1-D labels for Regression / Binary / Multi_class (no one-hot),
         2-D indicator matrix (n_samples, n_labels) for Multi_label
"""
import os
import time
import warnings

_n_jobs_env = os.environ.get("AUTOML_N_JOBS", "1")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, _n_jobs_env)

import numpy as np
from joblib import Parallel, delayed
from scipy.sparse import issparse, vstack as sparse_vstack
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.metrics import get_scorer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, ParameterGrid, RandomizedSearchCV, StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MaxAbsScaler

from classesProblems import Classes
from const import rand_state

TIME_BUDGET = int(os.environ.get('AUTOML_TIME_BUDGET', 3600))
N_JOBS = int(os.environ.get('AUTOML_N_JOBS', 1))
N_TRIALS = int(os.environ.get('AUTOML_N_TRIALS', 10))

CV_FOLDS = 5
N_TREES = 100

TIME_MARGIN = 0.8

EVAL_METRICS = {
    Classes.Regression: ["neg_root_mean_squared_error", "r2", "neg_mean_absolute_error"],
    Classes.Binary: ["roc_auc", "accuracy", "f1_macro"],
    Classes.Multi_class: ["f1_macro", "accuracy"],
    Classes.Multi_label: ["f1_macro", "jaccard_samples"]
}
SCORING = {problem: metrics[0] for problem, metrics in EVAL_METRICS.items()}


def _cfg(name, model, param_grid, scale=True, impute=True, dense_only=False):
    """
    Describe one candidate model (small helper that avoids rewriting every dictionary key).

    :param name: Name of the model, used in warnings and in the final report.
    :param model: Unfitted scikit-learn estimator.
    :param param_grid: Hyperparameters to test, as {"model__<param>": [values]}.
                       The "model__" prefix is required because the estimator ends up
                       in a Pipeline step named "model".
    :param scale: If True, put a scaler before the model (needed by linear models,
                  useless for tree-based models).
    :param impute: If True, put a median imputer before the model (for models that
                   cannot handle missing values themselves).
    :param dense_only: If True, the model cannot handle sparse input and is dropped
                       when the data is sparse.
    :return: Dictionary describing the candidate, consumed by get_model_configs.
    """
    return {"name": name, "model": model, "param_grid": param_grid, "scale": scale, "impute": impute, "dense_only": dense_only}

def get_model_configs(typeProblem, sparse=False):
    """
    Build the list of candidate models for a type of problem.

    Candidates are ordered from the fastest to the slowest: if the time budget runs out,
    the slow ones at the end are the ones skipped (see training).

    :param typeProblem: Type of problem (Classes enum).
    :param sparse: True if the training data is a sparse matrix. It changes the
                   preprocessing (no imputer, MaxAbsScaler) and removes the models that
                   cannot handle sparse data.
    :return: List of {"name", "model" (a Pipeline), "param_grid"} dictionaries.
    """
    if typeProblem == Classes.Regression:
        specs = [
            _cfg("Ridge", Ridge(random_state=rand_state), {"model__alpha": [0.1, 1.0, 10.0, 100.0]}),
            _cfg("Lasso", Lasso(max_iter=5000, random_state=rand_state), {"model__alpha": [0.001, 0.01, 0.1, 1.0]}),
            _cfg("Hist Gradient Boosting Regressor", HistGradientBoostingRegressor(random_state=rand_state), {"model__learning_rate": [0.01, 0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63]}, scale=False, impute=False, dense_only=True),
            _cfg("Random Forest Regressor", RandomForestRegressor(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20, None], "model__max_features": ["sqrt", "log2", None]}, scale=False)
        ]
    elif typeProblem in (Classes.Binary, Classes.Multi_class):
        specs = [
            _cfg("Logistic Regression", LogisticRegression(max_iter=1000, random_state=rand_state), {"model__C": [0.01, 0.1, 1.0, 10.0]}),
            _cfg("Hist Gradient Boosting Classifier", HistGradientBoostingClassifier(random_state=rand_state), {"model__learning_rate": [0.01, 0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63]}, scale=False, impute=False, dense_only=True),
            _cfg("Random Forest Classifier", RandomForestClassifier(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20, None], "model__max_features": ["sqrt", "log2", None]}, scale=False)
        ]
    elif typeProblem == Classes.Multi_label:
        specs = [
            _cfg("OneVsRest Logistic Regression", OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=rand_state)), {"model__estimator__C": [0.01, 0.1, 1.0, 10.0]}),
            _cfg("Random Forest Classifier", RandomForestClassifier(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20, None], "model__max_features": ["sqrt", "log2", None]}, scale=False)
        ]
    else:
        raise ValueError(f"Unknown type of problem : {typeProblem}.")

    configs = []
    for cfg in specs:
        if sparse and cfg["dense_only"]:
            continue
        steps = []
        if cfg["impute"] and not sparse:
            steps.append(("imputer", SimpleImputer(strategy="median")))
        if cfg["scale"]:
            steps.append(("scaler", MaxAbsScaler() if sparse else StandardScaler()))
        steps.append(("model", cfg["model"]))
        configs.append({"name": cfg["name"], "model": Pipeline(steps), "param_grid": cfg["param_grid"]})
    return configs

def get_cv(typeProblem):
    """
    Build the cross-validation splitter used by the hyperparameter search.

    Shuffling with a fixed seed keeps the folds identical from one run to the next.

    :param typeProblem: Type of problem (Classes enum).
    :return: StratifiedKFold for classification (keeps class proportions in every fold),
             kfold for regression and multi-label.
    """
    if typeProblem in (Classes.Binary, Classes.Multi_class):
        return StratifiedKFold(CV_FOLDS, shuffle=True, random_state=rand_state)
    return KFold(CV_FOLDS, shuffle=True, random_state=rand_state)

def train_model(name, model, param_grid, X_train, y_train, scoring, cv, deadline):
    """
    Tune the hyperparameters of one candidate with a cross-validated random search.

    :param name: Name of the model.
    :param model: Pipeline to tune.
    :param param_grid: Hyperparameters to test.
    :param X_train: Training features.
    :param y_train: Training targets.
    :param scoring: Name of the scikit-learn metric to maximize.
    :param cv: Cross-validation splitter.
    :param deadline: Timestamp (time.time()) after which new searches must not start.
    :return: {"name", "estimator", "cv_score", "best_params"} or None if the model was
             skipped (deadline passed) or failed.
    """
    if time.time() > deadline:
        warnings.warn(f"[{name}] ignored : time budget exceeded.")
        return None

    n_iter = min(N_TRIALS, len(ParameterGrid(param_grid)))
    search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = n_iter,
        cv = cv,
        scoring = scoring,
        random_state=rand_state,
        refit=True
    )

    try:
        search.fit(X_train, y_train)
    except Exception as exc:
        warnings.warn(f"[{name}] training failure : {exc}.")
        return None

    return {
        "name": name,
        "estimator": search.best_estimator_,
        "cv_score": search.best_score_,
        "best_params": search.best_params_
    }

def training(models, X_train, y_train, scoring, cv, deadline):
    trained_models = Parallel(N_JOBS)(
        delayed(train_model)(cfg["name"], cfg["model"], cfg["param_grid"], X_train, y_train, scoring, cv, deadline if i > 0 else float("inf"))
        for i, cfg in enumerate(models)
    )
    return [r for r in trained_models if r is not None]

def launch_training(typeProblem: Classes, X_train, y_train, deadline):
    configs = get_model_configs(typeProblem, sparse=issparse(X_train))
    return training(configs, X_train, y_train, SCORING[typeProblem], get_cv(typeProblem), deadline)

def eval_model(model, X, y, metrics):
    return {metric: float(get_scorer(metric)(model, X, y)) for metric in metrics}

def evaluation(models, X, y, metrics):
    return Parallel(N_JOBS)(
        delayed(eval_model)(mdl["estimator"], X, y, metrics) for mdl in models
    )

def launch_evaluation(typeProblem: Classes, trained_models, X, y):
    return evaluation(trained_models, X, y, EVAL_METRICS[typeProblem])

def _stack(a, b):
    if issparse(a) or issparse(b):
        return sparse_vstack([a,b]).tocsr()
    return np.concatenate([a,b], axis=0)

def best_model(typeProblem, X_train, X_val, y_train, y_val):

    deadline = time.time() + TIME_BUDGET * TIME_MARGIN

    trained_models = launch_training(typeProblem, X_train, y_train, deadline)
    if not trained_models:
        raise RuntimeError("No model could be trained.")

    val_scores = launch_evaluation(typeProblem, trained_models, X_val, y_val)
    for entry, scores in zip(trained_models, val_scores):
        entry["val_scores"] = scores

    main_metric = SCORING[typeProblem]
    ranking = sorted(trained_models, key=lambda e: e["val_scores"][main_metric], reverse=True)

    final_model = clone(ranking[0]["estimator"])
    final_model.fit(_stack(X_train, X_val), np.concatenate([y_train, y_val], axis=0))

    report = [{k: v for k, v in entry.items() if k != "estimator"} for entry in ranking]
    return final_model, report
