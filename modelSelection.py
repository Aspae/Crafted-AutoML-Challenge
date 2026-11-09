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
from tqdm import tqdm

from classesProblems import Classes
from const import rand_state, verbose, EVAL_METRICS

TIME_BUDGET = int(os.environ.get('AUTOML_TIME_BUDGET', 3600))
N_JOBS = int(os.environ.get('AUTOML_N_JOBS', 1))
N_TRIALS = int(os.environ.get('AUTOML_N_TRIALS', 10))

SHOW_PROGRESS = os.environ.get('AUTOML_PROGRESS', '1') != '0'

CV_FOLDS = 5
N_TREES = 100

TIME_MARGIN = 0.8

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
            _cfg("Random Forest Regressor", RandomForestRegressor(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20], "model__max_features": ["sqrt", "log2"]}, scale=False)
        ]
    elif typeProblem in (Classes.Binary, Classes.Multi_class):
        specs = [
            _cfg("Logistic Regression", LogisticRegression(max_iter=1000, random_state=rand_state), {"model__C": [0.01, 0.1, 1.0, 10.0]}),
            _cfg("Hist Gradient Boosting Classifier", HistGradientBoostingClassifier(random_state=rand_state), {"model__learning_rate": [0.01, 0.05, 0.1], "model__max_leaf_nodes": [15, 31, 63]}, scale=False, impute=False, dense_only=True),
            _cfg("Random Forest Classifier", RandomForestClassifier(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20], "model__max_features": ["sqrt", "log2"]}, scale=False)
        ]
    elif typeProblem == Classes.Multi_label:
        specs = [
            _cfg("OneVsRest Logistic Regression", OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=rand_state)), {"model__estimator__C": [0.01, 0.1, 1.0, 10.0]}),
            _cfg("Random Forest Classifier", RandomForestClassifier(n_estimators=N_TREES, random_state=rand_state), {"model__max_depth": [10, 20], "model__max_features": ["sqrt", "log2"]}, scale=False)
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
        refit=True,
        verbose=verbose
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
    """
    Run train_model on every candidate, in parallel on N_JOBS cores.

    :param models: Candidates, as returned by get_model_configs.
    :param X_train: Training features.
    :param y_train: Training targets.
    :param scoring: Name of the scikit-learn metric to maximize.
    :param cv: Cross-validation splitter.
    :param deadline: Timestamp after which new searches must not start.
    :return: List of train_model results, without the skipped / failed models.
    """
    jobs = Parallel(N_JOBS, return_as="generator")(
        delayed(train_model)(cfg["name"], cfg["model"], cfg["param_grid"], X_train, y_train, scoring, cv, deadline if i > 0 else float("inf"))
        for i, cfg in enumerate(models)
    )

    trained_models = []
    with tqdm(total=len(models), unit="model", disable=not SHOW_PROGRESS) as pbar:
        sequential = N_JOBS == 1
        pbar.set_description(f"Tuning {models[0]['name']}" if sequential else "Tuning models")
        for i, result in enumerate(jobs):
            if result is not None:
                trained_models.append(result)
                pbar.set_postfix(last=result["name"], cv=f"{result['cv_score']:.4f}")
            pbar.update(1)
            if sequential and i + 1 < len(models):
                pbar.set_description(f"Tuning {models[i+1]['name']}")

    return trained_models

def launch_training(typeProblem: Classes, X_train, y_train, deadline):
    """
    Build the candidates matching the problem and the data, then train them all.

    :param typeProblem: Type of problem (Classes enum).
    :param X_train: Training features.
    :param y_train: Training targets.
    :param deadline: Timestamp after which new searches must not start.
    :return: List of train_model results.
    """
    configs = get_model_configs(typeProblem, sparse=issparse(X_train))
    return training(configs, X_train, y_train, SCORING[typeProblem], get_cv(typeProblem), deadline)

def eval_model(model, X, y, metrics):
    """
    Score one fitted model on a dataset.

    :param model: Fitted estimator.
    :param X: Features to evaluate on.
    :param y: True targets.
    :param metrics: Names of the scikit-learn metrics to compute.
    :return: {metric name: score}, with higher = better for every metric.
    """
    return {metric: float(get_scorer(metric)(model, X, y)) for metric in metrics}

def evaluation(models, X, y, metrics):
    """
    Score every trained model on a dataset (in the same order as `models`).

    :param models: Results of train_model (dictionaries with an "estimator" key).
    :param X: Features to evaluate on.
    :param y: True targets.
    :param metrics: Names of the scikit-learn metrics to compute.
    :return: List of {metric name: score}, one per model.
    """
    return Parallel(N_JOBS)(
        delayed(eval_model)(mdl["estimator"], X, y, metrics) for mdl in models
    )

def launch_evaluation(typeProblem: Classes, trained_models, X, y):
    """
    Evaluate the trained models with the metrics of the given type of problem.

    :param typeProblem: Type of problem (Classes enum).
    :param trained_models: Results of launch_training.
    :param X: Features to evaluate on (validation set).
    :param y: True targets.
    :return: List of {metric name: score}, one per model.
    """
    return evaluation(trained_models, X, y, EVAL_METRICS[typeProblem])

def _stack(a, b):
    """
    Concatenate two sets of rows (e.g. X_train and X_val), sparse or dense.

    :param a: First array or sparse matrix.
    :param b: Second array or sparse matrix.
    :return: CSR matrix if any input is sparse, otherwise a NumPy array.
    """
    if issparse(a) or issparse(b):
        return sparse_vstack([a,b]).tocsr()
    return np.concatenate([a,b], axis=0)

def best_model(typeProblem, X_train, X_val, y_train, y_val):
    """
    Train and compare all candidates, then refit the best one on train + validation.

    The validation set is never seen by the hyperparameter search.

    :param typeProblem: Type of problem (Classes enum).
    :param X_train: Training features.
    :param X_val: Validation features.
    :param y_train: Training targets.
    :param y_val: Validation targets.
    :return: (final_model, report). final_model is the winner refitted on train + validation.
             report is the list of models sorted from best to worst, with their name,
             cv_score, best_params and val_scores (for automl.eval()).
    """
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
