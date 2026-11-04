from sklearn.linear_model import Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV, cross_val_score
from joblib import Parallel, delayed
from classesProblems import Classes
import optuna
import os

"""
Pour la régression classique:
    - neg_mean_squared_error ou r2 ou RMSE
    
Pour la classification binaire et multi-classe:
    - accuracy ou roc_auc => binaire
    - accuracy ou f1_macro => multi-class

Pour la classification multi-label:
    - f1_macro ou jaccard_score
"""

rand_state = 42

regression_models = [
    {
        "name": "Ridge",
        "model": Ridge(random_state=rand_state),
        "param_grid": {
            "alpha": [0.1, 1.0, 10.0, 100.0]
        }
    },
    {
        "name": "Lasso",
        "model": Lasso(random_state=rand_state),
        "param_grid": {
            "alpha": [0.001, 0.01, 0.1, 1.0]
        }
    },
    {
        "name": "Random Forest Regressor",
        "model": RandomForestRegressor(n_estimators=50, random_state=rand_state),
        "param_grid": {
            "max_depth": [10, 20, None],
            "max_features": ['sqrt', 'log2', None]
        }
    },
    {
        "name": "Hist Gradient Boosting Regressor",
        "model": HistGradientBoostingRegressor(random_state=rand_state),
        "param_grid": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63]
        }
    }
]

multi_class_models = [
    {
        "name": "Logistic Regression",
        "model": LogisticRegression(max_iter=1000, random_state=rand_state),
        "param_grid": {
            "C": [0.01, 0.1, 1.0, 10.0]
        }
    },
    {
        "name": "Random Forest Classifier",
        "model": RandomForestClassifier(random_state=rand_state),
        "param_grid": {
            "max_depth": [10, 20, None],
            "max_features": ['sqrt', 'log2']
        }
    },
    {
        "name": "Hist Gradient Boosting Classifier",
        "model": HistGradientBoostingClassifier(random_state=rand_state),
        "param_grid": {
            "learning_rate": [0.01, 0.05, 0.1],
            "max_leaf_nodes": [15, 31, 63]
        }
    }
]

multi_label_models = [
    {
        "name": "Random Forest Classifier",
        "model": RandomForestClassifier(random_state=rand_state),
        "param_grid": {
            "max_depth": [10, 20, None],
            "max_features": ['sqrt', 'log2']
        }
    }
]

TIME_BUDGET = int(os.environ.get('AUTOML_TIME_BUDGET', 3600))
N_JOBS = int(os.environ.get('AUTOML_N_JOBS', 1))
N_TRIALS = int(os.environ.get('AUTOML_N_TRIALS', 10))

def train_model(model, param_grid, X_train, y_train, score):
    search = RandomizedSearchCV(
        estimator = model,
        param_distributions = param_grid,
        n_iter = N_TRIALS,
        cv = 5,
        scoring = score,
        random_state=rand_state,
    )
    
    search.fit(X_train, y_train)
    return search

def training(models, X_train, y_train, score):
    trained_models = Parallel(N_JOBS)(
        delayed(train_model)(cfg["model"], cfg["param_grid"], X_train, y_train, score)
        for cfg in models
    )
    return trained_models

def launch_training(typeProblem: Classes, X_train, y_train):
    
    match typeProblem:
        case Classes.Regression:
            trained_models = training(regression_models, X_train, y_train, "rmse")
        case Classes.Binary:
            trained_models = training(multi_class_models, X_train, y_train, "roc_auc")
        case Classes.Multi_class:
            trained_models = training(multi_class_models, X_train, y_train, "f1_macro")
        case Classes.Multi_label:
            trained_models = training(multi_label_models, X_train, y_train, "f1_macro")
            
    return trained_models

def eval_model(model, X, y, metric):
    pass

def evaluation(models, X, y, metric):
    results = Parallel(N_JOBS)(
        delayed(eval_model)(mdl, X, y, metric) for mdl in models
    )
    return results
    
def launch_evaluation(typeProblem: Classes, X, y):
    pass

def best_model(typeProblem, X_train, X_val, y_train, y_val):
    
    trained_models = launch_training(typeProblem, X_train, y_train)
