"""
Defines the hyperparameter search space for each candidate model.
Centralizing this here means the tuner doesn't need to know
anything about what a good XGBoost learning_rate range looks like
-- that domain knowledge lives in exactly one place.
"""

from scipy.stats import randint, uniform

from ml_pipeline.data.schema_detector import ProblemType

# Prefixed with "model__" because the model lives inside a Pipeline
# step named "model" -- sklearn's parameter search tools address
# nested pipeline steps with a "stepname__paramname" convention.
_RANDOM_FOREST_SPACE = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 20),
    "model__min_samples_split": randint(2, 15),
    "model__min_samples_leaf": randint(1, 10),
}

_XGBOOST_SPACE = {
    "model__n_estimators": randint(100, 500),
    "model__max_depth": randint(3, 12),
    "model__learning_rate": uniform(0.01, 0.29),  # 0.01 to 0.30
    "model__subsample": uniform(0.6, 0.4),         # 0.6 to 1.0
}

_LIGHTGBM_SPACE = {
    "model__n_estimators": randint(100, 500),
    "model__num_leaves": randint(15, 100),
    "model__learning_rate": uniform(0.01, 0.29),
    "model__subsample": uniform(0.6, 0.4),
}

_CATBOOST_SPACE = {
    "model__iterations": randint(100, 500),
    "model__depth": randint(3, 10),
    "model__learning_rate": uniform(0.01, 0.29),
}

_LOGISTIC_REGRESSION_SPACE = {
    "model__C": uniform(0.01, 9.99),  # 0.01 to 10.0
}

_LINEAR_REGRESSION_SPACE: dict = {}  # LinearRegression has no meaningful hyperparameters to tune

SEARCH_SPACES: dict[str, dict] = {
    "random_forest": _RANDOM_FOREST_SPACE,
    "xgboost": _XGBOOST_SPACE,
    "lightgbm": _LIGHTGBM_SPACE,
    "catboost": _CATBOOST_SPACE,
    "logistic_regression": _LOGISTIC_REGRESSION_SPACE,
    "linear_regression": _LINEAR_REGRESSION_SPACE,
}


def get_search_space(model_name: str) -> dict:
    """
    Return the hyperparameter distribution dict for a given model name.

    Args:
        model_name: key matching get_candidate_models() naming, e.g.
            'random_forest', 'xgboost'.

    Returns:
        Dict of {param_name: scipy distribution or list}, empty if
        the model has no tunable space defined (e.g. linear_regression,
        or neural_network -- Keras architectures aren't tuned via
        this sklearn-style search; that's a separate, more involved
        technique intentionally out of scope here).
    """
    return SEARCH_SPACES.get(model_name, {})