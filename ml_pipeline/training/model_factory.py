"""
Defines the candidate model set per problem type. Centralizing the
list here means adding, removing, or swapping a candidate model is a
one-line change here, not a hunt through training/comparison code.
"""

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

from config.settings import RANDOM_STATE
from ml_pipeline.data.schema_detector import ProblemType
from utils.logger import get_logger

logger = get_logger(__name__)

def _build_keras_classifier_model(meta, hidden_units:int = 32):
    """
    Builds a small feed-forward neural network for classification.
    scikeras calls this lazily, at fit time, passing `meta` -- a dict
    it populates with facts about the data (feature count, class
    count) that we can't know until fit() actually runs.
    """
    from tensorflow import keras

    n_features = meta["n_features_in_"]
    n_classes = meta.get("n_classes_", 2)
    is_binary = n_classes <= 2

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(n_features,)),
            keras.layers.Dense(hidden_units, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_units // 2, activation="relu"),
            keras.layers.Dense(1 if is_binary else n_classes, activation="sigmoid" if is_binary else "softmax",),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy" if is_binary else "sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

def _build_keras_regressor_model(meta, hidden_units:int = 32):
    """Small feed-forward neural network for regression."""
    from tensorflow import keras

    n_features = meta["n_features_in_"]

    model = keras.Sequential(
        [
            keras.layers.InputLayer(input_shape=(n_features,)),
            keras.layers.Dense(hidden_units, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(hidden_units // 2, activation="relu"),
            keras.layers.Dense(1),
        ]
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def get_candidate_models(problem_type: ProblemType) -> dict:
    """
    Return the dict of {name: unfitted estimator} to compare for a
    given problem type. Every estimator here follows the sklearn
    fit/predict interface, including the Keras models once wrapped
    by scikeras -- which is exactly why they can slot into the same
    cross_validate() call as everything else with zero special-casing.
    """
    if problem_type == ProblemType.CLASSIFICATION:
        candidates: dict = {
            "logistic_regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
            "xgboost": XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss", verbosity=0),
            "lightgbm": LGBMClassifier(random_state=RANDOM_STATE, verbose=-1),
            "catboost": CatBoostClassifier(random_state=RANDOM_STATE, verbose=0),
        }
        try:
            from scikeras.wrappers import KerasClassifier

            candidates["neural_network"] = KerasClassifier(
                model=_build_keras_classifier_model,
                epochs=30,
                batch_size=32,
                verbose=0,
                random_state=RANDOM_STATE,
            )
        except ImportError:
            logger.warning("scikeras/tensorflow not installed -- skipping neural_network candidate")
        return candidates

    # Regression
    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE),
        "xgboost": XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
        "lightgbm": LGBMRegressor(random_state=RANDOM_STATE, verbose=-1),
        "catboost": CatBoostRegressor(random_state=RANDOM_STATE, verbose=0),
    }
    try:
        from scikeras.wrappers import KerasRegressor

        candidates["neural_network"] = KerasRegressor(
            model=_build_keras_regressor_model,
            epochs=30,
            batch_size=32,
            verbose=0,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        logger.warning("scikeras/tensorflow not installed -- skipping neural_network candidate")
    return candidates