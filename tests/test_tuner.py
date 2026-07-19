"""Tests for HyperparameterTuner."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.training.tuner import HyperparameterTuner
from ml_pipeline.training.search_spaces import get_search_space


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    n = 60
    X = pd.DataFrame(
        {
            "feature_a": [i * 1.3 % 17 for i in range(n)],
            "feature_b": ["Delhi", "Mumbai", "Pune"] * 20,
        }
    )
    y = pd.Series([1 if v > 8 else 0 for v in X["feature_a"]])
    return X, y


def test_get_search_space_returns_expected_keys():
    space = get_search_space("random_forest")
    assert "model__n_estimators" in space
    assert "model__max_depth" in space


def test_get_search_space_empty_for_unknown_model():
    assert get_search_space("some_model_that_does_not_exist") == {}


def test_randomized_search_returns_fitted_pipeline(classification_data):
    X, y = classification_data
    tuner = HyperparameterTuner(cv_folds=2)
    result = tuner.tune(
        X, y,
        model_name="random_forest",
        model=RandomForestClassifier(random_state=42),
        problem_type=ProblemType.CLASSIFICATION,
        method="randomized",
        n_iter=3,
    )

    assert result.method == "randomized"
    assert result.n_candidates_tried == 3
    assert "model__n_estimators" in result.best_params
    # The returned pipeline must already be fitted and usable.
    predictions = result.fitted_pipeline.predict(X)
    assert len(predictions) == len(X)


def test_model_without_search_space_skips_tuning_but_still_fits(classification_data):
    X, y = classification_data
    tuner = HyperparameterTuner(cv_folds=2)
    result = tuner.tune(
        X, y,
        model_name="linear_regression",  # deliberately empty search space
        model=LogisticRegression(max_iter=1000),
        problem_type=ProblemType.CLASSIFICATION,
        method="randomized",
    )

    assert result.method == "none"
    assert result.n_candidates_tried == 0
    assert result.best_params == {}
    predictions = result.fitted_pipeline.predict(X)
    assert len(predictions) == len(X)


def test_invalid_method_raises(classification_data):
    X, y = classification_data
    tuner = HyperparameterTuner(cv_folds=2)
    with pytest.raises(ValueError):
        tuner.tune(
            X, y,
            model_name="random_forest",
            model=RandomForestClassifier(),
            problem_type=ProblemType.CLASSIFICATION,
            method="not_a_real_method",
        )


def test_grid_search_tries_all_combinations():
    # Small, deliberate 2x2 grid so we know exactly how many fits to expect.
    n = 60
    X = pd.DataFrame({"feature_a": [i * 1.3 % 17 for i in range(n)]})
    y = pd.Series([1 if v > 8 else 0 for v in X["feature_a"]])

    from ml_pipeline.training import search_spaces
    original_space = search_spaces.SEARCH_SPACES["random_forest"]
    search_spaces.SEARCH_SPACES["random_forest"] = {
        "model__n_estimators": [50, 100],
        "model__max_depth": [3, 5],
    }
    try:
        tuner = HyperparameterTuner(cv_folds=2)
        result = tuner.tune(
            X, y,
            model_name="random_forest",
            model=RandomForestClassifier(random_state=42),
            problem_type=ProblemType.CLASSIFICATION,
            method="grid",
        )
        assert result.n_candidates_tried == 4  # 2 x 2 grid
    finally:
        search_spaces.SEARCH_SPACES["random_forest"] = original_space  # restore