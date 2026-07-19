"""Tests for ModelTrainer and the leaderboard builder."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor

from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.training.trainer import ModelTrainer, ModelResult
from ml_pipeline.training.comparison import build_leaderboard


@pytest.fixture
def classification_data() -> tuple[pd.DataFrame, pd.Series]:
    n = 60
    X = pd.DataFrame(
        {
            "feature_a": [i * 1.3 % 17 for i in range(n)],
            "feature_b": ["Delhi", "Mumbai", "Pune"] * 20,
        }
    )
    # feature_a alone almost perfectly separates the two classes,
    # so a real model should score meaningfully above a dummy baseline.
    y = pd.Series([1 if v > 8 else 0 for v in X["feature_a"]])
    return X, y


@pytest.fixture
def regression_data() -> tuple[pd.DataFrame, pd.Series]:
    n = 60
    X = pd.DataFrame({"feature_a": [i * 2.1 for i in range(n)]})
    y = pd.Series([v * 3 + 5 for v in X["feature_a"]])  # clean linear relationship
    return X, y


def test_compare_models_returns_result_per_candidate(classification_data):
    X, y = classification_data
    small_candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
    }
    trainer = ModelTrainer(cv_folds=2)
    results = trainer.compare_models(X, y, ProblemType.CLASSIFICATION, models=small_candidates)

    assert len(results) == 2
    names = {r.model_name for r in results}
    assert names == {"logistic_regression", "dummy_baseline"}


def test_real_model_beats_dummy_baseline_on_separable_data(classification_data):
    X, y = classification_data
    small_candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
    }
    trainer = ModelTrainer(cv_folds=3)
    results = trainer.compare_models(X, y, ProblemType.CLASSIFICATION, models=small_candidates)

    scores = {r.model_name: r.mean_scores["accuracy"] for r in results}
    assert scores["logistic_regression"] > scores["dummy_baseline"]


def test_regression_candidate_fits_clean_linear_relationship(regression_data):
    X, y = regression_data
    small_candidates = {
        "linear_regression": LinearRegression(),
        "dummy_baseline": DummyRegressor(strategy="mean"),
    }
    trainer = ModelTrainer(cv_folds=3)
    results = trainer.compare_models(X, y, ProblemType.REGRESSION, models=small_candidates)

    scores = {r.model_name: r.mean_scores["r2"] for r in results}
    assert scores["linear_regression"] > scores["dummy_baseline"]
    assert scores["linear_regression"] > 0.9  # near-perfect linear data


def test_fit_best_model_selects_highest_scoring(classification_data):
    X, y = classification_data
    small_candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "dummy_baseline": DummyClassifier(strategy="most_frequent"),
    }
    trainer = ModelTrainer(cv_folds=2)
    results = trainer.compare_models(X, y, ProblemType.CLASSIFICATION, models=small_candidates)
    best_name, fitted_pipeline = trainer.fit_best_model(
        X, y, results, ProblemType.CLASSIFICATION, models=small_candidates
    )

    assert best_name == "logistic_regression"
    # The returned pipeline should already be fitted and usable for prediction.
    predictions = fitted_pipeline.predict(X)
    assert len(predictions) == len(X)


def test_fit_best_model_raises_on_empty_results():
    trainer = ModelTrainer()
    with pytest.raises(ValueError):
        trainer.fit_best_model(
            pd.DataFrame({"a": [1]}), pd.Series([0]), [], ProblemType.CLASSIFICATION
        )


def test_leaderboard_sorted_best_first():
    results = [
        ModelResult(model_name="weak", mean_scores={"accuracy": 0.60, "f1_weighted": 0.55}, std_scores={"accuracy": 0.02, "f1_weighted": 0.02}, fit_time_seconds=0.1),
        ModelResult(model_name="strong", mean_scores={"accuracy": 0.92, "f1_weighted": 0.91}, std_scores={"accuracy": 0.01, "f1_weighted": 0.01}, fit_time_seconds=0.3),
    ]
    leaderboard = build_leaderboard(results, ProblemType.CLASSIFICATION)

    assert leaderboard.iloc[0]["model"] == "strong"
    assert leaderboard.iloc[-1]["model"] == "weak"


def test_leaderboard_ranks_regression_by_neg_rmse():
    results = [
        ModelResult(model_name="high_error", mean_scores={"neg_rmse": -500.0, "r2": 0.4}, std_scores={"neg_rmse": 10.0, "r2": 0.05}, fit_time_seconds=0.1),
        ModelResult(model_name="low_error", mean_scores={"neg_rmse": -50.0, "r2": 0.9}, std_scores={"neg_rmse": 2.0, "r2": 0.02}, fit_time_seconds=0.2),
    ]
    leaderboard = build_leaderboard(results, ProblemType.REGRESSION)

    # -50 > -500, so low_error should rank first despite the "negative" scores.
    assert leaderboard.iloc[0]["model"] == "low_error"