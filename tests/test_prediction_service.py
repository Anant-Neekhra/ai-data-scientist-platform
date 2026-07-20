"""Tests for PredictionService."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline
from ml_pipeline.registry.model_registry import ModelRegistry
from ml_pipeline.prediction.predictor import PredictionService, PredictionInputError


@pytest.fixture
def registry_with_model(tmp_path):
    n = 40
    X = pd.DataFrame({"age": [22 + (i * 4) % 50 for i in range(n)]})
    y = pd.Series([1 if v > 40 else 0 for v in X["age"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))
    pipeline.fit(X, y)

    registry = ModelRegistry(base_dir=tmp_path)
    registry.save_model(
        pipeline, "test.csv", "logistic_regression", "classification",
        {"accuracy": 0.8}, {}, list(X.columns), len(X),
    )
    return registry


def test_predict_single_returns_prediction_and_probability(registry_with_model):
    service = PredictionService(registry_with_model)
    result = service.predict_single("test.csv", {"age": 45})
    assert len(result.predictions) == 1
    assert result.probabilities is not None
    assert 0.0 <= result.probabilities[0] <= 1.0


def test_predict_batch_appends_prediction_columns(registry_with_model):
    service = PredictionService(registry_with_model)
    new_data = pd.DataFrame({"age": [25, 45, 60]})
    result = service.predict_batch("test.csv", new_data)

    assert "prediction" in result.columns
    assert "prediction_probability" in result.columns
    assert len(result) == 3
    assert "age" in result.columns  # original data preserved


def test_missing_required_column_raises(registry_with_model):
    service = PredictionService(registry_with_model)
    bad_data = pd.DataFrame({"wrong_column": [1, 2, 3]})
    with pytest.raises(PredictionInputError):
        service.predict_batch("test.csv", bad_data)


def test_extra_columns_are_tolerated(registry_with_model):
    service = PredictionService(registry_with_model)
    data_with_extra = pd.DataFrame({"age": [30, 40], "unrelated_col": ["x", "y"]})
    result = service.predict_batch("test.csv", data_with_extra)
    assert len(result) == 2  # doesn't raise, extra column just ignored downstream