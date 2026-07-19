"""Tests for ModelRegistry — save/load/list/versioning behavior."""

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline
from ml_pipeline.registry.model_registry import ModelRegistry, _sanitize_dataset_name


@pytest.fixture
def fitted_pipeline():
    n = 40
    X = pd.DataFrame({"age": [22 + (i * 4) % 50 for i in range(n)]})
    y = pd.Series([1 if v > 40 else 0 for v in X["age"]])
    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))
    pipeline.fit(X, y)
    return pipeline, X, y


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(base_dir=tmp_path)


def test_sanitize_dataset_name_strips_unsafe_characters():
    assert _sanitize_dataset_name("Titanic Survival Data.csv") == "titanic_survival_data"
    assert _sanitize_dataset_name("weird:name/here.csv") == "weird_name_here"


def test_save_and_load_model_roundtrip(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    registry.save_model(
        pipeline, dataset_name="titanic.csv", model_name="logistic_regression",
        problem_type="classification", metrics={"accuracy": 0.81},
        hyperparameters={"max_iter": 1000}, feature_columns=list(X.columns),
        n_train_rows=len(X),
    )

    loaded = registry.load_model("titanic.csv")
    predictions = loaded.predict(X)
    assert len(predictions) == len(X)


def test_versions_increment(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    meta1 = registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X),
    )
    meta2 = registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.83}, {}, list(X.columns), len(X),
    )
    assert meta1.version == 1
    assert meta2.version == 2


def test_datasets_have_independent_version_counters(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    meta_titanic = registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X),
    )
    meta_housing = registry.save_model(
        pipeline, "housing.csv", "logistic_regression", "classification",
        {"accuracy": 0.75}, {}, list(X.columns), len(X),
    )
    # Different datasets each start their own numbering at 1 -- one
    # dataset's history shouldn't affect another's.
    assert meta_titanic.version == 1
    assert meta_housing.version == 1


def test_load_without_version_gets_current_best(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X), set_as_best=True,
    )
    registry.save_model(
        pipeline, "titanic.csv", "random_forest", "classification",
        {"accuracy": 0.83}, {}, list(X.columns), len(X), set_as_best=False,
    )

    meta = registry.load_metadata("titanic.csv")
    # version 1 was explicitly kept as best; version 2 opted out via set_as_best=False
    assert meta.version == 1
    assert meta.model_name == "logistic_regression"


def test_set_best_promotes_a_version(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X),
    )
    registry.save_model(
        pipeline, "titanic.csv", "random_forest", "classification",
        {"accuracy": 0.83}, {}, list(X.columns), len(X), set_as_best=False,
    )

    registry.set_best("titanic.csv", version=2)
    meta = registry.load_metadata("titanic.csv")
    assert meta.version == 2
    assert meta.model_name == "random_forest"


def test_list_versions_returns_newest_first(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    for i in range(3):
        registry.save_model(
            pipeline, "titanic.csv", "logistic_regression", "classification",
            {"accuracy": 0.80 + i * 0.01}, {}, list(X.columns), len(X),
        )
    versions = registry.list_versions("titanic.csv")
    assert [v.version for v in versions] == [3, 2, 1]


def test_load_missing_dataset_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.load_model("nonexistent_dataset.csv")


def test_load_specific_missing_version_raises(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X),
    )
    with pytest.raises(FileNotFoundError):
        registry.load_model("titanic.csv", version=99)


def test_delete_version_falls_back_to_newest_remaining(registry, fitted_pipeline):
    pipeline, X, y = fitted_pipeline
    registry.save_model(
        pipeline, "titanic.csv", "logistic_regression", "classification",
        {"accuracy": 0.80}, {}, list(X.columns), len(X),
    )
    registry.save_model(
        pipeline, "titanic.csv", "random_forest", "classification",
        {"accuracy": 0.83}, {}, list(X.columns), len(X), set_as_best=True,
    )
    # Delete the current best (version 2) -- registry must fall back
    # to the newest remaining version (version 1) rather than leaving
    # current_best pointing at a version that no longer exists.
    registry.delete_version("titanic.csv", version=2)

    meta = registry.load_metadata("titanic.csv")
    assert meta.version == 1
    remaining = registry.list_versions("titanic.csv")
    assert len(remaining) == 1