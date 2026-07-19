"""Tests for ShapExplainer and its output normalization."""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression

from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline
from ml_pipeline.explainability.shap_explainer import ShapExplainer


@pytest.fixture
def classification_pipeline_and_data():
    n = 60
    X = pd.DataFrame(
        {
            "age": [22 + (i * 4) % 50 for i in range(n)],
            "city": ["Delhi", "Mumbai", "Pune"] * (n // 3),
        }
    )
    y = pd.Series([1 if v > 40 else 0 for v in X["age"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", RandomForestClassifier(n_estimators=50, random_state=42)))
    pipeline.fit(X, y)
    return pipeline, X, y


def test_explainer_rejects_non_tree_model():
    n = 40
    X = pd.DataFrame({"age": [22 + (i * 4) % 50 for i in range(n)]})
    y = pd.Series([1 if v > 40 else 0 for v in X["age"]])
    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))
    pipeline.fit(X, y)

    with pytest.raises(ValueError):
        ShapExplainer(pipeline, X)


def test_global_explanation_shapes_match(classification_pipeline_and_data):
    pipeline, X, y = classification_pipeline_and_data
    explainer = ShapExplainer(pipeline, X)
    explanation = explainer.explain_global(X, sample_size=30)

    n_features = len(explanation.feature_names)
    assert explanation.mean_abs_shap.shape == (n_features,)
    assert explanation.shap_values.shape[1] == n_features
    assert explanation.shap_values.shape[0] == len(explanation.feature_values)


def test_global_explanation_respects_sample_size(classification_pipeline_and_data):
    pipeline, X, y = classification_pipeline_and_data
    explainer = ShapExplainer(pipeline, X)
    explanation = explainer.explain_global(X, sample_size=20)
    assert explanation.shap_values.shape[0] == 20


def test_local_explanation_single_row(classification_pipeline_and_data):
    pipeline, X, y = classification_pipeline_and_data
    explainer = ShapExplainer(pipeline, X)
    row = X.iloc[[0]]  # double brackets -> keep as 1-row DataFrame
    explanation = explainer.explain_local(row)

    assert len(explanation.shap_values) == len(explanation.feature_names)


def test_local_explanation_rejects_multi_row(classification_pipeline_and_data):
    pipeline, X, y = classification_pipeline_and_data
    explainer = ShapExplainer(pipeline, X)
    with pytest.raises(ValueError):
        explainer.explain_local(X.iloc[0:3])


def test_local_explanation_sums_to_prediction(classification_pipeline_and_data):
    """The core mathematical guarantee of SHAP: base_value + sum(shap_values)
    should equal the model's actual predicted probability for that row."""
    pipeline, X, y = classification_pipeline_and_data
    explainer = ShapExplainer(pipeline, X)
    row = X.iloc[[0]]
    explanation = explainer.explain_local(row)

    actual_proba = pipeline.predict_proba(row)[0, 1]
    assert explanation.predicted_value == pytest.approx(actual_proba, abs=0.02)


def test_regression_explainer_works():
    n = 50
    X = pd.DataFrame({"size": [30 + i for i in range(n)]})
    y = pd.Series([100 + 5 * v for v in X["size"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", RandomForestRegressor(n_estimators=50, random_state=42)))
    pipeline.fit(X, y)

    explainer = ShapExplainer(pipeline, X)
    explanation = explainer.explain_global(X, sample_size=25)
    assert explanation.shap_values.shape[0] == 25