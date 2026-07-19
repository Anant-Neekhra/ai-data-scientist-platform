"""Tests for classification/regression metrics, feature importance, and learning curves."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier

from ml_pipeline.evaluation.classification_metrics import compute_classification_metrics
from ml_pipeline.evaluation.regression_metrics import compute_regression_metrics
from ml_pipeline.evaluation.feature_importance import extract_feature_importance
from ml_pipeline.evaluation.curves import compute_learning_curve
from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline


def test_classification_metrics_perfect_predictions():
    y_true = [0, 1, 0, 1, 1]
    y_pred = [0, 1, 0, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.confusion_matrix.trace() == len(y_true)  # all on diagonal


def test_classification_metrics_roc_auc_binary_only():
    y_true = [0, 1, 0, 1]
    y_proba = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]])
    metrics = compute_classification_metrics(y_true, [0, 1, 0, 1], y_proba)
    assert metrics.roc_auc is not None
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_classification_metrics_no_roc_auc_without_proba():
    metrics = compute_classification_metrics([0, 1, 0], [0, 1, 1])
    assert metrics.roc_auc is None


def test_regression_metrics_perfect_predictions():
    y_true = [10, 20, 30, 40]
    metrics = compute_regression_metrics(y_true, y_true)
    assert metrics.mae == pytest.approx(0.0)
    assert metrics.rmse == pytest.approx(0.0)
    assert metrics.r2 == pytest.approx(1.0)
    assert metrics.mape == pytest.approx(0.0)


def test_regression_metrics_mape_none_with_zero_target():
    y_true = [0, 10, 20]
    y_pred = [1, 11, 19]
    metrics = compute_regression_metrics(y_true, y_pred)
    assert metrics.mape is None  # zero in y_true -> MAPE undefined
    assert metrics.mae is not None  # other metrics still computed


def test_feature_importance_extracted_for_tree_model():
    n = 40
    X = pd.DataFrame(
        {
            "age": [22 + (i * 3) % 40 for i in range(n)],
            "city": ["Delhi", "Mumbai"] * (n // 2),
        }
    )
    y = pd.Series([1 if v > 35 else 0 for v in X["age"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", RandomForestClassifier(n_estimators=50, random_state=42)))
    pipeline.fit(X, y)

    importance_df = extract_feature_importance(pipeline, X)
    assert importance_df is not None
    assert "feature" in importance_df.columns
    assert "importance" in importance_df.columns
    # Sorted descending.
    assert (importance_df["importance"].diff().dropna() <= 0).all()


def test_feature_importance_extracted_for_linear_model():
    n = 40
    X = pd.DataFrame({"age": [22 + (i * 3) % 40 for i in range(n)]})
    y = pd.Series([1 if v > 35 else 0 for v in X["age"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))
    pipeline.fit(X, y)

    importance_df = extract_feature_importance(pipeline, X)
    assert importance_df is not None
    assert len(importance_df) > 0


def test_learning_curve_returns_expected_structure():
    n = 60
    X = pd.DataFrame({"age": [22 + (i * 3) % 40 for i in range(n)]})
    y = pd.Series([1 if v > 35 else 0 for v in X["age"]])

    pipeline = build_preprocessing_pipeline(X)
    pipeline.steps.append(("model", LogisticRegression(max_iter=1000)))

    curve = compute_learning_curve(pipeline, X, y, ProblemType.CLASSIFICATION, cv_folds=3)
    assert len(curve["train_sizes"]) == len(curve["train_scores_mean"])
    assert len(curve["train_sizes"]) == len(curve["val_scores_mean"])