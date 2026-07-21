"""Tests for MLflowTracker — verifying runs actually get logged."""

import pandas as pd
import pytest
import mlflow
from sklearn.linear_model import LogisticRegression

from ml_pipeline.tracking.mlflow_tracker import MLflowTracker


@pytest.fixture
def tracker(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.tracking.mlflow_tracker.MLRUNS_DIR", tmp_path / "mlruns"
    )
    return MLflowTracker(experiment_name="test_experiment")


def test_start_run_logs_params_and_metrics(tracker):
    with tracker.start_run(run_name="test_run") as run:
        tracker.log_params({"model_name": "logistic_regression", "cv_folds": 5})
        tracker.log_metrics({"accuracy": 0.81})
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    logged_run = client.get_run(run_id)
    assert logged_run.data.params["model_name"] == "logistic_regression"
    assert logged_run.data.metrics["accuracy"] == pytest.approx(0.81)


def test_tags_are_logged(tracker):
    with tracker.start_run(run_name="tagged_run") as run:
        tracker.log_tags({"dataset": "titanic.csv", "phase": "comparison"})
        run_id = run.info.run_id

    client = mlflow.tracking.MlflowClient()
    logged_run = client.get_run(run_id)
    assert logged_run.data.tags["dataset"] == "titanic.csv"


def test_compare_models_logs_a_run_per_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "ml_pipeline.tracking.mlflow_tracker.MLRUNS_DIR", tmp_path / "mlruns"
    )
    from ml_pipeline.training.trainer import ModelTrainer
    from ml_pipeline.data.schema_detector import ProblemType

    n = 50
    X = pd.DataFrame({"age": [22 + (i * 4) % 50 for i in range(n)]})
    y = pd.Series([1 if v > 40 else 0 for v in X["age"]])

    tracker = MLflowTracker(experiment_name="test_compare")
    small_candidates = {"logistic_regression": LogisticRegression(max_iter=1000)}

    trainer = ModelTrainer(cv_folds=2)
    trainer.compare_models(
        X, y, ProblemType.CLASSIFICATION, models=small_candidates,
        tracker=tracker, dataset_name="fake.csv",
    )

    client = mlflow.tracking.MlflowClient()
    experiment = client.get_experiment_by_name("test_compare")
    runs = client.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    assert runs[0].data.tags["model_name"] == "logistic_regression"