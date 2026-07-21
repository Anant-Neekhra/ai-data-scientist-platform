"""
MLflow experiment tracking wrapper. Uses a local SQLite database as
the tracking backend rather than the plain filesystem store -- MLflow
has moved the filesystem backend into maintenance mode as of recent
versions and recommends a database backend going forward. SQLite
keeps this just as simple to run locally (one file, zero extra
infrastructure) while staying on the officially supported path.
"""

from pathlib import Path
from contextlib import contextmanager

import mlflow
from sklearn.pipeline import Pipeline

from config.settings import BASE_DIR
from utils.logger import get_logger

logger = get_logger(__name__)

MLRUNS_DIR = BASE_DIR / "mlruns"


class MLflowTracker:
    """Wraps MLflow run logging for training experiments."""

    def __init__(self, experiment_name: str = "ai_data_scientist_platform") -> None:
        MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
        db_path = MLRUNS_DIR / "mlflow.db"
        mlflow.set_tracking_uri(f"sqlite:///{db_path.as_posix()}")
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str):
        """
        Context manager wrapping mlflow.start_run(). Using our own
        thin wrapper (rather than calling mlflow.start_run directly
        everywhere) means if we ever need to add platform-wide
        behavior around every run -- e.g. always tagging the dataset
        name -- there's one place to change it.
        """
        with mlflow.start_run(run_name=run_name) as run:
            logger.info(f"Started MLflow run '{run_name}' (id={run.info.run_id})")
            yield run

    def log_params(self, params: dict) -> None:
        # MLflow requires string-serializable values; stringify defensively
        # so an unexpected type (e.g. a numpy float) doesn't crash logging.
        mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

    def log_model(self, pipeline: Pipeline, artifact_path: str = "model") -> None:
        mlflow.sklearn.log_model(pipeline, artifact_path)

    def log_tags(self, tags: dict[str, str]) -> None:
        mlflow.set_tags(tags)