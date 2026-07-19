"""
Trains and cross-validates candidate models. Preprocessing is fused
into the SAME pipeline as the model for every cross-validation call
-- this is deliberate and load-bearing: it's what prevents leakage
across CV folds (see module docstring context in Day 7 notes).
"""

from dataclasses import dataclass, field
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_validate, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from config.settings import RANDOM_STATE
from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline
from ml_pipeline.training.model_factory import get_candidate_models
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CV_FOLDS = 5

CLASSIFICATION_SCORING = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
REGRESSION_SCORING = {"neg_rmse": "neg_root_mean_squared_error", "r2": "r2"}

PRIMARY_METRIC = {
    ProblemType.CLASSIFICATION: "accuracy",
    ProblemType.REGRESSION: "neg_rmse",
}


@dataclass
class ModelResult:
    """Cross-validation outcome for one candidate model."""

    model_name: str
    mean_scores: dict[str, float] = field(default_factory=dict)
    std_scores: dict[str, float] = field(default_factory=dict)
    fit_time_seconds: float = 0.0


class ModelTrainer:
    """Cross-validates candidate models and selects/fits the best one."""

    def __init__(self, cv_folds: int = DEFAULT_CV_FOLDS) -> None:
        self.cv_folds = cv_folds

    def compare_models(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        problem_type: ProblemType,
        models: dict | None = None,
    ) -> list[ModelResult]:
        """
        Cross-validate every candidate model and return their scores.

        Args:
            X_train: RAW (not preprocessed) training features. Passing
                already-transformed data here would reintroduce the
                cross-fold leakage this method is designed to prevent.
            y_train: training target.
            problem_type: classification or regression.
            models: optional override of the candidate dict, injected
                rather than always calling get_candidate_models() --
                this makes the method testable with a small, fast
                model set without needing to monkeypatch anything.

        Returns:
            List of ModelResult, one per model that trained successfully.
            Models that raise during cross-validation are logged and
            skipped rather than aborting the whole comparison.
        """
        candidates = models if models is not None else get_candidate_models(problem_type)
        scoring = (
            CLASSIFICATION_SCORING
            if problem_type == ProblemType.CLASSIFICATION
            else REGRESSION_SCORING
        )
        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
            if problem_type == ProblemType.CLASSIFICATION
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        )

        results: list[ModelResult] = []
        for name, model in candidates.items():
            # Preprocessing + model as ONE pipeline; cross_validate gets
            # RAW X_train so preprocessing is refit per fold, not once
            # up front. See leakage discussion above.
            full_pipeline = build_preprocessing_pipeline(X_train)
            full_pipeline.steps.append(("model", model))

            start = time.time()
            try:
                cv_results = cross_validate(
                    full_pipeline,
                    X_train,
                    y_train,
                    cv=cv,
                    scoring=scoring,
                    return_train_score=False,
                )
            except Exception as e:
                logger.warning(f"Model '{name}' failed during cross-validation: {e}")
                continue
            elapsed = round(time.time() - start, 3)

            mean_scores = {m: float(np.mean(cv_results[f"test_{m}"])) for m in scoring}
            std_scores = {m: float(np.std(cv_results[f"test_{m}"])) for m in scoring}

            results.append(
                ModelResult(
                    model_name=name,
                    mean_scores=mean_scores,
                    std_scores=std_scores,
                    fit_time_seconds=elapsed,
                )
            )
            logger.info(f"{name}: {mean_scores} (fit_time={elapsed}s)")

        return results

    def fit_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        results: list[ModelResult],
        problem_type: ProblemType,
        models: dict | None = None,
    ) -> tuple[str, Pipeline]:
        """
        Pick the best-performing model from compare_models() results and
        fit ONE final pipeline (preprocessing + that model) on the full
        training set -- this is the object Phase 9 will save to disk.

        Returns:
            (best_model_name, fitted_pipeline)
        """
        if not results:
            raise ValueError("No successful model results to select from.")

        primary_metric = PRIMARY_METRIC[problem_type]
        best_result = max(results, key=lambda r: r.mean_scores[primary_metric])

        candidates = models if models is not None else get_candidate_models(problem_type)
        best_model = candidates[best_result.model_name]

        full_pipeline = build_preprocessing_pipeline(X_train)
        full_pipeline.steps.append(("model", best_model))
        full_pipeline.fit(X_train, y_train)

        logger.info(
            f"Best model: {best_result.model_name} "
            f"({primary_metric}={best_result.mean_scores[primary_metric]:.4f})"
        )
        return best_result.model_name, full_pipeline