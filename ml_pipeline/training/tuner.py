"""
Hyperparameter tuning via RandomizedSearchCV (default) or
GridSearchCV (exhaustive, opt-in). Same leak-safe design as Day 7's
trainer: the FULL pipeline (preprocessing + model) is searched
against RAW X_train, so preprocessing is refit per fold rather than
computed once up front.
"""

from dataclasses import dataclass
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, StratifiedKFold, KFold
from sklearn.pipeline import Pipeline

from config.settings import RANDOM_STATE
from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline
from ml_pipeline.training.search_spaces import get_search_space
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_CV_FOLDS = 5
DEFAULT_RANDOM_SEARCH_ITER = 25

SCORING_METRIC = {
    ProblemType.CLASSIFICATION: "accuracy",
    ProblemType.REGRESSION: "neg_root_mean_squared_error",
}


@dataclass
class TuningResult:
    """Outcome of a hyperparameter search."""

    model_name: str
    method: str  # "randomized" or "grid"
    best_score: float
    best_params: dict
    fitted_pipeline: Pipeline
    n_candidates_tried: int


class HyperparameterTuner:
    """Searches for better hyperparameters for a single chosen model."""

    def __init__(self, cv_folds: int = DEFAULT_CV_FOLDS) -> None:
        self.cv_folds = cv_folds

    def tune(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        model_name: str,
        model,
        problem_type: ProblemType,
        method: str = "randomized",
        n_iter: int = DEFAULT_RANDOM_SEARCH_ITER,
    ) -> TuningResult:
        """
        Search for better hyperparameters for one model.

        Args:
            X_train: RAW training features (not preprocessed) -- see
                module docstring on why this is required for leak-safety.
            y_train: training target.
            model_name: key into SEARCH_SPACES (e.g. 'random_forest').
            model: an unfitted instance of that model (from
                get_candidate_models()) -- the object whose
                hyperparameters we're searching over.
            problem_type: classification or regression.
            method: 'randomized' (default, cheaper) or 'grid'
                (exhaustive, only sensible for small search spaces).
            n_iter: number of random combinations to try. Ignored
                for method='grid', which tries every combination.

        Returns:
            TuningResult with the best score, best params, and a
            fully fitted pipeline ready to use immediately.
        """
        search_space = get_search_space(model_name)
        if not search_space:
            logger.info(
                f"No search space defined for '{model_name}' -- skipping tuning, "
                f"fitting with default hyperparameters instead."
            )
            pipeline = build_preprocessing_pipeline(X_train)
            pipeline.steps.append(("model", model))
            pipeline.fit(X_train, y_train)
            return TuningResult(
                model_name=model_name,
                method="none",
                best_score=float("nan"),
                best_params={},
                fitted_pipeline=pipeline,
                n_candidates_tried=0,
            )

        pipeline = build_preprocessing_pipeline(X_train)
        pipeline.steps.append(("model", model))

        cv = (
            StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
            if problem_type == ProblemType.CLASSIFICATION
            else KFold(n_splits=self.cv_folds, shuffle=True, random_state=RANDOM_STATE)
        )
        scoring = SCORING_METRIC[problem_type]

        if method == "grid":
            search = GridSearchCV(
                pipeline, param_grid=search_space, cv=cv, scoring=scoring, n_jobs=-1
            )
            n_candidates = None  # grid determines this itself; logged after fitting
        elif method == "randomized":
            search = RandomizedSearchCV(
                pipeline,
                param_distributions=search_space,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            n_candidates = n_iter
        else:
            raise ValueError(f"method must be 'randomized' or 'grid', got '{method}'")

        logger.info(f"Starting {method} search for '{model_name}' ({scoring})")
        search.fit(X_train, y_train)

        n_candidates_tried = len(search.cv_results_["params"])
        logger.info(
            f"Tuning complete: best_score={search.best_score_:.4f}, "
            f"n_candidates={n_candidates_tried}, best_params={search.best_params_}"
        )

        return TuningResult(
            model_name=model_name,
            method=method,
            best_score=float(search.best_score_),
            best_params=search.best_params_,
            fitted_pipeline=search.best_estimator_,
            n_candidates_tried=n_candidates_tried,
        )