"""
Learning curves: diagnose whether a model would benefit from more
training data, and whether it's overfitting or underfitting.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve, StratifiedKFold, KFold

from config.settings import RANDOM_STATE
from ml_pipeline.data.schema_detector import ProblemType
from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_TRAIN_SIZES = np.linspace(0.1, 1.0, 6)


def compute_learning_curve(
    pipeline, X: pd.DataFrame, y: pd.Series, problem_type: ProblemType, cv_folds: int = 5
) -> dict:
    """
    Compute training and validation scores across increasing amounts
    of training data. See Day 9 notes for how to read a widening gap
    (overfitting) vs. a low plateau (underfitting) vs. still-rising
    curves (more data would help).

    Args:
        pipeline: an UNFITTED pipeline -- learning_curve refits it
            internally at every size/fold combination itself, so this
            is not a leakage concern; passing a pre-fitted pipeline
            would just be redundant, wasted work.
        X, y: RAW training data.
        problem_type: classification or regression.
        cv_folds: folds evaluated at each training size.
    """
    cv = (
        StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
        if problem_type == ProblemType.CLASSIFICATION
        else KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    )
    scoring = "accuracy" if problem_type == ProblemType.CLASSIFICATION else "r2"

    train_sizes, train_scores, val_scores = learning_curve(
        pipeline, X, y, cv=cv, scoring=scoring,
        train_sizes=DEFAULT_TRAIN_SIZES, random_state=RANDOM_STATE, n_jobs=-1,
    )

    result = {
        "train_sizes": train_sizes.tolist(),
        "train_scores_mean": train_scores.mean(axis=1).tolist(),
        "train_scores_std": train_scores.std(axis=1).tolist(),
        "val_scores_mean": val_scores.mean(axis=1).tolist(),
        "val_scores_std": val_scores.std(axis=1).tolist(),
    }
    logger.info(f"Learning curve computed across {len(train_sizes)} training sizes")
    return result