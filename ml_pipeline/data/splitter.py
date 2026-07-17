"""
Train/test splitting. A thin, deliberate wrapper around sklearn's
train_test_split -- centralizing it here means every part of the
platform splits data the exact same way (same random_state, same
stratification logic), rather than each phase reinventing it
slightly differently.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

from config.settings import RANDOM_STATE, TEST_SIZE
from ml_pipeline.data.schema_detector import SchemaDetector, ProblemType
from utils.logger import get_logger

logger = get_logger(__name__)


def split_dataset(
    df: pd.DataFrame, target_col: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split a dataset into train/test features and targets.

    For classification problems, stratifies on the target so both
    splits preserve the original class proportions -- important
    because a random split could otherwise leave a rare class
    severely underrepresented (or entirely absent) in the test set,
    making evaluation on that class meaningless. Regression targets
    can't be stratified this way (they're continuous, not classes),
    so a plain random split is used instead.

    Args:
        df: the full, already-imputed-and-clean-of-target-nulls dataset.
        target_col: name of the target column.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = SchemaDetector().detect_problem_type(df, target_col)
    stratify = y if problem_type == ProblemType.CLASSIFICATION else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify,
    )

    logger.info(
        f"Split dataset: train={X_train.shape}, test={X_test.shape}, "
        f"problem_type={problem_type.value}, stratified={stratify is not None}"
    )
    return X_train, X_test, y_train, y_test