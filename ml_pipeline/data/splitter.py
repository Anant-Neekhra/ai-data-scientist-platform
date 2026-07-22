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
    ...(existing docstring)...
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    problem_type = SchemaDetector().detect_problem_type(df, target_col)
    stratify = y if problem_type == ProblemType.CLASSIFICATION else None

    # Stratification requires every class to have at least 2 members
    # (one for train, one for test). A user is free to pick ANY column
    # as the target through the frontend, including one with rare
    # singleton classes (e.g. a car brand that appears exactly once) --
    # that's a legitimate, foreseeable situation, not a data error.
    # Degrade to a plain (non-stratified) split rather than crash.
    if stratify is not None:
        class_counts = y.value_counts()
        rare_classes = class_counts[class_counts < 2]
        if not rare_classes.empty:
            logger.warning(
                f"{len(rare_classes)} class(es) have fewer than 2 members "
                f"(e.g. {list(rare_classes.index[:5])}) -- stratification is "
                f"not possible for these; falling back to a non-stratified split."
            )
            stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=stratify,
    )

    logger.info(
        f"Split dataset: train={X_train.shape}, test={X_test.shape}, "
        f"problem_type={problem_type.value}, stratified={stratify is not None}"
    )
    return X_train, X_test, y_train, y_test