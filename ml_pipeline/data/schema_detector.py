"""
Infers feature types and the ML problem type (classification vs
regression) from a DataFrame — the core "automatic" part of the
platform's data understanding.
"""

from enum import Enum
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# If a numeric column has <= this many unique values, treat it as
# categorical rather than continuous (e.g. an encoded 0/1/2 label).
CATEGORICAL_UNIQUE_THRESHOLD = 20

# If a column's unique-value ratio exceeds this, treat it as an
# identifier rather than a usable feature.
ID_LIKE_UNIQUE_RATIO = 0.95


class FeatureType(str, Enum):
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    ID_LIKE = "id_like"


class ProblemType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class SchemaDetector:
    """Infers per-column feature types and the overall problem type."""

    def detect_feature_types(self, df: pd.DataFrame) -> dict[str, FeatureType]:
        """
        Infer the semantic type of every column in the DataFrame.

        Args:
            df: the dataset.

        Returns:
            Mapping of column name to inferred FeatureType.
        """
        schema: dict[str, FeatureType] = {}

        for col in df.columns:
            series = df[col]
            unique_ratio = series.nunique(dropna=True) / max(len(series), 1)

            if unique_ratio > ID_LIKE_UNIQUE_RATIO and len(series) > MIN_ROWS_FOR_ID_CHECK:
                schema[col] = FeatureType.ID_LIKE
            elif pd.api.types.is_bool_dtype(series):
                schema[col] = FeatureType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series):
                schema[col] = FeatureType.DATETIME
            elif pd.api.types.is_numeric_dtype(series):
                if series.nunique(dropna=True) <= CATEGORICAL_UNIQUE_THRESHOLD:
                    schema[col] = FeatureType.CATEGORICAL
                else:
                    schema[col] = FeatureType.NUMERICAL
            elif self._is_parseable_datetime(series):
                schema[col] = FeatureType.DATETIME
            else:
                schema[col] = FeatureType.CATEGORICAL

        logger.info(f"Detected schema: { {k: v.value for k, v in schema.items()} }")
        return schema

    def detect_problem_type(self, df: pd.DataFrame, target_col: str) -> ProblemType:
        """
        Infer whether this is a classification or regression problem
        based on the target column's dtype and cardinality.

        Args:
            df: the dataset.
            target_col: name of the target column.

        Returns:
            ProblemType.CLASSIFICATION or ProblemType.REGRESSION.
        """
        target = df[target_col].dropna()

        if not pd.api.types.is_numeric_dtype(target):
            problem_type = ProblemType.CLASSIFICATION
        elif target.nunique() <= CATEGORICAL_UNIQUE_THRESHOLD:
            problem_type = ProblemType.CLASSIFICATION
        else:
            problem_type = ProblemType.REGRESSION

        logger.info(
            f"Detected problem_type={problem_type.value} for target='{target_col}' "
            f"(nunique={target.nunique()}, dtype={target.dtype})"
        )
        return problem_type

    @staticmethod
    def _is_parseable_datetime(series: pd.Series) -> bool:
        """Best-effort check: can a string column be parsed as dates?"""
        try:
            sample = series.dropna().head(20)
            if sample.empty:
                return False
            pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False


MIN_ROWS_FOR_ID_CHECK = 20