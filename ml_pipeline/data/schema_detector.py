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
CATEGORICAL_UNIQUE_RATIO = 0.05

# If a column's unique-value ratio exceeds this, it's a *candidate*
# for being an identifier — but cardinality alone is not sufficient
# proof (continuous numeric features are naturally near-unique too).
ID_LIKE_UNIQUE_RATIO = 0.95

MIN_ROWS_FOR_ID_CHECK = 20

# Substrings/suffixes that strongly suggest a column is an identifier.
ID_NAME_HINTS = ("id", "uuid", "guid", "index")


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
        schema: dict[str, FeatureType] = {}

        for col in df.columns:
            series = df[col]

            if self._is_id_like(series, col):
                schema[col] = FeatureType.ID_LIKE
            elif pd.api.types.is_bool_dtype(series):
                schema[col] = FeatureType.BOOLEAN
            elif pd.api.types.is_datetime64_any_dtype(series):
                schema[col] = FeatureType.DATETIME
            elif pd.api.types.is_numeric_dtype(series):
                n_unique = series.nunique(dropna=True)
                unique_ratio = n_unique / len(series) if len(series) else 0.0
                if n_unique <= CATEGORICAL_UNIQUE_THRESHOLD or unique_ratio <= CATEGORICAL_UNIQUE_RATIO:
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
        import warnings

        try:
            sample = series.dropna().head(20)
            if sample.empty:
                return False
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                pd.to_datetime(sample, errors="raise")
            return True
        except (ValueError, TypeError):
            return False
        
    @staticmethod
    def _looks_like_id_name(col_name: str) -> bool:
        """Check if a column's *name* suggests it's an identifier."""
        normalized = col_name.strip().lower().replace("_", "").replace("-", "")
        return any(hint in normalized for hint in ID_NAME_HINTS)

    @staticmethod
    def _is_sequential_integers(series: pd.Series) -> bool:
        """
        Check if an integer column is a sequential run (1, 2, 3, 4...)
        when sorted — the structural signature of an auto-increment ID,
        as opposed to a continuous measurement that merely happens to
        have many unique values.
        """
        if not pd.api.types.is_integer_dtype(series):
            return False
        sorted_vals = series.dropna().sort_values().reset_index(drop=True)
        if len(sorted_vals) < 2:
            return False
        diffs = sorted_vals.diff().dropna()
        return bool((diffs == 1).all())

    def _is_id_like(self, series: pd.Series, col_name: str) -> bool:
        """
        Decide if a column is an identifier. Requires high cardinality
        AND a corroborating signal (name hint, sequential structure, or
        being non-numeric text) — cardinality alone is not enough.

        Known limitation: a genuinely categorical high-cardinality text
        column with zero repeated values in the current sample (e.g. a
        small dataset where a region code happens to be unique per row)
        is structurally indistinguishable from a true identifier, and
        will be misclassified as id_like. This is why schema detection
        results are meant to be user-overridable in the frontend rather
        than treated as final — no purely statistical rule can resolve
        this ambiguity from the data alone.
        """
        if len(series) <= MIN_ROWS_FOR_ID_CHECK:
            return False

        unique_ratio = series.nunique(dropna=True) / len(series)
        if unique_ratio <= ID_LIKE_UNIQUE_RATIO:
            return False

        # Floats are never treated as identifiers.
        if pd.api.types.is_float_dtype(series):
            return False

        name_suggests_id = self._looks_like_id_name(col_name)
        structurally_sequential = self._is_sequential_integers(series)
        is_high_cardinality_text = (
            not pd.api.types.is_numeric_dtype(series)
        )

        return name_suggests_id or structurally_sequential or is_high_cardinality_text


MIN_ROWS_FOR_ID_CHECK = 20