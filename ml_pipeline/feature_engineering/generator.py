"""
Generic, dataset-agnostic feature engineering: numeric interaction
features and datetime decomposition. Deliberately avoids
dataset-specific tricks (e.g. parsing titles out of a 'Name' column)
since this platform must work on arbitrary uploaded CSVs, not just
Titanic-shaped data.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)

# Cap on generated interaction features -- with N numeric columns,
# pairwise combinations grow as N*(N-1)/2, which explodes quickly.
# This keeps the pipeline from silently creating hundreds of columns
# on a wide dataset.
MAX_NUMERIC_COLS_FOR_INTERACTIONS = 6


class FeatureGenerator(BaseEstimator, TransformerMixin):
    """
    Generates two categories of new features:
      1. Pairwise products of numerical columns (interaction terms) —
         capture relationships a linear model can't see in raw columns
         (e.g. price alone and area alone matter less than price*area).
      2. Datetime decomposition (year, month, day, day-of-week) — raw
         datetime objects aren't directly usable by most ML models,
         but their components often are.
    No fitted statistics are involved (this doesn't compute means or
    frequencies from data), so leakage isn't a concern here the way
    it is for the imputer/encoder/scaler -- but we keep the same
    fit/transform interface for consistency within the Pipeline.
    """

    def __init__(self, schema: dict[str, FeatureType] | None = None) -> None:
        self.schema = schema
        self._schema_detector = SchemaDetector()
        self._numeric_cols_for_interactions: list[str] = []
        self._datetime_cols: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureGenerator":
        schema = self.schema or self._schema_detector.detect_feature_types(X)
        numeric_cols = [c for c, t in schema.items() if t == FeatureType.NUMERICAL]

        # Only generate interactions if the numeric column count is
        # small enough that pairwise combinations stay manageable.
        self._numeric_cols_for_interactions = (
            numeric_cols[:MAX_NUMERIC_COLS_FOR_INTERACTIONS]
            if len(numeric_cols) >= 2
            else []
        )
        self._datetime_cols = [c for c, t in schema.items() if t == FeatureType.DATETIME]

        logger.info(
            f"FeatureGenerator fitted: interactions from "
            f"{self._numeric_cols_for_interactions}, datetime decomposition "
            f"for {self._datetime_cols}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        cols = self._numeric_cols_for_interactions
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                col_a, col_b = cols[i], cols[j]
                X[f"{col_a}_x_{col_b}"] = X[col_a] * X[col_b]

        for col in self._datetime_cols:
            parsed = pd.to_datetime(X[col], errors="coerce")
            X[f"{col}_year"] = parsed.dt.year
            X[f"{col}_month"] = parsed.dt.month
            X[f"{col}_day"] = parsed.dt.day
            X[f"{col}_dayofweek"] = parsed.dt.dayofweek
            X = X.drop(columns=[col])  # raw datetime isn't directly usable by most models

        return X