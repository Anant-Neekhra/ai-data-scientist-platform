"""
Drops columns that shouldn't be fed to a model at all. Currently
targets id_like columns (identifiers, names, ticket numbers used as
raw text, etc).

Why this step needs to exist as its own thing: every other
preprocessing step deliberately SKIPS transforming id_like columns
(imputer doesn't impute them, encoder doesn't encode them, scaler
doesn't scale them) -- but "skip transforming" was never the same
thing as "remove from the DataFrame". Left alone, id_like columns
rode completely untouched, as raw strings, straight into model.fit()
-- which is exactly what broke every single candidate model on the
first real dataset run. This step is what actually removes them.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)


class ColumnDropper(BaseEstimator, TransformerMixin):
    """Drops columns whose FeatureType makes them unusable as raw model input."""

    DROP_TYPES = (FeatureType.ID_LIKE,)

    def __init__(self, schema: dict[str, FeatureType] | None = None) -> None:
        self.schema = schema
        self._schema_detector = SchemaDetector()
        self._columns_to_drop: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "ColumnDropper":
        schema = self.schema or self._schema_detector.detect_feature_types(X)
        self._columns_to_drop = [
            col for col, ftype in schema.items()
            if ftype in self.DROP_TYPES and col in X.columns
        ]
        self._is_fitted = True
        logger.info(f"ColumnDropper fitted: dropping {self._columns_to_drop}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("ColumnDropper has not been fitted yet. Call fit() first.")
        return X.drop(columns=self._columns_to_drop, errors="ignore")

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted