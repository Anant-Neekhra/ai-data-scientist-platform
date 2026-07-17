"""
Missing value imputation. Fill values are computed once via fit()
(intended to be called on training data only) and reapplied via
transform() to any dataset — this fit/transform split is what makes
leakage prevention structural rather than a matter of remembering.
"""

from typing import Any
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)


class MissingValueImputer(BaseEstimator, TransformerMixin):
    """
    Fills missing values: median for numerical columns (robust to
    outliers), mode for categorical/boolean columns. Datetime and
    id_like columns are left untouched — imputing an identifier or
    a date with a "typical value" doesn't make semantic sense.
    """

    def __init__(self, schema: dict[str, FeatureType] | None = None) -> None:
        self.schema = schema
        self._schema_detector = SchemaDetector()
        self._fill_values: dict[str, Any] = {}
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "MissingValueImputer":
        schema = self.schema or self._schema_detector.detect_feature_types(X)
        self._fill_values = {}

        for col in X.columns:
            ftype = schema.get(col)
            if ftype == FeatureType.NUMERICAL:
                self._fill_values[col] = float(X[col].median())
            elif ftype in (FeatureType.CATEGORICAL, FeatureType.BOOLEAN):
                mode = X[col].mode(dropna=True)
                self._fill_values[col] = mode.iloc[0] if not mode.empty else "missing"

        self._is_fitted = True
        logger.info(f"Imputer fitted on {len(self._fill_values)} columns")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Imputer has not been fitted yet. Call fit() first.")

        X = X.copy()
        for col, fill_value in self._fill_values.items():
            if col in X.columns:
                X[col] = X[col].fillna(fill_value)
        return X

    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted