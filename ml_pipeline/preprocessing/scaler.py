"""
Numerical feature scaling. Same fit/transform discipline as the
imputer and encoder — fit only on training data.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)


class FeatureScaler(BaseEstimator, TransformerMixin):
    """Scales numerical columns using StandardScaler (default) or MinMaxScaler."""

    def __init__(
        self,
        method: str = "standard",
        schema: dict[str, FeatureType] | None = None,
    ) -> None:
        if method not in ("standard", "minmax"):
            raise ValueError(f"method must be 'standard' or 'minmax', got '{method}'")
        self.method = method
        self.schema = schema
        self._schema_detector = SchemaDetector()
        self._scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self._numerical_cols: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "FeatureScaler":
        """Learn scaling parameters from X. Call ONLY on training data."""
        schema = self.schema or self._schema_detector.detect_feature_types(X)
        self._numerical_cols = [
            col for col, ftype in schema.items() if ftype == FeatureType.NUMERICAL
        ]
        if self._numerical_cols:
            self._scaler.fit(X[self._numerical_cols])
        self._is_fitted = True
        logger.info(f"Scaler ({self.method}) fitted on columns: {self._numerical_cols}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the scaling learned during fit()."""
        if not self._numerical_cols:
            return X.copy()

        X = X.copy()
        scaled_values = self._scaler.transform(X[self._numerical_cols])
        X[self._numerical_cols] = scaled_values
        return X
    
    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted
