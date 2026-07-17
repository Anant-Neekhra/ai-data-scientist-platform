"""
Categorical encoding. Routes columns to one-hot encoding (low
cardinality) or frequency encoding (high cardinality) to avoid the
dimensionality explosion one-hot would cause on high-cardinality
columns. Fit only on training data; transform reapplies learned
mappings, degrading gracefully on unseen categories.
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)

# Columns with <= this many unique categories get one-hot encoded.
# Above this, one-hot would create too many new columns, so we
# switch to frequency encoding instead.
ONE_HOT_CARDINALITY_THRESHOLD = 10


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical/boolean columns via one-hot or frequency encoding."""

    def __init__(self, schema: dict[str, FeatureType] | None = None) -> None:
        self.schema = schema
        self._schema_detector = SchemaDetector()
        self._onehot_encoders: dict[str, OneHotEncoder] = {}
        self._frequency_maps: dict[str, dict[str, float]] = {}
        self._onehot_columns: list[str] = []
        self._frequency_columns: list[str] = []
        self._is_fitted = False

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> "CategoricalEncoder":
        """
        Learn encoding mappings from X. Call this ONLY on training
        data — the one-hot category lists and frequency values learned
        here are reused as-is by transform(), never recomputed.
        """
        schema = self.schema or self._schema_detector.detect_feature_types(X)
        categorical_cols = [
            col
            for col, ftype in schema.items()
            if ftype in (FeatureType.CATEGORICAL, FeatureType.BOOLEAN)
        ]

        self._onehot_encoders = {}
        self._frequency_maps = {}
        self._onehot_columns = []
        self._frequency_columns = []

        for col in categorical_cols:
            cardinality = X[col].nunique(dropna=True)
            if cardinality <= ONE_HOT_CARDINALITY_THRESHOLD:
                encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
                encoder.fit(X[[col]].astype(str))
                self._onehot_encoders[col] = encoder
                self._onehot_columns.append(col)
            else:
                freq_map = X[col].astype(str).value_counts(normalize=True).to_dict()
                self._frequency_maps[col] = freq_map
                self._frequency_columns.append(col)

        self._is_fitted = True
        logger.info(
            f"Encoder fitted: one-hot={self._onehot_columns}, "
            f"frequency={self._frequency_columns}"
        )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply learned encodings. Unseen categories degrade gracefully:
        one-hot columns become all-zero, frequency-encoded values become 0.0.
        """
        if not self._is_fitted:
            raise RuntimeError("Encoder has not been fitted yet. Call fit() first.")

        encoded_cols = self._onehot_columns + self._frequency_columns
        untouched = X.drop(columns=encoded_cols, errors="ignore")
        output_frames = [untouched.reset_index(drop=True)]

        for col in self._onehot_columns:
            encoder = self._onehot_encoders[col]
            encoded_array = encoder.transform(X[[col]].astype(str))
            new_col_names = [f"{col}_{category}" for category in encoder.categories_[0]]
            encoded_df = pd.DataFrame(encoded_array, columns=new_col_names)
            output_frames.append(encoded_df.reset_index(drop=True))

        for col in self._frequency_columns:
            freq_map = self._frequency_maps[col]
            # Unseen category at transform time -> frequency 0.0, not an
            # error and not recomputed from the current data.
            freq_series = X[col].astype(str).map(freq_map).fillna(0.0)
            freq_series = freq_series.rename(f"{col}_freq").reset_index(drop=True)
            output_frames.append(freq_series)

        return pd.concat(output_frames, axis=1)
    
    def __sklearn_is_fitted__(self) -> bool:
        return self._is_fitted