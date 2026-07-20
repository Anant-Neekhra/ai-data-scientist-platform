"""
Prediction service: loads a saved pipeline from the registry and
serves single or batch predictions, validating input columns before
ever touching the pipeline itself.
"""

from dataclasses import dataclass
import pandas as pd

from ml_pipeline.registry.model_registry import ModelRegistry
from ml_pipeline.registry.metadata import ModelMetadata
from utils.logger import get_logger

logger = get_logger(__name__)


class PredictionInputError(Exception):
    """Raised when incoming data doesn't match what the model expects."""


@dataclass
class PredictionResult:
    predictions: list
    probabilities: list[float] | None  # positive-class probability, classification only
    model_name: str
    model_version: int


class PredictionService:
    """Loads a registered model and serves predictions on new data."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self._registry = registry or ModelRegistry()

    def _load(self, dataset_name: str, version: int | None = None):
        pipeline = self._registry.load_model(dataset_name, version)
        metadata = self._registry.load_metadata(dataset_name, version)
        return pipeline, metadata

    def _validate_columns(self, df: pd.DataFrame, metadata: ModelMetadata) -> None:
        """
        Check incoming columns against what the model was trained on.
        Missing columns are a hard error -- the pipeline has no way to
        guess a value for a column it never saw at all. Extra columns
        are just ignored (a superset is harmless; the pipeline only
        reads the columns it knows about).
        """
        missing = set(metadata.feature_columns) - set(df.columns)
        if missing:
            raise PredictionInputError(
                f"Input is missing required columns: {sorted(missing)}. "
                f"Model '{metadata.model_name}' (v{metadata.version}) expects: "
                f"{metadata.feature_columns}"
            )

    def predict_single(
        self, dataset_name: str, row: dict, version: int | None = None
    ) -> PredictionResult:
        """
        Predict on a single row, given as a dict of {column_name: value}.
        """
        pipeline, metadata = self._load(dataset_name, version)
        df = pd.DataFrame([row])
        self._validate_columns(df, metadata)

        prediction = pipeline.predict(df)
        probability = None
        if hasattr(pipeline, "predict_proba"):
            probability = float(pipeline.predict_proba(df)[0, 1])

        logger.info(f"Single prediction served using {metadata.model_name} v{metadata.version}")
        return PredictionResult(
            predictions=prediction.tolist(),
            probabilities=[probability] if probability is not None else None,
            model_name=metadata.model_name,
            model_version=metadata.version,
        )

    def predict_batch(
        self, dataset_name: str, df: pd.DataFrame, version: int | None = None
    ) -> pd.DataFrame:
        """
        Predict on a batch of rows. Returns the ORIGINAL DataFrame with
        prediction (and probability, if available) columns appended --
        so the user gets their data back with predictions attached,
        ready to download, rather than a bare array they'd have to
        realign themselves.
        """
        pipeline, metadata = self._load(dataset_name, version)
        self._validate_columns(df, metadata)

        result_df = df.copy()
        result_df["prediction"] = pipeline.predict(df)

        if hasattr(pipeline, "predict_proba"):
            result_df["prediction_probability"] = pipeline.predict_proba(df)[:, 1]

        logger.info(
            f"Batch prediction served for {len(df)} rows using "
            f"{metadata.model_name} v{metadata.version}"
        )
        return result_df