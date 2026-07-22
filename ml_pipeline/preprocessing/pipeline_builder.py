"""
Assembles imputation, feature engineering, encoding, and scaling
into a single sklearn Pipeline. This is the ONE object the rest of
the platform interacts with for preprocessing -- fit once on train
data, transform train/test/prediction data identically, and persist
it whole via joblib in Phase 9 so a saved model always carries its
exact preprocessing steps with it.
"""

from sklearn.pipeline import Pipeline
import pandas as pd

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from ml_pipeline.preprocessing.imputer import MissingValueImputer
from ml_pipeline.feature_engineering.generator import FeatureGenerator
from ml_pipeline.preprocessing.encoder import CategoricalEncoder
from ml_pipeline.preprocessing.scaler import FeatureScaler
from ml_pipeline.preprocessing.column_dropper import ColumnDropper
from utils.logger import get_logger

logger = get_logger(__name__)


def build_preprocessing_pipeline(
    df: pd.DataFrame,
    scaling_method: str = "standard",
    schema_override: dict[str, FeatureType] | None = None,
) -> Pipeline:
    """
    Build a preprocessing Pipeline configured for this dataset's schema.

    Args:
        df: a representative sample of the data (typically X_train)
            used to auto-detect the schema, UNLESS schema_override is
            provided.
        scaling_method: 'standard' or 'minmax'.
        schema_override: optional user-corrected schema (e.g. from the
            Streamlit upload page's editable schema table). When
            provided, this is used INSTEAD of auto-detection -- the
            whole reason every transformer accepts an optional schema
            parameter (since Day 5) is to make this override possible
            without touching any transformer's internals.

    Returns:
        An unfitted sklearn Pipeline.
    """
    schema = schema_override or SchemaDetector().detect_feature_types(df)

    pipeline = Pipeline(
        steps=[
            ("column_dropper", ColumnDropper(schema=schema)),
            ("imputer", MissingValueImputer(schema=schema)),
            ("feature_engineer", FeatureGenerator(schema=schema)),
            ("encoder", CategoricalEncoder(schema=schema)),
            ("scaler", FeatureScaler(method=scaling_method, schema=schema)),
        ]
    )
    logger.info(
        f"Built preprocessing pipeline with schema: {schema} "
        f"(override={'yes' if schema_override else 'no'})"
    )
    return pipeline