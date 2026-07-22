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
    ...(existing docstring)...
    """
    if schema_override:
        # Restrict the override to columns that actually exist in df.
        # This matters because schema_override is typically built by
        # the frontend from detecting schema on the FULL uploaded
        # dataset (before the target column is split off) -- so it
        # can legitimately contain the target column's own entry.
        # df here is X_train, which never contains the target (already
        # dropped by split_dataset), so any override key not present
        # in df -- most commonly the target itself -- is silently and
        # safely ignored rather than being handed to transformers that
        # would then try to operate on a column that doesn't exist.
        schema = {col: ftype for col, ftype in schema_override.items() if col in df.columns}
    else:
        schema = SchemaDetector().detect_feature_types(df)

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