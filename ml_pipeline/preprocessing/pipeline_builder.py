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
    df: pd.DataFrame, scaling_method: str = "standard"
) -> Pipeline:
    """
    Build a preprocessing Pipeline configured for this dataset's schema.

    Order matters and is deliberate:
      1. Impute missing values first -- later steps (encoding,
         interaction features) shouldn't have to handle NaNs.
      2. Engineer features next -- interaction terms should be built
         from clean numeric data, and datetime decomposition removes
         the raw datetime column before encoding would have to deal
         with it.
      3. Encode categoricals -- turns text into numeric columns.
      4. Scale last -- scaling should apply to the FINAL set of
         numeric columns, including any newly engineered interaction
         features, not just the original raw ones.

    Args:
        df: a representative sample of the data (typically X_train)
            used only to detect the schema that configures each step.
        scaling_method: 'standard' or 'minmax'.

    Returns:
        An unfitted sklearn Pipeline. Call .fit(X_train) then
        .transform(X_train) / .transform(X_test) — never fit twice.
    """
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
    logger.info(f"Built preprocessing pipeline with schema: {schema}")
    return pipeline