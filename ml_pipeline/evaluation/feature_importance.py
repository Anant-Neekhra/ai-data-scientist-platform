"""
Extracts global feature importance from a fitted pipeline. Different
model families expose this differently -- tree ensembles via
feature_importances_, linear models via coef_ -- normalized here
into one common shape.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from utils.logger import get_logger

logger = get_logger(__name__)


def extract_feature_importance(
    fitted_pipeline: Pipeline, X_sample: pd.DataFrame
) -> pd.DataFrame | None:
    """
    Extract feature importance from a fitted pipeline's final model step.

    Args:
        fitted_pipeline: a Pipeline already fit via .fit(X_train, y_train).
        X_sample: any DataFrame with the same raw columns the pipeline
            was trained on -- used only to recover the FINAL feature
            names after preprocessing (one-hot expansion, interaction
            terms, etc.), since those names don't exist until data has
            actually passed through every preprocessing step.

    Returns:
        DataFrame [feature, importance] sorted descending, or None if
        the model exposes neither feature_importances_ nor coef_ (e.g.
        the Keras neural network -- an intentionally opaque candidate,
        consistent with Day 1/7's scoping of it as an optional, less
        interpretable model).
    """
    model = fitted_pipeline.named_steps.get("model")
    if model is None:
        logger.warning("Pipeline has no step named 'model' -- cannot extract importance.")
        return None

    # Run the sample through every step EXCEPT the model to recover
    # the actual post-preprocessing feature names/order.
    preprocessing_only = fitted_pipeline[:-1]
    transformed_sample = preprocessing_only.transform(X_sample)
    feature_names = list(transformed_sample.columns)

    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        # coef_ is 1D for simple regression, 2D (n_classes_or_1, n_features)
        # for classification -- average across classes for one per-feature value.
        importances = np.abs(coef) if coef.ndim == 1 else np.abs(coef).mean(axis=0)
    else:
        logger.info(
            f"Model type {type(model).__name__} exposes neither "
            f"feature_importances_ nor coef_ -- skipping importance extraction."
        )
        return None

    if len(importances) != len(feature_names):
        logger.warning(
            f"Importance count ({len(importances)}) doesn't match feature "
            f"count ({len(feature_names)}) -- skipping."
        )
        return None

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)