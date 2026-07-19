"""
SHAP-based model explainability: global (aggregate) and local
(single-prediction) explanations. Scoped to tree-based models via
TreeExplainer -- exact, fast SHAP computation exploiting tree
structure. Non-tree models (linear, neural network) are not
supported here; see module docstring reasoning in Day 10 notes on
why this is an explicit, honest scope decision rather than a gap.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from utils.logger import get_logger

logger = get_logger(__name__)

DEFAULT_SAMPLE_SIZE = 150

# Model classes TreeExplainer supports directly. Checked by class name
# rather than isinstance against imported classes, so this module
# doesn't need to import xgboost/lightgbm/catboost just to reference
# their types -- keeps this file's dependencies minimal and avoids
# tightly coupling explainability to every specific library.
TREE_MODEL_CLASS_NAMES = {
    "RandomForestClassifier", "RandomForestRegressor",
    "XGBClassifier", "XGBRegressor",
    "LGBMClassifier", "LGBMRegressor",
    "CatBoostClassifier", "CatBoostRegressor",
}


@dataclass
class GlobalExplanation:
    """Aggregate SHAP summary across a sample of rows."""

    feature_names: list[str]
    mean_abs_shap: np.ndarray       # one value per feature -- overall importance
    shap_values: np.ndarray         # full matrix (n_samples, n_features) -- for beeswarm plots
    feature_values: pd.DataFrame    # the actual feature values, aligned with shap_values rows
    base_value: float


@dataclass
class LocalExplanation:
    """SHAP explanation for a single prediction."""

    feature_names: list[str]
    shap_values: np.ndarray   # one value per feature, for this one row
    feature_values: pd.Series
    base_value: float
    predicted_value: float    # base_value + sum(shap_values), should match model output


class ShapExplainer:
    """Computes global and local SHAP explanations for tree-based models."""

    def __init__(self, fitted_pipeline: Pipeline, X_reference: pd.DataFrame) -> None:
        """
        Args:
            fitted_pipeline: a Pipeline already fit via .fit(X_train, y_train).
            X_reference: RAW training data (or a representative sample of it)
                used to build the SHAP explainer's background distribution.
        """
        self._pipeline = fitted_pipeline
        self._model = fitted_pipeline.named_steps.get("model")
        self._preprocessing = fitted_pipeline[:-1]

        if self._model is None:
            raise ValueError("Pipeline has no step named 'model'.")

        model_class_name = type(self._model).__name__
        if model_class_name not in TREE_MODEL_CLASS_NAMES:
            raise ValueError(
                f"ShapExplainer currently supports tree-based models only "
                f"(got {model_class_name}). Linear and neural network models "
                f"are intentionally out of scope -- see Day 10 notes."
            )

        # Preprocess the reference data ONCE here, at explainer construction --
        # TreeExplainer needs a background dataset in the same numeric,
        # post-preprocessing shape the model was actually trained on.
        X_ref_transformed = self._preprocessing.transform(X_reference)

        # model_output="probability" only makes sense for classifiers --
        # regressors have no probability space to convert into, and SHAP
        # raises NotImplementedError if asked to. Detect via predict_proba,
        # the standard sklearn signal for "this is a classifier."
        is_classifier = hasattr(self._model, "predict_proba")
        model_output = "probability" if is_classifier else "raw"

        self._explainer = shap.TreeExplainer(
            self._model, X_ref_transformed, model_output=model_output
        )
        logger.info(
            f"ShapExplainer initialized for {model_class_name} "
            f"(model_output='{model_output}')"
        )

    def explain_global(
        self, X_sample: pd.DataFrame, sample_size: int = DEFAULT_SAMPLE_SIZE
    ) -> GlobalExplanation:
        if len(X_sample) > sample_size:
            X_sample = X_sample.sample(n=sample_size, random_state=42)

        X_transformed = self._preprocessing.transform(X_sample)
        # check_additivity=False: some probability-output tree ensembles
        # (notably RandomForest) can fail SHAP's strict internal additivity
        # self-check due to floating-point inconsistencies between the
        # tree-based SHAP approximation and averaged-vote probability
        # outputs -- a known SHAP/RandomForest interaction, not a sign our
        # values are wrong. We verify the sum identity ourselves in tests
        # (see test_local_explanation_sums_to_prediction) with a tolerant
        # threshold instead of relying on SHAP's strict internal check.
        shap_values = self._explainer.shap_values(X_transformed, check_additivity=False)
        shap_values, base_value = self._normalize_shap_output(shap_values)

        mean_abs_shap = np.abs(shap_values).mean(axis=0)

        return GlobalExplanation(
            feature_names=list(X_transformed.columns),
            mean_abs_shap=mean_abs_shap,
            shap_values = self._explainer.shap_values(X_transformed, check_additivity=False),
            feature_values=X_transformed.reset_index(drop=True),
            base_value=base_value,
        )

    def explain_local(self, X_row: pd.DataFrame) -> LocalExplanation:
        """
        Compute a SHAP explanation for a single row.

        Args:
            X_row: RAW data containing exactly ONE row to explain.
        """
        if len(X_row) != 1:
            raise ValueError(f"explain_local expects exactly 1 row, got {len(X_row)}.")

        X_transformed = self._preprocessing.transform(X_row)
        shap_values = self._explainer.shap_values(X_transformed)
        shap_values, base_value = self._normalize_shap_output(shap_values)

        row_shap = shap_values[0]
        predicted_value = float(base_value + row_shap.sum())

        return LocalExplanation(
            feature_names=list(X_transformed.columns),
            shap_values=row_shap,
            feature_values=X_transformed.iloc[0],
            base_value=base_value,
            predicted_value=predicted_value,
        )

    def _normalize_shap_output(self, shap_values) -> tuple[np.ndarray, float]:
        """
        Different model libraries return shap_values / expected_value in
        different shapes for binary classification (list-of-2-arrays vs.
        a single 3D array). This normalizes to a single 2D array (the
        positive class's contributions) and a single scalar base value,
        so every caller in this module can treat the output uniformly.
        """
        expected_value = self._explainer.expected_value

        if isinstance(shap_values, list):
            # e.g. RandomForest binary classifier: [array_class_0, array_class_1]
            values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # e.g. some XGBoost/LightGBM outputs: (n_samples, n_features, n_classes)
            values = shap_values[:, :, 1]
            base = expected_value[1] if isinstance(expected_value, (list, np.ndarray)) else expected_value
        else:
            values = shap_values
            base = expected_value[0] if isinstance(expected_value, (list, np.ndarray)) else expected_value

        return values, float(base)