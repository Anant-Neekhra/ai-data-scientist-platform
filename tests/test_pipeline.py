"""Tests for FeatureScaler, FeatureGenerator, split_dataset, and the
assembled preprocessing pipeline end-to-end."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.preprocessing.scaler import FeatureScaler
from ml_pipeline.feature_engineering.generator import FeatureGenerator
from ml_pipeline.data.splitter import split_dataset
from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline


@pytest.fixture
def numeric_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [22 + (i * 3) % 40 for i in range(30)],
            "salary": [30000 + i * 733 for i in range(30)],
        }
    )


def test_standard_scaler_produces_zero_mean(numeric_df):
    scaler = FeatureScaler(method="standard")
    scaler.fit(numeric_df)
    result = scaler.transform(numeric_df)
    assert result["age"].mean() == pytest.approx(0.0, abs=1e-8)
    assert result["salary"].std() == pytest.approx(1.0, abs=1e-1)


def test_scaler_reuses_train_stats_on_test_data(numeric_df):
    scaler = FeatureScaler(method="standard")
    scaler.fit(numeric_df)

    weird_test_df = pd.DataFrame({"age": [999], "salary": [1]})
    result = scaler.transform(weird_test_df)
    # A wildly different test value should NOT come out ~0 mean --
    # it should reflect train's mean/std, proving nothing was
    # recomputed from the test data itself.
    assert result["age"].iloc[0] > 5  # far from 0 given train's much smaller scale


def test_invalid_scaler_method_raises():
    with pytest.raises(ValueError):
        FeatureScaler(method="not_a_real_method")


def test_feature_generator_creates_interaction_columns(numeric_df):
    generator = FeatureGenerator()
    generator.fit(numeric_df)
    result = generator.transform(numeric_df)
    assert "age_x_salary" in result.columns
    assert (result["age_x_salary"] == numeric_df["age"] * numeric_df["salary"]).all()


def test_feature_generator_decomposes_datetime():
    df = pd.DataFrame(
        {
            "signup_date": pd.date_range("2023-01-01", periods=10, freq="D"),
            "value": range(10),
        }
    )
    generator = FeatureGenerator()
    generator.fit(df)
    result = generator.transform(df)

    assert "signup_date_year" in result.columns
    assert "signup_date_month" in result.columns
    assert "signup_date" not in result.columns  # raw column dropped
    assert result["signup_date_year"].iloc[0] == 2023


def test_split_dataset_stratifies_classification():
    df = pd.DataFrame(
        {
            "feature": range(100),
            "target": [0] * 80 + [1] * 20,  # imbalanced classes
        }
    )
    X_train, X_test, y_train, y_test = split_dataset(df, target_col="target")

    train_ratio = y_train.value_counts(normalize=True)[1]
    test_ratio = y_test.value_counts(normalize=True)[1]
    # Stratification should keep the minority class ratio close in both splits.
    assert abs(train_ratio - test_ratio) < 0.05


def test_full_pipeline_runs_end_to_end_without_leakage():
    df = pd.DataFrame(
        {
            "age": [22 + (i * 3) % 40 for i in range(50)] + [None] * 0,
            "city": (["Delhi", "Mumbai", "Pune"] * 17)[:50],
            "target": [0, 1] * 25,
        }
    )
    df.loc[0, "age"] = np.nan  # inject a missing value to prove imputation runs

    X_train, X_test, y_train, y_test = split_dataset(df, target_col="target")

    pipeline = build_preprocessing_pipeline(X_train)
    pipeline.fit(X_train)

    X_train_transformed = pipeline.transform(X_train)
    X_test_transformed = pipeline.transform(X_test)

    # No missing values should survive preprocessing.
    assert X_train_transformed.isnull().sum().sum() == 0
    assert X_test_transformed.isnull().sum().sum() == 0
    # City should have been one-hot encoded, not left as raw text.
    assert "city" not in X_train_transformed.columns
    # Both splits must end up with the exact same columns, in the
    # same preprocessing "shape" -- otherwise a model trained on one
    # couldn't even be evaluated on the other.
    assert list(X_train_transformed.columns) == list(X_test_transformed.columns)

    def test_full_pipeline_drops_id_like_and_leaves_only_numeric_columns():
        df = pd.DataFrame(
            {
                "name": [f"Person {i}" for i in range(30)],  # high-cardinality text -> id_like
                "age": [22 + (i * 3) % 40 for i in range(30)],
                "city": ["Delhi", "Mumbai", "Pune"] * 10,
                "target": [0, 1] * 15,
            }
        )
        X_train, X_test, y_train, y_test = split_dataset(df, target_col="target")

        pipeline = build_preprocessing_pipeline(X_train)
        pipeline.fit(X_train)
        X_train_transformed = pipeline.transform(X_train)

        assert "name" not in X_train_transformed.columns
        # Every remaining column must be numeric -- this is the exact
        # property that silently failed before ColumnDropper existed,
        # breaking every model at fit() time on the real Titanic run.
        assert all(
            pd.api.types.is_numeric_dtype(X_train_transformed[col])
            for col in X_train_transformed.columns
        )

def test_schema_override_takes_precedence_over_autodetection():
    from ml_pipeline.data.schema_detector import FeatureType

    # A column that would auto-detect as categorical (low cardinality)
    df = pd.DataFrame({"code": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1] * 3})
    override = {"code": FeatureType.NUMERICAL}

    pipeline = build_preprocessing_pipeline(df, schema_override=override)
    pipeline.fit(df)
    result = pipeline.transform(df)

    # Under override, 'code' is treated as numerical -> scaled in place,
    # NOT one-hot expanded into multiple columns.
    assert "code" in result.columns
    assert list(result.columns) == ["code"]

def test_schema_override_with_target_column_key_is_ignored():
    """A schema_override built from the FULL uploaded dataset (as the
    Streamlit Upload page does) can legitimately include an entry for
    the target column. Since X never contains the target (already
    dropped by split_dataset before reaching here), that key must be
    silently ignored rather than causing FeatureGenerator to try
    building an interaction feature against a column that doesn't
    exist -- this reproduces the real KeyError hit via the Streamlit
    Train page on a regression dataset."""
    from ml_pipeline.data.schema_detector import FeatureType

    X_train = pd.DataFrame(
        {
            "km_driven": [10000 + i * 500 for i in range(30)],
            "brand": ["Honda", "Toyota", "Ford"] * 10,
        }
    )
    # Override dict as it would arrive from the frontend: includes an
    # entry for a column ('selling_price') that is NOT in X_train.
    override = {
        "km_driven": FeatureType.NUMERICAL,
        "brand": FeatureType.CATEGORICAL,
        "selling_price": FeatureType.NUMERICAL,  # the target -- not in X_train
    }

    pipeline = build_preprocessing_pipeline(X_train, schema_override=override)
    pipeline.fit(X_train)
    result = pipeline.transform(X_train)  # must not raise KeyError

    assert "selling_price" not in result.columns
    assert "km_driven_x_selling_price" not in result.columns

def test_split_falls_back_to_unstratified_when_class_has_single_member():
    df = pd.DataFrame(
        {
            "feature": range(20),
            "target": ["common"] * 18 + ["common"] + ["rare_singleton"],
        }
    )
    # 'rare_singleton' appears exactly once -- stratification is impossible.
    # Must not raise; must fall back gracefully.
    X_train, X_test, y_train, y_test = split_dataset(df, target_col="target")
    assert len(X_train) + len(X_test) == 20