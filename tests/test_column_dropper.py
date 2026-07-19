"""Tests for ColumnDropper."""

import pandas as pd
import pytest

from ml_pipeline.preprocessing.column_dropper import ColumnDropper
from ml_pipeline.preprocessing.pipeline_builder import build_preprocessing_pipeline


@pytest.fixture
def df_with_id_column() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "passenger_id": range(30),  # name hints "id" -> id_like
            "age": [22 + (i * 3) % 40 for i in range(30)],
            "city": ["Delhi", "Mumbai", "Pune"] * 10,
        }
    )


def test_id_like_column_is_dropped(df_with_id_column):
    dropper = ColumnDropper()
    dropper.fit(df_with_id_column)
    result = dropper.transform(df_with_id_column)

    assert "passenger_id" not in result.columns
    assert "age" in result.columns
    assert "city" in result.columns


def test_dropper_raises_if_not_fitted():
    dropper = ColumnDropper()
    with pytest.raises(RuntimeError):
        dropper.transform(pd.DataFrame({"a": [1]}))


def test_dropper_is_noop_when_no_id_like_columns():
    df = pd.DataFrame({"age": [22, 25, 30], "city": ["Delhi", "Mumbai", "Pune"]})
    dropper = ColumnDropper()
    dropper.fit(df)
    result = dropper.transform(df)
    assert list(result.columns) == list(df.columns)

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