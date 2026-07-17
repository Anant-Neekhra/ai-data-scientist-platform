"""Tests for MissingValueImputer and CategoricalEncoder."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.preprocessing.imputer import MissingValueImputer
from ml_pipeline.preprocessing.encoder import CategoricalEncoder


@pytest.fixture
def train_df() -> pd.DataFrame:
    # age: 25 distinct values (well above categorical threshold, not sequential-by-1)
    # city: low cardinality (3 values) -> should route to one-hot
    # region_code: high cardinality (15 values, each repeating) -> should route
    #   to frequency encoding. Values repeat because a real region code would
    #   naturally cover multiple rows -- a code that's unique per row is
    #   structurally indistinguishable from a row ID, which is a genuine,
    #   unavoidable ambiguity, not something we can code around.
    return pd.DataFrame(
        {
            "age": [22 + (i * 3) % 40 for i in range(25)],
            "city": ["Delhi", "Mumbai", "Pune", None] * 6 + ["Delhi"],
            "region_code": [f"R{i % 15}" for i in range(25)],
        }
    )


def test_imputer_fills_numeric_with_median(train_df):
    train_df.loc[0, "age"] = np.nan
    expected_median = train_df["age"].median()

    imputer = MissingValueImputer()
    imputer.fit(train_df)
    result = imputer.transform(train_df)

    assert result["age"].isnull().sum() == 0
    assert result.loc[0, "age"] == expected_median


def test_imputer_fills_categorical_with_mode(train_df):
    imputer = MissingValueImputer()
    imputer.fit(train_df)
    result = imputer.transform(train_df)

    assert result["city"].isnull().sum() == 0
    assert result.loc[3, "city"] == "Delhi"  # Delhi is the mode (7 occurrences)


def test_imputer_reuses_train_stats_on_test_data(train_df):
    # Simulate correct usage: fit ONLY on train, then transform test.
    imputer = MissingValueImputer()
    imputer.fit(train_df)
    train_median = imputer._fill_values["age"]

    test_df = pd.DataFrame({"age": [np.nan, 999], "city": [None, "Pune"], "region_code": ["R1", "R2"]})
    result = imputer.transform(test_df)

    # The filled value must match the value learned from TRAIN, not
    # anything derived from test_df itself (which has a wildly
    # different distribution, e.g. 999).
    assert result.loc[0, "age"] == train_median


def test_imputer_raises_if_not_fitted():
    imputer = MissingValueImputer()
    with pytest.raises(RuntimeError):
        imputer.transform(pd.DataFrame({"age": [1, 2]}))


def test_low_cardinality_column_gets_onehot_encoded(train_df):
    encoder = CategoricalEncoder()
    encoder.fit(train_df.fillna({"city": "Delhi"}))
    result = encoder.transform(train_df.fillna({"city": "Delhi"}))

    onehot_cols = [c for c in result.columns if c.startswith("city_")]
    assert len(onehot_cols) == 3  # Delhi, Mumbai, Pune
    assert "region_code" not in result.columns  # replaced, not left raw

def test_high_cardinality_column_gets_frequency_encoded(train_df):
    encoder = CategoricalEncoder()
    filled = train_df.fillna({"city": "Delhi"})
    encoder.fit(filled)
    result = encoder.transform(filled)

    assert "region_code_freq" in result.columns
    # 15 unique codes across 25 rows, cycling via i % 15 -> R0..R9 appear
    # twice each (rows 0-14 then 15-24 wrap), R10..R14 appear once each.
    # Row 0 is "R0", which appears at i=0 and i=15 -> frequency = 2/25.
    assert result["region_code_freq"].iloc[0] == pytest.approx(2 / 25)


def test_unseen_category_at_transform_time_degrades_gracefully(train_df):
    filled = train_df.fillna({"city": "Delhi"})
    encoder = CategoricalEncoder()
    encoder.fit(filled)

    unseen_df = pd.DataFrame(
        {"age": [30], "city": ["Bangalore"], "region_code": ["R999"]}  # never seen in fit
    )
    result = encoder.transform(unseen_df)

    city_onehot_cols = [c for c in result.columns if c.startswith("city_")]
    assert result[city_onehot_cols].iloc[0].sum() == 0  # all-zero row, no crash
    assert result["region_code_freq"].iloc[0] == 0.0  # unseen -> 0.0, no crash