"""Tests for DataValidator and SchemaDetector."""

import pandas as pd
import pytest

from ml_pipeline.data.validator import DataValidator
from ml_pipeline.data.schema_detector import SchemaDetector, ProblemType, FeatureType


@pytest.fixture
def valid_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": range(1, 31),
            "salary": [30000 + i * 500 for i in range(30)],
            "city": ["Delhi", "Mumbai"] * 15,
            "purchased": [0, 1] * 15,
        }
    )


def test_valid_dataset_passes(valid_df):
    result = DataValidator().validate(valid_df)
    assert result.is_valid
    assert result.errors == []


def test_too_few_rows_fails():
    tiny_df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    result = DataValidator().validate(tiny_df)
    assert not result.is_valid
    assert any("rows" in e for e in result.errors)


def test_fully_null_column_fails(valid_df):
    valid_df["empty_col"] = None
    result = DataValidator().validate(valid_df)
    assert not result.is_valid
    assert any("empty_col" in e for e in result.errors)


def test_single_unique_target_fails(valid_df):
    valid_df["constant_target"] = 1
    result = DataValidator().validate_target_column(valid_df, "constant_target")
    assert not result.is_valid


def test_categorical_target_detected_as_classification(valid_df):
    problem_type = SchemaDetector().detect_problem_type(valid_df, "purchased")
    assert problem_type == ProblemType.CLASSIFICATION


def test_continuous_target_detected_as_regression(valid_df):
    problem_type = SchemaDetector().detect_problem_type(valid_df, "salary")
    assert problem_type == ProblemType.REGRESSION


def test_feature_types_detected_correctly(valid_df):
    schema = SchemaDetector().detect_feature_types(valid_df)
    assert schema["city"] == FeatureType.CATEGORICAL
    assert schema["salary"] == FeatureType.NUMERICAL