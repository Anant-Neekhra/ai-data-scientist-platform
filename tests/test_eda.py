"""Tests for StatisticsEngine and DataQualityAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.eda.statistics import StatisticsEngine
from ml_pipeline.eda.data_quality import DataQualityAnalyzer
from ml_pipeline.data.schema_detector import FeatureType


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [25, 30, 35, np.nan, 40, 45, 50, 22, 28, 33] * 3,
            "salary": [30000, 35000, 40000, 45000, 50000, 55000, 60000, 32000, 38000, 42000] * 3,
            "city": ["Delhi", "Mumbai", "Delhi", "Pune", "Mumbai"] * 6,
        }
    )


def test_dataset_stats_shape(sample_df):
    stats = StatisticsEngine().compute(sample_df)
    assert stats.n_rows == 30
    assert stats.n_columns == 3
    assert len(stats.column_stats) == 3


def test_numeric_column_gets_numeric_summary(sample_df):
    stats = StatisticsEngine().compute(sample_df)
    salary_stats = next(c for c in stats.column_stats if c.name == "salary")
    assert salary_stats.feature_type == FeatureType.NUMERICAL
    assert salary_stats.numeric_summary is not None
    assert "mean" in salary_stats.numeric_summary
    assert salary_stats.categorical_summary is None


def test_categorical_column_gets_categorical_summary(sample_df):
    stats = StatisticsEngine().compute(sample_df)
    city_stats = next(c for c in stats.column_stats if c.name == "city")
    assert city_stats.feature_type == FeatureType.CATEGORICAL
    assert city_stats.categorical_summary is not None
    assert city_stats.numeric_summary is None


def test_missing_value_report(sample_df):
    report = DataQualityAnalyzer().analyze_missing_values(sample_df)
    assert report.per_column["age"]["count"] == 3  # 1 NaN x 3 repeats
    assert report.per_column["salary"]["count"] == 0
    assert report.total_missing_cells == 3


def test_duplicate_report_detects_repeats(sample_df):
    # sample_df repeats the same 10 rows 3 times -> heavy duplication
    report = DataQualityAnalyzer().analyze_duplicates(sample_df)
    assert report.duplicate_count == 20  # 30 rows, 10 unique -> 20 dupes
    assert len(report.duplicate_row_indices) == 20


def test_high_risk_missing_column_flagged():
    df = pd.DataFrame(
        {
            "mostly_missing": [None] * 8 + [1, 2],
            "fine": range(10),
        }
    )
    report = DataQualityAnalyzer().analyze_missing_values(df)
    assert "mostly_missing" in report.high_risk_columns
    assert "fine" not in report.high_risk_columns