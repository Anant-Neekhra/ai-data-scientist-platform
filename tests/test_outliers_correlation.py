"""Tests for OutlierDetector and CorrelationAnalyzer."""

import numpy as np
import pandas as pd
import pytest

from ml_pipeline.eda.outliers import OutlierDetector
from ml_pipeline.eda.correlation import CorrelationAnalyzer


@pytest.fixture
def outlier_df() -> pd.DataFrame:
    # 20 normal values clustered around 50, plus 2 obvious outliers
    normal_values = list(range(45, 65))  # 45..64
    return pd.DataFrame({"score": normal_values + [500, -300]})


@pytest.fixture
def correlated_df() -> pd.DataFrame:
    n = 30
    x = pd.Series([i * 7 + 3 for i in range(n)])       # not sequential by 1 -> not id-like
    return pd.DataFrame(
        {
            "x": x,
            "y_strongly_correlated": x * 2 + 1,          # perfectly linear with x
            "z_uncorrelated": [((i * 41 + 17) % 97) for i in range(n)],  # noise-like, high cardinality
        }
    )


def test_outliers_detected_by_iqr(outlier_df):
    reports = OutlierDetector().detect(outlier_df)
    score_report = next(r for r in reports if r.column == "score")
    assert score_report.iqr_outlier_count == 2
    assert 500 in outlier_df.loc[score_report.outlier_row_indices, "score"].values
    assert -300 in outlier_df.loc[score_report.outlier_row_indices, "score"].values


def test_constant_column_has_no_zscore_outliers():
    # A constant column will always be classified categorical by schema
    # detection (nunique=1), so we test the std==0 guard directly on
    # the per-column analyzer rather than through detect().
    series = pd.Series([5] * 15)
    report = OutlierDetector().analyze_column(series, "constant")
    assert report.zscore_outlier_count == 0
    assert report.iqr_outlier_count == 0


def test_high_correlation_pair_detected(correlated_df):
    report = CorrelationAnalyzer().analyze(correlated_df)
    pair_columns = [{p[0], p[1]} for p in report.high_corr_pairs]
    assert {"x", "y_strongly_correlated"} in pair_columns


def test_uncorrelated_pair_not_flagged(correlated_df):
    report = CorrelationAnalyzer().analyze(correlated_df)
    pair_columns = [{p[0], p[1]} for p in report.high_corr_pairs]
    assert {"x", "z_uncorrelated"} not in pair_columns


def test_target_correlation_ranking(correlated_df):
    report = CorrelationAnalyzer().analyze(correlated_df, target_col="x")
    assert report.target_correlations is not None
    # y_strongly_correlated should rank above z_uncorrelated for target 'x'
    ranked_cols = list(report.target_correlations.keys())
    assert ranked_cols.index("y_strongly_correlated") < ranked_cols.index("z_uncorrelated")