"""
Computes descriptive statistics for a dataset, branching behavior
by feature type (numerical vs categorical) since the same summary
doesn't make sense for both.
"""

from dataclasses import dataclass, field
from typing import Any
import pandas as pd

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ColumnStats:
    """Summary statistics for a single column."""

    name: str
    feature_type: FeatureType
    count: int
    missing_count: int
    missing_pct: float
    numeric_summary: dict[str, float] | None = None
    categorical_summary: dict[str, Any] | None = None


@dataclass
class DatasetStats:
    """Full statistical summary of a dataset."""

    n_rows: int
    n_columns: int
    memory_usage_mb: float
    column_stats: list[ColumnStats] = field(default_factory=list)


class StatisticsEngine:
    """Computes per-column and dataset-level descriptive statistics."""

    def __init__(self) -> None:
        self._schema_detector = SchemaDetector()

    def compute(self, df: pd.DataFrame) -> DatasetStats:
        """
        Compute full descriptive statistics for a dataset.

        Args:
            df: the dataset to summarize.

        Returns:
            DatasetStats containing dataset-level and per-column summaries.
        """
        schema = self._schema_detector.detect_feature_types(df)

        column_stats = [
            self._compute_column_stats(df[col], col, schema[col]) for col in df.columns
        ]

        stats = DatasetStats(
            n_rows=df.shape[0],
            n_columns=df.shape[1],
            memory_usage_mb=round(df.memory_usage(deep=True).sum() / (1024**2), 3),
            column_stats=column_stats,
        )
        logger.info(
            f"Computed stats for {stats.n_rows} rows x {stats.n_columns} cols "
            f"({stats.memory_usage_mb} MB)"
        )
        return stats

    def _compute_column_stats(
        self, series: pd.Series, name: str, feature_type: FeatureType
    ) -> ColumnStats:
        missing_count = int(series.isnull().sum())
        count = int(series.notnull().sum())
        missing_pct = round(missing_count / len(series) * 100, 2) if len(series) else 0.0

        numeric_summary = None
        categorical_summary = None

        if feature_type == FeatureType.NUMERICAL:
            numeric_summary = self._numeric_summary(series)
        else:
            # Categorical, boolean, id_like, and datetime columns all get
            # a value-counts-based summary rather than a numeric one.
            categorical_summary = self._categorical_summary(series)

        return ColumnStats(
            name=name,
            feature_type=feature_type,
            count=count,
            missing_count=missing_count,
            missing_pct=missing_pct,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
        )

    @staticmethod
    def _numeric_summary(series: pd.Series) -> dict[str, float]:
        clean = series.dropna()
        if clean.empty:
            return {}
        return {
            "mean": round(float(clean.mean()), 4),
            "std": round(float(clean.std()), 4),
            "min": round(float(clean.min()), 4),
            "q25": round(float(clean.quantile(0.25)), 4),
            "median": round(float(clean.median()), 4),
            "q75": round(float(clean.quantile(0.75)), 4),
            "max": round(float(clean.max()), 4),
            "skewness": round(float(clean.skew()), 4),
            "kurtosis": round(float(clean.kurt()), 4),
        }

    @staticmethod
    def _categorical_summary(series: pd.Series, top_n: int = 10) -> dict[str, Any]:
        clean = series.dropna()
        if clean.empty:
            return {}
        value_counts = clean.value_counts().head(top_n)
        return {
            "unique_count": int(clean.nunique()),
            "mode": clean.mode().iloc[0] if not clean.mode().empty else None,
            "top_values": value_counts.to_dict(),
        }