"""
Outlier detection for numerical columns using two complementary
statistical methods (IQR and Z-score), since neither is universally
correct and their disagreement is itself informative.
"""

from dataclasses import dataclass, field
import numpy as np
import pandas as pd

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)

IQR_MULTIPLIER = 1.5
Z_SCORE_THRESHOLD = 3.0


@dataclass
class ColumnOutlierReport:
    """Outlier findings for a single numerical column."""

    column: str
    iqr_outlier_count: int
    iqr_outlier_pct: float
    iqr_bounds: tuple[float, float]
    zscore_outlier_count: int
    zscore_outlier_pct: float
    outlier_row_indices: list[int] = field(default_factory=list)


class OutlierDetector:
    """Detects outliers in numerical columns via IQR and Z-score methods."""

    def __init__(self) -> None:
        self._schema_detector = SchemaDetector()

    def detect(self, df: pd.DataFrame) -> list[ColumnOutlierReport]:
        """
        Run outlier detection on every numerical column in the dataset.

        Args:
            df: the dataset to analyze.

        Returns:
            List of ColumnOutlierReport, one per numerical column.
        """
        schema = self._schema_detector.detect_feature_types(df)
        numerical_cols = [
            col for col, ftype in schema.items() if ftype == FeatureType.NUMERICAL
        ]

        reports = [self.analyze_column(df[col], col) for col in numerical_cols]
        logger.info(f"Outlier detection ran on {len(reports)} numerical columns")
        return reports

    def analyze_column(self, series: pd.Series, name: str) -> ColumnOutlierReport:
        clean = series.dropna()

        # --- IQR method ---
        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr
        iqr_outlier_mask = (clean < lower_bound) | (clean > upper_bound)
        iqr_outlier_indices = clean.index[iqr_outlier_mask].tolist()

        # --- Z-score method ---
        std = clean.std()
        if std == 0 or pd.isna(std):
            # Constant column: no variation, so no meaningful z-score outliers.
            zscore_outlier_count = 0
        else:
            z_scores = (clean - clean.mean()) / std
            zscore_outlier_count = int((z_scores.abs() > Z_SCORE_THRESHOLD).sum())

        n = len(clean) if len(clean) else 1

        return ColumnOutlierReport(
            column=name,
            iqr_outlier_count=len(iqr_outlier_indices),
            iqr_outlier_pct=round(len(iqr_outlier_indices) / n * 100, 2),
            iqr_bounds=(round(float(lower_bound), 4), round(float(upper_bound), 4)),
            zscore_outlier_count=zscore_outlier_count,
            zscore_outlier_pct=round(zscore_outlier_count / n * 100, 2),
            outlier_row_indices=iqr_outlier_indices,  # IQR is our primary flag
        )