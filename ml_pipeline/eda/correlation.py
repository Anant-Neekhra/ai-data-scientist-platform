"""
Correlation analysis: feature-feature (multicollinearity) and
feature-target relationships, computed only over numerical columns
since Pearson correlation is undefined for categorical data.
"""

from dataclasses import dataclass, field
import pandas as pd

from ml_pipeline.data.schema_detector import SchemaDetector, FeatureType
from utils.logger import get_logger

logger = get_logger(__name__)

# Pairs of features correlated above this are flagged as redundant
# (multicollinearity risk) — 0.8 is a common practical rule of thumb.
HIGH_CORRELATION_THRESHOLD = 0.8


@dataclass
class CorrelationReport:
    """Full correlation analysis for a dataset."""

    correlation_matrix: pd.DataFrame
    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    target_correlations: dict[str, float] | None = None


class CorrelationAnalyzer:
    """Computes Pearson correlation across numerical features."""

    def __init__(self) -> None:
        self._schema_detector = SchemaDetector()

    def analyze(self, df: pd.DataFrame, target_col: str | None = None) -> CorrelationReport:
        """
        Compute the correlation matrix over numerical columns, flag
        highly-correlated feature pairs, and optionally rank features
        by correlation with a target column.

        Args:
            df: the dataset to analyze.
            target_col: optional target column name. If provided and
                numeric, feature-target correlations are also computed.

        Returns:
            CorrelationReport with the matrix, flagged pairs, and
            (optionally) target correlations.
        """
        schema = self._schema_detector.detect_feature_types(df)
        numerical_cols = [
            col for col, ftype in schema.items() if ftype == FeatureType.NUMERICAL
        ]

        if len(numerical_cols) < 2:
            logger.info("Fewer than 2 numerical columns — correlation matrix is trivial.")
            corr_matrix = df[numerical_cols].corr() if numerical_cols else pd.DataFrame()
            return CorrelationReport(correlation_matrix=corr_matrix)

        corr_matrix = df[numerical_cols].corr(method="pearson")
        high_pairs = self._find_high_correlation_pairs(corr_matrix)

        target_correlations = None
        if (
            target_col
            and target_col in numerical_cols
            and target_col in corr_matrix.columns
        ):
            target_correlations = (
                corr_matrix[target_col]
                .drop(labels=[target_col])
                .sort_values(key=abs, ascending=False)
                .round(4)
                .to_dict()
            )

        logger.info(
            f"Correlation computed over {len(numerical_cols)} numeric columns, "
            f"{len(high_pairs)} high-correlation pairs found"
        )
        return CorrelationReport(
            correlation_matrix=corr_matrix,
            high_corr_pairs=high_pairs,
            target_correlations=target_correlations,
        )

    @staticmethod
    def _find_high_correlation_pairs(
        corr_matrix: pd.DataFrame,
    ) -> list[tuple[str, str, float]]:
        """Find column pairs whose |correlation| exceeds the threshold."""
        pairs: list[tuple[str, str, float]] = []
        cols = corr_matrix.columns.tolist()

        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):  # upper triangle only, skip diagonal
                value = corr_matrix.iloc[i, j]
                if pd.notna(value) and abs(value) >= HIGH_CORRELATION_THRESHOLD:
                    pairs.append((cols[i], cols[j], round(float(value), 4)))

        return sorted(pairs, key=lambda p: abs(p[2]), reverse=True)