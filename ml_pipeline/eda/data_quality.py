"""
Data quality reporting: missing values and duplicate rows, with
enough detail for a user to decide how to handle them (not just
whether a problem exists).
"""

from dataclasses import dataclass, field
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

# Columns above this missing percentage are flagged as high-risk —
# imputation quality degrades badly once too much information is gone.
HIGH_MISSING_THRESHOLD_PCT = 40.0


@dataclass
class MissingValueReport:
    """Missing value breakdown, per column and overall."""

    total_missing_cells: int
    total_cells: int
    overall_missing_pct: float
    per_column: dict[str, dict[str, float]] = field(default_factory=dict)
    high_risk_columns: list[str] = field(default_factory=list)


@dataclass
class DuplicateReport:
    """Duplicate row breakdown."""

    duplicate_count: int
    duplicate_pct: float
    duplicate_row_indices: list[int] = field(default_factory=list)


class DataQualityAnalyzer:
    """Analyzes missing values and duplicate rows in a dataset."""

    def analyze_missing_values(self, df: pd.DataFrame) -> MissingValueReport:
        """
        Build a per-column and overall missing-value report.

        Args:
            df: the dataset to analyze.

        Returns:
            MissingValueReport with counts, percentages, and a list of
            columns whose missingness is severe enough to need special
            handling in Phase 4.
        """
        total_cells = df.size
        total_missing = int(df.isnull().sum().sum())
        overall_pct = round(total_missing / total_cells * 100, 2) if total_cells else 0.0

        per_column: dict[str, dict[str, float]] = {}
        high_risk: list[str] = []

        for col in df.columns:
            missing_count = int(df[col].isnull().sum())
            missing_pct = round(missing_count / len(df) * 100, 2) if len(df) else 0.0
            per_column[col] = {"count": missing_count, "pct": missing_pct}
            if missing_pct >= HIGH_MISSING_THRESHOLD_PCT:
                high_risk.append(col)

        report = MissingValueReport(
            total_missing_cells=total_missing,
            total_cells=total_cells,
            overall_missing_pct=overall_pct,
            per_column=per_column,
            high_risk_columns=high_risk,
        )
        logger.info(
            f"Missing values: {total_missing}/{total_cells} cells "
            f"({overall_pct}%), high-risk columns: {high_risk}"
        )
        return report

    def analyze_duplicates(self, df: pd.DataFrame) -> DuplicateReport:
        """
        Identify fully duplicate rows.

        Args:
            df: the dataset to analyze.

        Returns:
            DuplicateReport with count, percentage, and the actual
            row indices so the caller can display/inspect/drop them.
        """
        duplicate_mask = df.duplicated(keep="first")
        duplicate_indices = df.index[duplicate_mask].tolist()
        duplicate_count = len(duplicate_indices)
        duplicate_pct = round(duplicate_count / len(df) * 100, 2) if len(df) else 0.0

        report = DuplicateReport(
            duplicate_count=duplicate_count,
            duplicate_pct=duplicate_pct,
            duplicate_row_indices=duplicate_indices,
        )
        logger.info(f"Duplicates: {duplicate_count} rows ({duplicate_pct}%)")
        return report