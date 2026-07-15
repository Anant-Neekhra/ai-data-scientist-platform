"""
Validates DataFrames before they enter the ML pipeline.
Fails fast with clear, actionable error messages.
"""

from dataclasses import dataclass, field
import pandas as pd

from utils.logger import get_logger

logger = get_logger(__name__)

MIN_ROWS = 10
MAX_NULL_RATIO_WARNING = 0.5  # warn if a column is >50% missing


@dataclass
class ValidationResult:
    """Outcome of running validation checks on a dataset."""

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class DataValidator:
    """Runs a series of sanity checks on an uploaded dataset."""

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        """
        Run all validation checks on a DataFrame.

        Args:
            df: the dataset to validate.

        Returns:
            ValidationResult with errors (blocking) and warnings
            (non-blocking, but shown to the user).
        """
        errors: list[str] = []
        warnings: list[str] = []

        if df.shape[0] < MIN_ROWS:
            errors.append(
                f"Dataset has only {df.shape[0]} rows; at least "
                f"{MIN_ROWS} are required for a meaningful train/test split."
            )

        if df.shape[1] < 2:
            errors.append(
                "Dataset must have at least 2 columns "
                "(1+ feature column and a target column)."
            )

        duplicate_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicate_cols:
            errors.append(f"Duplicate column names found: {duplicate_cols}")

        fully_null_cols = df.columns[df.isnull().all()].tolist()
        if fully_null_cols:
            errors.append(f"Columns are entirely empty: {fully_null_cols}")

        duplicate_rows = df.duplicated().sum()
        if duplicate_rows > 0:
            warnings.append(
                f"{duplicate_rows} duplicate rows found "
                f"({duplicate_rows / len(df):.1%} of the dataset)."
            )

        high_null_cols = [
            col
            for col in df.columns
            if df[col].isnull().mean() > MAX_NULL_RATIO_WARNING
        ]
        if high_null_cols:
            warnings.append(
                f"Columns with >{MAX_NULL_RATIO_WARNING:.0%} missing values: "
                f"{high_null_cols}"
            )

        result = ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
        logger.info(
            f"Validation complete: is_valid={result.is_valid}, "
            f"errors={len(errors)}, warnings={len(warnings)}"
        )
        return result

    def validate_target_column(self, df: pd.DataFrame, target_col: str) -> ValidationResult:
        """
        Validate that a chosen target column is usable.

        Args:
            df: the dataset.
            target_col: name of the column the user selected as target.

        Returns:
            ValidationResult specific to the target column choice.
        """
        errors: list[str] = []
        warnings: list[str] = []

        if target_col not in df.columns:
            errors.append(f"'{target_col}' is not a column in this dataset.")
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)

        null_count = df[target_col].isnull().sum()
        if null_count > 0:
            warnings.append(
                f"Target column '{target_col}' has {null_count} missing values; "
                f"these rows will be dropped before training."
            )

        if df[target_col].nunique() == 1:
            errors.append(
                f"Target column '{target_col}' has only one unique value — "
                f"there is nothing for a model to learn."
            )

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)