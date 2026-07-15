"""
Handles reading uploaded CSV files into pandas DataFrames and
persisting them to the datasets directory.
"""

from pathlib import Path
from typing import Union
import pandas as pd

from config.settings import DATASETS_DIR
from utils.logger import get_logger

logger = get_logger(__name__)


class DatasetLoadError(Exception):
    """Raised when a CSV file cannot be parsed into a DataFrame."""


class DatasetLoader:
    """Loads CSV files from disk or an upload buffer into DataFrames."""

    @staticmethod
    def load_csv(source: Union[str, Path]) -> pd.DataFrame:
        """
        Load a CSV file into a DataFrame, trying common encodings.

        Args:
            source: path to a CSV file.

        Returns:
            Parsed DataFrame.

        Raises:
            DatasetLoadError: if the file cannot be parsed with any
                supported encoding, or is empty.
        """
        # Real-world CSVs are not always UTF-8 (Excel exports on Windows
        # often use Latin-1 / cp1252). Try the common ones in order
        # instead of crashing on the first mismatch.
        encodings_to_try = ["utf-8", "latin1", "cp1252"]

        last_error: Exception | None = None
        for encoding in encodings_to_try:
            try:
                df = pd.read_csv(source, encoding=encoding)
                logger.info(
                    f"Loaded '{source}' with encoding='{encoding}' "
                    f"shape={df.shape}"
                )
                if df.empty:
                    raise DatasetLoadError(f"'{source}' parsed but contains 0 rows.")
                return df
            except (UnicodeDecodeError, pd.errors.ParserError) as e:
                last_error = e
                continue

        raise DatasetLoadError(
            f"Could not parse '{source}' with any of {encodings_to_try}. "
            f"Last error: {last_error}"
        )

    @staticmethod
    def save_uploaded_file(file_bytes: bytes, filename: str) -> Path:
        """
        Persist raw uploaded bytes to the datasets directory.

        Args:
            file_bytes: raw content of the uploaded file.
            filename: original filename (used as-is; caller should
                sanitize if accepting untrusted filenames).

        Returns:
            Path to the saved file on disk.
        """
        destination = DATASETS_DIR / filename
        destination.write_bytes(file_bytes)
        logger.info(f"Saved uploaded file to {destination}")
        return destination