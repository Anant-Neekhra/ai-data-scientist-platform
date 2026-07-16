"""
Centralized logging configuration.
Import get_logger(__name__) in any module that needs to log.
"""

import logging
import sys
from config.settings import LOGS_DIR, LOG_LEVEL


def get_logger(name: str) -> logging.Logger:
    """
    Create (or retrieve) a configured logger.

    Args:
        name: typically __name__ of the calling module, so log
              lines show exactly which file produced them.

    Returns:
        A logger writing to both console and a rotating log file.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if this logger was already configured
    # (e.g. Streamlit re-runs scripts on every interaction)
    if logger.handlers:
        return logger

    logger.setLevel(LOG_LEVEL)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
    LOGS_DIR / "app.log",
    maxBytes=5 * 1024 * 1024,  # 5 MB
    backupCount=3,
    encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger