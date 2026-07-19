"""
Regression evaluation metrics: MAE, RMSE, R2, MAPE.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RegressionMetrics:
    mae: float
    rmse: float
    r2: float
    mape: float | None  # None if y_true contains zeros (MAPE mathematically undefined)


def compute_regression_metrics(y_true, y_pred) -> RegressionMetrics:
    """Compute a standard regression metric bundle. See Day 9 notes
    on why all four metrics are reported rather than just one."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mape = None
    if not np.any(y_true == 0):
        mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    else:
        logger.info("Target contains zero values -- MAPE is undefined, skipping.")

    return RegressionMetrics(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mean_squared_error(y_true, y_pred))),
        r2=float(r2_score(y_true, y_pred)),
        mape=mape,
    )