"""
Classification evaluation metrics: accuracy, precision, recall, F1,
ROC-AUC (binary only), and the confusion matrix.
"""

from dataclasses import dataclass
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    confusion_matrix: np.ndarray
    class_labels: list
    roc_auc: float | None = None  # only computed for binary classification


def compute_classification_metrics(
    y_true, y_pred, y_proba: np.ndarray | None = None
) -> ClassificationMetrics:
    """
    Compute a standard classification metric bundle.

    Args:
        y_true: ground-truth labels.
        y_pred: predicted labels.
        y_proba: predicted class probabilities (n_samples, n_classes),
            needed for ROC-AUC. Optional -- if omitted, roc_auc is None.

    Returns:
        ClassificationMetrics using WEIGHTED-average precision/recall/f1
        -- weighted accounts for class imbalance better than macro or
        micro averaging, appropriate for a platform that can't assume
        balanced classes on an arbitrary uploaded dataset.
    """
    class_labels = sorted(set(y_true) | set(y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    roc_auc = None
    if y_proba is not None and len(class_labels) == 2:
        try:
            roc_auc = float(roc_auc_score(y_true, y_proba[:, 1]))
        except (ValueError, IndexError) as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    return ClassificationMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        confusion_matrix=confusion_matrix(y_true, y_pred, labels=class_labels),
        class_labels=class_labels,
        roc_auc=roc_auc,
    )