"""
Builds a sorted leaderboard DataFrame from cross-validation results
-- the artifact the Streamlit frontend (Phase 13) will render as a
model comparison table.
"""

import pandas as pd

from ml_pipeline.data.schema_detector import ProblemType
from ml_pipeline.training.trainer import ModelResult, PRIMARY_METRIC


def build_leaderboard(results: list[ModelResult], problem_type: ProblemType) -> pd.DataFrame:
    """
    Convert ModelResult objects into a single, sorted DataFrame.

    Args:
        results: output of ModelTrainer.compare_models().
        problem_type: used to determine which metric ranks the table.

    Returns:
        DataFrame with one row per model, sorted best-first by the
        primary metric for this problem type.
    """
    rows = []
    for r in results:
        row: dict = {"model": r.model_name, "fit_time_seconds": r.fit_time_seconds}
        for metric, value in r.mean_scores.items():
            row[metric] = round(value, 4)
            row[f"{metric}_std"] = round(r.std_scores[metric], 4)
        rows.append(row)

    leaderboard = pd.DataFrame(rows)
    primary_metric = PRIMARY_METRIC[problem_type]
    if not leaderboard.empty:
        leaderboard = leaderboard.sort_values(
            by=primary_metric, ascending=False
        ).reset_index(drop=True)
    return leaderboard