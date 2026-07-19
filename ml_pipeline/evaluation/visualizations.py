"""
Plotly figures for the evaluation dashboard. Same separation-of-
concerns principle as Day 4's EDA visualizer: zero Streamlit
imports, returns Figure objects for the frontend to render.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve, auc

from ml_pipeline.evaluation.classification_metrics import ClassificationMetrics


class EvaluationVisualizer:
    """Builds evaluation plots for classification and regression."""

    def confusion_matrix_heatmap(self, metrics: ClassificationMetrics) -> go.Figure:
        labels = [str(l) for l in metrics.class_labels]
        fig = go.Figure(
            data=go.Heatmap(
                z=metrics.confusion_matrix,
                x=[f"Predicted: {l}" for l in labels],
                y=[f"Actual: {l}" for l in labels],
                text=metrics.confusion_matrix,
                texttemplate="%{text}",
                colorscale="Blues",
            )
        )
        fig.update_layout(title="Confusion Matrix", yaxis_autorange="reversed")
        return fig

    def roc_curve_plot(self, y_true, y_proba_positive) -> go.Figure:
        """ROC curve for BINARY classification only."""
        fpr, tpr, _ = roc_curve(y_true, y_proba_positive)
        auc_value = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"ROC (AUC={auc_value:.3f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline", line=dict(dash="dash")))
        fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
        return fig

    def precision_recall_curve_plot(self, y_true, y_proba_positive) -> go.Figure:
        precision, recall, _ = precision_recall_curve(y_true, y_proba_positive)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name="Precision-Recall"))
        fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall", yaxis_title="Precision")
        return fig

    def feature_importance_bar(self, importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
        top = importance_df.head(top_n).iloc[::-1]  # reverse -> biggest bar on top
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h"))
        fig.update_layout(title=f"Top {min(top_n, len(importance_df))} Feature Importances")
        return fig

    def learning_curve_plot(self, curve_data: dict) -> go.Figure:
        sizes = curve_data["train_sizes"]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sizes, y=curve_data["train_scores_mean"], mode="lines+markers", name="Training score"))
        fig.add_trace(go.Scatter(x=sizes, y=curve_data["val_scores_mean"], mode="lines+markers", name="Validation score"))
        fig.update_layout(title="Learning Curve", xaxis_title="Training examples", yaxis_title="Score")
        return fig

    def regression_actual_vs_predicted(self, y_true, y_pred) -> go.Figure:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", name="Predictions", opacity=0.6))
        fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Perfect prediction", line=dict(dash="dash")))
        fig.update_layout(title="Actual vs Predicted", xaxis_title="Actual", yaxis_title="Predicted")
        return fig

    def regression_residual_plot(self, y_true, y_pred) -> go.Figure:
        y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
        residuals = y_true - y_pred

        fig = go.Figure(go.Scatter(x=y_pred, y=residuals, mode="markers", opacity=0.6))
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title="Residual Plot", xaxis_title="Predicted", yaxis_title="Residual (Actual - Predicted)")
        return fig