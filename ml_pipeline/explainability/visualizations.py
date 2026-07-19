"""
Plotly figures for SHAP explanations. Same zero-Streamlit-imports
separation as every prior visualization module.
"""

import numpy as np
import plotly.graph_objects as go

from ml_pipeline.explainability.shap_explainer import GlobalExplanation, LocalExplanation


class ShapVisualizer:
    """Builds Plotly figures from SHAP explanations."""

    def global_importance_bar(self, explanation: GlobalExplanation, top_n: int = 15) -> go.Figure:
        """Mean |SHAP value| per feature -- the SHAP-based equivalent
        of Day 9's model.feature_importances_ bar chart, but derived
        from actual prediction contributions rather than internal
        tree-split statistics."""
        order = np.argsort(explanation.mean_abs_shap)[::-1][:top_n]
        names = [explanation.feature_names[i] for i in order][::-1]
        values = [explanation.mean_abs_shap[i] for i in order][::-1]

        fig = go.Figure(go.Bar(x=values, y=names, orientation="h"))
        fig.update_layout(
            title=f"Global Feature Importance (mean |SHAP value|, top {len(names)})",
            xaxis_title="Mean |SHAP value|",
        )
        return fig

    def beeswarm_plot(self, explanation: GlobalExplanation, top_n: int = 12) -> go.Figure:
        """
        One point per (row, feature): x = SHAP value (impact on
        prediction), color = the feature's actual value. Reveals not
        just THAT a feature matters, but the DIRECTION of its effect
        -- e.g. high Fare values (bright) clustering on the positive
        side means high fares push predictions up.
        """
        order = np.argsort(explanation.mean_abs_shap)[::-1][:top_n]
        fig = go.Figure()

        for rank, feature_idx in enumerate(order):
            feature_name = explanation.feature_names[feature_idx]
            shap_col = explanation.shap_values[:, feature_idx]
            value_col = explanation.feature_values.iloc[:, feature_idx]

            # Normalize feature values to 0-1 for consistent color scaling
            # across features with very different natural ranges (e.g. Age
            # 0-80 vs. a one-hot 0/1 column).
            v = value_col.astype(float)
            v_norm = (v - v.min()) / (v.max() - v.min()) if v.max() > v.min() else v * 0

            fig.add_trace(
                go.Scatter(
                    x=shap_col,
                    y=[rank] * len(shap_col),
                    mode="markers",
                    marker=dict(color=v_norm, colorscale="Bluered", size=6, opacity=0.6),
                    name=feature_name,
                    showlegend=False,
                )
            )

        fig.update_layout(
            title="SHAP Summary (beeswarm)",
            xaxis_title="SHAP value (impact on prediction)",
            yaxis=dict(
                tickmode="array",
                tickvals=list(range(len(order))),
                ticktext=[explanation.feature_names[i] for i in order],
            ),
        )
        return fig

    def local_waterfall(self, explanation: LocalExplanation, top_n: int = 10) -> go.Figure:
        """
        For ONE prediction: shows exactly how each feature pushed the
        result up or down from the base value to the final prediction.
        Bars sorted by |impact|, largest first.
        """
        order = np.argsort(np.abs(explanation.shap_values))[::-1][:top_n]
        names = [
            f"{explanation.feature_names[i]}={explanation.feature_values.iloc[i]:.2f}"
            for i in order
        ]
        values = [explanation.shap_values[i] for i in order]
        colors = ["#d62728" if v > 0 else "#1f77b4" for v in values]

        fig = go.Figure(go.Bar(x=values, y=names[::-1], orientation="h", marker_color=colors[::-1]))
        fig.add_vline(x=0, line_dash="dash")
        fig.update_layout(
            title=(
                f"Local Explanation (base={explanation.base_value:.3f}, "
                f"prediction={explanation.predicted_value:.3f})"
            ),
            xaxis_title="SHAP value (red = pushes prediction up, blue = pushes down)",
        )
        return fig