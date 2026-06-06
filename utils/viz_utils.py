"""
Visualization utility functions using Plotly for interactive charts.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Optional, List


def plot_histogram(data: pd.Series, title: str = "", color: str = "#2563eb") -> go.Figure:
    """Create an interactive histogram with KDE overlay."""
    fig = go.Figure()

    fig.add_trace(go.Histogram(
        x=data.dropna(),
        nbinsx=30,
        name='Histogram',
        marker_color=color,
        marker_line_color='white',
        marker_line_width=0.5,
        opacity=0.75,
        histnorm='probability density'
    ))

    # KDE using violin-like approach
    fig.add_trace(go.Violin(
        x=data.dropna(),
        name='Density',
        line_color=color,
        fillcolor='rgba(37, 99, 235, 0.15)',
        points=False,
        showlegend=False,
        side='positive',
        spanmode='hard'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title=data.name if data.name else "Value",
        yaxis_title="Density",
        template="plotly_white",
        bargap=0.05,
        height=380,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    return fig


def plot_correlation_heatmap(df: pd.DataFrame, title: str = "特征相关性热图") -> go.Figure:
    """Create an interactive correlation heatmap."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        return go.Figure()

    corr = numeric_df.corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns.tolist(),
        y=corr.index.tolist(),
        colorscale='RdBu_r',
        zmid=0,
        text=np.round(corr.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        hoverongaps=False,
        colorbar=dict(title="Correlation")
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=500,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_missing_values(df: pd.DataFrame, title: str = "缺失值分析") -> go.Figure:
    """Create a bar chart showing missing value counts."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="✅ 没有缺失值", showarrow=False, font=dict(size=16, color='green'))
        fig.update_layout(height=200)
        return fig

    fig = go.Figure(data=go.Bar(
        x=missing.values,
        y=missing.index.tolist(),
        orientation='h',
        marker_color='#ef4444',
        marker_line_color='white',
        marker_line_width=0.5,
        text=missing.values,
        textposition='outside'
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="缺失数量",
        template="plotly_white",
        height=max(200, len(missing) * 30 + 100),
        margin=dict(l=20, r=60, t=50, b=20)
    )
    return fig


def plot_class_distribution(labels: pd.Series, title: str = "类别分布") -> go.Figure:
    """Create a pie/bar chart for class distribution."""
    counts = labels.value_counts().sort_index()

    colors = ['#2563eb', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6']

    fig = go.Figure(data=[
        go.Pie(
            labels=[f"Class {int(k)}" for k in counts.index],
            values=counts.values,
            marker_colors=colors[:len(counts)],
            textinfo='label+percent',
            textfont_size=13,
            hole=0.4,
            hovertemplate='%{label}<br>Count: %{value}<br>Percentage: %{percent}'
        )
    ])

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        template="plotly_white",
        height=420,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_model_comparison(metrics_list: List[dict], model_names: List[str]) -> go.Figure:
    """Create a grouped bar chart comparing multiple models."""
    metric_keys = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']

    fig = go.Figure()

    colors = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626']
    width = 0.8 / len(model_names)

    for i, (name, metrics) in enumerate(zip(model_names, metrics_list)):
        values = [metrics.get(k, 0) or 0 for k in metric_keys]
        fig.add_trace(go.Bar(
            name=name,
            x=metric_labels,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[f'{v:.3f}' if v else 'N/A' for v in values],
            textposition='outside',
            textfont_size=10
        ))

    fig.update_layout(
        title=dict(text="模型性能对比", font=dict(size=16)),
        barmode='group',
        template="plotly_white",
        height=420,
        yaxis=dict(range=[0, 1.15], title="Score"),
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def plot_similarity_heatmap(similarities: List[float], labels: List[str],
                            title: str = "分子相似性") -> go.Figure:
    """Create a horizontal bar chart for similarity scores."""
    colors = ['#059669' if s > 0.7 else '#2563eb' if s > 0.4 else '#d97706'
              for s in similarities]

    fig = go.Figure(data=go.Bar(
        x=similarities,
        y=labels,
        orientation='h',
        marker_color=colors,
        marker_line_color='white',
        marker_line_width=0.5,
        text=[f'{s:.3f}' for s in similarities],
        textposition='outside',
        textfont_size=11
    ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        xaxis_title="Tanimoto Similarity",
        xaxis=dict(range=[0, 1.05]),
        template="plotly_white",
        height=max(200, len(similarities) * 35 + 100),
        margin=dict(l=20, r=80, t=50, b=20)
    )
    return fig
