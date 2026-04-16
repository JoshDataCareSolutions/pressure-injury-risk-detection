"""OSEMN Phase 3: Explore — EDA visualizations returning Plotly figures."""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config


def plot_class_distribution(df: pd.DataFrame) -> go.Figure:
    """Bar chart of pressure injury vs no pressure injury."""
    counts = df[config.TARGET].value_counts().reset_index()
    counts.columns = ["Class", "Count"]
    counts["Class"] = counts["Class"].map({0: "No PI", 1: "PI"})
    counts["Percentage"] = (counts["Count"] / counts["Count"].sum() * 100).round(1)

    fig = px.bar(counts, x="Class", y="Count", color="Class",
                 color_discrete_map={"No PI": "#2ecc71", "PI": "#e74c3c"},
                 text="Percentage",
                 title="Pressure Injury Class Distribution")
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, yaxis_title="Number of Admissions")
    return fig


def plot_feature_distribution(df: pd.DataFrame, feature: str) -> go.Figure:
    """Histogram of a feature split by outcome."""
    plot_df = df[[feature, config.TARGET]].dropna()
    plot_df["Outcome"] = plot_df[config.TARGET].map({0: "No PI", 1: "PI"})

    fig = px.histogram(plot_df, x=feature, color="Outcome",
                       color_discrete_map={"No PI": "#2ecc71", "PI": "#e74c3c"},
                       barmode="overlay", opacity=0.7,
                       title=f"Distribution of {feature} by Outcome")
    return fig


def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Correlation heatmap for numeric features."""
    numeric_cols = [c for c in config.NUMERIC_FEATURES if c in df.columns]
    if config.TARGET in df.columns:
        numeric_cols.append(config.TARGET)
    corr = df[numeric_cols].corr()

    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1,
                    title="Feature Correlation Heatmap")
    fig.update_layout(width=800, height=700)
    return fig


def plot_missing_data(df: pd.DataFrame) -> go.Figure:
    """Bar chart of missingness percentage per feature."""
    all_feats = config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES + config.BINARY_FEATURES
    available = [f for f in all_feats if f in df.columns]

    missing = df[available].isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=True)

    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(text="No missing data found", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        fig.update_layout(title="Missing Data Report")
        return fig

    fig = px.bar(x=missing.values, y=missing.index, orientation="h",
                 title="Missing Data by Feature",
                 labels={"x": "% Missing", "y": "Feature"})
    fig.update_traces(marker_color="#e67e22")
    return fig


def plot_prevalence_by_group(df: pd.DataFrame, group_col: str) -> go.Figure:
    """Bar chart of PI prevalence rate by demographic group."""
    if group_col not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text=f"Column '{group_col}' not found",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False)
        return fig

    grouped = df.groupby(group_col)[config.TARGET].agg(["mean", "count"]).reset_index()
    grouped.columns = [group_col, "PI Rate", "Count"]
    grouped["PI Rate"] = (grouped["PI Rate"] * 100).round(2)
    grouped = grouped.sort_values("PI Rate", ascending=False)

    fig = px.bar(grouped, x=group_col, y="PI Rate",
                 text="Count",
                 title=f"Pressure Injury Prevalence by {group_col}",
                 labels={"PI Rate": "PI Rate (%)"})
    fig.update_traces(texttemplate="n=%{text}", textposition="outside",
                      marker_color="#3498db")
    return fig


def plot_braden_boxplots(df: pd.DataFrame) -> go.Figure:
    """Box plots of Braden subscores by outcome."""
    braden_cols = [c for c in df.columns if c.startswith("braden_")]
    if not braden_cols:
        fig = go.Figure()
        fig.add_annotation(text="No Braden score data available",
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False)
        fig.update_layout(title="Braden Subscores by Outcome")
        return fig

    plot_df = df[braden_cols + [config.TARGET]].copy()
    plot_df["Outcome"] = plot_df[config.TARGET].map({0: "No PI", 1: "PI"})
    melted = plot_df.melt(id_vars=["Outcome"], value_vars=braden_cols,
                          var_name="Subscore", value_name="Value")
    melted["Subscore"] = melted["Subscore"].str.replace("braden_", "")

    fig = px.box(melted, x="Subscore", y="Value", color="Outcome",
                 color_discrete_map={"No PI": "#2ecc71", "PI": "#e74c3c"},
                 title="Braden Subscores by Pressure Injury Outcome")
    return fig


def plot_los_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of length of stay by outcome."""
    if "los_days" not in df.columns:
        fig = go.Figure()
        fig.add_annotation(text="LOS data not available", xref="paper",
                           yref="paper", x=0.5, y=0.5, showarrow=False)
        return fig

    plot_df = df[["los_days", config.TARGET]].dropna()
    # Cap at 60 days for visualization
    plot_df = plot_df[plot_df["los_days"] <= 60]
    plot_df["Outcome"] = plot_df[config.TARGET].map({0: "No PI", 1: "PI"})

    fig = px.histogram(plot_df, x="los_days", color="Outcome",
                       color_discrete_map={"No PI": "#2ecc71", "PI": "#e74c3c"},
                       barmode="overlay", opacity=0.7, nbins=60,
                       title="Length of Stay Distribution by Outcome",
                       labels={"los_days": "Length of Stay (days)"})
    return fig


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    """Histogram of age by outcome."""
    plot_df = df[["age", config.TARGET]].dropna()
    plot_df["Outcome"] = plot_df[config.TARGET].map({0: "No PI", 1: "PI"})

    fig = px.histogram(plot_df, x="age", color="Outcome",
                       color_discrete_map={"No PI": "#2ecc71", "PI": "#e74c3c"},
                       barmode="overlay", opacity=0.7, nbins=40,
                       title="Age Distribution by Outcome")
    return fig
