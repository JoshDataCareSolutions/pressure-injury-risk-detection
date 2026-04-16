"""Page 2: Exploratory Data Analysis — Interactive visualizations."""

import streamlit as st
import pandas as pd
import plotly.io as pio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.explore import (
    plot_feature_distribution, plot_correlation_heatmap,
    plot_prevalence_by_group, plot_braden_boxplots,
    plot_los_distribution, plot_age_distribution,
)

st.title("Exploratory Data Analysis")

display_dir = config.ARTIFACTS_DIR / "display"
has_live = config.ANALYTIC_DATASET.exists()
has_cached = display_dir.exists()

if not has_live and not has_cached:
    st.error("No data available.")
    st.stop()


def load_plot(filename, fallback_fn=None, *args):
    """Load pre-computed plot or generate live."""
    path = display_dir / filename
    if has_cached and path.exists():
        return pio.read_json(str(path))
    elif has_live and fallback_fn:
        df = pd.read_parquet(config.ANALYTIC_DATASET)
        return fallback_fn(df, *args) if args else fallback_fn(df)
    return None


# Feature distributions
st.subheader("Feature Distributions by Outcome")
features = ["age", "los_days", "hemoglobin", "creatinine", "albumin"]
selected = st.selectbox("Select a feature:", features)

fig = load_plot(f"plot_dist_{selected}.json")
if fig:
    st.plotly_chart(fig, use_container_width=True)
elif has_live:
    df = pd.read_parquet(config.ANALYTIC_DATASET)
    st.plotly_chart(plot_feature_distribution(df, selected), use_container_width=True)

st.markdown("---")

# Correlation heatmap
st.subheader("Correlation Heatmap")
fig = load_plot("plot_correlation.json")
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Age and LOS
col1, col2 = st.columns(2)
with col1:
    st.subheader("Age Distribution")
    fig = load_plot("plot_age.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)
with col2:
    st.subheader("Length of Stay")
    fig = load_plot("plot_los.json")
    if fig:
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Braden subscores
st.subheader("Braden Subscores by Outcome")
fig = load_plot("plot_braden.json")
if fig:
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Fairness analysis
st.subheader("Prevalence by Demographic Group")
groups = ["race", "gender", "insurance"]
group = st.selectbox("Group by:", groups)
fig = load_plot(f"plot_prev_{group}.json")
if fig:
    st.plotly_chart(fig, use_container_width=True)
