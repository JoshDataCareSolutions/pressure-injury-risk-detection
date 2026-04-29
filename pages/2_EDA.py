"""Page 2: Exploratory Data Analysis — Interactive visualizations."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
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

st.markdown("---")

# --- Choose Analysis Method ---
st.subheader("Choose Analysis Method")
st.caption("Select one or more on-demand analyses to run against the live dataset.")

analysis_methods = st.multiselect(
    "Analysis methods:",
    [
        "Descriptive Statistics",
        "Pearson Correlation Matrix",
        "Group Comparison by Outcome (t-test summary)",
        "Outlier Detection (IQR method)",
        "Missingness Profile",
        "Class Prevalence Summary",
    ],
)

if analysis_methods and has_live:
    df_live = pd.read_parquet(config.ANALYTIC_DATASET)

    if "Descriptive Statistics" in analysis_methods:
        st.markdown("**Descriptive Statistics**")
        st.dataframe(df_live.describe().T, use_container_width=True)

    if "Pearson Correlation Matrix" in analysis_methods:
        st.markdown("**Pearson Correlation (numeric features only)**")
        num_df = df_live.select_dtypes(include=[np.number])
        corr = num_df.corr(method="pearson")
        st.plotly_chart(
            px.imshow(corr, color_continuous_scale="RdBu_r", aspect="auto",
                      title="Pearson Correlation"),
            use_container_width=True,
        )

    if "Group Comparison by Outcome (t-test summary)" in analysis_methods:
        st.markdown("**Mean comparison: PI vs no-PI cohorts**")
        if config.TARGET in df_live.columns:
            num_cols = df_live.select_dtypes(include=[np.number]).columns
            num_cols = [c for c in num_cols if c != config.TARGET]
            rows = []
            for c in num_cols:
                grp = df_live.groupby(config.TARGET)[c]
                rows.append({
                    "feature": c,
                    "mean_no_pi": grp.mean().get(0, np.nan),
                    "mean_pi": grp.mean().get(1, np.nan),
                    "delta": grp.mean().get(1, np.nan) - grp.mean().get(0, np.nan),
                })
            comp = pd.DataFrame(rows).sort_values("delta", key=abs, ascending=False)
            st.dataframe(comp, use_container_width=True)

    if "Outlier Detection (IQR method)" in analysis_methods:
        st.markdown("**Outlier counts (values outside 1.5 × IQR)**")
        num_cols = df_live.select_dtypes(include=[np.number]).columns
        rows = []
        for c in num_cols:
            q1, q3 = df_live[c].quantile(0.25), df_live[c].quantile(0.75)
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out = ((df_live[c] < low) | (df_live[c] > high)).sum()
            rows.append({"feature": c, "lower_bound": low, "upper_bound": high,
                         "outlier_count": int(n_out)})
        st.dataframe(pd.DataFrame(rows).sort_values("outlier_count", ascending=False),
                     use_container_width=True)

    if "Missingness Profile" in analysis_methods:
        st.markdown("**Missing value percentages**")
        miss = (df_live.isna().mean() * 100).sort_values(ascending=False)
        miss = miss[miss > 0]
        st.bar_chart(miss)

    if "Class Prevalence Summary" in analysis_methods:
        st.markdown("**Outcome class prevalence**")
        if config.TARGET in df_live.columns:
            counts = df_live[config.TARGET].value_counts().rename(
                {0: "No PI", 1: "PI"})
            st.dataframe(counts, use_container_width=True)
            st.write(f"Prevalence: {df_live[config.TARGET].mean():.2%}")
