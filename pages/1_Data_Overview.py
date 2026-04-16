"""Page 1: Data Overview — Dataset statistics, class distribution, missing data."""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.explore import plot_class_distribution, plot_missing_data

st.title("Data Overview")

display_dir = config.ARTIFACTS_DIR / "display"
has_live_data = config.ANALYTIC_DATASET.exists()
has_cached = display_dir.exists()

if not has_live_data and not has_cached:
    st.error("No data available. Run the pipeline first.")
    st.stop()

# Load summary
if has_cached and (display_dir / "summary.json").exists():
    with open(display_dir / "summary.json") as f:
        summary = json.load(f)
else:
    df = pd.read_parquet(config.ANALYTIC_DATASET)
    summary = {
        "total_admissions": len(df),
        "total_features": df.shape[1] - 1,
        "pi_cases": int(df[config.TARGET].sum()),
        "pi_prevalence": df[config.TARGET].mean(),
    }

st.subheader("Dataset Summary")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Admissions", f"{summary['total_admissions']:,}")
col2.metric("Features", f"{summary['total_features']}")
col3.metric("PI Cases", f"{summary['pi_cases']:,}")
col4.metric("PI Prevalence", f"{summary['pi_prevalence']:.2%}")

st.markdown("---")

# Class distribution
st.subheader("Class Distribution")
if has_cached and (display_dir / "plot_class_dist.json").exists():
    import plotly.io as pio
    fig = pio.read_json(str(display_dir / "plot_class_dist.json"))
    st.plotly_chart(fig, use_container_width=True)
elif has_live_data:
    df = pd.read_parquet(config.ANALYTIC_DATASET)
    st.plotly_chart(plot_class_distribution(df), use_container_width=True)

# Missing data
st.subheader("Missing Data")
if has_cached and (display_dir / "plot_missing.json").exists():
    import plotly.io as pio
    fig = pio.read_json(str(display_dir / "plot_missing.json"))
    st.plotly_chart(fig, use_container_width=True)
elif has_live_data:
    df = pd.read_parquet(config.ANALYTIC_DATASET)
    st.plotly_chart(plot_missing_data(df), use_container_width=True)

# Sample rows
st.subheader("Sample Data")
if has_cached and (display_dir / "sample_rows.json").exists():
    sample = pd.read_json(display_dir / "sample_rows.json")
    st.dataframe(sample, use_container_width=True)
elif has_live_data:
    df = pd.read_parquet(config.ANALYTIC_DATASET)
    st.dataframe(df.head(20), use_container_width=True)
