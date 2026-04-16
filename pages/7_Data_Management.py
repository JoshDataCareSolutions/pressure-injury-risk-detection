"""Page 7: Data Management — Load data, display in various formats, clean/preprocess."""

import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

st.title("Data Management")

# --- Load Data ---
st.subheader("Load Data")

data_source = st.radio("Data source:", [
    "MIMIC-IV (processed dataset)",
    "Upload CSV file",
], horizontal=True)

df = None

if data_source == "MIMIC-IV (processed dataset)":
    if config.ANALYTIC_DATASET.exists():
        df = pd.read_parquet(config.ANALYTIC_DATASET)
        st.success(f"Loaded analytic dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        st.warning("Processed dataset not found. Run the pipeline first on the Pipeline page.")
else:
    uploaded = st.file_uploader("Upload a CSV file:", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Uploaded: {df.shape[0]:,} rows, {df.shape[1]} columns")

if df is None:
    st.stop()

st.markdown("---")

# --- Display Data in Various Formats ---
st.subheader("Display Data")

display_format = st.selectbox("Display format:", [
    "Data Table", "Summary Statistics", "Column Types",
    "Value Counts", "Head/Tail",
])

if display_format == "Data Table":
    n_rows = st.slider("Rows to display:", 5, 100, 20)
    st.dataframe(df.head(n_rows), use_container_width=True)

elif display_format == "Summary Statistics":
    st.dataframe(df.describe(include="all").T, use_container_width=True)

elif display_format == "Column Types":
    type_df = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null": df.notna().sum().values,
        "Null %": (df.isna().mean() * 100).round(1).values,
        "Unique": df.nunique().values,
    })
    st.dataframe(type_df, use_container_width=True)

elif display_format == "Value Counts":
    col = st.selectbox("Select column:", df.columns)
    vc = df[col].value_counts().reset_index()
    vc.columns = ["Value", "Count"]
    st.dataframe(vc, use_container_width=True)

elif display_format == "Head/Tail":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**First 5 rows**")
        st.dataframe(df.head(), use_container_width=True)
    with col2:
        st.markdown("**Last 5 rows**")
        st.dataframe(df.tail(), use_container_width=True)

st.markdown("---")

# --- Data Cleaning & Preprocessing ---
st.subheader("Data Cleaning & Preprocessing")

with st.expander("Clinical Range Validation"):
    st.markdown("Values outside clinically plausible ranges are set to NaN:")
    range_df = pd.DataFrame([
        {"Feature": k, "Min": v[0], "Max": v[1]}
        for k, v in config.CLINICAL_RANGES.items()
    ])
    st.dataframe(range_df, use_container_width=True)

    if st.checkbox("Show out-of-range counts"):
        for col, (low, high) in config.CLINICAL_RANGES.items():
            if col in df.columns:
                n_oor = ((df[col] < low) | (df[col] > high)).sum()
                if n_oor > 0:
                    st.write(f"  {col}: {n_oor} values outside [{low}, {high}]")

with st.expander("Missing Data Analysis"):
    missing = df.isnull().mean() * 100
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        st.bar_chart(missing)
    else:
        st.info("No missing data found.")

    st.markdown("""
    **Imputation strategy:**
    - Continuous features (< 50% missing): Median imputation + StandardScaler
    - Continuous features (> 50% missing): Passed through raw (XGBoost handles NaN natively)
    - Categorical features: Mode imputation + One-Hot Encoding
    - Missingness indicators added as binary features
    """)

with st.expander("Feature Engineering"):
    st.markdown("""
    **Derived features added during preprocessing:**
    | Feature | Logic | Rationale |
    |---|---|---|
    | `age_over_75` | age >= 75 | Non-linear age-risk relationship |
    | `los_over_7` | LOS > 7 days | Extended stay risk threshold |
    | `los_over_14` | LOS > 14 days | Prolonged stay risk threshold |
    | `albumin_low` | albumin < 3.0 g/dL | Clinical malnutrition flag |
    | `hemoglobin_low` | hemoglobin < 10.0 g/dL | Clinical anemia flag |
    | `braden_total` | Sum of 6 Braden subscores | Composite risk score |
    | `*_missing` | NaN indicator per feature | Missingness is informative |
    """)

with st.expander("Class Balance"):
    if config.TARGET in df.columns:
        counts = df[config.TARGET].value_counts()
        col1, col2 = st.columns(2)
        col1.metric("No PI", f"{counts.get(0, 0):,}")
        col2.metric("PI", f"{counts.get(1, 0):,}")
        st.markdown(f"**Prevalence:** {df[config.TARGET].mean():.2%}")
        st.markdown("""
        **Balancing strategy:** `scale_pos_weight` parameter in XGBoost
        (ratio of negative to positive samples) rather than SMOTE,
        which enables the model to handle NaN values natively.
        """)

st.markdown("---")

# --- Download / Export ---
st.subheader("Export Data")

export_format = st.selectbox("Export format:", ["CSV", "Parquet"])

if st.button("Download Dataset"):
    if export_format == "CSV":
        csv = df.to_csv(index=False)
        st.download_button("Download CSV", csv, "analytic_dataset.csv", "text/csv")
    else:
        parquet = df.to_parquet(index=False)
        st.download_button("Download Parquet", parquet, "analytic_dataset.parquet",
                          "application/octet-stream")
