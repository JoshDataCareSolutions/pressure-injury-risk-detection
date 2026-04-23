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
    "Load from URL",
], horizontal=True)

df = None

if data_source == "MIMIC-IV (processed dataset)":
    if config.ANALYTIC_DATASET.exists():
        df = pd.read_parquet(config.ANALYTIC_DATASET)
        st.success(f"Loaded analytic dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
    else:
        st.warning("Processed dataset not found. Run the pipeline first on the Pipeline page.")
elif data_source == "Upload CSV file":
    uploaded = st.file_uploader("Upload a CSV file:", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
        st.success(f"Uploaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
else:
    url = st.text_input(
        "Dataset URL (CSV or Parquet):",
        placeholder="https://example.com/data.csv",
    )
    if url and st.button("Fetch from URL"):
        try:
            if url.lower().endswith(".parquet"):
                df = pd.read_parquet(url)
            else:
                df = pd.read_csv(url)
            st.session_state["url_df"] = df
            st.success(f"Loaded from URL: {df.shape[0]:,} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Failed to load from URL: {e}")
    elif "url_df" in st.session_state:
        df = st.session_state["url_df"]
        st.info(f"Using previously fetched URL data: {df.shape[0]:,} rows, {df.shape[1]} columns")

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

# --- Interactive Preprocessing ---
st.subheader("Apply Preprocessing Methods")
st.caption("Select one or more methods to apply to the loaded data. A cleaned copy is "
           "shown below; the source data is not modified.")

method_options = [
    "Drop rows with any missing values",
    "Median imputation (numeric columns)",
    "Mean imputation (numeric columns)",
    "Mode imputation (categorical columns)",
    "Clip numeric columns to clinical ranges",
    "Min-max scaling (numeric columns)",
    "Z-score standardization (numeric columns)",
    "One-hot encode categorical columns",
    "Drop duplicate rows",
]
selected_methods = st.multiselect("Preprocessing methods:", method_options)

if selected_methods:
    cleaned = df.copy()
    numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = cleaned.select_dtypes(include=["object", "category"]).columns.tolist()
    log = []

    for method in selected_methods:
        if method == "Drop rows with any missing values":
            before = len(cleaned)
            cleaned = cleaned.dropna()
            log.append(f"Dropped {before - len(cleaned):,} rows with missing values.")
        elif method == "Median imputation (numeric columns)":
            for c in numeric_cols:
                cleaned[c] = cleaned[c].fillna(cleaned[c].median())
            log.append(f"Median-imputed {len(numeric_cols)} numeric columns.")
        elif method == "Mean imputation (numeric columns)":
            for c in numeric_cols:
                cleaned[c] = cleaned[c].fillna(cleaned[c].mean())
            log.append(f"Mean-imputed {len(numeric_cols)} numeric columns.")
        elif method == "Mode imputation (categorical columns)":
            for c in cat_cols:
                mode = cleaned[c].mode()
                if not mode.empty:
                    cleaned[c] = cleaned[c].fillna(mode.iloc[0])
            log.append(f"Mode-imputed {len(cat_cols)} categorical columns.")
        elif method == "Clip numeric columns to clinical ranges":
            clipped = 0
            for col, (lo, hi) in config.CLINICAL_RANGES.items():
                if col in cleaned.columns:
                    cleaned[col] = cleaned[col].clip(lower=lo, upper=hi)
                    clipped += 1
            log.append(f"Clipped {clipped} columns to clinical ranges.")
        elif method == "Min-max scaling (numeric columns)":
            for c in numeric_cols:
                col_min, col_max = cleaned[c].min(), cleaned[c].max()
                if pd.notna(col_min) and pd.notna(col_max) and col_max > col_min:
                    cleaned[c] = (cleaned[c] - col_min) / (col_max - col_min)
            log.append(f"Min-max scaled {len(numeric_cols)} numeric columns.")
        elif method == "Z-score standardization (numeric columns)":
            for c in numeric_cols:
                mu, sd = cleaned[c].mean(), cleaned[c].std()
                if pd.notna(sd) and sd > 0:
                    cleaned[c] = (cleaned[c] - mu) / sd
            log.append(f"Z-score standardized {len(numeric_cols)} numeric columns.")
        elif method == "One-hot encode categorical columns":
            before_cols = cleaned.shape[1]
            cleaned = pd.get_dummies(cleaned, columns=cat_cols, drop_first=False)
            log.append(f"One-hot encoded {len(cat_cols)} columns "
                       f"(shape change: {before_cols} -> {cleaned.shape[1]} columns).")
        elif method == "Drop duplicate rows":
            before = len(cleaned)
            cleaned = cleaned.drop_duplicates()
            log.append(f"Dropped {before - len(cleaned):,} duplicate rows.")

    st.markdown("**Preprocessing log:**")
    for entry in log:
        st.write(f"- {entry}")
    st.markdown(f"**Resulting shape:** {cleaned.shape[0]:,} rows × {cleaned.shape[1]} columns")
    st.dataframe(cleaned.head(20), use_container_width=True)

    csv = cleaned.to_csv(index=False)
    st.download_button("Download Cleaned Dataset (CSV)", csv,
                       "cleaned_dataset.csv", "text/csv")

st.markdown("---")

# --- Preprocessing Reference Documentation ---
st.subheader("Pipeline Preprocessing Reference")

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
