"""Page 8: Reports — Generate downloadable summary reports."""

import streamlit as st
import pandas as pd
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

st.title("Report Generation")
st.markdown("Generate downloadable reports summarizing the analysis.")

# --- Report Type Selection ---
report_type = st.selectbox("Report type:", [
    "Model Performance Summary",
    "Dataset Overview",
    "Feature Importance Report",
    "Full Pipeline Report",
])

if st.button("Generate Report", type="primary"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("PRESSURE INJURY RISK DETECTION SYSTEM")
    report_lines.append(f"Report Type: {report_type}")
    report_lines.append(f"Generated: {timestamp}")
    report_lines.append("=" * 70)
    report_lines.append("")

    if report_type == "Model Performance Summary":
        metrics_path = config.ARTIFACTS_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)

            report_lines.append("MODEL PERFORMANCE METRICS")
            report_lines.append("-" * 40)
            report_lines.append(f"  AUROC:       {metrics['auroc']:.4f}  "
                              f"{'[PASS]' if metrics['auroc'] >= 0.80 else '[BELOW TARGET]'}")
            report_lines.append(f"  Sensitivity: {metrics['sensitivity']:.4f}  "
                              f"{'[PASS]' if metrics['sensitivity'] >= 0.80 else '[BELOW TARGET]'}")
            report_lines.append(f"  Specificity: {metrics['specificity']:.4f}")
            report_lines.append(f"  Precision:   {metrics['precision']:.4f}")
            report_lines.append(f"  F1 Score:    {metrics['f1']:.4f}")
            report_lines.append(f"  Brier Score: {metrics['brier_score']:.4f}")
            report_lines.append(f"  Threshold:   {metrics.get('threshold', 'N/A')}")
            report_lines.append("")
            report_lines.append("TARGET THRESHOLDS")
            report_lines.append(f"  AUROC target:       >= {config.AUROC_TARGET}")
            report_lines.append(f"  Sensitivity target: >= {config.SENSITIVITY_TARGET}")
        else:
            report_lines.append("No model metrics found. Run the pipeline first.")

    elif report_type == "Dataset Overview":
        if config.ANALYTIC_DATASET.exists():
            df = pd.read_parquet(config.ANALYTIC_DATASET)
            report_lines.append("DATASET SUMMARY")
            report_lines.append("-" * 40)
            report_lines.append(f"  Total admissions:  {len(df):,}")
            report_lines.append(f"  Total features:    {df.shape[1] - 1}")
            report_lines.append(f"  PI cases:          {df[config.TARGET].sum():,}")
            report_lines.append(f"  PI prevalence:     {df[config.TARGET].mean():.2%}")
            report_lines.append(f"  Date range:        MIMIC-IV (2008-2019)")
            report_lines.append("")
            report_lines.append("MISSING DATA")
            report_lines.append("-" * 40)
            for col in df.columns:
                pct = df[col].isna().mean() * 100
                if pct > 0:
                    report_lines.append(f"  {col}: {pct:.1f}%")
            report_lines.append("")
            report_lines.append("COLUMNS")
            report_lines.append("-" * 40)
            for col in df.columns:
                report_lines.append(f"  {col} ({df[col].dtype})")
        else:
            report_lines.append("No dataset found. Run the pipeline first.")

    elif report_type == "Feature Importance Report":
        fn_path = config.ARTIFACTS_DIR / "feature_names.json"
        model_path = config.ARTIFACTS_DIR / "xgb_model.json"
        if fn_path.exists() and model_path.exists():
            from pipeline.model import load_model, load_feature_names
            import numpy as np
            model = load_model()
            feature_names = load_feature_names()
            importances = model.feature_importances_
            sorted_idx = np.argsort(importances)[::-1]

            report_lines.append("FEATURE IMPORTANCE (XGBoost gain)")
            report_lines.append("-" * 40)
            report_lines.append(f"{'Rank':<6}{'Feature':<40}{'Importance':<12}")
            report_lines.append("-" * 58)
            for rank, i in enumerate(sorted_idx, 1):
                report_lines.append(f"{rank:<6}{feature_names[i]:<40}{importances[i]:<12.4f}")
        else:
            report_lines.append("No model found. Run the pipeline first.")

    elif report_type == "Full Pipeline Report":
        report_lines.append("PIPELINE CONFIGURATION")
        report_lines.append("-" * 40)
        report_lines.append(f"  Data source:    MIMIC-IV v3.1")
        report_lines.append(f"  Algorithm:      XGBoost (gradient boosted trees)")
        report_lines.append(f"  Test split:     {config.TEST_SIZE:.0%}")
        report_lines.append(f"  Random state:   {config.RANDOM_STATE}")
        report_lines.append(f"  AUROC target:   >= {config.AUROC_TARGET}")
        report_lines.append(f"  Sens. target:   >= {config.SENSITIVITY_TARGET}")
        report_lines.append("")

        if config.ANALYTIC_DATASET.exists():
            df = pd.read_parquet(config.ANALYTIC_DATASET)
            report_lines.append("DATASET")
            report_lines.append("-" * 40)
            report_lines.append(f"  Admissions:    {len(df):,}")
            report_lines.append(f"  PI cases:      {df[config.TARGET].sum():,}")
            report_lines.append(f"  Prevalence:    {df[config.TARGET].mean():.2%}")

        metrics_path = config.ARTIFACTS_DIR / "metrics.json"
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            report_lines.append("")
            report_lines.append("RESULTS")
            report_lines.append("-" * 40)
            for k, v in metrics.items():
                report_lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    report_text = "\n".join(report_lines)
    st.text(report_text)

    st.download_button(
        "Download Report",
        report_text,
        f"pi_risk_report_{report_type.lower().replace(' ', '_')}.txt",
        "text/plain",
    )
