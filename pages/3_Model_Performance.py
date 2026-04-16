"""Page 3: Model Performance — Metrics, curves, and SHAP summary."""

import streamlit as st
import json
import plotly.io as pio
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

st.title("Model Performance")

model_path = config.ARTIFACTS_DIR / "xgb_model.json"
if not model_path.exists():
    st.error("No trained model found. Run `python run_pipeline.py` first.")
    st.stop()

display_dir = config.ARTIFACTS_DIR / "display"

# Metrics
metrics_path = config.ARTIFACTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        metrics = json.load(f)

    st.subheader("Key Metrics")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    col1.metric("AUROC", f"{metrics['auroc']:.3f}",
                delta="PASS" if metrics["auroc"] >= config.AUROC_TARGET else "BELOW")
    col2.metric("Sensitivity", f"{metrics['sensitivity']:.3f}",
                delta="PASS" if metrics["sensitivity"] >= config.SENSITIVITY_TARGET else "BELOW")
    col3.metric("Specificity", f"{metrics['specificity']:.3f}")
    col4.metric("Precision", f"{metrics['precision']:.3f}")
    col5.metric("F1 Score", f"{metrics['f1']:.3f}")
    col6.metric("Brier Score", f"{metrics['brier_score']:.3f}")

    if "threshold" in metrics:
        st.caption(f"Optimized decision threshold: {metrics['threshold']:.2f}")
    st.markdown("---")

# Performance curves from pre-computed data
def show_plot(filename, title=None):
    path = display_dir / filename
    if path.exists():
        fig = pio.read_json(str(path))
        st.plotly_chart(fig, use_container_width=True)
        return True
    return False

st.subheader("Performance Curves")
col1, col2 = st.columns(2)
with col1:
    show_plot("plot_roc.json")
with col2:
    show_plot("plot_pr.json")

col3, col4 = st.columns(2)
with col3:
    show_plot("plot_cm.json")
with col4:
    show_plot("plot_cal.json")

st.markdown("---")

# SHAP from pre-computed images
st.subheader("SHAP Feature Importance")
col1, col2 = st.columns(2)

shap_bar = display_dir / "shap_bar.png"
shap_summary = display_dir / "shap_summary.png"

with col1:
    st.markdown("**Global Feature Importance**")
    if shap_bar.exists():
        st.image(str(shap_bar))
    else:
        st.info("SHAP bar plot not available. Run precompute.py.")

with col2:
    st.markdown("**SHAP Summary (Beeswarm)**")
    if shap_summary.exists():
        st.image(str(shap_summary))
    else:
        st.info("SHAP summary plot not available. Run precompute.py.")
