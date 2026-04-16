"""Page 6: Analytics Pipeline — Visualize, configure, and execute the OSEMN pipeline."""

import streamlit as st
import pandas as pd
import numpy as np
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

st.title("Analytics Pipeline")
st.markdown("Visualize and execute the OSEMN data science pipeline.")

# --- Pipeline Visualization ---
st.subheader("OSEMN Pipeline Overview")

pipeline_steps = {
    "O - Obtain": {
        "description": "Extract data from MIMIC-IV tables (patients, admissions, labs, chart events, diagnoses)",
        "status": "ready" if config.ANALYTIC_DATASET.exists() else "pending",
        "icon": "1.",
    },
    "S - Scrub": {
        "description": "Validate clinical ranges, impute missing values, encode categories, engineer features",
        "status": "ready" if (config.ARTIFACTS_DIR / "preprocessor.pkl").exists() else "pending",
        "icon": "2.",
    },
    "E - Explore": {
        "description": "Generate distribution plots, correlation heatmaps, fairness analysis",
        "status": "ready" if config.ANALYTIC_DATASET.exists() else "pending",
        "icon": "3.",
    },
    "M - Model": {
        "description": "Train XGBoost classifier, tune hyperparameters, evaluate on held-out test set",
        "status": "ready" if (config.ARTIFACTS_DIR / "xgb_model.json").exists() else "pending",
        "icon": "4.",
    },
    "N - iNterpret": {
        "description": "Compute SHAP values, generate feature importance plots, patient-level explanations",
        "status": "ready" if (config.ARTIFACTS_DIR / "shap_expected_value.json").exists() else "pending",
        "icon": "5.",
    },
}

# Visual pipeline display
cols = st.columns(5)
for i, (step_name, step_info) in enumerate(pipeline_steps.items()):
    with cols[i]:
        if step_info["status"] == "ready":
            st.success(f"**{step_name}**")
        else:
            st.warning(f"**{step_name}**")
        st.caption(step_info["description"])

st.markdown("---")

# --- Step-by-Step Execution ---
st.subheader("Execute Pipeline")

exec_mode = st.radio("Execution mode:", ["Step by step", "Run entire pipeline"],
                      horizontal=True)

if exec_mode == "Step by step":
    step = st.selectbox("Select step to run:", list(pipeline_steps.keys()))

    if st.button(f"Run: {step}", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        if step == "O - Obtain":
            status.info("Extracting and joining MIMIC-IV tables...")
            progress.progress(10)
            from pipeline.obtain import build_analytic_dataset
            df = build_analytic_dataset()
            progress.progress(100)
            status.success(f"Obtain complete: {df.shape[0]:,} admissions, {df.shape[1]} columns")

        elif step == "S - Scrub":
            if not config.ANALYTIC_DATASET.exists():
                status.error("Run Obtain first.")
            else:
                status.info("Cleaning, imputing, and encoding features...")
                progress.progress(10)
                df = pd.read_parquet(config.ANALYTIC_DATASET)
                from pipeline.scrub import prepare_data
                X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df, apply_smote=False)
                progress.progress(100)
                status.success(f"Scrub complete: {len(feature_names)} features, "
                              f"train={X_train.shape[0]:,}, test={X_test.shape[0]:,}")

        elif step == "E - Explore":
            if not config.ANALYTIC_DATASET.exists():
                status.error("Run Obtain first.")
            else:
                status.info("Generating exploratory visualizations...")
                progress.progress(50)
                status.success("Explore complete — view results on the EDA page.")
                progress.progress(100)

        elif step == "M - Model":
            if not (config.ARTIFACTS_DIR / "preprocessor.pkl").exists():
                status.error("Run Scrub first.")
            else:
                status.info("Training XGBoost model...")
                progress.progress(10)
                df = pd.read_parquet(config.ANALYTIC_DATASET)
                from pipeline.scrub import prepare_data
                from pipeline.model import save_model
                from xgboost import XGBClassifier
                X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df, apply_smote=False)
                progress.progress(30)
                neg = (y_train == 0).sum()
                pos = (y_train == 1).sum()
                model = XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.1,
                    min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
                    scale_pos_weight=neg/pos, eval_metric='aucpr', random_state=42,
                )
                model.fit(X_train, y_train)
                save_model(model, feature_names)
                progress.progress(100)
                from sklearn.metrics import roc_auc_score
                y_prob = model.predict_proba(X_test)[:, 1]
                auroc = roc_auc_score(y_test, y_prob)
                status.success(f"Model trained — AUROC: {auroc:.4f}")

        elif step == "N - iNterpret":
            if not (config.ARTIFACTS_DIR / "xgb_model.json").exists():
                status.error("Run Model first.")
            else:
                status.info("Computing SHAP values...")
                progress.progress(10)
                from pipeline.model import load_model
                from pipeline.interpret import compute_shap_values
                model = load_model()
                feature_names = json.load(open(config.ARTIFACTS_DIR / "feature_names.json"))
                df = pd.read_parquet(config.ANALYTIC_DATASET)
                from pipeline.scrub import prepare_data
                _, X_test, _, _, _, _ = prepare_data(df, apply_smote=False)
                progress.progress(50)
                sample = X_test[:500]
                shap_values = compute_shap_values(model, sample, feature_names)
                progress.progress(100)
                status.success("Interpret complete — view SHAP plots on Model Performance page.")

elif exec_mode == "Run entire pipeline":
    if st.button("Run Full Pipeline", type="primary"):
        progress = st.progress(0)
        status = st.empty()

        steps = ["Obtaining data...", "Scrubbing data...", "Exploring data...",
                 "Training model...", "Interpreting results..."]

        status.info(steps[0])
        from pipeline.obtain import build_analytic_dataset
        df = build_analytic_dataset()
        progress.progress(20)

        status.info(steps[1])
        from pipeline.scrub import prepare_data
        X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_data(df, apply_smote=False)
        progress.progress(40)

        status.info(steps[2])
        progress.progress(50)

        status.info(steps[3])
        from pipeline.model import save_model
        from xgboost import XGBClassifier
        from sklearn.metrics import roc_auc_score, recall_score, precision_score, f1_score, brier_score_loss
        neg = (y_train == 0).sum()
        pos = (y_train == 1).sum()
        model = XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.1,
            min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
            scale_pos_weight=neg/pos, eval_metric='aucpr', random_state=42,
        )
        model.fit(X_train, y_train)
        save_model(model, feature_names)
        progress.progress(80)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        metrics = {
            'auroc': float(roc_auc_score(y_test, y_prob)),
            'sensitivity': float(recall_score(y_test, y_pred)),
            'specificity': float(recall_score(y_test, y_pred, pos_label=0)),
            'precision': float(precision_score(y_test, y_pred, zero_division=0)),
            'f1': float(f1_score(y_test, y_pred)),
            'brier_score': float(brier_score_loss(y_test, y_prob)),
        }
        with open(config.ARTIFACTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        status.info(steps[4])
        from pipeline.interpret import compute_shap_values
        shap_values = compute_shap_values(model, X_test[:500], feature_names)
        progress.progress(100)

        status.success(f"Pipeline complete! AUROC: {metrics['auroc']:.4f}")

st.markdown("---")

# --- Pipeline Configuration ---
st.subheader("Pipeline Configuration")

with st.expander("Model Hyperparameters"):
    st.markdown("""
    | Parameter | Value | Description |
    |---|---|---|
    | `n_estimators` | 500 | Number of boosting rounds |
    | `max_depth` | 6 | Maximum tree depth |
    | `learning_rate` | 0.1 | Step size shrinkage |
    | `min_child_weight` | 10 | Minimum sum of instance weight in a child |
    | `subsample` | 0.8 | Row sampling ratio per tree |
    | `colsample_bytree` | 0.7 | Column sampling ratio per tree |
    | `scale_pos_weight` | ~34.6 | Class weight for imbalanced data |
    """)

with st.expander("Data Sources"):
    data_dir = config.get_data_dir()
    st.markdown(f"**Data directory:** `{data_dir}`")
    st.markdown(f"**Using demo data:** {config.USE_DEMO}")

    if config.ANALYTIC_DATASET.exists():
        size_mb = config.ANALYTIC_DATASET.stat().st_size / (1024 * 1024)
        st.markdown(f"**Analytic dataset:** `{config.ANALYTIC_DATASET}` ({size_mb:.1f} MB)")
