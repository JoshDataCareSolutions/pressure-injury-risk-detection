"""Page 4: Risk Assessment — Enter patient data, get a risk prediction with SHAP."""

import streamlit as st
import numpy as np
import pandas as pd
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from pipeline.model import load_model, load_feature_names
from pipeline.interpret import plot_risk_gauge, compute_shap_values, get_top_risk_factors, plot_shap_waterfall

st.title("Patient Risk Assessment")
st.markdown("Enter patient clinical data below to generate a pressure injury risk score.")

model_path = config.ARTIFACTS_DIR / "xgb_model.json"
pconfig_path = config.ARTIFACTS_DIR / "preprocessor_config.json"

if not model_path.exists() or not pconfig_path.exists():
    st.error("Model artifacts not found. Run the training pipeline first.")
    st.stop()

model = load_model()
feature_names = load_feature_names()

with open(pconfig_path) as f:
    pcfg = json.load(f)

metrics_path = config.ARTIFACTS_DIR / "metrics.json"
if metrics_path.exists():
    with open(metrics_path) as f:
        opt_threshold = json.load(f).get("threshold", 0.5)
else:
    opt_threshold = 0.5

has_braden = any("braden" in f for f in feature_names)

# --- Input Form ---
st.subheader("Patient Information")

if has_braden:
    col1, col2, col3 = st.columns(3)
else:
    col1, col2 = st.columns(2)

with col1:
    st.markdown("**Demographics**")
    age = st.number_input("Age", min_value=18, max_value=120, value=65)
    gender = st.selectbox("Gender", ["M", "F"])
    race = st.selectbox("Race", ["WHITE", "BLACK/AFRICAN AMERICAN",
                                  "HISPANIC/LATINO", "ASIAN", "Other"])
    insurance = st.selectbox("Insurance", ["Medicare", "Medicaid", "Other"])

with col2:
    st.markdown("**Clinical Indicators**")
    los_days = st.number_input("Length of Stay (days)", min_value=0.0,
                                max_value=365.0, value=5.0, step=0.5)
    icu_flag = st.checkbox("ICU Admission", value=False)

    st.markdown("**Laboratory Values**")
    albumin = st.number_input("Albumin (g/dL)", min_value=0.5, max_value=7.0,
                               value=3.5, step=0.1)
    hemoglobin = st.number_input("Hemoglobin (g/dL)", min_value=2.0,
                                  max_value=25.0, value=12.0, step=0.5)
    creatinine = st.number_input("Creatinine (mg/dL)", min_value=0.1,
                                  max_value=30.0, value=1.0, step=0.1)
    wbc = st.number_input("White Blood Cells (K/uL)", min_value=0.1,
                           max_value=500.0, value=8.0, step=0.5)
    platelets = st.number_input("Platelet Count (K/uL)", min_value=1.0,
                                 max_value=2000.0, value=200.0, step=10.0)

if has_braden:
    with col3:
        st.markdown("**Braden Scale Subscores**")
        st.caption("Lower scores = higher risk. 1 is worst.")
        braden_sensory = st.slider("Sensory Perception", 1, 4, 3)
        braden_moisture = st.slider("Moisture", 1, 4, 3)
        braden_activity = st.slider("Activity", 1, 4, 3)
        braden_mobility = st.slider("Mobility", 1, 4, 3)
        braden_nutrition = st.slider("Nutrition", 1, 4, 3)
        braden_friction = st.slider("Friction/Shear", 1, 3, 2)

st.markdown("---")

if st.button("Predict Risk", type="primary", use_container_width=True):
    # Warnings
    warnings = []
    if albumin < 2.5:
        warnings.append("Albumin is critically low (< 2.5 g/dL)")
    if hemoglobin < 7.0:
        warnings.append("Hemoglobin is critically low (< 7.0 g/dL)")
    if has_braden:
        braden_total = (braden_sensory + braden_moisture + braden_activity +
                        braden_mobility + braden_nutrition + braden_friction)
        if braden_total <= 12:
            warnings.append(f"Braden total score is {braden_total} (high risk per Braden Scale)")
    for w in warnings:
        st.warning(w)

    try:
        # Build feature vector manually (no pickle dependency)
        num_vals = {
            "age": age, "los_days": los_days, "hemoglobin": hemoglobin,
            "creatinine": creatinine, "white_blood_cells": wbc,
            "platelet_count": platelets,
        }

        # Impute (use training medians) and scale
        num_imputed = []
        for i, feat in enumerate(pcfg["num_features"]):
            val = num_vals.get(feat, pcfg["imputer_fill"][i])
            if val is None or np.isnan(val):
                val = pcfg["imputer_fill"][i]
            scaled = (val - pcfg["scaler_mean"][i]) / pcfg["scaler_scale"][i]
            num_imputed.append(scaled)

        # One-hot encode categoricals
        cat_vals = {"gender": gender, "race": race, "insurance": insurance}
        cat_encoded = []
        for i, feat in enumerate(pcfg["cat_features"]):
            val = cat_vals.get(feat, "")
            cats = pcfg["ohe_categories"][i]
            for cat in cats:
                cat_encoded.append(1.0 if val == cat else 0.0)

        # Passthrough numeric (Braden scores — NaN if not provided)
        passthru = []
        if has_braden:
            braden_vals = {
                "braden_sensory_perception": braden_sensory,
                "braden_moisture": braden_moisture,
                "braden_activity": braden_activity,
                "braden_mobility": braden_mobility,
                "braden_nutrition": braden_nutrition,
                "braden_friction_shear": braden_friction,
                "braden_total": braden_total,
                "albumin": albumin,
            }
            for feat in pcfg.get("passthru_features", []):
                passthru.append(float(braden_vals.get(feat, albumin if feat == "albumin" else np.nan)))
        else:
            for feat in pcfg.get("passthru_features", []):
                if feat == "albumin":
                    passthru.append(float(albumin))
                else:
                    passthru.append(np.nan)

        # Binary features
        binary_vals = {
            "icu_flag": int(icu_flag),
            "age_over_75": int(age >= 75),
            "los_over_7": int(los_days > 7),
            "los_over_14": int(los_days > 14),
            "albumin_low": int(albumin < 3.0),
            "hemoglobin_low": int(hemoglobin < 10.0),
            "albumin_missing": 0, "hemoglobin_missing": 0,
            "creatinine_missing": 0, "white_blood_cells_missing": 0,
            "platelet_count_missing": 0,
        }
        if has_braden:
            for feat in pcfg.get("passthru_features", []):
                binary_vals[f"{feat}_missing"] = 0

        binary = [float(binary_vals.get(f, 0)) for f in pcfg.get("binary_features", [])]

        # Combine into feature vector
        X_input = np.array([num_imputed + cat_encoded + passthru + binary])

        if X_input.shape[1] != len(feature_names):
            st.error(f"Feature mismatch: expected {len(feature_names)}, got {X_input.shape[1]}")
            st.stop()

        probability = model.predict_proba(X_input)[0, 1]

        # Display results
        st.subheader("Risk Assessment Result")
        col1, col2 = st.columns([1, 1])

        with col1:
            st.plotly_chart(plot_risk_gauge(probability), use_container_width=True)

        with col2:
            st.markdown("**What's Driving This Risk Score?**")
            shap_values = compute_shap_values(model, X_input, feature_names)
            factors = get_top_risk_factors(shap_values, 0, feature_names, n=5)
            for f in factors:
                icon = "+" if f["direction"] == "increases risk" else "-"
                color = "red" if f["direction"] == "increases risk" else "green"
                st.markdown(f":{color}[{icon}] **{f['feature']}** — {f['direction']} "
                           f"(SHAP: {f['shap_value']:+.3f})")

        st.subheader("SHAP Waterfall Chart")
        st.caption("Shows how each feature contributes to the prediction.")
        fig = plot_shap_waterfall(shap_values, 0)
        st.pyplot(fig)

        st.subheader("Suggested Interventions")
        suggestions = []
        if albumin < 3.0:
            suggestions.append("Consider nutrition consultation — low albumin indicates malnutrition risk")
        if has_braden and braden_mobility <= 2:
            suggestions.append("Increase repositioning frequency — patient has limited mobility")
        if has_braden and braden_moisture <= 2:
            suggestions.append("Implement moisture management protocol")
        if los_days > 7:
            suggestions.append("Extended stay increases PI risk — ensure prevention bundle is active")
        if icu_flag:
            suggestions.append("ICU patients have elevated PI risk — verify pressure redistribution surface")
        if not suggestions:
            suggestions.append("Continue standard prevention protocols")
        for s in suggestions:
            st.info(s)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
