"""Page 5: User Guide — Feature definitions, interpretation help, disclaimers."""

import streamlit as st

st.title("User Guide")

st.subheader("What Are Pressure Injuries?")
st.markdown("""
Pressure injuries (also called pressure ulcers or bedsores) are localized damage
to the skin and underlying tissue, usually over a bony prominence, resulting from
sustained pressure or pressure in combination with shear. They are one of the most
common and costly hospital-acquired conditions, affecting approximately 2.5 million
patients annually in the United States.
""")

st.markdown("---")

st.subheader("How the Model Works")
st.markdown("""
This system uses **XGBoost** (eXtreme Gradient Boosting), a machine learning algorithm
that learns patterns from thousands of historical patient records to identify which
combinations of clinical factors are most predictive of pressure injury development.

The model was trained on de-identified data from the **MIMIC-IV** database, which
contains real clinical records from Beth Israel Deaconess Medical Center.

**Key steps:**
1. Patient features (demographics, lab values, Braden scores) are collected
2. The model calculates a probability (0-100%) of pressure injury development
3. SHAP analysis explains which factors contributed most to the prediction
4. The result is displayed as a color-coded risk gauge with clinical context
""")

st.markdown("---")

st.subheader("Interpreting the Risk Score")
st.markdown("""
| Risk Level | Probability | Color | Recommended Action |
|---|---|---|---|
| **Low** | < 15% | Green | Standard prevention protocols |
| **Moderate** | 15% - 40% | Yellow | Enhanced monitoring, consider prevention bundle |
| **High** | > 40% | Red | Immediate prevention bundle, frequent reassessment |
""")

st.markdown("---")

st.subheader("Understanding SHAP Charts")
st.markdown("""
**SHAP (SHapley Additive exPlanations)** values show how each feature contributes
to the prediction:

- **Red/positive values** push the prediction toward higher risk
- **Blue/negative values** push the prediction toward lower risk
- The length of each bar shows how much that feature matters for this patient

For example, if "albumin" has a large red bar, it means the patient's albumin level
is significantly increasing their predicted risk.
""")

st.markdown("---")

st.subheader("Feature Definitions")
st.markdown("""
**Demographics:**
- **Age**: Patient age in years
- **Gender**: Biological sex (M/F)
- **Race**: Self-reported race/ethnicity
- **Insurance**: Insurance type (Medicare, Medicaid, Other)

**Clinical Indicators:**
- **Length of Stay (LOS)**: Days since hospital admission
- **ICU Admission**: Whether the patient was admitted to an intensive care unit

**Laboratory Values:**
- **Albumin** (g/dL): Protein indicating nutritional status. Normal: 3.5-5.0. Low values suggest malnutrition.
- **Hemoglobin** (g/dL): Oxygen-carrying protein. Normal: 12.0-17.5. Low values indicate anemia.
- **Creatinine** (mg/dL): Kidney function marker. Normal: 0.7-1.3.
- **White Blood Cells** (K/uL): Immune marker. Normal: 4.5-11.0. Elevated values suggest infection.
- **Platelet Count** (K/uL): Clotting cells. Normal: 150-400.

**Braden Scale Subscores** (lower = higher risk):
- **Sensory Perception** (1-4): Ability to respond to pressure-related discomfort
- **Moisture** (1-4): Degree of skin exposure to moisture
- **Activity** (1-4): Degree of physical activity
- **Mobility** (1-4): Ability to change and control body position
- **Nutrition** (1-4): Usual food intake pattern
- **Friction/Shear** (1-3): Degree of friction and shear on skin

*Braden total score <= 18 indicates at-risk; <= 12 indicates high risk*
""")

st.markdown("---")

st.subheader("Limitations and Disclaimers")
st.warning("""
**This is a research prototype and is NOT intended for clinical decision-making.**

- The model was trained on data from a single medical center (BIDMC) and may not
  generalize to all patient populations or clinical settings.
- Predictions are probabilistic estimates, not diagnoses.
- Clinical judgment should always take precedence over model predictions.
- The model does not account for all possible risk factors (e.g., specific
  medications, surgical procedures, or skin condition assessments).
- This tool uses de-identified research data and complies with the MIMIC-IV
  Data Use Agreement.
""")

st.caption(
    "Built for DSC-580: Designing and Creating Data Products at Grand Canyon University. "
    "Data source: MIMIC-IV (Johnson et al., 2023)."
)
