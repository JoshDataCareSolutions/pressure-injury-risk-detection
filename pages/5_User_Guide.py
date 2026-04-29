"""Page 5: User Guide — Feature definitions, interpretation help, disclaimers."""

import streamlit as st

st.title("User Guide")

st.subheader("How to Use This Application")
st.markdown("""
Use the sidebar on the left to navigate between pages. Each page is described below.

| Page | Purpose | Key Controls |
|---|---|---|
| **Home** | Overview and navigation index | Sidebar links |
| **Data Overview** | High-level dataset stats and class distribution | Static charts |
| **EDA** | Feature distributions, correlations, fairness, and on-demand analysis methods | Feature dropdown, demographic-group dropdown, analysis-method multiselect |
| **Model Performance** | AUROC, sensitivity, specificity, ROC/PR curves, calibration, SHAP summary | Read-only metrics; downloadable charts via Plotly toolbar |
| **Risk Assessment** | Enter a single patient's data and receive a probability + SHAP explanation | Numeric inputs, sliders, "Predict Risk" button |
| **User Guide** *(this page)* | Help and how-to instructions | Reference content |
| **Pipeline** | Visualize OSEMN phases and run them step-by-step or end-to-end | Radio mode selector, "Run" buttons |
| **Data Management** | Load data from file/URL, choose preprocessing methods, export | Radio source, multiselect for cleaning methods |
| **Reports** | Generate and download text reports | Report-type dropdown, "Generate" + "Download" buttons |
| **Tests** | Run the validation test suite and view pass/fail outcomes | "Run Tests" button |
| **References** | Citations and technology stack | Read-only |
""")

st.markdown("---")

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

st.markdown("---")

st.subheader("Security and Privacy")
st.markdown("""
**Data anonymization.** This application uses only the publicly released MIMIC-IV
v3.1 dataset, which has been de-identified per HIPAA Safe Harbor before release by
the MIT Laboratory for Computational Physiology. No protected health information
(PHI) is processed by this prototype.

**Data access.** Access to the underlying MIMIC-IV files requires PhysioNet
credentialing, completion of CITI human-subjects training, and a signed Data Use
Agreement. The repository excludes raw data via .gitignore.

**Transport security.** When deployed to Streamlit Community Cloud, all traffic
is served over HTTPS.

**Authorized access.** For a production deployment, role-based access control via
enterprise single sign-on would separate clinicians, informatics staff, and
administrators. Federated identity (e.g., SAML/OIDC) is the recommended pattern
for multi-institution deployment.

**Backup and recovery.** Trained model artifacts and the analytic dataset are
versioned in Git/Git LFS and re-buildable from the OSEMN pipeline.

**Incident response.** If a security event is detected, the response is to
rotate credentials, take the application offline, audit logs, notify affected
users, and publish a postmortem before redeployment (NIST SP 800-61 Rev. 2).

**Ethical use.** This is a research prototype and must not be used to make
clinical decisions. Outputs should support, not replace, professional nursing
judgment.
""")

st.caption(
    "Built for DSC-580: Designing and Creating Data Products at Grand Canyon University. "
    "Data source: MIMIC-IV (Johnson et al., 2023)."
)
