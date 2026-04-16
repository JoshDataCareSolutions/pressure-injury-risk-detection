import streamlit as st

st.set_page_config(page_title="PI Risk Detector", page_icon=":hospital:", layout="wide")

st.title("Pressure Injury Risk Detection System")
st.markdown("A clinical decision-support tool powered by XGBoost and SHAP, "
            "built on MIMIC-IV electronic health record data.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("About This Tool")
    st.markdown("""
    This system uses machine learning to predict which hospitalized patients
    are at elevated risk for developing pressure injuries (bedsores).

    **Key capabilities:**
    - XGBoost classification trained on 546K+ hospital admissions
    - SHAP-based explainability for every prediction
    - Interactive risk assessment with clinical input validation
    - Demographic fairness analysis across patient subgroups
    """)

with col2:
    st.subheader("Navigation")
    st.markdown("""
    Use the sidebar to navigate:

    1. **Data Overview** — Dataset statistics and class distribution
    2. **EDA** — Exploratory visualizations and feature analysis
    3. **Model Performance** — Metrics, ROC/PR curves, SHAP summary
    4. **Risk Assessment** — Enter patient data and get a risk score
    5. **User Guide** — Interpretation help and feature definitions
    6. **Pipeline** — Visualize and execute the OSEMN pipeline
    7. **Data Management** — Load, display, clean, and export data
    8. **Reports** — Generate downloadable analysis reports
    9. **References** — Data sources, citations, and acknowledgments
    """)

st.markdown("---")
st.caption("Research prototype using de-identified MIMIC-IV data. "
           "Not intended for clinical decision-making.")
