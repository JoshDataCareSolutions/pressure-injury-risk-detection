"""Page 9: References — Data sources, methodology, and academic citations."""

import streamlit as st

st.title("References & Acknowledgments")

st.subheader("Data Source")
st.markdown("""
This system was developed using the **MIMIC-IV** (Medical Information Mart for
Intensive Care IV) clinical database, version 3.1.

> Johnson, A. E. W., Bulgarelli, L., Shen, L., Gayles, A., Shammout, A., Horng, S.,
> Pollard, T. J., Hao, S., Moody, B., Gow, B., Lehman, L. H., Celi, L. A., & Mark, R. G.
> (2023). MIMIC-IV, a freely accessible electronic health record dataset.
> *Scientific Data, 10*(1), 1. https://doi.org/10.1038/s41597-022-01899-x

MIMIC-IV contains de-identified electronic health records from over 300,000 hospital
admissions at Beth Israel Deaconess Medical Center (2008–2019). Access requires
completion of CITI human subjects training and a PhysioNet Data Use Agreement.

- **PhysioNet:** https://physionet.org/content/mimiciv/3.1/
""")

st.markdown("---")

st.subheader("Clinical Background")
st.markdown("""
- **Pressure injury prevalence and cost:**
  Padula, W. V., & Delarmente, B. A. (2019). The national cost of hospital-acquired
  pressure injuries in the United States. *International Wound Journal, 16*(3), 634–640.
  https://doi.org/10.1111/iwj.13071

- **Braden Scale validation:**
  Bergstrom, N., Braden, B. J., Laguzza, A., & Holman, V. (1987). The Braden Scale for
  predicting pressure sore risk. *Nursing Research, 36*(4), 205–210.
  https://doi.org/10.1097/00006199-198707000-00002

- **Braden Scale limitations:**
  Pancorbo-Hidalgo, P. L., Garcia-Fernandez, F. P., Lopez-Medina, I. M., & Alvarez-Nieto, C.
  (2006). Risk assessment scales for pressure ulcer prevention: A systematic review.
  *Journal of Advanced Nursing, 54*(1), 94–110.
  https://doi.org/10.1111/j.1365-2648.2006.03794.x

- **Inpatient prevalence data:**
  Bauer, K., Rock, K., Nazzal, M., Jones, O., & Qu, W. (2016). Pressure ulcers in
  the United States' inpatient population from 2008 to 2012: Results of a retrospective
  nationwide study. *Ostomy/Wound Management, 62*(11), 30–38.
  https://pubmed.ncbi.nlm.nih.gov/27861135/
""")

st.markdown("---")

st.subheader("Machine Learning Methodology")
st.markdown("""
- **XGBoost for pressure injury prediction:**
  Song, W., Kang, M.-J., Zhang, L., Jung, W., Song, J., Bates, D. W., & Dykes, P. C.
  (2021). Predicting pressure injury using nursing assessment phenotypes and machine
  learning methods. *Journal of the American Medical Informatics Association, 28*(4),
  759–765. https://doi.org/10.1093/jamia/ocaa336

- **Bayesian network models for PI prediction:**
  Kaewprag, P., Newton, C., Vermillion, B., Hyun, S., Huang, K., & Machiraju, R.
  (2017). Predictive models for pressure ulcers from intensive care unit electronic
  health records using Bayesian networks. *BMC Medical Informatics and Decision Making,
  17*(Suppl 2), 65. https://doi.org/10.1186/s12911-017-0471-z

- **SHAP explainability:**
  Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model
  predictions. *Advances in Neural Information Processing Systems, 30*, 4765–4774.
  https://arxiv.org/abs/1705.07874

- **XGBoost algorithm:**
  Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system.
  *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery
  and Data Mining*, 785–794. https://doi.org/10.1145/2939672.2939785

- **Missing data imputation:**
  Sterne, J. A. C., White, I. R., Carlin, J. B., Spratt, M., Royston, P., Kenward, M. G.,
  Wood, A. M., & Carpenter, J. R. (2009). Multiple imputation for missing data in
  epidemiological and clinical research: Potential and pitfalls. *BMJ, 338*, b2393.
  https://doi.org/10.1136/bmj.b2393

- **Prediction model evaluation:**
  Steyerberg, E. W., Vickers, A. J., Cook, N. R., Gerds, T., Gonen, M., Obuchowski, N.,
  Pencina, M. J., & Kattan, M. W. (2010). Assessing the performance of prediction models:
  A framework for traditional and novel measures. *Epidemiology, 21*(1), 128–138.
  https://doi.org/10.1097/EDE.0b013e3181c30fb2
""")

st.markdown("---")

st.subheader("Algorithmic Fairness")
st.markdown("""
- **Racial bias in healthcare algorithms:**
  Obermeyer, Z., Powers, B., Vogeli, C., & Mullainathan, S. (2019). Dissecting racial
  bias in an algorithm used to manage the health of populations. *Science, 366*(6464),
  447–453. https://doi.org/10.1126/science.aax2342

- **Interpretability in healthcare ML:**
  Ahmad, M. A., Eckert, C., & Teredesai, A. (2018). Interpretable machine learning
  in healthcare. *Proceedings of the 2018 ACM International Conference on Bioinformatics,
  Computational Biology, and Health Informatics*, 559–560.
  https://doi.org/10.1145/3233547.3233667
""")

st.markdown("---")

st.subheader("User Interface Design")
st.markdown("""
- **Usability heuristics:**
  Nielsen, J. (1994). 10 usability heuristics for user interface design.
  *Nielsen Norman Group*. https://www.nngroup.com/articles/ten-usability-heuristics/

- **Clinical decision support UI:**
  Pickering, B. W., Dong, Y., Ahmed, A., Giri, J., Kilickaya, O., Guber, A.,
  Gajic, O., & Herasevich, V. (2015). The implementation of clinician designed,
  human-centered electronic medical record viewer in the intensive care unit.
  *International Journal of Medical Informatics, 84*(5), 299–307.
  https://doi.org/10.1016/j.ijmedinf.2015.01.017
""")

st.markdown("---")

st.subheader("Technology Stack")
st.markdown("""
| Component | Technology | Reference |
|-----------|-----------|-----------|
| Web framework | [Streamlit](https://streamlit.io/) | Streamlit, Inc. |
| ML algorithm | [XGBoost](https://xgboost.readthedocs.io/) | Chen & Guestrin, 2016 |
| Explainability | [SHAP](https://shap.readthedocs.io/) | Lundberg & Lee, 2017 |
| Visualization | [Plotly](https://plotly.com/python/) | Plotly Technologies |
| Data processing | [pandas](https://pandas.pydata.org/) | McKinney, 2010 |
| ML utilities | [scikit-learn](https://scikit-learn.org/) | Pedregosa et al., 2011 |
| Class balancing | [imbalanced-learn](https://imbalanced-learn.org/) | Lemaitre et al., 2017 |
""")

st.markdown("---")

st.caption("Developed for DSC-580: Designing and Creating Data Products — Grand Canyon University")
