# Pressure Injury Risk Detection System

A machine learning-based clinical decision support tool that predicts hospital-acquired pressure injury risk using electronic health record data from MIMIC-IV.

## Overview

Pressure injuries affect approximately 2.5 million hospitalized patients annually in the United States. This system uses XGBoost gradient-boosted trees trained on 546,000+ hospital admissions to identify patients at elevated risk, enabling proactive clinical intervention.

**Performance:** AUROC 0.842 | Sensitivity 0.800 | Specificity 0.715

## Features

- Binary risk classification using patient demographics, lab values, Braden Scale subscores, and clinical indicators
- SHAP-based explainability showing which factors drive each prediction
- Interactive web interface built with Streamlit
- Full OSEMN analytics pipeline (Obtain, Scrub, Explore, Model, iNterpret)

## Tech Stack

- **Data processing:** pandas, NumPy
- **Machine learning:** XGBoost, scikit-learn, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** Plotly, Matplotlib
- **Web framework:** Streamlit
- **Data source:** MIMIC-IV (PhysioNet)

## Setup

```bash
pip install -r requirements.txt
```

### Data

This project uses [MIMIC-IV](https://physionet.org/content/mimiciv/3.1/) which requires PhysioNet credentialing. Place the following files in `data/raw/`:

```
data/raw/
├── hosp/
│   ├── patients.csv.gz
│   ├── admissions.csv.gz
│   ├── diagnoses_icd.csv.gz
│   ├── d_labitems.csv.gz
│   └── labevents.csv.gz
└── icu/
    ├── d_items.csv.gz
    ├── chartevents.csv.gz
    └── icustays.csv.gz
```

### Run Pipeline

```bash
python run_pipeline.py --full
```

### Launch App

```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Streamlit entry point
├── config.py              # Configuration and constants
├── run_pipeline.py        # End-to-end pipeline runner
├── pipeline/
│   ├── obtain.py          # MIMIC-IV data extraction and joining
│   ├── scrub.py           # Cleaning, imputation, feature engineering
│   ├── explore.py         # EDA visualizations
│   ├── model.py           # XGBoost training and evaluation
│   └── interpret.py       # SHAP explanations and risk gauge
├── pages/                 # Streamlit multi-page app
│   ├── 1_Data_Overview.py
│   ├── 2_EDA.py
│   ├── 3_Model_Performance.py
│   ├── 4_Risk_Assessment.py
│   ├── 5_User_Guide.py
│   ├── 6_Pipeline.py
│   ├── 7_Data_Management.py
│   └── 8_Reports.py
└── artifacts/             # Saved model and preprocessor
```

## Disclaimer

This is a research prototype developed for academic purposes. It uses de-identified data and is not intended for clinical decision-making.
