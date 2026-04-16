"""Project config - paths, item IDs, feature lists, etc."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
DEMO_DIR = PROJECT_ROOT / "data" / "demo"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ANALYTIC_DATASET = PROCESSED_DIR / "analytic_dataset.parquet"

USE_DEMO = False

def get_data_dir():
    return DEMO_DIR if USE_DEMO else RAW_DIR

# verified these against d_labitems.csv
LAB_ITEM_IDS = {
    "albumin": 50862,
    "hemoglobin": 51222,
    "creatinine": 50912,
    "white_blood_cells": 51301,
    "platelet_count": 51265,
}

# verified against icu/d_items.csv
BRADEN_ITEM_IDS = {
    "sensory_perception": 224054,
    "moisture": 224055,
    "activity": 224056,
    "mobility": 224057,
    "nutrition": 224058,
    "friction_shear": 224059,
}

ICD9_PI_PREFIXES = ["7070", "7071", "7072", "7073", "7074", "7075",
                    "7076", "7077", "7078", "7079"]
ICD10_PI_PREFIX = "L89"

CHUNK_SIZE = 500_000

CLINICAL_RANGES = {
    "albumin": (0.5, 7.0),
    "hemoglobin": (2.0, 25.0),
    "creatinine": (0.1, 30.0),
    "white_blood_cells": (0.1, 500.0),
    "platelet_count": (1.0, 2000.0),
    "los_days": (0.0, 365.0),
    "age": (18, 120),
}

BRADEN_RANGES = {
    "sensory_perception": (1, 4),
    "moisture": (1, 4),
    "activity": (1, 4),
    "mobility": (1, 4),
    "nutrition": (1, 4),
    "friction_shear": (1, 3),
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
AUROC_TARGET = 0.80
SENSITIVITY_TARGET = 0.80

RISK_THRESHOLDS = {"low": 0.15, "moderate": 0.40}

NUMERIC_FEATURES = [
    "age", "los_days",
    "albumin", "hemoglobin", "creatinine", "white_blood_cells", "platelet_count",
    "braden_sensory_perception", "braden_moisture", "braden_activity",
    "braden_mobility", "braden_nutrition", "braden_friction_shear",
]

CATEGORICAL_FEATURES = ["gender", "race", "insurance"]
BINARY_FEATURES = ["icu_flag"]
TARGET = "pi_flag"
