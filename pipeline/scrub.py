"""
Preprocessing pipeline: validation, imputation, encoding, feature engineering.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import config


def validate_clinical_ranges(df):
    """Set clinically implausible values to NaN."""
    df = df.copy()
    for col, (lo, hi) in config.CLINICAL_RANGES.items():
        if col in df.columns:
            bad = (df[col] < lo) | (df[col] > hi)
            if bad.sum() > 0:
                print(f"  {col}: {bad.sum()} out-of-range -> NaN")
                df.loc[bad, col] = np.nan

    for col, (lo, hi) in config.BRADEN_RANGES.items():
        bc = f"braden_{col}"
        if bc in df.columns:
            bad = (df[bc] < lo) | (df[bc] > hi)
            if bad.sum() > 0:
                print(f"  {bc}: {bad.sum()} out-of-range -> NaN")
                df.loc[bad, bc] = np.nan
    return df


def report_missing(df):
    print("\nMissing data:")
    for col in df.columns:
        pct = df[col].isna().mean() * 100
        if pct > 0:
            print(f"  {col}: {pct:.1f}%")
    print()


def consolidate_race(df, top_n=5):
    """Lump rare race categories into 'Other'."""
    df = df.copy()
    if "race" not in df.columns:
        return df
    top = df["race"].value_counts().nlargest(top_n).index.tolist()
    df["race"] = df["race"].where(df["race"].isin(top), other="Other")
    return df


def add_missingness_indicators(df, cols):
    """Whether a lab was even ordered can be informative on its own."""
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isna().astype(int)
    return df


def add_engineered_features(df):
    """Derived features based on clinical thresholds."""
    df = df.copy()
    braden_cols = [c for c in df.columns if c.startswith("braden_")]
    if braden_cols:
        df["braden_total"] = df[braden_cols].sum(axis=1, min_count=1)

    if "age" in df.columns:
        df["age_over_75"] = (df["age"] >= 75).astype(int)
    if "los_days" in df.columns:
        df["los_over_7"] = (df["los_days"] > 7).astype(int)
        df["los_over_14"] = (df["los_days"] > 14).astype(int)
    if "albumin" in df.columns:
        df["albumin_low"] = (df["albumin"] < 3.0).astype(int)
    if "hemoglobin" in df.columns:
        df["hemoglobin_low"] = (df["hemoglobin"] < 10.0).astype(int)
    return df


def build_preprocessor(num_feats, cat_feats, passthru_feats=None, bin_feats=None):
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = [("num", num_pipe, num_feats), ("cat", cat_pipe, cat_feats)]
    if passthru_feats:
        transformers.append(("passthru_num", "passthrough", passthru_feats))
    if bin_feats:
        transformers.append(("binary", "passthrough", bin_feats))
    return ColumnTransformer(transformers=transformers, remainder="drop")


def prepare_data(df, apply_smote=True):
    """
    Full preprocessing: validate ranges, engineer features, split, transform.
    Returns X_train, X_test, y_train, y_test, preprocessor, feature_names
    """
    print("Preprocessing...\n")
    df = validate_clinical_ranges(df)
    df = consolidate_race(df)

    lab_cols = [c for c in ["albumin", "hemoglobin", "creatinine",
                            "white_blood_cells", "platelet_count"] if c in df.columns]
    df = add_missingness_indicators(df, lab_cols)
    df = add_engineered_features(df)
    report_missing(df)

    # figure out which features we actually have
    num_feats = [f for f in config.NUMERIC_FEATURES if f in df.columns]
    cat_feats = [f for f in config.CATEGORICAL_FEATURES if f in df.columns]
    bin_feats = [f for f in config.BINARY_FEATURES if f in df.columns]

    # engineered features
    if "braden_total" in df.columns:
        num_feats.append("braden_total")
    for f in ["age_over_75", "los_over_7", "los_over_14", "albumin_low", "hemoglobin_low"]:
        if f in df.columns:
            bin_feats.append(f)
    for c in lab_cols:
        ind = f"{c}_missing"
        if ind in df.columns and ind not in bin_feats:
            bin_feats.append(ind)

    # high-missingness features go through as-is (XGBoost handles NaN)
    imputable = [f for f in num_feats if df[f].isna().mean() <= 0.50]
    passthru = [f for f in num_feats if df[f].isna().mean() > 0.50]

    if passthru:
        print(f"  High-missingness (passthrough): {passthru}")
        for f in passthru:
            ind = f"{f}_missing"
            if ind not in df.columns:
                df[ind] = df[f].isna().astype(int)
            if ind not in bin_feats:
                bin_feats.append(ind)

    all_feats = imputable + passthru + cat_feats + bin_feats
    print(f"Total features: {len(all_feats)} ({len(imputable)} num, {len(passthru)} passthru, "
          f"{len(cat_feats)} cat, {len(bin_feats)} binary)")

    X = df[all_feats].copy()
    y = df[config.TARGET].copy()
    print(f"Classes: {int((y==0).sum()):,} neg / {int(y.sum()):,} pos ({y.mean():.2%})\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE)

    preprocessor = build_preprocessor(imputable, cat_feats,
                                       passthru or None, bin_feats or None)
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # feature names after encoding
    feat_names = list(imputable)
    if cat_feats:
        ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        feat_names.extend(ohe.get_feature_names_out(cat_feats).tolist())
    if passthru:
        feat_names.extend(passthru)
    feat_names.extend(bin_feats)

    print(f"After encoding: {len(feat_names)} features")

    # SMOTE only if no NaNs (it can't handle them)
    if apply_smote and y_train.sum() >= 5 and not np.isnan(X_train_proc).any():
        majority = int((y_train == 0).sum())
        target = int(majority * 0.20)
        minority = int(y_train.sum())
        if target > minority:
            smote = SMOTE(sampling_strategy=target / majority,
                          random_state=config.RANDOM_STATE)
            X_train_proc, y_train = smote.fit_resample(X_train_proc, y_train)
            print(f"SMOTE: {int((y_train==0).sum()):,} neg / {int(y_train.sum()):,} pos")
    elif apply_smote and np.isnan(X_train_proc).any():
        print("Skipping SMOTE (NaN in data, using scale_pos_weight instead)")

    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, config.ARTIFACTS_DIR / "preprocessor.pkl")
    print(f"Train: {X_train_proc.shape}, Test: {X_test_proc.shape}\n")
    return X_train_proc, X_test_proc, y_train, y_test, preprocessor, feat_names
