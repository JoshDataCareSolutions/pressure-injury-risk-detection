"""
Data extraction from MIMIC-IV tables.
Handles chunked reading for large files (labevents, chartevents).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


def find_csv(data_dir, *parts):
    """Check for both .csv and .csv.gz versions of a file."""
    base = Path(data_dir).joinpath(*parts)
    if base.exists():
        return base
    gz = base.parent / (base.name + ".gz")
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Not found: {base} or {gz}")


def lookup_item_ids(data_dir=None):
    """Quick utility to find itemids for labs and braden scores in MIMIC."""
    data_dir = data_dir or config.get_data_dir()

    try:
        d_lab = pd.read_csv(find_csv(data_dir, "hosp", "d_labitems.csv"))
        print("=== Lab items ===")
        for name in ["albumin", "hemoglobin", "creatinine", "white blood", "platelet"]:
            matches = d_lab[d_lab["label"].str.contains(name, case=False, na=False)]
            print(f"\n{name}:")
            print(matches[["itemid", "label"]].to_string(index=False))
    except FileNotFoundError:
        print("d_labitems not found")

    try:
        d_items = pd.read_csv(find_csv(data_dir, "icu", "d_items.csv"))
        braden = d_items[d_items["label"].str.contains("Braden", case=False, na=False)]
        print("\n=== Braden items ===")
        print(braden[["itemid", "label"]].to_string(index=False))
    except FileNotFoundError:
        print("d_items not found")


def load_patients(data_dir=None):
    data_dir = data_dir or config.get_data_dir()
    df = pd.read_csv(find_csv(data_dir, "hosp", "patients.csv"))
    df = df[["subject_id", "gender", "anchor_age"]].copy()
    df.rename(columns={"anchor_age": "age"}, inplace=True)
    print(f"Patients: {len(df):,}")
    return df


def load_admissions(data_dir=None):
    data_dir = data_dir or config.get_data_dir()
    df = pd.read_csv(find_csv(data_dir, "hosp", "admissions.csv"),
                     parse_dates=["admittime", "dischtime"])
    keep = [c for c in ["subject_id", "hadm_id", "admittime", "dischtime",
                         "race", "insurance", "hospital_expire_flag"] if c in df.columns]
    df = df[keep].copy()
    df["los_days"] = (df["dischtime"] - df["admittime"]).dt.total_seconds() / 86400
    df.drop(columns=["admittime", "dischtime"], inplace=True)
    print(f"Admissions: {len(df):,}")
    return df


def load_diagnoses(data_dir=None):
    """Pull PI diagnosis codes and create a flag per admission."""
    data_dir = data_dir or config.get_data_dir()
    df = pd.read_csv(find_csv(data_dir, "hosp", "diagnoses_icd.csv"),
                     dtype={"icd_code": str})

    icd9 = (df["icd_version"] == 9) & df["icd_code"].str.startswith(tuple(config.ICD9_PI_PREFIXES))
    icd10 = (df["icd_version"] == 10) & df["icd_code"].str.startswith(config.ICD10_PI_PREFIX)

    pi_ids = df.loc[icd9 | icd10, "hadm_id"].unique()
    result = pd.DataFrame({"hadm_id": pi_ids, "pi_flag": 1})
    print(f"PI diagnoses: {len(result):,} admissions")
    return result


def load_labevents(data_dir=None):
    """Read lab events, filtering to target analytes. Uses chunking for large files."""
    data_dir = data_dir or config.get_data_dir()
    path = find_csv(data_dir, "hosp", "labevents.csv")

    targets = list(config.LAB_ITEM_IDS.values())
    id_to_name = {v: k for k, v in config.LAB_ITEM_IDS.items()}
    usecols = ["subject_id", "hadm_id", "itemid", "valuenum"]

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"labevents: {size_mb:.0f} MB")

    if size_mb > 500:  # chunk if file is big
        print("  reading in chunks...")
        chunks = []
        for i, chunk in enumerate(pd.read_csv(path, usecols=usecols,
                                               chunksize=config.CHUNK_SIZE)):
            filtered = chunk[chunk["itemid"].isin(targets)]
            if len(filtered) > 0:
                chunks.append(filtered)
            if (i + 1) % 10 == 0:
                print(f"  {(i+1) * config.CHUNK_SIZE:,} rows...")
        labs = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    else:
        labs = pd.read_csv(path, usecols=usecols)
        labs = labs[labs["itemid"].isin(targets)]

    if labs.empty:
        print("  no matching labs found")
        return pd.DataFrame(columns=["hadm_id"])

    labs["lab_name"] = labs["itemid"].map(id_to_name)
    labs = labs.dropna(subset=["hadm_id", "valuenum"])
    labs["hadm_id"] = labs["hadm_id"].astype(int)

    # pivot to one row per admission, take min (worst) value
    pivoted = labs.pivot_table(index="hadm_id", columns="lab_name",
                               values="valuenum", aggfunc="min").reset_index()
    print(f"  {len(pivoted):,} admissions with lab data")
    return pivoted


def load_chartevents(data_dir=None):
    """Read chart events for Braden subscores. Chunked for the huge file."""
    data_dir = data_dir or config.get_data_dir()
    try:
        path = find_csv(data_dir, "icu", "chartevents.csv")
    except FileNotFoundError:
        print("chartevents not found, skipping Braden scores")
        return pd.DataFrame(columns=["hadm_id"])

    targets = list(config.BRADEN_ITEM_IDS.values())
    id_to_name = {v: f"braden_{k}" for k, v in config.BRADEN_ITEM_IDS.items()}
    usecols = ["subject_id", "hadm_id", "itemid", "valuenum"]

    size_mb = path.stat().st_size / (1024 * 1024)
    print(f"chartevents: {size_mb:.0f} MB")

    try:
        if size_mb > 500:
            print("  reading in chunks...")
            chunks = []
            for i, chunk in enumerate(pd.read_csv(path, usecols=usecols,
                                                   chunksize=config.CHUNK_SIZE,
                                                   dtype={"valuenum": "float32",
                                                          "itemid": "int32"})):
                filtered = chunk[chunk["itemid"].isin(targets)]
                if len(filtered) > 0:
                    chunks.append(filtered)
                if (i + 1) % 10 == 0:
                    print(f"  {(i+1) * config.CHUNK_SIZE:,} rows...")
            charts = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        else:
            charts = pd.read_csv(path, usecols=usecols)
            charts = charts[charts["itemid"].isin(targets)]
    except (EOFError, pd.errors.ParserError) as e:
        print(f"  chartevents file seems incomplete ({e}), skipping")
        return pd.DataFrame(columns=["hadm_id"])

    if charts.empty:
        print("  no Braden scores found")
        return pd.DataFrame(columns=["hadm_id"])

    charts["braden_name"] = charts["itemid"].map(id_to_name)
    charts = charts.dropna(subset=["hadm_id", "valuenum"])
    charts["hadm_id"] = charts["hadm_id"].astype(int)

    pivoted = charts.pivot_table(index="hadm_id", columns="braden_name",
                                  values="valuenum", aggfunc="min").reset_index()
    print(f"  {len(pivoted):,} admissions with Braden data")
    return pivoted


def load_icu_stays(data_dir=None):
    data_dir = data_dir or config.get_data_dir()
    try:
        path = find_csv(data_dir, "icu", "icustays.csv")
    except FileNotFoundError:
        print("icustays not found, skipping ICU flag")
        return pd.DataFrame(columns=["hadm_id"])

    df = pd.read_csv(path, usecols=["hadm_id"])
    icu = pd.DataFrame({"hadm_id": df["hadm_id"].unique(), "icu_flag": 1})
    print(f"ICU stays: {len(icu):,} admissions")
    return icu


def build_analytic_dataset(data_dir=None, save=True):
    """Join all the MIMIC tables into one flat dataset at the admission level."""
    data_dir = data_dir or config.get_data_dir()
    print(f"\nBuilding dataset from {data_dir}\n")

    patients = load_patients(data_dir)
    admissions = load_admissions(data_dir)
    diagnoses = load_diagnoses(data_dir)
    labs = load_labevents(data_dir)
    charts = load_chartevents(data_dir)
    icu = load_icu_stays(data_dir)

    # join everything onto admissions
    df = admissions.merge(patients, on="subject_id", how="left")
    df = df.merge(diagnoses, on="hadm_id", how="left")
    df = df.merge(icu, on="hadm_id", how="left")
    if "hadm_id" in labs.columns and len(labs) > 0:
        df = df.merge(labs, on="hadm_id", how="left")
    if "hadm_id" in charts.columns and len(charts) > 0:
        df = df.merge(charts, on="hadm_id", how="left")

    df["pi_flag"] = df["pi_flag"].fillna(0).astype(int)
    df["icu_flag"] = df["icu_flag"].fillna(0).astype(int)

    # adults only
    df = df[df["age"] >= 18].copy()

    print(f"\nDataset: {df.shape[0]:,} rows x {df.shape[1]} cols")
    print(f"PI prevalence: {df['pi_flag'].mean():.2%} ({df['pi_flag'].sum():,} cases)\n")

    if save:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(config.ANALYTIC_DATASET, index=False)
        print(f"Saved to {config.ANALYTIC_DATASET}")

    return df


if __name__ == "__main__":
    lookup_item_ids()
    print()
    build_analytic_dataset()
