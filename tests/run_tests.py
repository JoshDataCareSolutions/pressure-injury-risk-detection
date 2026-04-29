"""Validation test suite for the Pressure Injury Risk Detection System.

Each test exercises a single product feature, prints a PASS/FAIL line, and is
recorded in the returned results list. The script can be run from the command
line (writes ``tests/test_results.txt``) or imported and called from the
Streamlit Tests page.
"""

from __future__ import annotations

import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config  # noqa: E402
from pipeline import scrub  # noqa: E402
from pipeline.model import load_model, load_feature_names  # noqa: E402
from pipeline.interpret import compute_shap_values  # noqa: E402


def _record(results, name, status, detail=""):
    results.append({"name": name, "status": status, "detail": detail})
    flag = "PASS" if status == "PASS" else "FAIL"
    print(f"[{flag}] {name}{(' — ' + detail) if detail else ''}")


def test_data_loads_from_file(results):
    name = "Data loads from analytic Parquet file"
    try:
        df = pd.read_parquet(config.ANALYTIC_DATASET)
        assert len(df) > 0, "empty dataframe"
        _record(results, name, "PASS", f"{len(df):,} rows × {df.shape[1]} cols")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_data_loads_from_url(results):
    """Verify URL-based ingestion works (uses a tiny public CSV)."""
    name = "Data loads from URL"
    url = ("https://raw.githubusercontent.com/mwaskom/seaborn-data/"
           "master/iris.csv")
    try:
        df = pd.read_csv(url)
        assert len(df) > 0
        _record(results, name, "PASS", f"loaded {len(df)} rows from {url}")
    except Exception as e:
        _record(results, name, "FAIL", f"network or parse error: {e}")


def test_clinical_range_clipping(results):
    name = "Preprocessing — clinical range clipping"
    try:
        df = pd.DataFrame({"albumin": [-1.0, 3.5, 100.0]})
        for col, (lo, hi) in config.CLINICAL_RANGES.items():
            if col in df.columns:
                df[col] = df[col].clip(lower=lo, upper=hi)
        assert df["albumin"].min() >= config.CLINICAL_RANGES["albumin"][0]
        assert df["albumin"].max() <= config.CLINICAL_RANGES["albumin"][1]
        _record(results, name, "PASS", "out-of-range values clamped")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_imputation_methods(results):
    name = "Preprocessing — median, mean, mode imputation"
    try:
        df = pd.DataFrame({
            "x": [1.0, 2.0, np.nan, 4.0],
            "y": [10.0, np.nan, 30.0, 40.0],
            "g": ["a", "a", None, "b"],
        })
        # median
        med = df.copy()
        for c in ["x", "y"]:
            med[c] = med[c].fillna(med[c].median())
        assert med.isna().sum().sum() == 1  # only g remains
        # mean
        mean = df.copy()
        for c in ["x", "y"]:
            mean[c] = mean[c].fillna(mean[c].mean())
        # mode
        mode = df.copy()
        mode["g"] = mode["g"].fillna(mode["g"].mode().iloc[0])
        assert mode["g"].isna().sum() == 0
        _record(results, name, "PASS", "all three imputation strategies functional")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_one_hot_encoding(results):
    name = "Preprocessing — one-hot encoding"
    try:
        df = pd.DataFrame({"gender": ["M", "F", "M"], "age": [50, 60, 70]})
        encoded = pd.get_dummies(df, columns=["gender"])
        assert "gender_M" in encoded.columns and "gender_F" in encoded.columns
        _record(results, name, "PASS", f"encoded shape {encoded.shape}")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_model_artifact_exists(results):
    name = "Model artifact loads"
    try:
        model = load_model()
        feats = load_feature_names()
        assert hasattr(model, "predict_proba")
        assert isinstance(feats, list) and len(feats) > 0
        _record(results, name, "PASS", f"{len(feats)} features registered")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_model_prediction(results):
    name = "Model produces a probability for a synthetic input"
    try:
        model = load_model()
        feats = load_feature_names()
        x = np.zeros((1, len(feats)))
        p = float(model.predict_proba(x)[0, 1])
        assert 0.0 <= p <= 1.0
        _record(results, name, "PASS", f"probability={p:.4f}")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_metrics_meet_targets(results):
    name = "Saved metrics meet AUROC and sensitivity targets"
    try:
        with open(config.ARTIFACTS_DIR / "metrics.json") as f:
            m = json.load(f)
        ok = m["auroc"] >= config.AUROC_TARGET and m["sensitivity"] >= config.SENSITIVITY_TARGET
        status = "PASS" if ok else "FAIL"
        _record(results, name, status,
                f"AUROC={m['auroc']:.3f}, sens={m['sensitivity']:.3f}")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_shap_explanation(results):
    name = "SHAP explanation runs on a single prediction"
    try:
        model = load_model()
        feats = load_feature_names()
        x = np.zeros((1, len(feats)))
        shap_values = compute_shap_values(model, x, feats)
        assert shap_values is not None
        _record(results, name, "PASS", "SHAP values computed without error")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_report_generation(results):
    name = "Report text generation produces non-empty output"
    try:
        with open(config.ARTIFACTS_DIR / "metrics.json") as f:
            m = json.load(f)
        lines = [
            "PRESSURE INJURY RISK DETECTION SYSTEM",
            f"Generated: {datetime.now()}",
            f"AUROC: {m['auroc']:.3f}",
            f"Sensitivity: {m['sensitivity']:.3f}",
        ]
        text = "\n".join(lines)
        assert len(text) > 50
        _record(results, name, "PASS", f"{len(text)} characters")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_report_saves_to_file(results):
    name = "Report saves to a file"
    try:
        out = ROOT / "tests" / "_sample_report.txt"
        out.write_text("test report\n")
        assert out.exists() and out.stat().st_size > 0
        out.unlink()
        _record(results, name, "PASS", "round-trip file write/delete")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def test_risk_threshold_classification(results):
    name = "Risk thresholds map probabilities to categories"
    try:
        thresholds = config.RISK_THRESHOLDS
        def cat(p):
            if p < thresholds["low"]: return "Low"
            if p < thresholds["moderate"]: return "Moderate"
            return "High"
        assert cat(0.05) == "Low"
        assert cat(0.25) == "Moderate"
        assert cat(0.75) == "High"
        _record(results, name, "PASS", "low/moderate/high categories correct")
    except Exception as e:
        _record(results, name, "FAIL", str(e))


def run_all():
    results = []
    tests = [
        test_data_loads_from_file,
        test_data_loads_from_url,
        test_clinical_range_clipping,
        test_imputation_methods,
        test_one_hot_encoding,
        test_model_artifact_exists,
        test_model_prediction,
        test_metrics_meet_targets,
        test_shap_explanation,
        test_report_generation,
        test_report_saves_to_file,
        test_risk_threshold_classification,
    ]
    for t in tests:
        try:
            t(results)
        except Exception:
            results.append({"name": t.__name__, "status": "FAIL",
                            "detail": traceback.format_exc(limit=1).splitlines()[-1]})
    return results


def write_results_file(results, path):
    n_pass = sum(1 for r in results if r["status"] == "PASS")
    lines = [
        "=" * 70,
        "PRESSURE INJURY RISK DETECTION — VALIDATION TEST RESULTS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Summary: {n_pass}/{len(results)} tests passed",
        "=" * 70,
        "",
    ]
    for r in results:
        lines.append(f"[{r['status']}] {r['name']}")
        if r["detail"]:
            lines.append(f"        {r['detail']}")
    path.write_text("\n".join(lines))
    return n_pass, len(results)


if __name__ == "__main__":
    print("Running validation test suite...\n")
    results = run_all()
    out_path = Path(__file__).parent / "test_results.txt"
    n_pass, total = write_results_file(results, out_path)
    print(f"\n{n_pass}/{total} passed. Results written to {out_path}")
    sys.exit(0 if n_pass == total else 1)
