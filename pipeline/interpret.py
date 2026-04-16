"""SHAP explanations and the risk gauge visualization."""

import json
import numpy as np
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import config


def compute_shap_values(model, X, feature_names=None):
    explainer = shap.TreeExplainer(model)
    sv = explainer(X)
    if feature_names:
        sv.feature_names = feature_names

    # save expected value for later use
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ev = explainer.expected_value
    if isinstance(ev, np.ndarray):
        ev = float(ev[0]) if len(ev) == 1 else float(ev[1])
    with open(config.ARTIFACTS_DIR / "shap_expected_value.json", "w") as f:
        json.dump({"expected_value": float(ev)}, f)
    return sv


def plot_shap_summary(sv, max_display=15):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.summary_plot(sv, show=False, max_display=max_display)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_bar(sv, max_display=15):
    fig, ax = plt.subplots(figsize=(10, 6))
    shap.plots.bar(sv, show=False, max_display=max_display)
    plt.tight_layout()
    return plt.gcf()


def plot_shap_waterfall(sv, idx):
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.waterfall_plot(sv[idx], show=False)
    plt.tight_layout()
    return plt.gcf()


def get_top_risk_factors(sv, idx, feature_names, n=5):
    vals = sv[idx].values
    names = feature_names or [f"f{i}" for i in range(len(vals))]
    top = np.argsort(np.abs(vals))[::-1][:n]
    return [{"feature": names[i], "shap_value": float(vals[i]),
             "direction": "increases risk" if vals[i] > 0 else "decreases risk"}
            for i in top]


def plot_risk_gauge(prob):
    """Color-coded gauge: green < 15%, yellow 15-40%, red > 40%."""
    if prob < config.RISK_THRESHOLDS["low"]:
        label, color = "Low Risk", "#2ecc71"
    elif prob < config.RISK_THRESHOLDS["moderate"]:
        label, color = "Moderate Risk", "#f39c12"
    else:
        label, color = "High Risk", "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%", "font": {"size": 48}},
        title={"text": label, "font": {"size": 24}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color, "thickness": 0.75},
            "steps": [
                {"range": [0, 15], "color": "#d5f5e3"},
                {"range": [15, 40], "color": "#fdebd0"},
                {"range": [40, 100], "color": "#fadbd8"},
            ],
        },
    ))
    fig.update_layout(height=350, margin=dict(t=80, b=20))
    return fig
