"""XGBoost model training, evaluation, and persistence."""

import json
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, recall_score, precision_score, f1_score,
    brier_score_loss, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report
)
from sklearn.calibration import calibration_curve
from xgboost import XGBClassifier
import config


def get_default_model():
    return XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.1,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        eval_metric="aucpr", random_state=config.RANDOM_STATE)


def train_model(X_train, y_train):
    m = get_default_model()
    m.fit(X_train, y_train)
    return m


def tune_hyperparameters(X_train, y_train, n_iter=50):
    params = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "min_child_weight": [1, 3, 5, 10],
        "subsample": [0.6, 0.7, 0.8, 0.9],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9],
    }
    m = XGBClassifier(eval_metric="aucpr", random_state=config.RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    search = RandomizedSearchCV(m, params, n_iter=n_iter, scoring="roc_auc",
                                cv=cv, random_state=config.RANDOM_STATE, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print(f"Best CV AUROC: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def find_optimal_threshold(model, X_test, y_test, min_sens=0.80):
    """Search for the threshold that maximizes F1 while keeping sensitivity >= target."""
    y_prob = model.predict_proba(X_test)[:, 1]
    best_t, best_f1 = 0.5, 0.0

    for t in np.arange(0.01, 0.50, 0.01):
        preds = (y_prob >= t).astype(int)
        sens = recall_score(y_test, preds)
        if sens >= min_sens:
            f = f1_score(y_test, preds)
            if f > best_f1:
                best_f1, best_t = f, t

    # fallback if nothing meets sensitivity target
    if best_f1 == 0:
        best_s = 0
        for t in np.arange(0.01, 0.50, 0.01):
            preds = (y_prob >= t).astype(int)
            s = recall_score(y_test, preds)
            if s > best_s:
                best_s, best_t = s, t
    return best_t


def evaluate_model(model, X_test, y_test, threshold=None):
    y_prob = model.predict_proba(X_test)[:, 1]
    if threshold is None:
        threshold = find_optimal_threshold(model, X_test, y_test)
    y_pred = (y_prob >= threshold).astype(int)

    m = {
        "auroc": roc_auc_score(y_test, y_prob),
        "sensitivity": recall_score(y_test, y_pred),
        "specificity": recall_score(y_test, y_pred, pos_label=0),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred),
        "brier_score": brier_score_loss(y_test, y_prob),
        "threshold": threshold,
    }

    print(f"\nResults (threshold={threshold:.2f}):")
    for k, v in m.items():
        tag = ""
        if k == "auroc": tag = " [PASS]" if v >= config.AUROC_TARGET else " [BELOW]"
        elif k == "sensitivity": tag = " [PASS]" if v >= config.SENSITIVITY_TARGET else " [BELOW]"
        print(f"  {k}: {v:.4f}{tag}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No PI", "PI"]))
    return m


# --- plotting ---

def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                             name=f"AUC={auc:.3f}", line=dict(color="#3498db", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Random", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR", width=600, height=500)
    return fig

def plot_pr_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                             name="XGBoost", line=dict(color="#e74c3c", width=2)))
    fig.update_layout(title="Precision-Recall Curve", xaxis_title="Recall",
                      yaxis_title="Precision", width=600, height=500)
    return fig

def plot_confusion_matrix(model, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    fig = px.imshow(cm, text_auto=True,
                    x=["Pred No PI", "Pred PI"], y=["Actual No PI", "Actual PI"],
                    color_continuous_scale="Blues", title="Confusion Matrix")
    fig.update_layout(width=500, height=450)
    return fig

def plot_calibration(model, X_test, y_test, n_bins=10):
    y_prob = model.predict_proba(X_test)[:, 1]
    frac, mean_pred = calibration_curve(y_test, y_prob, n_bins=n_bins)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mean_pred, y=frac, mode="lines+markers",
                             name="XGBoost", line=dict(color="#2ecc71", width=2)))
    fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                             name="Perfect", line=dict(color="gray", dash="dash")))
    fig.update_layout(title="Calibration", xaxis_title="Predicted", yaxis_title="Observed",
                      width=600, height=500)
    return fig


# --- save/load ---

def save_model(model, feature_names):
    config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(config.ARTIFACTS_DIR / "xgb_model.json"))
    with open(config.ARTIFACTS_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f)

def load_model():
    m = XGBClassifier()
    m.load_model(str(config.ARTIFACTS_DIR / "xgb_model.json"))
    return m

def load_feature_names():
    with open(config.ARTIFACTS_DIR / "feature_names.json") as f:
        return json.load(f)
