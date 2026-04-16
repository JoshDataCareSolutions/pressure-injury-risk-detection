"""
Runs the full OSEMN pipeline end to end.
    python run_pipeline.py          # default (uses config.USE_DEMO setting)
    python run_pipeline.py --full   # forces full MIMIC-IV dataset
"""

import json, sys
import numpy as np
import config


def main():
    if "--full" in sys.argv:
        config.USE_DEMO = False

    # obtain
    from pipeline.obtain import build_analytic_dataset
    df = build_analytic_dataset()

    # scrub
    from pipeline.scrub import prepare_data
    X_train, X_test, y_train, y_test, preprocessor, feat_names = prepare_data(
        df, apply_smote=(df[config.TARGET].sum() >= 5))

    # model
    from xgboost import XGBClassifier
    from pipeline.model import save_model, evaluate_model

    neg, pos = int((y_train == 0).sum()), int(y_train.sum())
    model = XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.1,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.7,
        scale_pos_weight=neg / pos if pos > 0 else 1,
        eval_metric="aucpr", random_state=config.RANDOM_STATE)

    if len(y_train) > 1000:
        from pipeline.model import tune_hyperparameters
        model = tune_hyperparameters(X_train, y_train, n_iter=30)
    else:
        model.fit(X_train, y_train)

    metrics = evaluate_model(model, X_test, y_test)
    save_model(model, feat_names)
    with open(config.ARTIFACTS_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # interpret
    from pipeline.interpret import compute_shap_values
    sv = compute_shap_values(model, X_test[:500], feat_names)
    mean_abs = np.abs(sv.values).mean(axis=0)
    top = np.argsort(mean_abs)[::-1][:5]
    print("Top risk factors:")
    for i in top:
        print(f"  {feat_names[i]}: {mean_abs[i]:.4f}")

    print(f"\nDone. Launch with: streamlit run app.py")


if __name__ == "__main__":
    main()
