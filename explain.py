#!/usr/bin/env python3
"""SHAP summary and threshold plot for XGBoost fraud model."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "creditcard_sample.csv"
REPORTS = ROOT / "reports"
MODEL = REPORTS / "xgboost_model.joblib"


def main() -> None:
    if not MODEL.exists():
        raise SystemExit("Run train_benchmark.py first")

    df = pd.read_csv(DATA)
    label = "Class"
    features = [c for c in df.columns if c != label]
    clf = joblib.load(MODEL)
    sample = df.sample(n=min(2000, len(df)), random_state=42)
    X = sample[features]
    y = sample[label].values
    proba = clf.predict_proba(X)[:, 1]

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, show=False, max_display=15)
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(REPORTS / "shap_summary.png", dpi=120)
    plt.close()

    precision, recall, thresholds = precision_recall_curve(y, proba)
    plt.figure(figsize=(8, 5))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve (XGBoost)")
    plt.savefig(REPORTS / "pr_curve.png", dpi=120)
    plt.close()

    # business threshold: max recall where precision >= 0.9
    best = {"threshold": 0.5, "precision": 0.0, "recall": 0.0}
    for t in np.linspace(0.01, 0.99, 99):
        pred = (proba >= t).astype(int)
        tp = ((pred == 1) & (y == 1)).sum()
        fp = ((pred == 1) & (y == 0)).sum()
        fn = ((pred == 0) & (y == 1)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        if prec >= 0.9 and rec > best["recall"]:
            best = {"threshold": float(t), "precision": float(prec), "recall": float(rec)}

    metrics = json.loads((REPORTS / "metrics.json").read_text()) if (REPORTS / "metrics.json").exists() else {}
    metrics["threshold_tuning"] = best
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(best, indent=2))


if __name__ == "__main__":
    main()
