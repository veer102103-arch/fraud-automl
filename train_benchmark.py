#!/usr/bin/env python3
"""Benchmark AutoGluon vs tuned XGBoost on imbalanced fraud data."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "creditcard_sample.csv"
REPORTS = ROOT / "reports"
AUTODIR = ROOT / "autogluon_models"


def recall_at_precision(y_true, y_score, min_precision: float = 0.9) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    mask = precision >= min_precision
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def train_xgb(X_train, y_train, X_test, y_test) -> dict:
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)
    proba = clf.predict_proba(X_test)[:, 1]
    return {
        "model": "xgboost",
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "recall_at_precision_0.9": recall_at_precision(y_test, proba, 0.9),
        "estimator": clf,
        "proba": proba,
        "y_test": y_test,
    }


def train_autogluon(train_df, test_df, label: str = "Class", time_limit: int = 120) -> dict:
    from autogluon.tabular import TabularPredictor

    AUTODIR.mkdir(parents=True, exist_ok=True)
    predictor = TabularPredictor(
        label=label,
        eval_metric="average_precision",
        path=str(AUTODIR),
        verbosity=1,
    )
    predictor.fit(
        train_df,
        time_limit=time_limit,
        presets="medium_quality",
        ag_args_fit={"num_gpus": 0},
    )
    proba = predictor.predict_proba(test_df.drop(columns=[label]))[1]
    y_test = test_df[label].values
    return {
        "model": "autogluon",
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "pr_auc": float(average_precision_score(y_test, proba)),
        "recall_at_precision_0.9": recall_at_precision(y_test, proba, 0.9),
        "predictor": predictor,
        "proba": proba,
        "y_test": y_test,
    }


def main() -> None:
    if not DATA.exists():
        import subprocess
        import sys

        subprocess.check_call([sys.executable, str(ROOT / "scripts" / "download_data.py")])

    df = pd.read_csv(DATA)
    label = "Class"
    features = [c for c in df.columns if c != label]
    stratify = df[label] if df[label].sum() > 1 else None
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=stratify
    )
    X_train, y_train = train_df[features], train_df[label]
    X_test, y_test = test_df[features], test_df[label]

    xgb_res = train_xgb(X_train.values, y_train.values, X_test.values, y_test.values)
    ag_res = train_autogluon(train_df, test_df)

    REPORTS.mkdir(parents=True, exist_ok=True)
    summary = {
        "xgboost": {k: v for k, v in xgb_res.items() if k not in ("estimator", "proba", "y_test")},
        "autogluon": {k: v for k, v in ag_res.items() if k not in ("predictor", "proba", "y_test")},
        "data_rows": len(df),
        "fraud_rate_train": float(train_df[label].mean()),
    }
    (REPORTS / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    import joblib

    joblib.dump(xgb_res["estimator"], REPORTS / "xgboost_model.joblib")
    print(f"Saved metrics -> {REPORTS / 'metrics.json'}")


if __name__ == "__main__":
    main()
