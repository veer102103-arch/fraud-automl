#!/usr/bin/env python3
"""Download or synthesize an imbalanced fraud-style dataset."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DEFAULT = Path(__file__).resolve().parents[1] / "data" / "raw" / "creditcard_sample.csv"


def synthesize(n_rows: int = 20000, fraud_rate: float = 0.002, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_rows * fraud_rate))
    n_ok = n_rows - n_fraud
    cols = [f"V{i}" for i in range(1, 29)]
    ok = rng.normal(0, 1, size=(n_ok, len(cols)))
    fraud = rng.normal(2.5, 1.2, size=(n_fraud, len(cols)))
    x = np.vstack([ok, fraud])
    y = np.array([0] * n_ok + [1] * n_fraud)
    idx = rng.permutation(n_rows)
    x, y = x[idx], y[idx]
    amount = np.abs(rng.lognormal(mean=3, sigma=1, size=n_rows))
    df = pd.DataFrame(x, columns=cols)
    df["Amount"] = amount
    df["Class"] = y
    return df


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=RAW_DEFAULT)
    p.add_argument("--rows", type=int, default=20000)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df = synthesize(n_rows=args.rows)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows ({df['Class'].mean():.4f} fraud rate) -> {args.out}")
    print("Replace with full ULB CSV from Kaggle if available: mlg-ulb/creditcardfraud")


if __name__ == "__main__":
    main()
