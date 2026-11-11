# fraud-automl

Benchmark **AutoGluon** vs hand-tuned **XGBoost** on highly imbalanced card-fraud-style data. Reports **PR-AUC**, recall at fixed precision, **SHAP** explanations, and threshold tuning for payments teams.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_data.py
python train_benchmark.py
python explain.py
```

Outputs: `reports/metrics.json`, `reports/shap_summary.png`, `reports/pr_curve.png`.

## Replace data

Drop the full [ULB creditcardfraud CSV](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) at `data/raw/creditcard_sample.csv` and re-run.

## Resume mapping

Portfolio project for Rakuten-style **fraud detection** and **AutoML** evaluation (not production FinTech scale).

## Limitations

Synthetic sample by default; CPU-only AutoGluon with a short time budget. Tune `time_limit` in `train_benchmark.py` for heavier runs.

## Docker

```bash
docker build -t fraud-automl .
docker run --rm -v "$PWD/reports:/app/reports" fraud-automl
```
