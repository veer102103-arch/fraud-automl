.PHONY: data train explain all
data:
	python scripts/download_data.py
train: data
	python train_benchmark.py
explain: train
	python explain.py
all: explain
