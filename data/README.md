# Data Preparation

This folder contains dataset download and partitioning tools for the federated learning experiments.

## Scripts
- `download_and_prepare.py`: downloads UCI Heart Disease and Pima Diabetes datasets and saves cleaned CSV files to `data/raw/`.
- `create_partitions.py`: loads each dataset and creates four client partitions for IID, Dirichlet non-IID, and label-skew non-IID splits.

## Usage
```bash
python data/download_and_prepare.py
python data/create_partitions.py
```

The partitions are saved as pickle files under `data/heart/`, `data/diabetes/`, and `data/mnist/`.
