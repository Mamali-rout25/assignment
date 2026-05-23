import os
import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.data.loaders import (
    load_heart_disease,
    load_diabetes,
    load_mnist,
    save_dataset_split,
    split_dataset,
)
from src.data.partitions import (
    create_dirichlet_partitions,
    create_iid_partitions,
    create_label_skew_partitions,
    save_partitions,
    seed_everything,
)

PARTITION_ROOT = Path(__file__).resolve().parent


def partition_dataset(name: str, x, y, num_clients: int = 4, seed: int = 42):
    target_root = PARTITION_ROOT / name
    target_root.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.2, seed=seed)
    save_dataset_split(name, x_train, y_train, x_test, y_test)

    iid = create_iid_partitions(x_train, y_train, num_clients)
    save_partitions(iid, target_root / "iid_partition.pkl")

    noniid = create_dirichlet_partitions(x_train, y_train, num_clients, alpha=0.3)
    save_partitions(noniid, target_root / "dirichlet_partition.pkl")

    label_skew = create_label_skew_partitions(x_train, y_train, num_clients, labels_per_client=2)
    save_partitions(label_skew, target_root / "label_skew_partition.pkl")

    print(f"Saved partitions and splits for {name} at {target_root}")


def main():
    seed_everything(42)
    print("Loading raw datasets...")
    x_heart, y_heart = load_heart_disease(return_numpy=True)
    x_diabetes, y_diabetes = load_diabetes(return_numpy=True)
    x_mnist_train, y_mnist_train = load_mnist(return_numpy=True, train=True)
    x_mnist_test, y_mnist_test = load_mnist(return_numpy=True, train=False)
    x_mnist = np.concatenate([x_mnist_train, x_mnist_test], axis=0)
    y_mnist = np.concatenate([y_mnist_train, y_mnist_test], axis=0)

    partition_dataset("heart", x_heart, y_heart)
    partition_dataset("diabetes", x_diabetes, y_diabetes)
    partition_dataset("mnist", x_mnist, y_mnist)
    print("Partition files created.")


if __name__ == "__main__":
    main()
