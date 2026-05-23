import os
import pickle
from typing import Dict, List
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def seed_everything(seed: int = 42):
    import random
    import numpy as _np

    random.seed(seed)
    _np.random.seed(seed)


def create_iid_partitions(x, y, num_clients: int = 4):
    num_examples = len(y)
    indices = np.arange(num_examples)
    np.random.shuffle(indices)
    splits = np.array_split(indices, num_clients)
    return {f"client_{i}": split.tolist() for i, split in enumerate(splits)}


def create_dirichlet_partitions(x, y, num_clients: int = 4, alpha: float = 0.3):
    labels = np.unique(y)
    num_samples = len(y)
    client_indices = {f"client_{i}": [] for i in range(num_clients)}
    label_indices = [np.where(y == label)[0] for label in labels]
    for label_ind in label_indices:
        proportions = np.random.dirichlet(alpha=np.repeat(alpha, num_clients))
        proportions = np.array([p * (len(idx) > 0) for p, idx in zip(proportions, [label_ind] * num_clients)])
        proportions = proportions / proportions.sum()
        counts = (proportions * len(label_ind)).astype(int)
        counts[-1] = len(label_ind) - counts[:-1].sum()
        np.random.shuffle(label_ind)
        start = 0
        for client_id, count in enumerate(counts):
            selected = label_ind[start : start + count]
            client_indices[f"client_{client_id}"].extend(selected.tolist())
            start += count
    return client_indices


def create_label_skew_partitions(x, y, num_clients: int = 4, labels_per_client: int = 2):
    unique_labels = np.unique(y)
    client_indices = {f"client_{i}": [] for i in range(num_clients)}
    for i in range(num_clients):
        labels = np.roll(unique_labels, -i)[:labels_per_client]
        for label in labels:
            inds = np.where(y == label)[0]
            selected = np.random.choice(inds, size=max(1, len(inds) // num_clients), replace=False)
            client_indices[f"client_{i}"].extend(selected.tolist())
    for client_id, indices in client_indices.items():
        if len(indices) == 0:
            client_indices[client_id] = list(np.random.choice(len(y), size=len(y) // num_clients, replace=False))
    return client_indices


def save_partitions(partitions: Dict[str, List[int]], path: str):
    with open(path, "wb") as fp:
        pickle.dump(partitions, fp)


def load_partitions(path: str):
    with open(path, "rb") as fp:
        return pickle.load(fp)
