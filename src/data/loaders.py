import os
from typing import Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

DATA_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "data")


def _normalize_tabular(x: np.ndarray) -> np.ndarray:
    x = x.astype(float)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std[std == 0] = 1.0
    return (x - mean) / std


def _dataset_dir(dataset_name: str) -> str:
    return os.path.join(DATA_ROOT, dataset_name)


def save_dataset_split(dataset_name: str, x_train, y_train, x_test, y_test):
    path = _dataset_dir(dataset_name)
    os.makedirs(path, exist_ok=True)
    np.savez_compressed(os.path.join(path, "train.npz"), x=x_train, y=y_train)
    np.savez_compressed(os.path.join(path, "test.npz"), x=x_test, y=y_test)


def load_dataset_split(dataset_name: str):
    path = _dataset_dir(dataset_name)
    train_path = os.path.join(path, "train.npz")
    test_path = os.path.join(path, "test.npz")
    if os.path.exists(train_path) and os.path.exists(test_path):
        train = np.load(train_path)
        test = np.load(test_path)
        return train["x"], train["y"], test["x"], test["y"]
    x, y = {
        "heart": load_heart_disease(return_numpy=True),
        "diabetes": load_diabetes(return_numpy=True),
        "mnist": load_mnist(return_numpy=True, train=True),
    }[dataset_name]
    if dataset_name == "mnist":
        x_test, y_test = load_mnist(return_numpy=True, train=False)
        x = np.concatenate([x, x_test], axis=0)
        y = np.concatenate([y, y_test], axis=0)
    return split_dataset(x, y)


def load_heart_disease(return_numpy: bool = False) -> Tuple[Any, Any]:
    path = os.path.join(DATA_ROOT, "raw", "heart.csv")
    df = pd.read_csv(path)
    x = df.drop(columns=["target"]).values
    y = df["target"].values.astype(int)
    x = _normalize_tabular(x)
    if return_numpy:
        return x, y
    return x, y


def load_diabetes(return_numpy: bool = False) -> Tuple[Any, Any]:
    path = os.path.join(DATA_ROOT, "raw", "diabetes.csv")
    df = pd.read_csv(path)
    x = df.drop(columns=["target"]).values
    y = df["target"].values.astype(int)
    x = _normalize_tabular(x)
    if return_numpy:
        return x, y
    return x, y


def load_mnist(return_numpy: bool = False, train: bool = True) -> Any:
    data_root = os.path.join(DATA_ROOT, "raw", "mnist")
    transform = transforms.Compose([transforms.ToTensor()])
    if return_numpy:
        dataset = datasets.MNIST(data_root, train=train, download=True, transform=transform)
        x = np.stack([np.array(img).astype(float) for img, _ in dataset])
        y = np.array([label for _, label in dataset], dtype=int)
        x = x.reshape(len(x), -1) / 255.0
        return x, y
    return datasets.MNIST(data_root, train=train, download=True, transform=transform)


def split_dataset(x: np.ndarray, y: np.ndarray, test_size: float = 0.2, seed: int = 42):
    return train_test_split(x, y, test_size=test_size, stratify=y, random_state=seed)
