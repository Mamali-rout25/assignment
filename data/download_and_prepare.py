import io
import os
import pandas as pd
import numpy as np
import requests
from torchvision import datasets, transforms

RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
HEART_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
DIABETES_URL = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"

os.makedirs(RAW_DIR, exist_ok=True)


def download_csv(url: str, path: str, header: list = None):
    if os.path.exists(path):
        print(f"Found cached file: {path}")
        return

    print(f"Downloading {url} -> {path}")
    response = requests.get(url)
    response.raise_for_status()
    text = response.text.strip()
    df = pd.read_csv(io.StringIO(text), header=None)
    if header is not None:
        df.columns = header
    df.to_csv(path, index=False)
    print(f"Saved {path}")


def prepare_heart():
    path = os.path.join(RAW_DIR, "heart.csv")
    header = [
        "age",
        "sex",
        "cp",
        "trestbps",
        "chol",
        "fbs",
        "restecg",
        "thalach",
        "exang",
        "oldpeak",
        "slope",
        "ca",
        "thal",
        "target",
    ]
    download_csv(HEART_URL, path, header)
    df = pd.read_csv(path)
    df = df.replace("?", np.nan).dropna().astype(float)
    df["target"] = (df["target"] > 0).astype(int)
    df.to_csv(path, index=False)
    print(f"Prepared heart disease dataset with {len(df)} rows")


def prepare_diabetes():
    path = os.path.join(RAW_DIR, "diabetes.csv")
    header = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "target",
    ]
    download_csv(DIABETES_URL, path, header)
    df = pd.read_csv(path)
    df.to_csv(path, index=False)
    print(f"Prepared diabetes dataset with {len(df)} rows")


def prepare_mnist():
    mnist_dir = os.path.join(RAW_DIR, "mnist")
    os.makedirs(mnist_dir, exist_ok=True)
    transform = transforms.Compose([transforms.ToTensor()])
    datasets.MNIST(mnist_dir, train=True, download=True, transform=transform)
    datasets.MNIST(mnist_dir, train=False, download=True, transform=transform)
    print(f"Downloaded MNIST into {mnist_dir}")


def main():
    print("Preparing raw datasets...")
    prepare_heart()
    prepare_diabetes()
    prepare_mnist()
    print("Datasets are ready in data/raw/")


if __name__ == "__main__":
    main()
