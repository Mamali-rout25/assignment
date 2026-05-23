import os
import sys
sys.path.insert(0, os.getcwd())
from src.data.loaders import load_heart_disease, split_dataset
from src.fl.server import run_federated_experiment

x, y = load_heart_disease(return_numpy=True)
x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.2, seed=42)
res = run_federated_experiment(
    dataset_name='heart',
    x_train=x_train,
    y_train=y_train,
    x_test=x_test,
    y_test=y_test,
    partitions_path='data/heart/iid_partition.pkl',
    num_clients=2,
    num_rounds=1,
    local_epochs=1,
    batch_size=16,
    lr=1e-3,
    use_dp=False,
    seed=42,
)
print(res)
