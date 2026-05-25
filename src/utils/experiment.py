import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from src.data.loaders import load_heart_disease, load_diabetes, load_mnist, split_dataset
from src.data.partitions import load_partitions
from src.fl.server import run_federated_experiment, get_model_for_dataset
from src.utils.plotting import plot_accuracy_loss, plot_communication
from src.utils.seed import seed_everything


class ExperimentRunner:
    def __init__(
        self,
        data_root: str = "data",
        results_root: str = "results",
        num_clients: int = 4,
        num_rounds: int = 5,
        local_epochs: int = 2,
        central_epochs: int = 10,
        dp_target_epsilon: float = 8.0,
        dp_target_delta: float = 1e-5,
        batch_size: int = 32,
        lr: float = 1e-3,
        seed: int = 42,
    ):
        self.data_root = data_root
        self.results_root = results_root
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.central_epochs = central_epochs
        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.batch_size = batch_size
        self.lr = lr
        self.seed = seed
        seed_everything(self.seed)

    def run_all(self):
        results = []
        for dataset_name in ["heart", "diabetes", "mnist"]:
            print(f"Running experiments for {dataset_name}")
            x, y = self._load_dataset(dataset_name)
            x_train, x_test, y_train, y_test = split_dataset(x, y, test_size=0.2, seed=self.seed)

            results.extend(self._run_dataset_experiments(dataset_name, x_train, y_train, x_test, y_test))
        return pd.DataFrame(results)

    def _load_dataset(self, dataset_name: str):
        if dataset_name == "heart":
            return load_heart_disease(return_numpy=True)
        if dataset_name == "diabetes":
            return load_diabetes(return_numpy=True)
        if dataset_name == "mnist":
            x_train, y_train = load_mnist(return_numpy=True, train=True)
            x_test, y_test = load_mnist(return_numpy=True, train=False)
            x = np.concatenate([x_train[:5000], x_test[:1000]], axis=0)
            y = np.concatenate([y_train[:5000], y_test[:1000]], axis=0)
            return x, y
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def _run_dataset_experiments(self, dataset_name, x_train, y_train, x_test, y_test):
        outputs = []
        if dataset_name == "mnist":
            x_train = x_train.reshape(-1, 28, 28)
            x_test = x_test.reshape(-1, 28, 28)

        print(f"  Running centralized baseline for {dataset_name}")
        central_results = self._run_centralized(dataset_name, x_train, y_train, x_test, y_test)
        outputs.append({**central_results, "variant": "centralized", "dataset": dataset_name})

        for label in ["iid_partition.pkl", "dirichlet_partition.pkl", "label_skew_partition.pkl"]:
            variant = self._variant_label(label)
            print(f"  Running federated variant {variant} for {dataset_name}")
            partitions_path = os.path.join(self.data_root, dataset_name, label)
            federated_results = run_federated_experiment(
                dataset_name=dataset_name,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                partitions_path=partitions_path,
                num_clients=self.num_clients,
                num_rounds=self.num_rounds,
                local_epochs=self.local_epochs,
                batch_size=self.batch_size,
                lr=self.lr,
                use_dp=False,
                seed=self.seed,
            )
            outputs.append({**federated_results, "variant": variant, "dataset": dataset_name})

        dp_results = run_federated_experiment(
            dataset_name=dataset_name,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            partitions_path=os.path.join(self.data_root, dataset_name, "dirichlet_partition.pkl"),
            num_clients=self.num_clients,
            num_rounds=self.num_rounds,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            lr=self.lr,
            use_dp=True,
            dp_target_epsilon=self.dp_target_epsilon,
            dp_target_delta=self.dp_target_delta,
            seed=self.seed,
        )
        outputs.append({**dp_results, "variant": "fl_dp", "dataset": dataset_name})

        plot_accuracy_loss(
            history=central_results.get("history", {}),
            title=f"{dataset_name} Centralized",
            output_path=os.path.join(self.results_root, dataset_name, "centralized_accuracy_loss.png"),
        )
        for entry in outputs:
            if entry["variant"] != "centralized":
                plot_accuracy_loss(
                    history=entry.get("history", {}),
                    title=f"{dataset_name} {entry['variant']}",
                    output_path=os.path.join(self.results_root, dataset_name, f"{entry['variant']}_accuracy_loss.png"),
                )

        plot_communication(
            [entry for entry in outputs if entry["variant"] != "centralized"],
            output_path=os.path.join(self.results_root, dataset_name, "communication_overhead.png"),
        )
        return outputs

    def _run_centralized(self, dataset_name, x_train, y_train, x_test, y_test):
        input_dim = x_train.shape[1]
        num_classes = 10 if dataset_name == "mnist" else len(np.unique(y_train))
        model = get_model_for_dataset(dataset_name, input_dim=input_dim, num_classes=num_classes)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        train_loader = DataLoader(
            TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=False,
        )

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)

        history = {"accuracy": [], "loss": []}
        for epoch in range(self.central_epochs):
            model.train()
            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if dataset_name == "mnist":
                    x_batch = x_batch.view(x_batch.size(0), 1, 28, 28)
                optimizer.zero_grad()
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            total = 0
            correct = 0
            test_loss = 0.0
            with torch.no_grad():
                for x_batch, y_batch in test_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    if dataset_name == "mnist":
                        x_batch = x_batch.view(x_batch.size(0), 1, 28, 28)
                    logits = model(x_batch)
                    test_loss += criterion(logits, y_batch).item() * x_batch.size(0)
                    preds = logits.argmax(dim=1)
                    correct += (preds == y_batch).sum().item()
                    total += x_batch.size(0)
            history["loss"].append(test_loss / total)
            history["accuracy"].append(correct / total)

        return {
            "accuracy": history["accuracy"][-1],
            "loss": history["loss"][-1],
            "uploaded_bytes": 0,
            "downloaded_bytes": 0,
            "rounds": 0,
            "history": history,
        }

    @staticmethod
    def _variant_label(filename: str):
        if filename.startswith("iid"):
            return "fl_iid"
        if filename.startswith("dirichlet"):
            return "fl_noniid"
        if filename.startswith("label_skew"):
            return "fl_label_skew"
        return filename
