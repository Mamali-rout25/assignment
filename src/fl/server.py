import numpy as np
import torch
from typing import Dict, List, Tuple
from src.fl.client import FlowerClient
from src.models.cnn import CNN
from src.models.mlp import MLP
from src.data.partitions import load_partitions


def get_model_for_dataset(dataset_name: str, input_dim: int = None, num_classes: int = 2):
    if dataset_name == "mnist":
        return CNN(num_classes=num_classes)
    return MLP(input_dim=input_dim, num_classes=num_classes)


def get_evaluate_fn(model, x_val, y_val, device):
    def evaluate(parameters):
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        model.load_state_dict(state_dict, strict=True)
        model.to(device)
        model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        x_tensor = torch.tensor(x_val, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
        with torch.no_grad():
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor).item()
            preds = outputs.argmax(dim=1)
            accuracy = (preds == y_tensor).float().mean().item()
        return float(loss), {"accuracy": float(accuracy)}

    return evaluate


def _get_parameters(model: torch.nn.Module) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _parameters_size(parameters: List[np.ndarray]) -> int:
    return sum(arr.nbytes for arr in parameters)


def _aggregate_fit_results(results: List[Tuple[List[np.ndarray], int]]) -> List[np.ndarray]:
    total_examples = sum(num_examples for _, num_examples in results)
    if total_examples == 0:
        return results[0][0]

    averaged = []
    num_params = len(results[0][0])
    for idx in range(num_params):
        weighted_sum = np.zeros_like(results[0][0][idx])
        for params, num_examples in results:
            weighted_sum += params[idx] * num_examples
        averaged.append(weighted_sum / total_examples)
    return averaged


def create_client_fn(
    dataset_name: str,
    partitions_path: str,
    x_train_full,
    y_train_full,
    x_val_full,
    y_val_full,
    input_dim: int,
    num_classes: int,
    device: torch.device,
    local_epochs: int,
    batch_size: int,
    lr: float,
    use_dp: bool,
    dp_target_epsilon: float,
    dp_target_delta: float,
):
    partitions = load_partitions(partitions_path)

    def client_fn(cid: str):
        index = int(cid)
        client_key = f"client_{index}"
        indices = partitions[client_key]
        x_client = x_train_full[indices]
        y_client = y_train_full[indices]

        model = get_model_for_dataset(dataset_name, input_dim=input_dim, num_classes=num_classes)
        if dataset_name == "mnist":
            x_client = x_client.reshape(-1, 1, 28, 28)
            x_val = x_val_full.reshape(-1, 1, 28, 28)
        else:
            x_val = x_val_full
        y_val = y_val_full

        return FlowerClient(
            cid=cid,
            model=model,
            x_train=x_client,
            y_train=y_client,
            x_val=x_val,
            y_val=y_val,
            device=device,
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr,
            use_dp=use_dp,
            dp_target_epsilon=dp_target_epsilon,
            dp_target_delta=dp_target_delta,
        )

    return client_fn


def run_federated_experiment(
    dataset_name: str,
    x_train,
    y_train,
    x_test,
    y_test,
    partitions_path: str,
    num_clients: int = 4,
    num_rounds: int = 5,
    local_epochs: int = 1,
    batch_size: int = 32,
    lr: float = 1e-3,
    use_dp: bool = False,
    dp_target_epsilon: float = 8.0,
    dp_target_delta: float = 1e-5,
    seed: int = 42,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 if dataset_name == "mnist" else len(np.unique(y_train))
    model = get_model_for_dataset(dataset_name, input_dim=x_train.shape[1] if dataset_name != "mnist" else None, num_classes=num_classes)
    evaluate_fn = get_evaluate_fn(model, x_test.reshape(-1, 1, 28, 28) if dataset_name == "mnist" else x_test, y_test, device)

    client_fn = create_client_fn(
        dataset_name=dataset_name,
        partitions_path=partitions_path,
        x_train_full=x_train,
        y_train_full=y_train,
        x_val_full=x_test,
        y_val_full=y_test,
        input_dim=x_train.shape[1] if dataset_name != "mnist" else None,
        num_classes=num_classes,
        device=device,
        local_epochs=local_epochs,
        batch_size=batch_size,
        lr=lr,
        use_dp=use_dp,
        dp_target_epsilon=dp_target_epsilon,
        dp_target_delta=dp_target_delta,
    )

    clients = [client_fn(str(i)) for i in range(num_clients)]
    parameters = _get_parameters(model)
    downloaded_bytes = 0
    uploaded_bytes = 0
    rounds = 0
    history = {"accuracy": [], "loss": []}

    for _ in range(num_rounds):
        round_results: List[Tuple[List[np.ndarray], int]] = []
        for client in clients:
            downloaded_bytes += _parameters_size(parameters)
            client_params, num_examples, _ = client.fit(parameters, config={})
            uploaded_bytes += _parameters_size(client_params)
            round_results.append((client_params, num_examples))

        parameters = _aggregate_fit_results(round_results)
        rounds += 1

        loss, metrics = evaluate_fn(parameters)
        history["loss"].append(loss)
        history["accuracy"].append(metrics.get("accuracy", 0.0))

    return {
        "accuracy": history["accuracy"][-1] if len(history["accuracy"]) > 0 else 0.0,
        "loss": history["loss"][-1] if len(history["loss"]) > 0 else 0.0,
        "downloaded_bytes": downloaded_bytes,
        "uploaded_bytes": uploaded_bytes,
        "rounds": rounds,
        "history": history,
    }
if __name__ == "__main__":

    import flwr as fl

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        min_fit_clients=5,
        min_available_clients=5,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )