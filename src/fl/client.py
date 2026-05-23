import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import flwr as fl

from src.utils.dp_utils import attach_privacy_engine
from src.utils.seed import seed_everything


class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        cid: str,
        model: nn.Module,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        device: torch.device,
        batch_size: int = 32,
        local_epochs: int = 1,
        lr: float = 1e-3,
        use_dp: bool = False,
        dp_target_epsilon: float = 8.0,
        dp_target_delta: float = 1e-5,
    ):
        seed_everything(int(cid) + 123)
        self.cid = cid
        self.device = device
        self.model = model.to(device)
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.use_dp = use_dp
        self.dp_target_epsilon = dp_target_epsilon
        self.dp_target_delta = dp_target_delta
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()
        self.train_loader = DataLoader(
            TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.tensor(x_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long)),
            batch_size=self.batch_size,
            shuffle=False,
        )
        self.privacy_engine = None

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        if self.use_dp:
            self._attach_privacy_engine()

        self.model.train()
        for _ in range(self.local_epochs):
            for x_batch, y_batch in self.train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()

        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for x_batch, y_batch in self.val_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = self.model(x_batch)
                loss += self.criterion(logits, y_batch).item() * x_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += x_batch.size(0)
        loss /= total
        accuracy = correct / total if total > 0 else 0.0
        return float(loss), total, {"accuracy": float(accuracy)}

    def _attach_privacy_engine(self):
        if self.privacy_engine is not None:
            return
        self.model, self.optimizer, self.train_loader, self.privacy_engine = attach_privacy_engine(
            model=self.model,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            criterion=self.criterion,
            noise_multiplier=1.1,
            max_grad_norm=1.0,
            target_epsilon=self.dp_target_epsilon,
            target_delta=self.dp_target_delta,
        )

if __name__ == "__main__":

    print("Client module loaded successfully")