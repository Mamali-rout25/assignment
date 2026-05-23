import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=None, num_classes: int = 2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]
        layers = []
        dims = [input_dim] + hidden_dims
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU(inplace=False))
            layers.append(nn.Dropout(p=0.2))
        layers.append(nn.Linear(dims[-1], num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)
