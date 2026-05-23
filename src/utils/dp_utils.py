import torch
from opacus import PrivacyEngine
from torch.utils.data import DataLoader
from typing import Optional, Tuple


def attach_privacy_engine(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    criterion: Optional[torch.nn.Module] = None,
    noise_multiplier: float = 1.1,
    max_grad_norm: float = 1.0,
    target_epsilon: Optional[float] = None,
    target_delta: Optional[float] = None,
) -> Tuple[torch.nn.Module, torch.optim.Optimizer, DataLoader, PrivacyEngine]:
    """Wrap the training objects with Opacus DP support."""
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    privacy_engine = PrivacyEngine()
    private_model, private_optimizer, private_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        criterion=criterion,
        data_loader=train_loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        target_epsilon=target_epsilon,
        target_delta=target_delta,
    )
    return private_model, private_optimizer, private_loader, privacy_engine
