"""Unified forecaster: model registry and predictability-weighted loss."""
import torch
import torch.nn as nn
from .s4 import S4Model
from .mamba import MambaModel
from .linear import DLinear, NLinear
from .patchtst import PatchTST

MODEL_REGISTRY = {
    "s4": S4Model,
    "mamba": MambaModel,
    "dlinear": DLinear,
    "nlinear": NLinear,
    "patchtst": PatchTST,
}


class PredictabilityWeightedLoss(nn.Module):
    """MSE loss re-weighted by local predictability score.

    Windows with high predictability get weight alpha, low predictability
    get weight beta. A curriculum ramp (first 20% of training is uniform)
    prevents the model from ignoring noisy windows entirely at the start.
    """
    def __init__(self, alpha: float = 1.0, beta: float = 0.3, curriculum: bool = True):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.curriculum = curriculum
        self.current_epoch = 0
        self.total_epochs = 100

    def set_epoch(self, epoch: int, total_epochs: int):
        self.current_epoch = epoch
        self.total_epochs = total_epochs

    def forward(self, pred, target, pred_scores):
        mse = ((pred - target) ** 2).mean(dim=(1, 2))
        weights = self.beta + (self.alpha - self.beta) * pred_scores

        if self.curriculum:
            progress = self.current_epoch / max(self.total_epochs, 1)
            blend = min(progress / 0.2, 1.0)
            weights = (1 - blend) * torch.ones_like(weights) + blend * weights

        weights = weights / (weights.mean() + 1e-8)
        return (mse * weights).mean()


class Forecaster(nn.Module):
    def __init__(self, model_name: str, n_features: int, config: dict):
        super().__init__()
        model_cls = MODEL_REGISTRY[model_name.lower()]
        kwargs = {
            "n_features": n_features,
            "seq_len": config.get("seq_len", 336),
            "pred_len": config.get("pred_len", 96),
            "d_model": config.get("d_model", 128),
            "n_layers": config.get("n_layers", 4),
            "dropout": config.get("dropout", 0.1),
        }
        if model_name.lower() == "s4":
            kwargs["d_state"] = config.get("s4_d_state", 64)
        elif model_name.lower() == "mamba":
            kwargs["d_state"] = config.get("mamba_d_state", 16)
            kwargs["d_conv"] = config.get("mamba_d_conv", 4)
            kwargs["expand"] = config.get("mamba_expand", 2)
        elif model_name.lower() == "patchtst":
            kwargs["n_heads"] = config.get("n_heads", 8)
            kwargs["patch_len"] = config.get("patch_len", 16)
            kwargs["stride"] = config.get("stride", 8)

        self.model = model_cls(**kwargs)
        self.model_name = model_name

    def forward(self, x):
        return self.model(x)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
