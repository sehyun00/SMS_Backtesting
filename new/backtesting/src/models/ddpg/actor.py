import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
from .encoder import SharedFactorEncoder


class Actor(nn.Module):
    """
    Base DDPG Actor: State -> Action (Portfolio Weights)
    """

    def __init__(self, num_stocks: int, num_features: int, hidden_dim: int = 128):
        super().__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features

        self.encoder = SharedFactorEncoder(num_features, 64)

        input_dim = num_stocks * 64
        self.global_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks),
        )

        # Constraints
        self.min_weight = 0.02
        self.max_weight = 0.30

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


class DDPGActor(Actor):
    # Wrapper to handle input shape adaptation
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [Batch, N, T, F_dim]
        # We flatten T*F implicit in usage
        batch, N, T, feat_dim = x.shape

        x_flat = x.reshape(batch, N, -1)  # [Batch, N, T*F]

        enc = self.encoder(x_flat)  # [Batch, N, 64]
        enc_flat = enc.reshape(batch, -1)
        scores = self.global_net(enc_flat)
        weights = F.softmax(scores, dim=-1)

        # Constraints
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)

        return weights, entropy
