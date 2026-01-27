import torch
import torch.nn as nn
from .encoder import SharedFactorEncoder


class Critic(nn.Module):
    def __init__(
        self, num_stocks: int, num_features: int, action_dim: int, hidden_dim: int = 128
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.encoder = SharedFactorEncoder(num_features, 64)

        input_dim = (num_stocks * 64) + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # state: [Batch, N, T, F] OR [Batch, Flat]
        if state.dim() == 2:
            batch = state.shape[0]
            # Reshape to [Batch, N, Effectve_F]
            # We know Num Stocks.
            x = state.reshape(batch, self.num_stocks, -1)
        else:
            batch, N, T, feat_dim = state.shape
            x = state.reshape(batch, N, -1)

        enc = self.encoder(x)  # [Batch, N, 64]
        enc_flat = enc.reshape(batch, -1)

        xa = torch.cat([enc_flat, action], dim=-1)
        return self.net(xa)
