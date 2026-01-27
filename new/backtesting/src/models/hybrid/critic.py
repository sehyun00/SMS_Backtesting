import torch
import torch.nn as nn


class HybridCritic(nn.Module):
    """
    Critic Network for DDPG:
    Input: State + Action (Portfolio Weights)
    Output: Q-Value (Scalar)
    """

    def __init__(
        self, num_stocks: int, window_size: int, num_features: int, action_dim: int
    ):
        super().__init__()

        state_dim = num_stocks * window_size * num_features + num_stocks * num_stocks
        input_dim = state_dim + action_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        # State: Flat vector of Features + Adj
        x = torch.cat([state, action], dim=-1)
        return self.net(x)
