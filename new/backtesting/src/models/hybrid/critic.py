import torch
import torch.nn as nn
from src.models.layers import SharedFactorEncoder, GlobalPoolHead


class HybridCritic(nn.Module):
    """
    Hybrid Critic Network (Asset-Agnostic):
    - Uses Deep Sets architecture to evaluate State + Action pair for variable N stocks.
    - Input: Portfolio State (N stocks) + Portfolio Weights (N weights)
    - Output: Q-Value (Scalar)
    """

    def __init__(
        self,
        num_features: int,
        window_size: int,
        hidden_dim: int = 128,
        **kwargs,  # Ignore unused args like num_stocks
    ):
        super().__init__()

        # 1. State Encoder (Shared)
        # Input: T*F per stock -> Hidden
        self.input_dim = num_features * window_size
        self.state_encoder = SharedFactorEncoder(self.input_dim, hidden_dim)

        # 2. Global Pooling Head (Deep Sets)
        # Input to Pooler: Hidden (State) + 1 (Action)
        # Local -> Global -> Q-Value (Scalar)
        self.pool_net = GlobalPoolHead(
            input_dim=hidden_dim + 1, hidden_dim=hidden_dim, output_dim=1
        )

    def forward(
        self, features: torch.Tensor, actions: torch.Tensor, adj: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            features: [Batch, N, T, F]
            actions: [Batch, N] (Portfolio Weights)
            adj: [Batch, N, N] (Unused in Asset-Agnostic Critic, kept for API compatibility)
        """
        batch = features.shape[0]
        if features.dim() == 4:
            N = features.shape[1]
        else:
            N = 1  # Fallback, though usually 4D

        # 1. Encode States
        # Flatten time: [Batch, N, T, F] -> [Batch, N, T*F]
        if features.dim() == 4:
            features_flat = features.reshape(batch, N, -1)
        else:
            features_flat = features

        state_emb = self.state_encoder(features_flat)  # [Batch, N, H]

        # 2. Concat Action
        # actions: [Batch, N] -> [Batch, N, 1]
        if actions.dim() == 2:
            actions_exp = actions.unsqueeze(-1)
        else:
            actions_exp = actions

        # [Batch, N, H+1]
        state_action = torch.cat([state_emb, actions_exp], dim=-1)

        # 3. Global Pooling & Prediction -> [Batch, 1]
        q_value = self.pool_net(state_action)

        return q_value
