import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .encoders import TGNNEncoder, DDPGEncoder
from .heads import TGNNHead, DDPGHead, EnsembleHead
from .constraints import PortfolioConstraints


class HybridActor(nn.Module):
    """
    Hybrid Actor Network:
    - TGNN Path: GCN + Temporal Attention -> Weights
    - DDPG Path: MLP -> Weights
    - Ensemble: Alpha * TGNN + (1-Alpha) * DDPG
    """

    def __init__(
        self,
        num_stocks: int,
        window_size: int,
        num_features: int,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_stocks = num_stocks

        # 1. TGNN Path
        self.tgnn_encoder = TGNNEncoder(num_features, hidden_dim // 2)

        # TGNN Head Input: N * (Hidden/2)
        tgnn_flat_dim = num_stocks * (hidden_dim // 2)
        self.tgnn_head = TGNNHead(tgnn_flat_dim, hidden_dim, num_stocks)

        # 2. DDPG Path
        # State: Features (N*T*F) + Adj (N*N)
        self.state_dim_features = num_stocks * window_size * num_features
        self.state_dim_adj = num_stocks * num_stocks
        total_state_dim = self.state_dim_features + self.state_dim_adj

        self.ddpg_encoder = DDPGEncoder(total_state_dim, hidden_dim)

        self.ddpg_head = DDPGHead(hidden_dim, hidden_dim, num_stocks)

        # 3. Ensemble (Alpha)
        # Input: State + TGNN_W + DDPG_W
        ensemble_input_dim = total_state_dim + num_stocks * 2
        self.ensemble_net = EnsembleHead(ensemble_input_dim)

        # 4. Constraints
        self.constraints = PortfolioConstraints(num_stocks)

    def forward(
        self, features: torch.Tensor, adj: torch.Tensor, current_mdd: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = features.shape[0]

        # --- TGNN Path ---
        tgnn_emb = self.tgnn_encoder(features, adj)  # [B, N, H/2]
        tgnn_flat = tgnn_emb.reshape(batch, -1)
        tgnn_logits = self.tgnn_head(tgnn_flat)
        tgnn_weights = F.softmax(tgnn_logits, dim=-1)

        # --- DDPG Path ---
        features_flat = features.reshape(batch, -1)
        adj_flat = adj.reshape(batch, -1)
        state_flat = torch.cat([features_flat, adj_flat], dim=-1)

        ddpg_feat = self.ddpg_encoder(state_flat)
        ddpg_logits = self.ddpg_head(ddpg_feat)
        ddpg_weights = F.softmax(ddpg_logits, dim=-1)

        # --- Ensemble ---
        ensemble_in = torch.cat([state_flat, tgnn_weights, ddpg_weights], dim=-1)
        alpha = self.ensemble_net(ensemble_in)
        # alpha constraints (0.2 ~ 0.8)
        alpha = torch.clamp(alpha, 0.2, 0.8)

        # --- Dynamic Constraint Adjustment (MDD based) ---
        if current_mdd > 0.15:
            self.constraints.max_weight = 0.15
            self.constraints.min_weight = 0.07
        else:
            self.constraints.max_weight = 0.25  # Reset to default
            self.constraints.min_weight = 0.00

        # Mixing
        final_weights = alpha * tgnn_weights + (1 - alpha) * ddpg_weights

        # --- Constraint Enforcement ---
        final_weights = self.constraints.enforce(final_weights)

        return final_weights, alpha
