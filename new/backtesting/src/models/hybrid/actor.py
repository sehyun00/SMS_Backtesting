import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from .encoders import TGNNEncoder
from src.models.layers import (
    SharedFactorEncoder,
    ScoreHead,
    GlobalPoolHead,
    PortfolioSoftmax,
)
from .constraints import PortfolioConstraints


class HybridActor(nn.Module):
    """
    Hybrid Actor Network (Asset-Agnostic):
    - TGNN Path: GCN + Temporal Attention -> Score -> Weights
    - DDPG Path: Shared Encoder -> Score -> Weights
    - Ensemble: Global Pooling -> Alpha -> Mixing
    """

    def __init__(
        self,
        num_stocks: int,  # Kept for interface compatibility, but unused for sizing
        window_size: int,
        num_features: int,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        # self.num_stocks = num_stocks # Removed dependency

        # 1. TGNN Path (Graph-based, naturally handles variable N)
        self.tgnn_encoder = TGNNEncoder(num_features, hidden_dim // 2)

        # TGNN Head: (Hidden/2) -> Score (Scalar)
        # Shared weights across all stocks
        self.tgnn_head = ScoreHead(hidden_dim // 2, hidden_dim)

        # 2. DDPG Path (Asset-Agnostic)
        # Shared Encoder: Features -> Hidden
        self.ddpg_encoder = SharedFactorEncoder(num_features, hidden_dim)

        # DDPG Head: Hidden -> Score (Scalar)
        self.ddpg_head = ScoreHead(hidden_dim, hidden_dim)

        # Softmax Layer with Temperature
        self.softmax_layer = PortfolioSoftmax(temperature)

        # 3. Ensemble (Alpha)
        # Input to Pooler:
        #   Feature State (DDPG Enc: Hidden)
        #   + TGNN Score (1)
        #   + DDPG Score (1)
        #   = Hidden + 2
        ensemble_input_dim = hidden_dim + 2

        # Global Pooling -> Scalar Alpha
        self.ensemble_net = GlobalPoolHead(ensemble_input_dim, hidden_dim, output_dim=1)

        # 4. Constraints (Stateless)
        self.constraints = PortfolioConstraints()

    def forward(
        self, features: torch.Tensor, adj: torch.Tensor, current_mdd: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [Batch, N, T, F]
            adj: [Batch, N, N]
            current_mdd: float
        """
        batch = features.shape[0]
        N = features.shape[1]

        # --- TGNN Path ---
        # Encoder: [B, N, T, F] -> [B, N, H/2]
        tgnn_emb = self.tgnn_encoder(features, adj)
        # Head: [B, N, H/2] -> [B, N, 1]
        tgnn_scores = self.tgnn_head(tgnn_emb).squeeze(-1)  # [B, N]
        tgnn_weights = self.softmax_layer(tgnn_scores)

        # --- DDPG Path ---
        # Flatten time: [B, N, T, F] -> [B, N, T*F]
        if features.dim() == 4:
            features_flat = features.reshape(batch, N, -1)
        else:
            features_flat = features

        # Encoder: [B, N, T*F] -> [B, N, H]
        ddpg_emb = self.ddpg_encoder(features_flat)
        # Head: [B, N, H] -> [B, N, 1]
        ddpg_scores = self.ddpg_head(ddpg_emb).squeeze(-1)  # [B, N]
        ddpg_weights = self.softmax_layer(ddpg_scores)

        # --- Ensemble ---
        # Combine State + Scores for Alpha calculation
        # [B, N, H], [B, N, 1], [B, N, 1] -> [B, N, H+2]
        ensemble_in = torch.cat(
            [ddpg_emb, tgnn_scores.unsqueeze(-1), ddpg_scores.unsqueeze(-1)], dim=-1
        )

        # Global Pooling -> [B, 1] (Alpha)
        raw_alpha = self.ensemble_net(ensemble_in)
        alpha = torch.sigmoid(raw_alpha)  # Sigmoid for 0~1 range

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
        # [B, 1] * [B, N] + [B, 1] * [B, N] -> [B, N]
        final_weights = alpha * tgnn_weights + (1 - alpha) * ddpg_weights

        # --- Constraint Enforcement ---
        final_weights = self.constraints.enforce(final_weights)

        return final_weights, alpha
