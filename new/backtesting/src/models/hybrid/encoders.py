import torch
import torch.nn as nn
from src.models.layers import GraphConvLayer, TemporalAttention


class TGNNEncoder(nn.Module):
    """
    Encoder part of TGNN. Returns node embeddings.
    """

    def __init__(self, num_features: int, hidden_dim: int = 64, num_heads: int = 4):
        super().__init__()
        self.HIDDEN_DIM = hidden_dim
        self.input_proj = nn.Linear(num_features, hidden_dim)

        # 2 GCN Layers
        self.gcn1 = GraphConvLayer(hidden_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.gcn2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

        self.temporal_attn = TemporalAttention(hidden_dim, num_heads)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        batch, N, T, F = features.shape
        gcn_outputs = []
        for t in range(T):
            x_t = features[:, :, t, :]
            h = self.input_proj(x_t)

            # GCN 1
            h1 = self.gcn1(h, adj)
            h1 = self.ln1(h1)
            h = h + h1

            # GCN 2
            h2 = self.gcn2(h, adj)
            h2 = self.ln2(h2)
            h = h + h2

            gcn_outputs.append(self.dropout(h))

        # Stack time steps: [batch, N, T, D] -> Permute handled in Attn if needed
        # TemporalAttention expects [batch, T, N, D]
        temporal_features = torch.stack(gcn_outputs, dim=2).permute(0, 2, 1, 3)
        node_embeddings = self.temporal_attn(temporal_features)
        return node_embeddings


class DDPGEncoder(nn.Module):
    """
    Encoder part of DDPG.
    Processes flat state (features + adj) into a hidden representation.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
