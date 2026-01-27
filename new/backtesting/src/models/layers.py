import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer.
    Z = ReLU(D^-1/2 * A * D^-1/2 * X * W)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        return F.relu(output)


class TemporalAttention(nn.Module):
    """
    Multi-head Self Attention for Temporal features.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, T, N, D] where N is number of nodes
        batch, T, N, D = x.shape
        # Permute to [batch*N, T, D] for processing time series of each node independently
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)

        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        # Take the last time step: [batch*N, D] -> [batch, N, D]
        return attn_out[:, -1, :].reshape(batch, N, D)
