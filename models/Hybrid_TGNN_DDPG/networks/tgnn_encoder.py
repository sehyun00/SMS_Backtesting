"""
TGNN Encoder Module
시공간적 특성을 추출하는 인코더
"""

import torch
import torch.nn as nn
from .graph_layers import GraphConvLayer, TemporalAttention


class TGNNEncoder(nn.Module):
    """
    Spatiotemporal Encoder
    - TGNN 구조를 사용하여 공간적 및 시간적 특성을 추출합니다.
    """

    def __init__(self, num_features, hidden_dims=[64, 64], num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        self.input_norm = nn.BatchNorm1d(hidden_dims[0])

        # 여러 GCN 레이어 스택
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.dropout = nn.Dropout(0.1)
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)
        self.out_dim = hidden_dims[-1]

    def forward(self, features, adj):
        """
        Args:
            features: (Batch, N, T, F) - 노드별 시계열 특성
            adj: (Batch, N, N) - 인접 행렬

        Returns:
            node_embeddings: (Batch, N, D) - 노드 임베딩
        """
        batch, N, T, F = features.shape
        gcn_outputs = []

        # 각 시점(t)에 대해 GCN 적용
        for t in range(T):
            x_t = features[:, :, t, :]  # (Batch, N, F)
            h = self.input_proj(x_t)

            # Batch Norm 적용 (Batch size > 1일 때만)
            if batch > 1:
                h = h.permute(0, 2, 1)  # (B, N, D) → (B, D, N)
                h = self.input_norm(h)
                h = h.permute(0, 2, 1)  # (B, D, N) → (B, N, D)

            for gcn in self.gcn_layers:
                h = gcn(h, adj)
                h = self.dropout(h)

            gcn_outputs.append(h)

        # 시간 축을 따라 결과 스택: (Batch, T, N, D)
        temporal_features = torch.stack(gcn_outputs, dim=1)

        # 시간적 어텐션 적용
        node_embeddings = self.temporal_attn(temporal_features)

        return node_embeddings
