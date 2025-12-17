"""
Graph Neural Network Layers for TGNN
그래프 컨볼루션과 시간적 어텐션 레이어
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer
    - 인접 행렬을 사용하여 이웃 노드로부터 정보를 집계합니다.
    - 공식: Output = Activation(Normalized_Adj * X * W)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, N, In_F) - 노드 특성 행렬
            adj: (Batch, N, N) - 인접 행렬

        Returns:
            output: (Batch, N, Out_F) - 변환된 노드 특성
        """
        # Degree Matrix 계산 및 정규화 준비
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        # 인접 행렬 정규화: D^(-1/2) * A * D^(-1/2)
        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)

        # 선형 변환 및 정보 전파
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        return F.relu(output)


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism
    - 시계열 데이터에서 중요한 시점에 가중치를 부여합니다.
    - Multi-head Attention을 사용하여 시간 축의 중요도를 학습합니다.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, T, N, D) - 시간별 노드 특성

        Returns:
            output: (Batch, N, D) - 마지막 시점의 어텐션 적용 특성
        """
        batch, T, N, D = x.shape

        # 각 노드에 대해 시간 축을 따라 Attention 적용
        # (Batch * N, T, D) 형태로 변환
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)

        # Self-Attention 수행
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)

        # 마지막 시점의 어텐션 적용된 특성 추출 및 원래 배치 구조로 복원
        # (Batch, N, D)
        return attn_out[:, -1, :].reshape(batch, N, D)
