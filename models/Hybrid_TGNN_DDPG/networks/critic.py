"""
Hybrid Critic Network
Q-Value를 평가하는 Critic 네트워크
"""

import torch
import torch.nn as nn
from .tgnn_encoder import TGNNEncoder


class HybridCritic(nn.Module):
    """
    Hybrid Critic Network
    - Actor의 이중 경로 구조에 맞춰 Q-Value를 평가합니다.
    """

    def __init__(
        self, num_stocks, window_size, num_features, action_dim, hidden_dim=128
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features

        # TGNN 특성 추출기
        self.tgnn_encoder = TGNNEncoder(num_features)

        # DDPG 특성 추출기
        state_dim = num_stocks * window_size * num_features + num_stocks * num_stocks
        self.ddpg_encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, hidden_dim)
        )

        # Q-Value 예측 헤드
        # 입력: TGNN 임베딩 + DDPG 임베딩 + Action
        tgnn_emb_dim = num_stocks * self.tgnn_encoder.out_dim
        input_dim = tgnn_emb_dim + hidden_dim + action_dim

        self.q_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        """
        Args:
            state: (Batch, state_dim) - 상태 벡터
            action: (Batch, action_dim) - 행동 벡터

        Returns:
            q_value: (Batch, 1) - Q-Value
        """
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features

        # 상태 복원
        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]

        features = features_flat.reshape(
            batch, self.num_stocks, self.window_size, self.num_features
        )
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)

        # 양쪽 경로에서 특성 추출
        tgnn_embeddings = self.tgnn_encoder(features, adj)
        tgnn_embeddings_flat = tgnn_embeddings.reshape(batch, -1)

        ddpg_embeddings = self.ddpg_encoder(state)

        # Q-Value 계산 (양쪽 경로의 특성 + 행동 결합)
        qa = torch.cat([tgnn_embeddings_flat, ddpg_embeddings, action], dim=-1)
        q_value = self.q_net(qa)

        return q_value
