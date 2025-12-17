"""
Hybrid Actor Network
TGNN과 DDPG 경로를 결합한 Actor 네트워크
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .tgnn_encoder import TGNNEncoder


class HybridActor(nn.Module):
    """
    Hybrid Actor: TGNN + DDPG 경로를 동적으로 결합
    - 제약 조건을 엄격히 적용하며 학습합니다.
    """

    def __init__(self, num_stocks, window_size, num_features, hidden_dim=128):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features

        # 제약 파라미터
        self.MIN_WEIGHT = 0.05  # 5% 최소 비중
        self.MAX_WEIGHT = 0.20  # 20% 최대 비중

        # 동적 제약을 위한 기본값 저장
        self.BASE_MIN_WEIGHT = 0.05
        self.BASE_MAX_WEIGHT = 0.20
        self.current_mdd = 0.0

        # TGNN 경로
        self.tgnn_encoder = TGNNEncoder(num_features)
        tgnn_input_dim = num_stocks * self.tgnn_encoder.out_dim

        self.tgnn_head = nn.Sequential(
            nn.Linear(tgnn_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_stocks),
        )

        # DDPG 경로
        state_dim = num_stocks * window_size * num_features + num_stocks * num_stocks

        self.ddpg_encoder = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        self.ddpg_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_stocks),
        )

        # 앙상블 가중치 네트워크
        ensemble_input_dim = state_dim + num_stocks * 2

        self.ensemble_weight_net = nn.Sequential(
            nn.Linear(ensemble_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        # Weight 초기화 (Xavier)
        self._initialize_weights()

    def _initialize_weights(self):
        """네트워크 가중치 초기화"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _enforce_constraints(self, weights, max_iter=10):
        """
        반복적 투영으로 제약 조건 강제

        보장사항:
        1. MIN_WEIGHT <= w_i <= MAX_WEIGHT (모든 종목)
        2. sum(w) = 1.0
        3. 정규화를 통한 우회 방지

        알고리즘:
        - 제약을 위반하는 가중치를 반복적으로 조정
        - 초과/부족분을 실행 가능한 종목에 재분배
        - 유효한 해로 수렴
        """
        MIN_W = self.MIN_WEIGHT
        MAX_W = self.MAX_WEIGHT
        eps = 1e-4  # 수렴 허용 오차

        for iteration in range(max_iter):
            # Step 1: [MIN_W, MAX_W]로 클램핑
            weights_clamped = torch.clamp(weights, MIN_W, MAX_W)

            # Step 2: 현재 합계 확인
            current_sum = weights_clamped.sum(dim=-1, keepdim=True)

            # Step 3: 합계가 1.0에 가까우면 완료
            if torch.allclose(current_sum, torch.ones_like(current_sum), atol=eps):
                weights = weights_clamped
                break

            # Step 4: 초과/부족분 재분배
            deficit = 1.0 - current_sum  # (Batch, 1)

            # 증가/감소 필요 여부 확인
            need_increase = deficit > 0  # (Batch, 1) boolean

            # 조정 가능한 종목 찾기
            room_to_grow = MAX_W - weights_clamped  # (Batch, N)
            room_to_shrink = weights_clamped - MIN_W  # (Batch, N)

            # 증가용: 성장 여지가 있는 종목에 분배
            total_room_grow = room_to_grow.sum(dim=-1, keepdim=True)  # (Batch, 1)
            adjustment_grow = torch.where(
                total_room_grow > eps,
                deficit * (room_to_grow / (total_room_grow + 1e-8)),
                deficit / self.num_stocks,
            )

            # 감소용: 축소 여지가 있는 종목에서 차감
            total_room_shrink = room_to_shrink.sum(dim=-1, keepdim=True)  # (Batch, 1)
            adjustment_shrink = torch.where(
                total_room_shrink > eps,
                deficit * (room_to_shrink / (total_room_shrink + 1e-8)),
                deficit / self.num_stocks,
            )

            # need_increase에 따라 적절한 조정 적용
            adjustment = torch.where(need_increase, adjustment_grow, adjustment_shrink)

            weights = weights_clamped + adjustment

        # 최종 안전 장치: 클램핑 및 정규화
        weights = torch.clamp(weights, MIN_W, MAX_W)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights

    def forward(self, state):
        """
        Args:
            state: (Batch, state_dim) - 상태 벡터

        Returns:
            final_weights: (Batch, N) - 최종 포트폴리오 가중치
            alpha: (Batch,) - 앙상블 가중치
            tgnn_weights: (Batch, N) - TGNN 경로 가중치
            ddpg_weights: (Batch, N) - DDPG 경로 가중치
        """
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features

        # MDD 기반 동적 제약 조정
        if self.current_mdd > 0.15:
            self.MAX_WEIGHT = 0.15
            self.MIN_WEIGHT = 0.07
            temperature = 8.0
        elif self.current_mdd > 0.10:
            self.MAX_WEIGHT = 0.18
            self.MIN_WEIGHT = 0.06
            temperature = 6.0
        else:
            self.MAX_WEIGHT = self.BASE_MAX_WEIGHT
            self.MIN_WEIGHT = self.BASE_MIN_WEIGHT
            temperature = 5.0

        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]
        features = features_flat.reshape(
            batch, self.num_stocks, self.window_size, self.num_features
        )
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)

        # TGNN 경로
        tgnn_embeddings = self.tgnn_encoder(features, adj)
        tgnn_embeddings_flat = tgnn_embeddings.reshape(batch, -1)
        tgnn_logits = self.tgnn_head(tgnn_embeddings_flat)
        tgnn_weights = F.softmax(tgnn_logits / temperature, dim=-1)

        # DDPG 경로
        ddpg_features = self.ddpg_encoder(state)
        ddpg_logits = self.ddpg_head(ddpg_features)
        ddpg_weights = F.softmax(ddpg_logits / temperature, dim=-1)

        # 앙상블
        ensemble_input = torch.cat([state, tgnn_weights, ddpg_weights], dim=-1)
        alpha_raw = self.ensemble_weight_net(ensemble_input)

        # Alpha 범위 제한
        alpha = torch.clamp(alpha_raw, min=0.2, max=0.8)

        # 최종 결합
        final_weights = alpha * tgnn_weights + (1 - alpha) * ddpg_weights
        final_weights = self._enforce_constraints(final_weights)

        return final_weights, alpha.squeeze(-1), tgnn_weights, ddpg_weights
