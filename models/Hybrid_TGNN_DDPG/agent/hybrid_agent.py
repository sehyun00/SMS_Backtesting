"""
Hybrid DDPG Agent
TGNN과 DDPG를 결합한 강화학습 에이전트
"""

import torch
import torch.nn as nn
import numpy as np
from agent.replay_buffer import ReplayBuffer
from networks import HybridActor, HybridCritic
from utils.constraints import apply_concentration_limit


class HybridAgent:
    """
    Hybrid DDPG Agent
    - Actor와 Critic 네트워크를 관리하고 학습합니다.
    - Target Network를 사용하여 학습 안정성을 확보합니다.
    - Learning Rate Separation: Alpha Network는 10배 느린 학습률 사용
    - Concentration Limit: 단일 종목 최대 15% 비중 제한
    """

    def __init__(
        self,
        num_stocks,
        window_size,
        num_features,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        entropy_coef=0.01,
        max_concentration=0.25,  # 🔥 0.15 -> 0.25
        device="cuda",
    ):
        """
        Args:
            num_stocks: 종목 수
            window_size: 시계열 윈도우 크기
            num_features: 특성 개수
            lr_actor: Actor 학습률 (TGNN/DDPG encoder 및 head)
            lr_critic: Critic 학습률
            gamma: 할인 계수
            tau: Target network soft update 비율 (README: 0.001)
            entropy_coef: 엔트로피 정규화 계수 (README: 0.01)
            max_concentration: 단일 종목 최대 비중 (README: 0.15 = 15%)
            device: 학습 디바이스 (cuda/cpu)
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.max_concentration = max_concentration

        # 네트워크 초기화
        self.actor = HybridActor(num_stocks, window_size, num_features).to(device)
        self.critic = HybridCritic(
            num_stocks, window_size, num_features, num_stocks
        ).to(device)

        # Target 네트워크 초기화
        self.actor_target = HybridActor(num_stocks, window_size, num_features).to(
            device
        )
        self.critic_target = HybridCritic(
            num_stocks, window_size, num_features, num_stocks
        ).to(device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # ===== Actor Optimizer: Learning Rate Separation =====
        # Strategy:
        # - TGNN/DDPG encoders & heads: Standard lr_actor (1e-4)
        # - Alpha Network (ensemble_weight_net): 5x faster (5e-5) 🔥 10x → 5x 변경
        #
        # Rationale:
        # - Alpha network controls ensemble weighting between TGNN and DDPG
        # - Faster learning allows dynamic strategy switching 🔥
        # - Allows TGNN/DDPG to stabilize before alpha adjusts weighting
        self.actor_optimizer = torch.optim.Adam(
            [
                {
                    "params": self.actor.tgnn_encoder.parameters(),
                    "lr": lr_actor,
                    "name": "tgnn_encoder",
                },
                {
                    "params": self.actor.ddpg_encoder.parameters(),
                    "lr": lr_actor,
                    "name": "ddpg_encoder",
                },
                {
                    "params": self.actor.tgnn_head.parameters(),
                    "lr": lr_actor,
                    "name": "tgnn_head",
                },
                {
                    "params": self.actor.ddpg_head.parameters(),
                    "lr": lr_actor,
                    "name": "ddpg_head",
                },
                {
                    "params": self.actor.ensemble_weight_net.parameters(),
                    "lr": lr_actor * 0.5,  # 🔥 0.1 -> 0.5 (5배 증가, 5e-5)
                    "name": "alpha_network",
                },
            ]
        )

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_std=0.1):
        """
        행동 선택 (Dirichlet 노이즈 + 집중도 제한)

        Args:
            state: 현재 상태
            noise_std: 탐색용 노이즈 강도 (Dirichlet concentration 조절)

        Returns:
            action: 선택된 행동 (numpy array, sum=1, max_weight<=0.15 보장)
            alpha_value: 앵상블 가중치 값
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Actor 이제 5개 값 반환: weights, alpha, tgnn_weights, ddpg_weights, entropy
            final_weights, alpha, tgnn_weights, ddpg_weights, entropy = self.actor(
                state
            )
            action = final_weights.cpu().numpy()[0]
            alpha_value = alpha.cpu().item()

        if noise_std > 0:
            # ===== Dirichlet Noise (개선된 탐색) =====
            # 기존 Gaussian 문제점:
            # - noise 추가 후 재정규화하면 원래 분포가 왜곡됨
            # - sum=1 제약 유지를 위한 추가 처리 필요
            #
            # Dirichlet 장점:
            # - 자동으로 sum=1 유지 (probability simplex)
            # - 재정규화 불필요
            # - 포트폴리오 탐색에 더 자연스러움

            # Concentration 파라미터 계산
            # noise_std가 클수록 더 균등한 분포로 탐색
            concentration = action / (noise_std + 1e-8)
            concentration = np.clip(concentration, 0.1, 100.0)

            # Dirichlet 샘플링
            action = np.random.dirichlet(concentration)

        # ===== 집중도 제한 적용 =====
        # 단일 종목 최대 15% 비중 제한 (README 기준)
        # 과도한 집중을 방지하여 리스크 분산
        action = apply_concentration_limit(action, max_weight=self.max_concentration)

        return action, alpha_value

    def train(self, batch_size=64):
        """
        에이전트 학습 (Gradient Clipping + Entropy Regularization)

        Args:
            batch_size: 미니배치 크기
        """
        if len(self.replay_buffer) < batch_size:
            return

        # 배치 샘플링
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # NaN 체크 및 제거
        if torch.isnan(states).any() or torch.isinf(states).any():
            print("⚠️  NaN/Inf detected in states, skipping batch")
            return

        if torch.isnan(rewards).any() or torch.isinf(rewards).any():
            print("⚠️  NaN/Inf detected in rewards, skipping batch")
            return

        # ----------------------
        # Critic 업데이트
        # ----------------------
        with torch.no_grad():
            next_actions, _, _, _, _ = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions)

        # NaN 체크
        if torch.isnan(current_q).any() or torch.isnan(target_q).any():
            print("⚠️  NaN detected in Q-values, skipping batch")
            return

        critic_loss = nn.MSELoss()(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)

        self.critic_optimizer.step()

        # ----------------------
        # Actor 업데이트 (Entropy Regularization 포함)
        # ----------------------
        pred_actions, _, _, _, pred_entropy = self.actor(states)

        # Q-value 기반 손실
        actor_loss = -self.critic(states, pred_actions).mean()

        # Entropy 보너스 (다양성 장려)
        # Entropy가 높을수록 손실 감소 (보너스 효과)
        entropy_bonus = self.entropy_coef * pred_entropy.mean()

        # 최종 Actor 손실
        total_actor_loss = actor_loss - entropy_bonus

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()
        # Note: 단일 optimizer 내에서 learning rate 분리
        # - tgnn/ddpg: lr_actor (1e-4)
        # - alpha_network: lr_actor * 0.1 (1e-5)

        # ----------------------
        # Target Network Soft Update
        # ----------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source, target):
        """
        Target 네트워크를 천천히 업데이트 (Polyak Averaging)

        Args:
            source: 소스 네트워크
            target: 타겟 네트워크
        """
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
