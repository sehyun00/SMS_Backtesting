"""
Hybrid DDPG Agent
TGNN과 DDPG를 결합한 강화학습 에이전트
"""

import torch
import torch.nn as nn
import numpy as np
from agent.replay_buffer import ReplayBuffer
from networks import HybridActor, HybridCritic


class HybridAgent:
    """
    Hybrid DDPG Agent
    - Actor와 Critic 네트워크를 관리하고 학습합니다.
    - Target Network를 사용하여 학습 안정성을 확보합니다.
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
        device="cuda",
    ):
        """
        Args:
            num_stocks: 종목 수
            window_size: 시계열 윈도우 크기
            num_features: 특성 개수
            lr_actor: Actor 학습률
            lr_critic: Critic 학습률
            gamma: 할인 계수
            tau: Target network soft update 비율
            entropy_coef: 엔트로피 정규화 계수
            device: 학습 디바이스 (cuda/cpu)
        """
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef

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

        # Alpha Network 별도 학습률 (10배 느리게)
        self.actor_optimizer = torch.optim.Adam(
            [
                {"params": self.actor.tgnn_encoder.parameters(), "lr": lr_actor},
                {"params": self.actor.ddpg_encoder.parameters(), "lr": lr_actor},
                {"params": self.actor.tgnn_head.parameters(), "lr": lr_actor},
                {"params": self.actor.ddpg_head.parameters(), "lr": lr_actor},
                {
                    "params": self.actor.ensemble_weight_net.parameters(),
                    "lr": lr_actor * 0.1,
                },
            ]
        )

        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_std=0.1):
        """
        행동 선택 (탐색 노이즈 포함)

        Args:
            state: 현재 상태
            noise_std: 탐색용 노이즈 표준편차

        Returns:
            action: 선택된 행동 (numpy array)
            alpha_value: 앙상블 가중치 값
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_weights, alpha, tgnn_weights, ddpg_weights = self.actor(state)
            action = final_weights.cpu().numpy()[0]
            alpha_value = alpha.cpu().item()

        if noise_std > 0:
            # 탐색 노이즈 추가
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)

        return action, alpha_value

    def train(self, batch_size=64):
        """
        에이전트 학습 (Gradient Clipping 포함)

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
            next_actions, _, _, _ = self.actor_target(next_states)
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
        # Actor 업데이트
        # ----------------------
        pred_actions, _, _, _ = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)

        self.actor_optimizer.step()

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
