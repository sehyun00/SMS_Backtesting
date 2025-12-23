import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ==================== 1. Enhanced Temporal Attention Module ====================
class TemporalAttention(nn.Module):
    """강화된 시계열 Attention 메커니즘 (Multi-head 증가 + Residual)"""

    def __init__(self, hidden_dim, num_heads=8, dropout=0.15):
        super(TemporalAttention, self).__init__()
        # Multi-head 수 증가: 4 -> 8 (더 다양한 패턴 학습)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Feed-forward network with residual connection
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (Batch, Seq_Len, Hidden_Dim)
        
        # Multi-head Attention with residual
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.layer_norm2(x + ffn_out)
        
        return x, attn_weights


# ==================== 2. Enhanced Factor Encoder with Residual LSTM ====================
class TemporalFactorEncoder(nn.Module):
    """Residual Connection이 추가된 LSTM 기반 인코더"""

    def __init__(self, num_features, hidden_dim=64, num_layers=2, dropout=0.15):
        super(TemporalFactorEncoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_features = num_features

        # LSTM for temporal processing
        self.lstm = nn.LSTM(
            num_features,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Input projection for residual connection
        self.input_proj = nn.Linear(num_features, hidden_dim) if num_features != hidden_dim else None
        
        # Enhanced feature extraction with residual
        self.feature_net1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.feature_net2 = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        # x: (Batch, Seq_Len, Num_Features)
        batch_size = x.shape[0]
        
        # Store input for residual
        if self.input_proj is not None:
            residual = self.input_proj(x[:, -1, :])  # Use last timestep
        else:
            residual = x[:, -1, :]
        
        # LSTM processing
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # (Batch, Hidden_Dim)
        
        # Feature extraction with residual
        features = self.feature_net1(last_hidden)
        features = self.feature_net2(features)
        
        # Residual connection
        output = self.layer_norm(features + residual)
        
        return output


# ==================== 3. Shared Factor Encoder (Legacy Support) ====================
class SharedFactorEncoder(nn.Module):
    """모든 종목에 공유되는 팩터 분석 레이어 (Universal Rules Learner)"""

    def __init__(self, num_features, hidden_dim=64, dropout=0.15):
        super(SharedFactorEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Input: (Batch, Num_Stocks, Num_Features)
        return self.net(x)
        # Output: (Batch, Num_Stocks, Hidden_Dim)


# ==================== 4. Enhanced Actor Network ====================
class Actor(nn.Module):
    """개선된 포트폴리오 비중 결정 네트워크 (Enhanced Attention + Residual)"""

    def __init__(
        self,
        num_stocks,
        num_features,
        window_size=12,
        hidden_dim=128,
        min_weight=0.02,
        max_weight=0.30,
        use_temporal=True,
        dropout=0.15,
    ):
        super(Actor, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.window_size = window_size
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.use_temporal = use_temporal

        if use_temporal:
            # Temporal processing with enhanced LSTM
            self.temporal_encoder = TemporalFactorEncoder(
                num_features, hidden_dim=64, dropout=dropout
            )

            # Enhanced attention mechanism (8 heads)
            self.attention = TemporalAttention(64, num_heads=8, dropout=dropout)

            # Global context integration with residual
            input_dim = num_stocks * 64
        else:
            # Legacy mode
            self.encoder = SharedFactorEncoder(num_features, 64, dropout=dropout)
            input_dim = num_stocks * 64

        # Portfolio weight generation with residual connections
        self.global_net1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.global_net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_layer = nn.Linear(hidden_dim, num_stocks)
        
        # Skip connection projection
        self.skip_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, state):
        batch_size = state.shape[0]

        if self.use_temporal:
            # State: (Batch, Num_Stocks * Window_Size * Num_Features)
            x = state.reshape(
                batch_size, self.num_stocks, self.window_size, self.num_features
            )

            # Process each stock's time series
            stock_features = []
            for i in range(self.num_stocks):
                stock_ts = x[:, i, :, :]  # (Batch, Window_Size, Num_Features)
                features = self.temporal_encoder(stock_ts)  # (Batch, 64)
                stock_features.append(features)

            # Stack: (Batch, Num_Stocks, 64)
            x = torch.stack(stock_features, dim=1)

            # Apply enhanced attention across stocks
            x, _ = self.attention(x)  # (Batch, Num_Stocks, 64)

            # Flatten for global processing
            x = x.reshape(batch_size, -1)
        else:
            # Legacy mode
            x = state.reshape(batch_size, self.num_stocks, -1)
            last_features = x[:, :, -self.num_features :]
            x = self.encoder(last_features)
            x = x.reshape(batch_size, -1)

        # Store input for skip connection
        skip = self.skip_proj(x)
        
        # Generate portfolio weights with residual
        x = self.global_net1(x)
        x = self.global_net2(x + skip)  # Residual connection
        scores = self.output_layer(x)
        
        weights = F.softmax(scores, dim=-1)

        # Apply constraints
        weights = torch.clamp(weights, min=self.min_weight, max=self.max_weight)
        weights = weights / weights.sum(dim=-1, keepdim=True)

        # Calculate entropy for exploration bonus
        entropy = -torch.sum(weights * torch.log(weights + 1e-8), dim=-1)

        return weights, entropy


# ==================== 5. Enhanced Critic Network ====================
class Critic(nn.Module):
    """개선된 Q-Value 추정 네트워크 (Residual Connections)"""

    def __init__(
        self,
        num_stocks,
        num_features,
        action_dim,
        window_size=12,
        hidden_dim=128,
        use_temporal=True,
        dropout=0.15,
    ):
        super(Critic, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        self.window_size = window_size
        self.use_temporal = use_temporal

        if use_temporal:
            self.temporal_encoder = TemporalFactorEncoder(
                num_features, hidden_dim=64, dropout=dropout
            )
            input_dim = (num_stocks * 64) + action_dim
        else:
            self.encoder = SharedFactorEncoder(num_features, 64, dropout=dropout)
            input_dim = (num_stocks * 64) + action_dim

        # Q-value estimation network with residual
        self.net1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.net2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_layer = nn.Linear(hidden_dim, 1)
        
        # Skip connection
        self.skip_proj = nn.Linear(input_dim, hidden_dim)

    def forward(self, state, action):
        batch_size = state.shape[0]

        if self.use_temporal:
            # Reshape for temporal processing
            x = state.reshape(
                batch_size, self.num_stocks, self.window_size, self.num_features
            )

            # Process each stock's time series
            stock_features = []
            for i in range(self.num_stocks):
                stock_ts = x[:, i, :, :]
                features = self.temporal_encoder(stock_ts)
                stock_features.append(features)

            x = torch.stack(stock_features, dim=1)
            x = x.reshape(batch_size, -1)
        else:
            # Legacy mode
            x = state.reshape(batch_size, self.num_stocks, -1)
            last_features = x[:, :, -self.num_features :]
            x = self.encoder(last_features)
            x = x.reshape(batch_size, -1)

        # Combine state and action
        xa = torch.cat([x, action], dim=-1)
        
        # Store for skip connection
        skip = self.skip_proj(xa)
        
        # Estimate Q-value with residual
        x = self.net1(xa)
        x = self.net2(x + skip)  # Residual connection
        q_value = self.output_layer(x)
        
        return q_value


# ==================== 6. Replay Buffer ====================
class ReplayBuffer:
    """경험 저장 및 샘플링"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards).reshape(-1, 1),
            np.array(next_states),
            np.array(dones).reshape(-1, 1),
        )

    def __len__(self):
        return len(self.buffer)


# ==================== 7. Enhanced DDPG Agent ====================
class DDPGAgent:
    def __init__(
        self,
        num_stocks,
        num_features,
        window_size=12,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        entropy_coef=0.01,
        use_temporal=True,
        use_scheduler=True,
        device="cuda",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef
        self.use_scheduler = use_scheduler

        action_dim = num_stocks

        # Main Networks
        self.actor = Actor(
            num_stocks, num_features, window_size, use_temporal=use_temporal
        ).to(device)
        self.critic = Critic(
            num_stocks, num_features, action_dim, window_size, use_temporal=use_temporal
        ).to(device)

        # Target Networks
        self.actor_target = Actor(
            num_stocks, num_features, window_size, use_temporal=use_temporal
        ).to(device)
        self.critic_target = Critic(
            num_stocks, num_features, action_dim, window_size, use_temporal=use_temporal
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=lr_critic
        )

        # Learning Rate Schedulers
        if use_scheduler:
            self.actor_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.actor_optimizer, T_max=1000, eta_min=lr_actor * 0.1
            )
            self.critic_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.critic_optimizer, T_max=1000, eta_min=lr_critic * 0.1
            )

        # Replay Buffer
        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_std=0.1):
        """행동 선택 (탐색 노이즈 포함)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, _ = self.actor(state)
            action = action.cpu().numpy()[0]

        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)  # Normalize

        return action

    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return 0.0, 0.0

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Critic Update
        with torch.no_grad():
            next_actions, _ = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(
                next_states, next_actions
            )

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        # NaN Check
        if torch.isnan(critic_loss) or torch.isinf(critic_loss):
            print("⚠️ NaN/Inf detected in critic loss")
            return 0.0, 0.0

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        if self.use_scheduler:
            self.critic_scheduler.step()

        # Actor Update
        new_actions, entropy = self.actor(states)
        actor_loss = -self.critic(states, new_actions).mean()

        # Entropy bonus for exploration
        entropy_bonus = -self.entropy_coef * entropy.mean()
        total_actor_loss = actor_loss + entropy_bonus

        # NaN Check
        if torch.isnan(total_actor_loss) or torch.isinf(total_actor_loss):
            print("⚠️ NaN/Inf detected in actor loss")
            return critic_loss.item(), 0.0

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        if self.use_scheduler:
            self.actor_scheduler.step()

        # Target Network Soft Update
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def get_lr(self):
        """현재 학습률 반환"""
        return {
            "actor_lr": self.actor_optimizer.param_groups[0]["lr"],
            "critic_lr": self.critic_optimizer.param_groups[0]["lr"],
        }
