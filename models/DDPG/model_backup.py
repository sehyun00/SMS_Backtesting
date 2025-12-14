import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ==================== 1. Factor-Aware Network Building Block ====================
class SharedFactorEncoder(nn.Module):
    """모든 종목에 공유되는 팩터 분석 레이어 (Universal Rules Learner)"""
    def __init__(self, num_features, hidden_dim=64):
        super(SharedFactorEncoder, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Input: (Batch, Num_Stocks, Num_Features)
        # Linear layer applies to the last dimension
        return self.net(x) 
        # Output: (Batch, Num_Stocks, Hidden_Dim)

# ==================== 2. Actor Network ====================
class Actor(nn.Module):
    """포트폴리오 비중 결정 (Factor Analysis -> Portfolio Weighting)"""
    def __init__(self, num_stocks, num_features, hidden_dim=128):
        super(Actor, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        
        # 1. 팩터 분석 (공유 가중치)
        self.encoder = SharedFactorEncoder(num_features, 64)
        
        # 2. 글로벌 문맥 통합 (Global Context)
        input_dim = num_stocks * 64
        self.global_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_stocks) # 각 종목별 점수 출력
        )
        
    def forward(self, state):
        # State: (Batch, Num_Stocks * Num_Features) -> Flat Vector
        batch_size = state.shape[0]
        
        # 1. 구조화 (Reshape)
        x = state.reshape(batch_size, self.num_stocks, self.num_features)
        
        # 2. 개별 종목 팩터 분석
        x = self.encoder(x) # (Batch, Num_Stocks, 64)
        
        # 3. 전체 시장 상황 종합
        x = x.reshape(batch_size, -1) # Flatten
        scores = self.global_net(x)
        
        # 4. 포트폴리오 비중 (Softmax)
        weights = F.softmax(scores, dim=-1)
        return weights

# ==================== 3. Critic Network ====================
class Critic(nn.Module):
    """Q-Value 추정 (Factor Analysis + Action -> Q-Value)"""
    def __init__(self, num_stocks, num_features, action_dim, hidden_dim=128):
        super(Critic, self).__init__()
        self.num_stocks = num_stocks
        self.num_features = num_features
        
        # 1. 팩터 분석 (공유 가중치)
        self.encoder = SharedFactorEncoder(num_features, 64)
        
        # 2. State + Action 통합
        input_dim = (num_stocks * 64) + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, state, action):
        batch_size = state.shape[0]
        
        # 1. 구조화 및 팩터 분석
        x = state.reshape(batch_size, self.num_stocks, self.num_features)
        x = self.encoder(x)
        x = x.reshape(batch_size, -1)
        
        # 2. 행동(Action)과 결합
        xa = torch.cat([x, action], dim=-1)
        
        # 3. 가치 추정
        q_value = self.net(xa)
        return q_value

# ==================== 4. Replay Buffer ====================
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
            np.array(dones).reshape(-1, 1)
        )
    
    def __len__(self):
        return len(self.buffer)

# ==================== 5. DDPG Agent ====================
class DDPGAgent:
    def __init__(self, num_stocks, num_features, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.001, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        
        action_dim = num_stocks
        
        # Main Networks
        self.actor = Actor(num_stocks, num_features).to(device)
        self.critic = Critic(num_stocks, num_features, action_dim).to(device)
        
        # Target Networks
        self.actor_target = Actor(num_stocks, num_features).to(device)
        self.critic_target = Critic(num_stocks, num_features, action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Replay Buffer
        self.replay_buffer = ReplayBuffer()
    
    def select_action(self, state, noise_std=0.1):
        """행동 선택 (탐색 노이즈 포함)"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)  # 0 나누기 방지
        
        return action
    
    def train(self, batch_size=64):
        if len(self.replay_buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Critic 업데이트
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        
        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Actor 업데이트
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Target Soft Update
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
    
    def _soft_update(self, source, target):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
