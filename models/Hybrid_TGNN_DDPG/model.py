import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random

# ==================== 1. TGNN 로직 (상태 인코더) ====================

class GraphConvLayer(nn.Module):
    """
    그래프 합성곱 레이어 (Graph Convolutional Layer)
    - 인접 행렬(Adjacency Matrix)을 사용하여 이웃 노드의 정보를 집계합니다.
    - 수식: Output = Activation(Normalized_Adj * X * W)
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, In_F) - 노드 특징 행렬
        # adj: (Batch, N, N) - 인접 행렬
        
        # 차수 행렬(Degree Matrix) 계산 및 정규화 준비
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
    시간적 주의 메커니즘 (Temporal Attention)
    - 시계열 데이터에서 중요한 시간 시점에 가중치를 부여합니다.
    - Multi-head Attention을 사용하여 시간 축(Time axis)에 대한 중요도를 학습합니다.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, N, D)
        batch, T, N, D = x.shape
        
        # 각 노드별로 시간 축에 대해 Attention 적용을 위해 차원 변경
        # (Batch * N, T, D) 형태로 변환
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)
        
        # Self-Attention 수행
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # 마지막 시점의 attended feature만 추출하여 원래 배치 구조로 복원
        # (Batch, N, D)
        return attn_out[:, -1, :].reshape(batch, N, D)

class TGNNEncoder(nn.Module):
    """
    시공간 인코더 (Spatiotemporal Encoder)
    - TGNN 구조를 활용하여 주가 데이터의 공간적(종목 간 관계) 및 시간적(시계열) 특징을 추출합니다.
    - 구조: GCN Layers (공간 정보) -> Temporal Attention (시간 정보)
    """
    def __init__(self, num_features, hidden_dims=[64, 64], num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        
        # 여러 층의 GCN 쌓기
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dims[i], hidden_dims[i+1])
            for i in range(len(hidden_dims)-1)
        ])
        
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)
        self.out_dim = hidden_dims[-1]

    def forward(self, features, adj):
        # features: (Batch, N, T, F) - 배치, 종목수, 시간(윈도우), 특징수
        batch, N, T, F = features.shape
        
        gcn_outputs = []
        # 각 시간 단계(t)별로 GCN 적용
        for t in range(T):
            x_t = features[:, :, t, :] # (Batch, N, F)
            h = self.input_proj(x_t)
            
            for gcn in self.gcn_layers:
                h = gcn(h, adj)
            
            gcn_outputs.append(h)
        
        # 시간 축으로 결과 쌓기: (Batch, T, N, D)
        temporal_features = torch.stack(gcn_outputs, dim=1)
        
        # 시간적 주의 메커니즘 적용하여 최종 임베딩 생성 -> (Batch, N, D)
        node_embeddings = self.temporal_attn(temporal_features)
        return node_embeddings

# ==================== 2. Hybrid 네트워크 컴포넌트 (Actor-Critic) ====================

class HybridActor(nn.Module):
    """
    Hybrid Actor 네트워크
    - 상태(State)를 입력받아 각 종목의 포트폴리오 비중(Action)을 결정합니다.
    - TGNNEncoder를 통해 특징을 추출하고, Fully Connected Layer로 비중을 계산합니다.
    """
    def __init__(self, num_stocks, window_size, num_features, hidden_dim=128):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features
        
        # 특징 추출기 (TGNN)
        self.encoder = TGNNEncoder(num_features)
        
        # 정책 결정 네트워크 (Global Context 반영)
        input_dim = num_stocks * self.encoder.out_dim
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
        # Flatten된 상태 벡터를 원래 구조로 복원
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features
        
        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]
        
        features = features_flat.reshape(batch, self.num_stocks, self.window_size, self.num_features)
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)
        
        # 인코딩 (특징 추출)
        embeddings = self.encoder(features, adj) # (Batch, N, D)
        
        # 정책(비중) 계산
        embeddings_flat = embeddings.reshape(batch, -1)
        scores = self.global_net(embeddings_flat)
        weights = F.softmax(scores, dim=-1) # 합이 1이 되도록 Softmax 적용
        return weights

class HybridCritic(nn.Module):
    """
    Hybrid Critic 네트워크
    - 상태(State)와 행동(Action)을 입력받아 해당 행동의 가치(Q-Value)를 평가합니다.
    """
    def __init__(self, num_stocks, window_size, num_features, action_dim, hidden_dim=128):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features
        
        # 특징 추출기 (TGNN) - Critic을 위한 별도 인스턴스
        self.encoder = TGNNEncoder(num_features)
        
        # Q-Value 예측 헤드
        # 입력: Flattened Embeddings + Action 벡터
        input_dim = (num_stocks * self.encoder.out_dim) + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # 단일 Q-Value 출력
        )
        
    def forward(self, state, action):
        # 상태 벡터 복원
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features
        
        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]
        
        features = features_flat.reshape(batch, self.num_stocks, self.window_size, self.num_features)
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)
        
        # 인코딩
        embeddings = self.encoder(features, adj) # (Batch, N, D)
        embeddings_flat = embeddings.reshape(batch, -1)
        
        # Q-Value 계산 (상태 임베딩 + 행동 결합)
        xa = torch.cat([embeddings_flat, action], dim=-1)
        q_value = self.net(xa)
        return q_value

# ==================== 3. RL 인프라 (Replay Buffer & Agent) ====================

class ReplayBuffer:
    """
    경험 재생 버퍼 (Experience Replay Buffer)
    - 학습 데이터를 저장하고 무작위로 샘플링하여 데이터 간 상관관계를 끊고 학습 안정성을 높입니다.
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """새로운 경험 저장"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """학습을 위한 미니배치 샘플링"""
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

class HybridAgent:
    """
    Hybrid DDPG 에이전트
    - Actor와 Critic 네트워크를 관리하고 학습시킵니다.
    - Target Network를 사용하여 학습 안정성을 확보합니다.
    """
    def __init__(self, num_stocks, window_size, num_features, 
                 lr_actor=1e-4, lr_critic=1e-3, gamma=0.99, tau=0.001, device='cuda'):
        self.device = device
        self.gamma = gamma # 할인율
        self.tau = tau     # Soft Update 계수
        
        # 네트워크 초기화
        self.actor = HybridActor(num_stocks, window_size, num_features).to(device)
        self.critic = HybridCritic(num_stocks, window_size, num_features, num_stocks).to(device)
        
        # 타겟 네트워크 초기화 (학습 대상 네트워크 복사)
        self.actor_target = HybridActor(num_stocks, window_size, num_features).to(device)
        self.critic_target = HybridCritic(num_stocks, window_size, num_features, num_stocks).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 옵티마이저 설정
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.replay_buffer = ReplayBuffer()
        
    def select_action(self, state, noise_std=0.1):
        """주어진 상태에서 행동 선택 (탐색을 위한 노이즈 추가 가능)"""
        # state는 1차원으로 Flatten된 벡터
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        
        if noise_std > 0:
            # 탐색 노이즈 추가 (OU Noise 대신 간단한 Gaussian Noise 사용 가능)
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
            # 비중 제약 조건 적용 (0~1 사이, 합 1)
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)
            
        return action
        
    def train(self, batch_size=64):
        """미니배치를 이용한 네트워크 업데이트"""
        if len(self.replay_buffer) < batch_size:
            return
            
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # ----------------------------
        # 1. Critic 네트워크 업데이트
        # ----------------------------
        with torch.no_grad():
            # 타겟 Actor로 다음 상태 행동 예측
            next_actions = self.actor_target(next_states)
            # 타겟 Q값 계산: Reward + Gamma * Q_target(next_state, next_action)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)
        
        # 현재 Q값 예측
        current_q = self.critic(states, actions)
        
        # Critic 손실함수 (MSE) 계산 및 역전파
        critic_loss = F.mse_loss(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # ----------------------------
        # 2. Actor 네트워크 업데이트
        # ----------------------------
        # Actor 손실함수: Critic이 평가한 가치의 음수 (가치 최대화)
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # ----------------------------
        # 3. 타겟 네트워크 Soft Update
        # ----------------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)
        
        return critic_loss.item(), actor_loss.item()
        
    def _soft_update(self, source, target):
        """타겟 네트워크를 천천히 업데이트 (Polyak Averaging)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
