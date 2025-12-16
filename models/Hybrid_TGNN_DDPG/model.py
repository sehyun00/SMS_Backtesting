import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


# ==================== 1. TGNN Logic (State Encoder) ====================


class GraphConvLayer(nn.Module):
    """
    Graph Convolutional Layer
    - Aggregates information from neighboring nodes using the Adjacency Matrix.
    - Formula: Output = Activation(Normalized_Adj * X * W)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, In_F) - Node feature matrix
        # adj: (Batch, N, N) - Adjacency matrix

        # Calculate Degree Matrix and prepare normalization
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        # Normalize adjacency matrix: D^(-1/2) * A * D^(-1/2)
        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)

        # Linear transformation and information propagation
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        return F.relu(output)


class TemporalAttention(nn.Module):
    """
    Temporal Attention Mechanism
    - Assigns weights to important time steps in time series data.
    - Uses Multi-head Attention to learn importance along the time axis.
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, T, N, D)
        batch, T, N, D = x.shape

        # Reshape for applying Attention along time axis for each node
        # Convert to (Batch * N, T, D) format
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)

        # Perform Self-Attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)

        # Extract attended feature of the last time step and restore to original batch structure
        # (Batch, N, D)
        return attn_out[:, -1, :].reshape(batch, N, D)


class TGNNEncoder(nn.Module):
    """
    Spatiotemporal Encoder
    - Uses TGNN structure to extract spatial (inter-stock relationships) and temporal (time series) features.
    - Structure: GCN Layers (spatial info) -> Temporal Attention (temporal info)
    """

    def __init__(self, num_features, hidden_dims=[64, 64], num_heads=4):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dims[0])

        # Stack multiple GCN layers
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)
        self.out_dim = hidden_dims[-1]

    def forward(self, features, adj):
        # features: (Batch, N, T, F) - Batch, num_stocks, time (window), num_features
        batch, N, T, F = features.shape

        gcn_outputs = []
        # Apply GCN for each time step (t)
        for t in range(T):
            x_t = features[:, :, t, :]  # (Batch, N, F)
            h = self.input_proj(x_t)

            for gcn in self.gcn_layers:
                h = gcn(h, adj)

            gcn_outputs.append(h)

        # Stack results along time axis: (Batch, T, N, D)
        temporal_features = torch.stack(gcn_outputs, dim=1)

        # Apply temporal attention mechanism to generate final embedding -> (Batch, N, D)
        node_embeddings = self.temporal_attn(temporal_features)
        return node_embeddings


# ==================== 2. Hybrid Network Components (Actor-Critic) ====================


class HybridActor(nn.Module):
    """Pure learning-based: minimize hard constraints, control with soft constraints"""

    def __init__(self, num_stocks, window_size, num_features, hidden_dim=128):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features

        # Constraint parameters
        self.MIN_WEIGHT = 0.05  # 5% minimum per stock
        self.MAX_WEIGHT = 0.20  # 20% maximum per stock

        # TGNN path
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

        # DDPG path
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

        # Ensemble weight
        ensemble_input_dim = state_dim + num_stocks * 2

        self.ensemble_weight_net = nn.Sequential(
            nn.Linear(ensemble_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def _enforce_constraints(self, weights, max_iter=10):
        """
        🔥 Iterative Projection: Enforce constraints while maintaining sum=1
        
        This method ensures:
        1. MIN_WEIGHT <= w_i <= MAX_WEIGHT for all stocks
        2. sum(w) = 1.0
        3. No bypass through normalization
        
        Algorithm:
        - Iteratively adjust weights that violate constraints
        - Redistribute excess/deficit to feasible stocks
        - Converge to a valid solution
        """
        MIN_W = self.MIN_WEIGHT
        MAX_W = self.MAX_WEIGHT
        eps = 1e-4  # Convergence tolerance
        
        for iteration in range(max_iter):
            # Step 1: Clamp to [MIN_W, MAX_W]
            weights_clamped = torch.clamp(weights, MIN_W, MAX_W)
            
            # Step 2: Check current sum
            current_sum = weights_clamped.sum(dim=-1, keepdim=True)
            
            # Step 3: If sum is close to 1.0, we're done
            if torch.allclose(current_sum, torch.ones_like(current_sum), atol=eps):
                weights = weights_clamped
                break
            
            # Step 4: Redistribute excess/deficit
            deficit = 1.0 - current_sum  # How much we need to add/subtract
            
            if deficit > 0:  # Need to increase weights
                # Find stocks that have room to grow (below MAX_W)
                room_to_grow = MAX_W - weights_clamped
                total_room = room_to_grow.sum(dim=-1, keepdim=True)
                
                # Distribute deficit proportionally to available room
                if total_room > eps:
                    adjustment = deficit * (room_to_grow / (total_room + 1e-8))
                    weights = weights_clamped + adjustment
                else:
                    # No room to grow - uniformly distribute
                    weights = weights_clamped + deficit / self.num_stocks
                    
            else:  # deficit < 0, need to decrease weights
                # Find stocks that have room to shrink (above MIN_W)
                room_to_shrink = weights_clamped - MIN_W
                total_room = room_to_shrink.sum(dim=-1, keepdim=True)
                
                # Distribute excess proportionally to available room
                if total_room > eps:
                    adjustment = deficit * (room_to_shrink / (total_room + 1e-8))
                    weights = weights_clamped + adjustment
                else:
                    # No room to shrink - uniformly distribute
                    weights = weights_clamped + deficit / self.num_stocks
        
        # Final safety: clamp and normalize
        weights = torch.clamp(weights, MIN_W, MAX_W)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
        
        return weights

    def forward(self, state):
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features

        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]

        features = features_flat.reshape(
            batch, self.num_stocks, self.window_size, self.num_features
        )
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)

        # Temperature for softmax
        temperature = 5.0

        # TGNN path
        tgnn_embeddings = self.tgnn_encoder(features, adj)
        tgnn_embeddings_flat = tgnn_embeddings.reshape(batch, -1)
        tgnn_logits = self.tgnn_head(tgnn_embeddings_flat)
        tgnn_weights = F.softmax(tgnn_logits / temperature, dim=-1)

        # DDPG path
        ddpg_features = self.ddpg_encoder(state)
        ddpg_logits = self.ddpg_head(ddpg_features)
        ddpg_weights = F.softmax(ddpg_logits / temperature, dim=-1)

        # Ensemble
        ensemble_input = torch.cat([state, tgnn_weights, ddpg_weights], dim=-1)
        alpha_raw = self.ensemble_weight_net(ensemble_input)
        alpha = 0.3 + 0.4 * torch.sigmoid(alpha_raw)

        # Final combination
        final_weights = alpha * tgnn_weights + (1 - alpha) * ddpg_weights

        # 🔥 NEW: Use iterative projection instead of clamp+normalize
        final_weights = self._enforce_constraints(final_weights)

        return final_weights, alpha.squeeze(-1), tgnn_weights, ddpg_weights


class HybridCritic(nn.Module):
    """
    Improved Hybrid Critic Network
    - Evaluates Q-Value aligned with Actor's dual-path structure
    """

    def __init__(
        self, num_stocks, window_size, num_features, action_dim, hidden_dim=128
    ):
        super().__init__()
        self.num_stocks = num_stocks
        self.window_size = window_size
        self.num_features = num_features

        # TGNN feature extractor
        self.tgnn_encoder = TGNNEncoder(num_features)

        # DDPG feature extractor
        state_dim = num_stocks * window_size * num_features + num_stocks * num_stocks
        self.ddpg_encoder = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(), nn.Linear(256, hidden_dim)
        )

        # Q-Value prediction head
        # Input: TGNN embedding + DDPG embedding + Action
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
        batch = state.shape[0]
        feat_size = self.num_stocks * self.window_size * self.num_features

        # Restore state
        features_flat = state[:, :feat_size]
        adj_flat = state[:, feat_size:]

        features = features_flat.reshape(
            batch, self.num_stocks, self.window_size, self.num_features
        )
        adj = adj_flat.reshape(batch, self.num_stocks, self.num_stocks)

        # Extract features from both paths
        tgnn_embeddings = self.tgnn_encoder(features, adj)
        tgnn_embeddings_flat = tgnn_embeddings.reshape(batch, -1)

        ddpg_embeddings = self.ddpg_encoder(state)

        # Calculate Q-Value (combine features from both paths + action)
        qa = torch.cat([tgnn_embeddings_flat, ddpg_embeddings, action], dim=-1)
        q_value = self.q_net(qa)

        return q_value


# ==================== 3. RL Infrastructure (Replay Buffer & Agent) ====================


class ReplayBuffer:
    """
    Experience Replay Buffer
    - Stores training data and samples randomly to break correlations between data and improve training stability.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Store new experience"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample mini-batch for training"""
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


class HybridAgent:
    """
    Hybrid DDPG Agent
    - Manages and trains Actor and Critic networks.
    - Uses Target Network to ensure training stability.
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
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.entropy_coef = entropy_coef

        # Initialize networks
        self.actor = HybridActor(num_stocks, window_size, num_features).to(device)
        self.critic = HybridCritic(
            num_stocks, window_size, num_features, num_stocks
        ).to(device)

        # Initialize target networks (copy training networks)
        self.actor_target = HybridActor(num_stocks, window_size, num_features).to(
            device
        )
        self.critic_target = HybridCritic(
            num_stocks, window_size, num_features, num_stocks
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Set up optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.replay_buffer = ReplayBuffer()

    def select_action(self, state, noise_std=0.1):
        """Improved: also return alpha value"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            final_weights, alpha, tgnn_weights, ddpg_weights = self.actor(state)
            action = final_weights.cpu().numpy()[0]
            alpha_value = alpha.cpu().item()

        if noise_std > 0:
            # Add exploration noise
            noise = np.random.normal(0, noise_std, size=action.shape)
            action = action + noise
            action = np.clip(action, 0, 1)
            action = action / (action.sum() + 1e-8)

        # Return debugging information (if needed)
        return action, alpha_value  # also return alpha value

    def train(self, batch_size=64):
        """Training method - with entropy regularization"""
        if len(self.replay_buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # ----------------------------
        # 1. Update Critic Network
        # ----------------------------
        with torch.no_grad():
            next_actions, _, _, _ = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(
                next_states, next_actions
            )

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ----------------------------
        # 2. Update Actor Network (with entropy)
        # ----------------------------
        predicted_actions, alpha, _, _ = self.actor(states)

        # Basic Actor Loss (maximize Q-Value)
        actor_loss = -self.critic(states, predicted_actions).mean()

        # ⭐ Entropy regularization: encourage Alpha to stay near 0.5
        # Prevent Alpha from going to extreme values (0 or 1)
        alpha_entropy = -(
            alpha * torch.log(alpha + 1e-8) + (1 - alpha) * torch.log(1 - alpha + 1e-8)
        ).mean()

        # Total Loss = Actor Loss - Entropy Bonus
        total_actor_loss = actor_loss - self.entropy_coef * alpha_entropy

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ----------------------------
        # 3. Soft Update Target Networks
        # ----------------------------
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """Slowly update target network (Polyak Averaging)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )
