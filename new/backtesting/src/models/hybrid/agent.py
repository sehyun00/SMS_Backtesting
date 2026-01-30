import torch
import numpy as np
import torch.optim as optim
from typing import Dict, Any, Tuple
from src.models.base_model import BaseModel
from .actor import HybridActor
from .critic import HybridCritic
from src.training.replay_buffer import ReplayBuffer


class HybridAgent(BaseModel):
    """
    Hybrid TGNN-DDPG Agent (Asset-Agnostic).
    Wraps Actor and Critic.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_features = len(config["data"]["features"])
        if "factors" in config["data"]:
            self.num_features += len(config["data"]["factors"]["weights"])

        # Note: num_stocks is derived from config for context,
        # but the underlying Actor/Critic are Asset-Agnostic (Dynamic N).
        self.num_stocks = len(config["data"]["stock_universes"])
        self.window_size = config["data"]["window_size"]

        # Hyperparameters
        self.gamma = config["training"].get("gamma", 0.99)
        self.tau = config["training"].get("tau", 0.005)
        self.lr_actor = config["training"].get("lr_actor", 1e-4)
        self.lr_critic = config["training"].get("lr_critic", 1e-3)
        self.batch_size = config["training"].get("batch_size", 64)
        self.temperature = config["model"].get("softmax_temperature", 1.0)

        # Initialize Actor & Critic
        self.actor = HybridActor(
            self.num_stocks,
            self.window_size,
            self.num_features,
            temperature=self.temperature,
        )
        self.critic = HybridCritic(self.num_features, self.window_size, hidden_dim=128)

        # Target Networks (Copy)
        self.actor_target = HybridActor(
            self.num_stocks,
            self.window_size,
            self.num_features,
            temperature=self.temperature,
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = HybridCritic(
            self.num_features, self.window_size, hidden_dim=128
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic)

        # Replay Buffer
        self.buffer = ReplayBuffer(capacity=10000)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for prediction/inference.
        Returns portfolio weights.
        """
        weights, _ = self.actor(x, adj)
        return weights

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prediction helper for evaluation.
        """
        self.actor.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)
            adj = batch["adj_matrix"].to(self.device)
            weights, _ = self.actor(x, adj)
        return weights.cpu()

    def select_action(
        self, state_feat: np.ndarray, state_adj: np.ndarray, noise_std: float = 0.1
    ) -> np.ndarray:
        """
        Select action with Dirichlet noise for exploration.
        """
        self.actor.eval()
        with torch.no_grad():
            feat_tensor = (
                torch.FloatTensor(state_feat).unsqueeze(0).to(self.device)
            )  # [1, N, T, F]
            adj_tensor = (
                torch.FloatTensor(state_adj).unsqueeze(0).to(self.device)
            )  # [1, N, N]

            weights, _ = self.actor(feat_tensor, adj_tensor)
            action = weights.cpu().numpy()[0]  # [N]

        if noise_std > 0:
            concentration = action / (noise_std + 1e-8)
            concentration = np.clip(concentration, 0.1, 100.0)
            action = np.random.dirichlet(concentration)

        return action
