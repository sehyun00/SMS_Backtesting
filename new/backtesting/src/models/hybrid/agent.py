import torch
from typing import Dict, Any, Tuple
from src.models.base_model import BaseModel
from .actor import HybridActor
from .critic import HybridCritic


class HybridAgent(BaseModel):
    """
    Hybrid TGNN-DDPG Agent (Main Interface).
    Wraps Actor and Critic.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_features = len(config["data"]["features"])
        if "factors" in config["data"]:
            self.num_features += len(config["data"]["factors"]["weights"])

        self.num_stocks = len(config["data"]["stock_universes"])
        self.window_size = config["data"]["window_size"]

        # Initialize Actor & Critic
        self.actor = HybridActor(self.num_stocks, self.window_size, self.num_features)
        self.critic = HybridCritic(
            self.num_stocks,
            self.window_size,
            self.num_features,
            action_dim=self.num_stocks,
        )

        # Target Networks (Copy)
        self.actor_target = HybridActor(
            self.num_stocks, self.window_size, self.num_features
        )
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = HybridCritic(
            self.num_stocks,
            self.window_size,
            self.num_features,
            action_dim=self.num_stocks,
        )
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.to(self.device)

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
