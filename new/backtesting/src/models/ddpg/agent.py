import torch
from typing import Dict, Any, Tuple
from src.models.base_model import BaseModel
from .actor import DDPGActor
from .critic import Critic


class DDPGAgent(BaseModel):
    """
    DDPG Agent Wrapper.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.num_stocks = len(config["data"]["stock_universes"])

        # Determine effective Num Features (F * T)
        # Because we flatten time window
        raw_features = len(config["data"]["features"])
        if "factors" in config["data"]:
            raw_features += len(config["data"]["factors"]["weights"])

        window_size = config["data"]["window_size"]
        self.effective_num_features = raw_features * window_size

        self.actor = DDPGActor(self.num_stocks, self.effective_num_features)
        self.critic = Critic(
            self.num_stocks, self.effective_num_features, action_dim=self.num_stocks
        )

        # Target Networks (RL specific)
        self.actor_target = DDPGActor(self.num_stocks, self.effective_num_features)
        self.critic_target = Critic(
            self.num_stocks, self.effective_num_features, action_dim=self.num_stocks
        )
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.to(self.device)
        self.actor_target.to(self.device)
        self.critic_target.to(self.device)

    def forward(self, x: torch.Tensor, adj: torch.Tensor = None) -> torch.Tensor:
        # Return prediction for compatibility
        # x is [B, N, T, F]
        weights, _ = self.actor(x)
        return weights

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self.actor.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)  # [B, N, T, F]
            # adj not used in DDPG
            weights, _ = self.actor(x)
        return weights.cpu()
