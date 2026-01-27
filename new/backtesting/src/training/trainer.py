import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any
import logging
from datetime import datetime

from ..models.base_model import BaseModel
from .dataset import FinancialDataset


class Trainer:
    """
    Standard Trainer for Research Models.
    Handles training loop, logging, and checkpointing.
    """

    def __init__(
        self, config: Dict[str, Any], model: BaseModel, dataset: FinancialDataset
    ):
        self.config = config
        self.model = model
        self.device = model.device

        # Data Loader
        self.batch_size = config["model"]["ddpg"][
            "batch_size"
        ]  # Using DDPG batch size as default
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimizer
        self.lr = config["model"]["ddpg"]["actor_lr"]  # Default to generic LR
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)

        # Loss
        # Support TGNN Combined Loss
        from ..models.tgnn.loss import combined_loss

        if config["project"]["selected_model"] == "tgnn":
            self.criterion = combined_loss
        else:
            self.criterion = nn.MSELoss()

        # Logging
        # Organize results by model name
        model_name = config["project"].get("selected_model", "default")
        self.results_dir = os.path.join(config["paths"]["results_dir"], model_name)
        os.makedirs(self.results_dir, exist_ok=True)

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = self._setup_logger()

    def _setup_logger(self):
        logger = logging.getLogger(f"Trainer_{self.timestamp}")
        logger.setLevel(logging.INFO)
        fh = logging.FileHandler(
            os.path.join(self.results_dir, f"train_{self.timestamp}.log")
        )
        ch = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        logger.addHandler(fh)
        logger.addHandler(ch)
        return logger

    def train(self):
        """
        Executes the training loop.
        """
        epochs = self.config["training"]["episodes"]
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")

        best_loss = float("inf")

        for epoch in range(epochs):
            self.model.train()
            total_loss = 0

            for batch in self.dataloader:
                # Move to device
                features = batch["features"].to(self.device)
                adj = batch["adj_matrix"].to(self.device)
                labels = batch["labels"].to(self.device)  # [B, N]

                self.optimizer.zero_grad()

                # Forward
                # Check model type to determine output
                # TGNN returns (predictions, embeddings)
                # Hybrid returns ... (it's an Agent, specialized training needed?)
                # This Trainer assumes a generic supervised API for now.
                # If Hybrid/DDPG, we need a specialized RL loop.
                # Let's assume this generic Trainer is for TGNN pre-training or simple supervision.
                # For RL, we might need 'RLTrainer'.

                output = self.model(features, adj)

                # Handle tuple return
                if isinstance(output, tuple):
                    preds = output[0]
                else:
                    preds = output

                # Verify shape match for generic MSE
                # Labels might be [B, N], Preds [B, N]
                if preds.shape != labels.shape:
                    # Simple fix if one is [B, N, 1]
                    preds = preds.squeeze()
                    labels = labels.squeeze()

                loss = self.criterion(preds, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.dataloader)

            if epoch % 10 == 0:
                self.logger.info(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.6f}")

            # Checkpoint
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_path = os.path.join(
                    self.results_dir, f"best_model_{self.timestamp}.pth"
                )
                self.model.save(save_path)

        self.logger.info(f"Training Complete. Best Loss: {best_loss:.6f}")
