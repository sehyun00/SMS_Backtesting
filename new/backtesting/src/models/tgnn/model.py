import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
from src.models.base_model import BaseModel
from src.models.layers import GraphConvLayer, TemporalAttention


class TGNN(BaseModel):
    """
    TGNN (Temporal Graph Neural Network) with Multi-Head Prediction.
    Restored features:
    - Batch Normalization
    - Residual Connections
    - 4 Independent Prediction Heads (1M, 3M, 6M, 12M)
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        # Config parsing
        self.num_features = len(config["data"]["features"])
        if "factors" in config["data"]:
            self.num_features += len(config["data"]["factors"]["weights"])

        self.num_stocks = len(config["data"]["stock_universes"])
        self.tgnn_cfg = config["model"]["tgnn"]
        self.hidden_dim = self.tgnn_cfg.get("hidden_dim", 128)
        self.num_heads = self.tgnn_cfg.get("num_heads", 8)
        self.dropout_rate = self.tgnn_cfg.get("dropout", 0.3)

        # Architecture (Hardcoded structure from legacy, or parametrized)
        hidden_dims = [128, 128, 64]  # Legacy default

        # 1. Input Projection
        self.input_proj = nn.Linear(self.num_features, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])

        # 2. GCN Layers
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.gcn_lns = nn.ModuleList(
            [nn.LayerNorm(hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )

        # 3. Temporal Attention
        self.temporal_attn = TemporalAttention(hidden_dims[-1], self.num_heads)

        # 4. Multi-Task Predictors
        # Keys correspond to dataset label columns
        self.heads = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
        self.predictors = nn.ModuleDict(
            {head: self._make_predictor(hidden_dims[-1]) for head in self.heads}
        )

        self.to(self.device)

    def _make_predictor(self, input_dim: int) -> nn.Sequential:
        """
        Legacy predictor structure:
        Linear -> LayerNorm -> Dropout -> Linear -> ReLU -> Dropout -> Linear
        """
        return nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
        )

    def forward(
        self, x: torch.Tensor, adj: torch.Tensor, target_type: str = "Momentum1M"
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        Args:
            target_type: Which head to use for prediction (default: Momentum1M)
        """
        batch, N, T, F = x.shape

        gcn_outputs = []
        for t in range(T):
            x_t = x[:, :, t, :]  # [B, N, F]
            h = self.input_proj(x_t)
            h = self.input_ln(h)

            # GCN + Residual
            for gcn, ln in zip(self.gcn_layers, self.gcn_lns):
                h_new = gcn(h, adj)
                h_new = ln(h_new)

                if h.shape[-1] == h_new.shape[-1]:
                    h = h + h_new
                else:
                    h = h_new

            gcn_outputs.append(h)

        # Temporal Attention
        # Stack: [B, T, N, D]
        temporal_features = torch.stack(gcn_outputs, dim=1)
        node_embeddings = self.temporal_attn(temporal_features)  # [B, N, D]

        # Prediction Head
        # If target_type is 'all', return dict?
        # For now, support single target per forward call as per legacy
        if target_type in self.predictors:
            predictions = self.predictors[target_type](node_embeddings).squeeze(-1)
        else:
            # Fallback or Error
            predictions = self.predictors["Momentum1M"](node_embeddings).squeeze(-1)

        return predictions, node_embeddings

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Evaluation helper (Default to 1M Momentum)
        """
        self.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)
            adj = batch["adj_matrix"].to(self.device)
            preds, _ = self.forward(x, adj, target_type="Momentum1M")
        return preds.cpu()
