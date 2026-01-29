import torch
import numpy as np
from typing import Dict, Any, Tuple


class StrategyHandler:
    """
    Handles model inference and weight calculation strategies.
    """

    def __init__(self, n_stocks: int, device):
        self.n_stocks = n_stocks
        self.device = device

    def get_weights(
        self, strategy_type: str, model, window: Dict[str, Any], target_head: str = None
    ) -> np.ndarray:
        """
        Determines portfolio weights based on strategy type.
        """
        if strategy_type == "buy_and_hold":
            return np.ones(self.n_stocks) / self.n_stocks

        elif strategy_type == "model":
            return self._model_strategy(model, window, target_head)

        else:
            # Default fallback
            return np.ones(self.n_stocks) / self.n_stocks

    def _model_strategy(
        self, model, window: Dict[str, Any], target_head: str
    ) -> np.ndarray:
        """
        Executes model inference and applies weighting logic (Top-K / Softmax).
        """
        # 1. Prepare Input
        features = torch.FloatTensor(window["features"]).to(self.device)
        if features.dim() == 2:  # [N, F] -> [1, N, F] ? No, expected [N, T, F]
            # If window features are [N, T, F], unsqueeze to [1, N, T, F]
            features = features.unsqueeze(0)
        elif features.dim() == 3:  # [N, T, F] -> [1, N, T, F]
            features = features.unsqueeze(0)

        adj = None
        if "adj_matrix" in window:
            adj = torch.FloatTensor(window["adj_matrix"]).unsqueeze(0).to(self.device)

        # 2. Inference
        model.eval()
        with torch.no_grad():
            output = None

            # Check for generic 'actor' (RL Agents) vs forward (Supervised)
            if hasattr(model, "actor"):
                # DDPG / Hybrid
                try:
                    if adj is not None:
                        output = model.actor(features, adj)
                    else:
                        output = model.actor(features)
                except TypeError:
                    output = model.actor(features)

                # Actor returns (weights, hidden) or just weights?
                # DDPGActor returns (action, hidden)
                if isinstance(output, tuple):
                    # Direct weights from Actor
                    raw_weights = output[0].cpu().numpy()[0]  # [N]
                    return raw_weights
                else:
                    return output.cpu().numpy()[0]

            else:
                # TGNN / Supervised Models
                # Returns (preds, embeddings)
                preds, _ = model(features, adj, target_type=target_head)
                scores = preds.cpu().numpy()[0]  # [N]

                return self._calculate_top_k_weights(scores)

    def _calculate_top_k_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Dynamic Top-K + Softmax Weighting Strategy.
        """
        # Dynamic Top-K: Select top 30% of stocks, at least 1, max 10.
        ratio_k = int(self.n_stocks * 0.3)
        TOP_K = max(1, min(10, ratio_k))

        current_weights = np.zeros(self.n_stocks)

        # Get indices of top K scores
        top_k_indices = np.argsort(scores)[-TOP_K:]
        top_k_scores = scores[top_k_indices]

        # Softmax Weighting (Score-based)
        if top_k_scores.std() > 1e-6:
            z_scores = (top_k_scores - top_k_scores.mean()) / top_k_scores.std()
            exp_scores = np.exp(z_scores)
            top_k_weights = exp_scores / np.sum(exp_scores)
        else:
            top_k_weights = np.ones(TOP_K) / TOP_K

        current_weights[top_k_indices] = top_k_weights

        # Fallback for NaNs
        if np.isnan(current_weights).any():
            current_weights[top_k_indices] = 1.0 / TOP_K

        return current_weights
