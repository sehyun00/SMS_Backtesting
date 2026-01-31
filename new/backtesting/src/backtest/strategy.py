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
        Executes model inference and applies weighting logic.

        [모델별 출력 형식 차이]
        - DDPG/Hybrid: Actor가 이미 Softmax 적용된 weights 반환 → 그대로 사용
        - TGNN: Raw scores 반환 → _calculate_softmax_weights()로 변환 필요

        Returns:
            np.ndarray: 포트폴리오 비중 (sum ≈ 1.0)
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

                return self._calculate_softmax_weights(scores)

    def _calculate_softmax_weights(self, scores: np.ndarray) -> np.ndarray:
        """
        Full Universe Softmax Strategy (Research Standard).
        Applies Softmax to ALL stock scores to determine weights.
        """
        # score-based weighting (Softmax)
        # 1. Normalize scores (Z-score) to prevent softmax saturation
        if scores.std() > 1e-6:
            z_scores = (scores - scores.mean()) / scores.std()
        else:
            z_scores = scores - scores.mean()

        # 2. Apply Softmax with temperature (optional, default 1.0)
        # Using a slight temperature > 1 can smooth out extreme bets if needed, but 1.0 is standard.
        exp_scores = np.exp(z_scores)
        weights = exp_scores / np.sum(exp_scores)

        return weights
