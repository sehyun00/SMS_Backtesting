import torch


class PortfolioConstraints:
    """
    Enforces portfolio weight constraints using Iterative Projection.
    """

    def __init__(
        self, num_stocks: int, max_weight: float = 0.25, min_weight: float = 0.0
    ):
        self.num_stocks = num_stocks
        self.max_weight = max_weight
        self.min_weight = min_weight

    def enforce(self, weights: torch.Tensor, max_iter: int = 10) -> torch.Tensor:
        """
        Iterative projection:
        1. Clamp weights between min_weight and max_weight
        2. Redistribute excess/deficit to satisfy sum(weights) = 1.0
        """
        min_w = self.min_weight
        max_w = self.max_weight
        eps = 1e-4

        for _ in range(max_iter):
            # Step 1: Clamp
            weights_clamped = torch.clamp(weights, min_w, max_w)

            # Step 2: Check sum
            current_sum = weights_clamped.sum(dim=-1, keepdim=True)
            if torch.allclose(current_sum, torch.ones_like(current_sum), atol=eps):
                weights = weights_clamped
                break

            # Step 3: Redistribute deficit
            deficit = 1.0 - current_sum
            need_increase = deficit > 0

            room_to_grow = max_w - weights_clamped
            room_to_shrink = weights_clamped - min_w

            total_room_grow = room_to_grow.sum(dim=-1, keepdim=True)
            adjustment_grow = torch.where(
                total_room_grow > eps,
                deficit * (room_to_grow / (total_room_grow + 1e-8)),
                deficit / self.num_stocks,
            )

            total_room_shrink = room_to_shrink.sum(dim=-1, keepdim=True)
            adjustment_shrink = torch.where(
                total_room_shrink > eps,
                deficit * (room_to_shrink / (total_room_shrink + 1e-8)),
                deficit / self.num_stocks,
            )

            adjustment = torch.where(need_increase, adjustment_grow, adjustment_shrink)
            weights = weights_clamped + adjustment

        # Final safety clamp & normalize
        weights = torch.clamp(weights, min_w, max_w)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

        return weights
