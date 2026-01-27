import torch
import torch.nn.functional as F


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor = None,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Vectorized Pairwise Ranking Loss.
    Ensures correct relative ordering of stock returns.
    """

    if active_mask is not None:
        # Mask inactive stocks (set to 0, though slicing is better if indices matched)
        # Here we use masking logic similar to legacy
        predictions = predictions.clone()
        labels = labels.clone()
        predictions[~active_mask] = 0
        labels[~active_mask] = 0

    # Vectorized pairwise comparison
    # [batch, n, 1] - [batch, 1, n] = [batch, n, n]
    pred_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)

    # Violation: label_diff > 0 but pred_diff < margin
    # We want pred_diff to be large when label_diff is positive
    # Use sign of label_diff to determine direction
    violation = torch.relu(margin - pred_diff * torch.sign(label_diff))

    # Exclude ties (label_diff == 0)
    mask = (label_diff != 0).float()

    # Upper triangular only (avoid double counting)
    triu_mask = torch.triu(torch.ones_like(mask), diagonal=1)
    mask = mask * triu_mask

    if active_mask is not None:
        # Mask out pairs where either stock is inactive
        active_pair_mask = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
        mask = mask * active_pair_mask.float()

    loss = (violation * mask).sum() / (mask.sum() + 1e-8)

    return loss


def combined_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    active_mask: torch.Tensor = None,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> torch.Tensor:
    """
    Combined MSE + Ranking Loss.
    """
    # MSE Loss
    if active_mask is not None:
        mse = F.mse_loss(predictions[active_mask], labels[active_mask])
    else:
        mse = F.mse_loss(predictions, labels)

    # Ranking Loss
    rank_loss = pairwise_ranking_loss(predictions, labels, active_mask)

    return alpha * mse + beta * rank_loss
