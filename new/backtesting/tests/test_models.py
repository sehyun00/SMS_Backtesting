import sys
import os
import torch
import pytest
import yaml

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.tgnn import TGNN
from src.models.hybrid import HybridAgent
from src.models.ddpg import DDPGAgent


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_tgnn_forward_pass():
    config = load_config()
    config["project"]["device"] = "cpu"

    model = TGNN(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    num_features = model.num_features

    x = torch.randn(batch_size, num_stocks, seq_len, num_features)
    adj = torch.ones(batch_size, num_stocks, num_stocks)

    print(f"Testing TGNN with input shape: {x.shape}")

    preds, embeddings = model(x, adj)

    assert preds.shape == (batch_size, num_stocks)
    assert embeddings.shape == (
        batch_size,
        num_stocks,
        config["model"]["tgnn"]["hidden_dim"],
    )

    print("✅ TGNN Forward Pass Successful")


def test_hybrid_forward_pass():
    config = load_config()
    config["project"]["device"] = "cpu"

    model = HybridAgent(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    num_features = model.num_features

    x = torch.randn(batch_size, num_stocks, seq_len, num_features)
    adj = torch.ones(batch_size, num_stocks, num_stocks)

    print(f"Testing Hybrid Agent with input shape: {x.shape}")

    # Test Predict (Actor)
    weights = model.predict({"features": x, "adj_matrix": adj})

    assert weights.shape == (batch_size, num_stocks)
    # Check sum = 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    print("✅ Hybrid Agent Predict Successful")


def test_ddpg_forward_pass():
    config = load_config()
    config["project"]["device"] = "cpu"

    model = DDPGAgent(config)

    batch_size = 2
    num_stocks = len(config["data"]["stock_universes"])
    seq_len = config["data"]["window_size"]
    # For DDPG, Input Num Features is implicitly handled in dataset but here we mock
    # Wait, TGNN/Hybrid num_features logic in test was simple. DDPG flattens window.
    # We must ensure x has same shape [B, N, T, F]

    # Helper to calculate num_features as in Dataset
    raw_features = len(config["data"]["features"])
    if "factors" in config["data"]:
        raw_features += 4

    x = torch.randn(batch_size, num_stocks, seq_len, raw_features)

    print(f"Testing DDPG Agent with input shape: {x.shape}")

    weights = model.predict({"features": x})

    assert weights.shape == (batch_size, num_stocks)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(batch_size), atol=1e-5)

    print("✅ DDPG Agent Predict Successful")


if __name__ == "__main__":
    test_tgnn_forward_pass()
    test_hybrid_forward_pass()
    test_ddpg_forward_pass()
