import torch
import numpy as np
from src.models.ddpg.agent import DDPGAgent


def test_ddpg_refactor():
    print("🚀 Verifying Asset-Agnostic DDPG...")

    # Mock Config
    config = {
        "data": {
            "stock_universes": ["S" + str(i) for i in range(55)],  # Init with 55
            "features": ["Close", "Volume", "High", "Low"],
            "window_size": 20,
            "factors": {"weights": {"Momenum": 1.0}},
        },
        "training": {
            "gamma": 0.99,
            "tau": 0.005,
            "lr_actor": 1e-4,
            "lr_critic": 1e-3,
            "batch_size": 32,
        },
        "model": {"ddpg": {}},  # Empty/Default
    }

    # Initialize Agent
    agent = DDPGAgent(config)
    print("✅ Agent Initialized (Default N=55 in config, but model should be agnostic)")

    # Test Case 1: Train-like Input (N=55)
    B, N, T, F = 2, 55, 20, 5  # F=4 + 1(Factor)
    state_55 = torch.randn(B, N, T, F).to(agent.device)

    print(f"\n🧪 Testing Input N={N}...")
    weights_55 = agent.forward(state_55)
    print(f"   Actor Output Shape: {weights_55.shape} (Expected: [{B}, {N}])")
    assert weights_55.shape == (B, N), f"Shape Mismatch: {weights_55.shape}"

    # Critic Test
    action_55 = torch.randn(B, N).to(agent.device)
    q_55 = agent.critic(state_55, action_55)
    print(f"   Critic Output Shape: {q_55.shape} (Expected: [{B}, 1])")
    assert q_55.shape == (B, 1), f"Critic Shape Mismatch: {q_55.shape}"

    # Test Case 2: Test-like Input (N=10)
    N_test = 10
    state_10 = torch.randn(B, N_test, T, F).to(agent.device)

    print(f"\n🧪 Testing Input N={N_test} (New Universe)...")
    weights_10 = agent.forward(state_10)
    print(f"   Actor Output Shape: {weights_10.shape} (Expected: [{B}, {N_test}])")
    assert weights_10.shape == (B, N_test), f"Shape Mismatch: {weights_10.shape}"

    # Critic Test N=10
    action_10 = torch.randn(B, N_test).to(agent.device)
    q_10 = agent.critic(state_10, action_10)
    print(f"   Critic Output Shape: {q_10.shape} (Expected: [{B}, 1])")
    assert q_10.shape == (B, 1), f"Critic Shape Mismatch: {q_10.shape}"

    print("\n✅ All Tests Passed! Model is Asset-Agnostic.")


if __name__ == "__main__":
    try:
        test_ddpg_refactor()
    except Exception as e:
        print(f"\n❌ Test Failed: {e}")
        import traceback

        traceback.print_exc()
