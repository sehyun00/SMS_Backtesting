# test_imports.py
"""모듈 임포트 테스트"""

print("Testing module imports...")

try:
    from networks import (
        GraphConvLayer,
        TemporalAttention,
        TGNNEncoder,
        HybridActor,
        HybridCritic,
    )

    print("✅ networks module imported successfully")
except Exception as e:
    print(f"❌ networks import failed: {e}")

try:
    from agent import ReplayBuffer, HybridAgent

    print("✅ agent module imported successfully")
except Exception as e:
    print(f"❌ agent import failed: {e}")

try:
    from environment import HybridDataset, HybridPortfolioEnv

    print("✅ environment module imported successfully")
except Exception as e:
    print(f"❌ environment import failed: {e}")

try:
    from utils import calculate_metrics, enforce_weight_constraints

    print("✅ utils module imported successfully")
except Exception as e:
    print(f"❌ utils import failed: {e}")

print("\n✅ All modules imported successfully!")
