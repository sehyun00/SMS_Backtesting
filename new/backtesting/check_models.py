import torch
import yaml
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import traceback

# Add new directory to path
sys.path.append(str(Path(__file__).parent))

from src.models.ddpg import DDPGAgent
from src.models.hybrid import HybridAgent
from src.models.tgnn import TGNN
from src.training.dataset import FinancialDataset
from src.training.rl_trainer import RLTrainer
from src.training.trainer import Trainer
from src.training.environment import PortfolioEnvironment


def create_mock_data():
    dates = pd.date_range(start="2010-01-01", periods=100, freq="ME")
    symbols = ["AAPL", "GOOGL", "MSFT"]
    feature_cols = ["Momentum1M", "Momentum6M", "Volatility", "RSI"]
    data = []

    for date in dates:
        for sym in symbols:
            row = {
                "Date": date,
                "Symbol": sym,
                "Sector": "Technology",
                "Open": 100,
                "High": 105,
                "Low": 95,
                "Close": 102,
                "Volume": 1000,
                "Momentum1M": np.random.randn(),
                "Momentum_Factor": np.random.randn(),
                "Value_Factor": np.random.randn(),
                "Beta_Factor": np.random.randn(),
                "Volatility_Factor": np.random.randn(),
            }
            # Add features
            for f_col in feature_cols:
                row[f_col] = np.random.randn()
            data.append(row)

    df = pd.DataFrame(data)
    df.set_index("Date", inplace=True)
    return df, symbols, feature_cols


def verify_model(model_name: str, df, symbols, feature_cols):
    print(f"\n🔍 Verifying Model: [{model_name.upper()}] ...")

    # 1. Load & Mod Config
    config_path = Path(__file__).parent / "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["project"]["selected_model"] = model_name
    config["data"]["stock_universes"] = symbols
    config["data"]["features"] = feature_cols
    # Remove factors from config to avoid Dataset looking for them (if missing in mock)
    # Mock data HAS factors now, but let's be safe if dataset logic changed
    if "factors" in config["data"]:
        # We supplied factor cols in mock data, so we can keep it?
        # Dataset.py looks for "Beta_Factor" etc.
        # Mock data has them.
        pass

    # 2. Create Dataset
    # TGNN/Generic Dataset
    # DDPG/Hybrid use FinancialDataset too.
    dataset = FinancialDataset(config=config, data=df, mode="train")
    print(f"   ✅ Dataset created ({len(dataset)} windows)")

    # 3. Initialize Model & Trainer
    try:
        if model_name == "tgnn":
            model = TGNN(config)
            print("   ✅ TGNN Initialized")
            # TGNN uses Supervised Trainer
            trainer = Trainer(config, model, dataset)
            print("   ✅ Supervised Trainer Initialized")

            # Run 1 Epoch
            model.train()
            batch = next(iter(trainer.dataloader))
            feat = batch["features"].to(model.device)
            adj = batch["adj_matrix"].to(model.device)
            labels = batch["labels"].to(model.device)

            preds, _ = model(feat, adj)
            loss = trainer.criterion(preds, labels)
            loss.backward()
            print(f"   ✅ Forward/Backward Pass (Loss: {loss.item():.4f})")

        elif model_name in ["hybrid", "ddpg"]:
            if model_name == "hybrid":
                agent = HybridAgent(config)
            else:  # ddpg
                agent = DDPGAgent(config)
            print(f"   ✅ {model_name.upper()} Agent Initialized")

            env = PortfolioEnvironment(dataset)
            trainer = RLTrainer(agent, env, config)
            print("   ✅ RL Trainer Initialized")

            stats = trainer.train_episode(noise_std=0.1)
            print(f"   ✅ Episode Complete (Reward: {stats['episode_reward']:.2f})")

        print(f"🎉 Verification PASSED for {model_name}")
        return True

    except Exception as e:
        print(f"❌ Verification FAILED for {model_name}")
        traceback.print_exc()
        return False


def verify_asset_agnostic():
    print(f"\n🔍 Verifying Asset-Agnostic Properties ...")

    # 1. Create Data with DIFFERENT number of stocks
    # Init Config has 3 stocks (AAPL, GOOGL, MSFT)
    # We want to test if model accepts 5 stocks
    df, _, features = create_mock_data()  # Default 3

    # Create 5-stock mock data
    dates = pd.date_range(start="2010-01-01", periods=10, freq="ME")
    symbols_5 = ["A", "B", "C", "D", "E"]
    data_5 = []
    for date in dates:
        for sym in symbols_5:
            row = {
                "Date": date,
                "Symbol": sym,
                "Sector": "Tech",
                "Open": 100,
                "High": 105,
                "Low": 95,
                "Close": 100,
                "Volume": 1000,
            }
            for f in features:
                row[f] = np.random.randn()
            # Factors
            for f in [
                "Momentum_Factor",
                "Value_Factor",
                "Beta_Factor",
                "Volatility_Factor",
            ]:
                row[f] = np.random.randn()
            data_5.append(row)
    df_5 = pd.DataFrame(data_5)
    df_5.set_index("Date", inplace=True)

    # Config
    config_path = Path(__file__).parent / "config/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config["data"]["stock_universes"] = symbols_5  # 5 stocks
    config["data"]["features"] = features

    # Dataset with 5 stocks
    dataset_5 = FinancialDataset(config=config, data=df_5, mode="train")

    # Initialize Models (Agent will see config has 5 stocks, but we want to see if it runs)
    # Actually, let's init with 3, and run with 5.
    config["data"]["stock_universes"] = ["A", "B", "C"]  # Agent thinks 3

    print("   Testing DDPG (Init 3 -> Run 5)...")
    try:
        agent = DDPGAgent(config)
        # Manually create batch of 5
        batch = next(iter(torch.utils.data.DataLoader(dataset_5, batch_size=2)))
        # batch features: [B, 5, T, F]
        x = batch["features"].to(agent.device)
        w = agent.forward(x)  # Should output [B, 5]
        if w.shape[1] == 5:
            print("   ✅ DDPG handled 5 stocks successfully!")
        else:
            print(f"   ❌ DDPG output shape mismatch: {w.shape}")
            return False

    except Exception as e:
        print(f"   ❌ DDPG Failed: {e}")
        traceback.print_exc()
        return False

    print("   Testing Hybrid (Init 3 -> Run 5)...")
    try:
        agent = HybridAgent(config)
        batch = next(iter(torch.utils.data.DataLoader(dataset_5, batch_size=2)))
        x = batch["features"].to(agent.device)
        adj = batch["adj_matrix"].to(agent.device)  # [B, 5, 5]

        w = agent.forward(x, adj)
        if w.shape[1] == 5:
            print("   ✅ Hybrid handled 5 stocks successfully!")
        else:
            print(f"   ❌ Hybrid output shape mismatch: {w.shape}")
            return False

    except Exception as e:
        print(f"   ❌ Hybrid Failed: {e}")
        traceback.print_exc()
        return False

    print("🎉 Asset-Agnostic Verification PASSED")
    return True


def verify_all():
    df, symbols, features = create_mock_data()

    models = ["tgnn", "ddpg", "hybrid"]
    results = {}

    for m in models:
        results[m] = verify_model(m, df, symbols, features)

    # ADDED: Test Asset Agnostic
    results["asset_agnostic"] = verify_asset_agnostic()

    print("\n" + "=" * 30)
    print("FINAL RESULTS")
    print("=" * 30)
    all_pass = True
    for m, res in results.items():
        status = "PASSED" if res else "FAILED"
        print(f"{m.upper():<15}: {status}")
        if not res:
            all_pass = False

    if all_pass:
        print("\n✅ All models verified successfully!")
    else:
        print("\n⚠️ Some models failed verification.")


if __name__ == "__main__":
    verify_all()
