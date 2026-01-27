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


def verify_all():
    df, symbols, features = create_mock_data()

    models = ["tgnn", "ddpg", "hybrid"]
    results = {}

    for m in models:
        results[m] = verify_model(m, df, symbols, features)

    print("\n" + "=" * 30)
    print("FINAL RESULTS")
    print("=" * 30)
    all_pass = True
    for m, res in results.items():
        status = "PASSED" if res else "FAILED"
        print(f"{m.upper():<10}: {status}")
        if not res:
            all_pass = False

    if all_pass:
        print("\n✅ All models verified successfully!")
    else:
        print("\n⚠️ Some models failed verification.")


if __name__ == "__main__":
    verify_all()
