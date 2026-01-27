import sys
import os
import yaml
import torch
import pandas as pd

# Path setup to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.data_processor import DataProcessor
from src.training.dataset import FinancialDataset
from src.training.trainer import Trainer
from src.models.tgnn import TGNN
from src.models.hybrid import HybridAgent
from src.models.ddpg import DDPGAgent


def load_config(path="config/config.yaml"):
    # Resolve absolute path relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    abs_path = os.path.join(script_dir, path)
    with open(abs_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    # 1. Load Config
    config = load_config()
    print(f"🚀 Starting Project: {config['project']['name']}")
    print(f"🔧 Device: {config['project']['device']}")

    # 2. Process Data (In-memory for demo, or load from file)
    print("\n[1/4] Preparing Data...")
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    symbols = config["data"]["stock_universes"]

    dfs = []
    for sym in symbols:
        df = pd.DataFrame(
            {
                "Date": dates,
                "Symbol": sym,
                "Open": 100 + torch.randn(100).numpy(),
                "High": 110 + torch.randn(100).numpy(),
                "Low": 90 + torch.randn(100).numpy(),
                "Close": 105 + torch.randn(100).numpy(),
                "Volume": 10000,
                "Sector": "Technology",
            }
        ).set_index("Date")
        dfs.append(df)

    full_df = pd.concat(dfs)

    # Run Processor
    processor = DataProcessor(config)
    full_df = processor.add_technical_indicators(full_df)
    full_df = processor.calculate_factors(full_df)
    full_df = processor.calculate_weighted_score(full_df)

    # Attempt to merge FF factors (might be NaN for dummy dates if no match)
    try:
        full_df = processor.merge_fama_french_factors(full_df)
    except Exception as e:
        print(f"⚠️ Fama-French merge failed (expected for dummy data): {e}")

    full_df.dropna(inplace=True)

    print(f"      Data Shape: {full_df.shape}")

    # 3. Create Dataset
    print("\n[2/4] Creating Dataset...")
    dataset = FinancialDataset(config, full_df)
    print(f"      Windows Created: {len(dataset)}")

    # 4. Initialize Model
    model_type = config["project"].get("selected_model", "tgnn").lower()
    print(f"\n[3/4] Initializing Model ({model_type.upper()})...")

    if model_type == "hybrid":
        model = HybridAgent(config)
    elif model_type == "ddpg":
        model = DDPGAgent(config)
    else:
        model = TGNN(config)

    # 5. Train
    print("\n[4/4] Starting Training...")
    trainer = Trainer(config, model, dataset)
    trainer.train()

    print("\n🎉 Experiment Completed Successfully!")


if __name__ == "__main__":
    main()
