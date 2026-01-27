import sys
import os
import yaml
import torch
import pandas as pd
from pathlib import Path

# Path setup
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing.data_processor import DataProcessor
from src.training.dataset import FinancialDataset
from src.utils.backtester import Backtester
from src.models.tgnn import TGNN
from src.models.hybrid import HybridAgent
from src.models.ddpg import DDPGAgent


def load_config(path="backtesting/config/config.yaml"):
    # Fix path relative to project root if needed
    if not os.path.exists(path):
        # Try finding it relative to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(script_dir, "config", "config.yaml")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    print("🚀 Starting Backtest Simulation...")

    # 1. Load Config
    config = load_config()
    model_name = config["project"].get("selected_model", "tgnn")
    print(f"🔧 Model: {model_name.upper()}")

    # 2. Data Preparation (Same as main.py)
    # Ideally reuse data loading logic. For now, minimal mock or file load.
    # We need REAL data for meaningful logs.
    # Logic copied from main.py, but using actual file if exists?
    # main.py generated mock data for demo.
    # If the user wants REAL results like in `results/03.../hybrid_trade_logs.csv`,
    # we assume `dataset` holds the relevant data.

    # Let's try to verify if we have data source.
    # config['paths']['processed_data'] points to "../data/processed_daily_5factor_model.csv"
    # DDPG comparison script used `processed_daily_5factor_model_10stocks_10years_20251127.csv`.

    # Let's use `main.py` logic to create data/dataset.
    # Note: If main.py used random data, the backtest results will be random.
    # But the user asked to reproduce the structure.

    print("\n[1/3] Loading Data...")
    # Resolve relative to project root
    project_root = Path(__file__).parent.parent.parent
    abs_data_path = (
        project_root / "data" / "processed_daily_5factor_model.csv"
    ).resolve()

    if abs_data_path.exists():
        print(f"      Loading from {abs_data_path}")
        full_df = pd.read_csv(abs_data_path)
    else:
        print("⚠️ Real data not found. Generating Mock Data for Demo.")
        dates = pd.date_range(start="2020-01-01", periods=1000, freq="D")
        symbols = config["data"]["stock_universes"]

        dfs = []
        for sym in symbols:
            df = pd.DataFrame(
                {
                    "Date": dates,
                    "Symbol": sym,
                    "Open": 100 + torch.randn(1000).numpy(),
                    "High": 110 + torch.randn(1000).numpy(),
                    "Low": 90 + torch.randn(1000).numpy(),
                    "Close": 105 + torch.randn(1000).numpy(),
                    "Volume": 10000,
                    "Sector": "Technology",
                }
            ).set_index("Date")
            dfs.append(df)
        full_df = pd.concat(dfs)

        # Process Mock Data
        processor = DataProcessor(config)
        full_df = processor.add_technical_indicators(full_df)
        full_df = processor.calculate_factors(full_df)
        full_df = processor.calculate_weighted_score(full_df)
        try:
            full_df = processor.merge_fama_french_factors(full_df)
        except Exception:
            pass  # Fama-French 데이터 로드 실패 시 무시
        full_df.dropna(inplace=True)

    # 3. Create Dataset
    dataset = FinancialDataset(config, full_df)
    print(f"      Windows Created: {len(dataset)}")

    # 4. Load Model
    print(f"\n[2/3] Loading Model ({model_name})...")

    if model_name == "hybrid":
        model = HybridAgent(config)
    elif model_name == "ddpg":
        model = DDPGAgent(config)
    else:
        model = TGNN(config)

    # Load Checkpoint logic
    # trainer saves to `results/{model_name}/best_model.pth` (or timestamped)
    # We try to find the latest 'best_model.pth' or generic name
    results_dir = Path(config["paths"]["results_dir"]) / model_name
    # Try different patterns
    checkpoints = list(results_dir.glob("best_model*.pth"))
    if checkpoints:
        # Sort by modification time
        latest_ckpt = max(checkpoints, key=os.path.getmtime)
        print(f"      Loading checkpoint: {latest_ckpt.name}")
        try:
            if hasattr(model, "load_state_dict"):
                # TGNN / Base Wrapper
                # Note: TGNN save might save full model or state_dict?
                # Trainer calls `model.save` which typically wraps torch.save.
                # Let's check BaseModel.save
                # It does torch.save(self.state_dict())
                model.load_state_dict(
                    torch.load(latest_ckpt, map_location=model.device)
                )
            elif hasattr(model, "actor"):
                # RL Agents might need special Loading if saved as a dict
                # Trainer.save_checkpoint saves dict {'actor': ..., 'critic': ...}
                checkpoint = torch.load(latest_ckpt, map_location=model.device)
                if isinstance(checkpoint, dict) and "actor" in checkpoint:
                    model.actor.load_state_dict(checkpoint["actor"])
                    model.critic.load_state_dict(checkpoint["critic"])
                else:
                    model.load_state_dict(checkpoint)
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}")
            print("      Running with initialized weights.")
    else:
        print("⚠️ No checkpoint found. Running with initialized weights.")

    # 5. Run Backtest
    print("\n[3/3] Running Backtest Strategies...")
    backtester = Backtester(config, model, dataset)

    # A. Buy & Hold
    res_bn = backtester.run_strategy("buy_and_hold")

    # B. Model Strategies
    # Check if TGNN (Multi-head)
    results_map = {"1/N Buy & Hold": res_bn}

    if hasattr(model, "heads"):
        for (
            head
        ) in model.heads:  # ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
            print(f"\n🔍 Testing Head: {head}")
            res = backtester.run_strategy("model", target_head=head)
            results_map[f"Model_{head}"] = res
    else:
        # RL or Hybrid (Single Policy usually, or default)
        res_model = backtester.run_strategy("model")
        results_map[f"{model_name.upper()} Strategy"] = res_model

    # 6. Save & Plot
    backtester.save_and_plot(results_map)

    print("\n🎉 Backtest Completed!")


if __name__ == "__main__":
    main()
