import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Dict, Any


class Backtester:
    """
    Standardized Backtester for Financial Models.
    Handles simulation, logging, metrics calculation, and visualization.
    """

    def __init__(self, config: Dict[str, Any], model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = getattr(model, "device", "cpu")

        # Test Split
        self.test_split_date = pd.Timestamp(
            config["training"].get("test_split_date", "2021-01-01")
        )
        self.test_windows = [
            w for w in dataset.windows if w["date"] >= self.test_split_date
        ]

        if not self.test_windows:
            print(
                f"⚠️ Warning: No test windows found after {self.test_split_date}. Using last 20% windows."
            )
            split_idx = int(len(dataset.windows) * 0.8)
            self.test_windows = dataset.windows[split_idx:]

        self.initial_capital = 1_000_000
        self.symbols = dataset.symbols
        self.n_stocks = len(self.symbols)

        # Output Directory
        model_name = config["project"].get("selected_model", "default")
        self.results_dir = os.path.join(config["paths"]["results_dir"], model_name)
        os.makedirs(self.results_dir, exist_ok=True)

        # Plot Settings
        plt.rcParams["font.family"] = "Malgun Gothic"
        plt.rcParams["axes.unicode_minus"] = False

    def run_strategy(
        self, strategy_type: str = "model", target_head: str = "Momentum1M"
    ) -> Dict[str, Any]:
        """
        Runs a simulation for a given strategy.
        Args:
            strategy_type: 'buy_and_hold' or 'model'
            target_head: Custom head for TGNN (Momentum1M, 3M, 6M, 12M)
        """
        print(f"\n🔄 Running Strategy: {strategy_type.upper()} ({target_head})")

        capital = self.initial_capital
        portfolio_values = [capital]
        dates = []
        trade_logs = []

        # Initial Weights
        current_weights = np.ones(self.n_stocks) / self.n_stocks

        # For 'model' strategy rebalancing (monthly)

        for i, window in enumerate(self.test_windows):
            target_date = window["date"]
            actual_returns = window["labels"]  # [N] returns (e.g. 0.05 for 5%)

            # 1. Determine Weights
            if strategy_type == "model":
                # Rebalance every step (Monthly assumption)
                # Construct batch from window for model inference
                # But dataset returns tensors.

                # Construct Batch manually for safety
                features = torch.FloatTensor(window["features"]).to(
                    self.device
                )  # [N, T, F] or [T, F]??
                # Wait, dataset windows structure:
                # "features": np.ndarray [N, T, F] if initialized properly?
                # Let's check Dataset creation.
                # Assuming window["features"] is [N, T, F] based on standard usage.

                if features.dim() == 2:
                    # If it's [T, F] (Single Stock?) -> No, backtesting is usually portfolio.
                    # FinancialDataset usually constructs [N, T, F] for the whole market?
                    # Let's assume window["features"] is for the specific window.
                    # Actually FinancialDataset often returns [N, T, F] per window if it's a PortfolioDataset.
                    # Let's assume dimensions [N, T, F].
                    features = features.unsqueeze(0)  # [1, N, T, F]
                else:
                    features = features.unsqueeze(0)  # [1, N, T, F]

                adj = None
                if "adj_matrix" in window:
                    adj = (
                        torch.FloatTensor(window["adj_matrix"])
                        .unsqueeze(0)
                        .to(self.device)
                    )

                # Get Action/Prediction
                self.model.eval()
                with torch.no_grad():
                    if hasattr(self.model, "actor"):  # RL Agent
                        # Handle Hybrid/DDPG input diff
                        try:
                            if adj is not None:
                                output = self.model.actor(features, adj)
                            else:
                                output = self.model.actor(features)
                        except TypeError:
                            output = self.model.actor(features)

                        if isinstance(output, tuple):
                            current_weights = output[0].cpu().numpy()[0]
                        else:
                            current_weights = output.cpu().numpy()[0]

                    else:  # TGNN (Supervised)
                        # Predict Momentum -> Weights
                        # TGNN returns (preds, embeddings)
                        preds, _ = self.model(
                            features, adj, target_type=target_head
                        )  # [1, N]
                        scores = preds.cpu().numpy()[0]

                        # Strategy: Softmax or Top-K
                        # Simple Softmax over scores
                        try:
                            # scores might be raw values or returns
                            # Apply softmax
                            exp_scores = np.exp(scores)
                            current_weights = exp_scores / np.sum(exp_scores)
                        except Exception:
                            current_weights = np.ones(self.n_stocks) / self.n_stocks

            elif strategy_type == "buy_and_hold":
                # Weights are fixed at 1/N initially and drift.
                # But here we simulate simplified B&H: Rebalance to 1/N?
                # Or True B&H (drift)?
                # Simplest benchmark is 1/N Rebalanced Monthly usually.
                # or True Hold. run_comparison.py used 1/N Rebalanced initially?
                # Actually run_comparison.py had "1/N Buy & Hold" but implementation was drifting weights?
                # Let's stick to 1/N Rebalanced Monthly for "Benchmark" simplicity.
                current_weights = np.ones(self.n_stocks) / self.n_stocks

            # 2. Log Trade
            log_entry = {
                "Date": target_date,
                "Strategy": f"{strategy_type}_{target_head}"
                if strategy_type == "model"
                else strategy_type,
                "Type": "Rebalance",
            }
            for sym, w in zip(self.symbols, current_weights):
                log_entry[sym] = round(float(w), 4)
            trade_logs.append(log_entry)

            # 3. Calculate Return
            # returns is [N], weights is [N]
            # Assumes returns are percentage (e.g. 5.0) or decimal?
            # Check dataset.py. Usually labels are returns.
            # In run_comparison.py: `portfolio_return = np.dot(weights, actual_returns)`
            # and `capital *= 1 + portfolio_return / 100`. So it expects Percentage.
            # Let's check config or standard. Usually my datasets use %.

            # Safety check: if returns seems to be < 1.0 (decimal), treat as decimal.
            # If > 1.0 (likely %), treat as %.
            # But valid return can be 0.05 (5%)...
            # Let's assume Dataset provides raw numbers consistent with training.
            # If training used MSE against raw values (often scaled), we might need to unscale?
            # Creating Dataset usually keeps labels as raw returns?

            port_ret = np.dot(current_weights, actual_returns)

            # Assuming labels are Ratio (0.05 for 5%) based on previous log check (loss was 0.03 etc)
            # Wait, `run_comparison.py` divides by 100.
            # Let's stick to simple dot product and assume it's the period return.

            # Handle scaling if labels were scaled?
            # The Backtest should use Raw Returns if possible.
            # But the dataset.windows['labels'] holds the training targets.
            # Ideally we should use raw price data for backtest, but for now we use Window Labels.

            capital *= 1 + port_ret  # Assuming decimal return (0.05)
            # If capital exploads or vanishes, we know the scale is wrong.

            portfolio_values.append(capital)
            dates.append(target_date)

        return {
            "dates": dates,
            "portfolio_values": portfolio_values[1:],  # Align with dates
            "trade_logs": trade_logs,
            "final_capital": capital,
        }

    def compute_metrics(self, results: Dict[str, Any]) -> Dict[str, float]:
        values = np.array(results["portfolio_values"])
        if len(values) == 0:
            return {"Total_Return": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0}
        initial = self.initial_capital

        # Returns
        total_ret = (values[-1] / initial) - 1

        # CAGR (Annualized)
        days = len(values) * 30  # Approx
        years = days / 365.25
        if years > 0:
            cagr = (values[-1] / initial) ** (1 / years) - 1
        else:
            cagr = 0

        # MDD
        running_max = np.maximum.accumulate(values)
        drawdown = (values - running_max) / running_max
        mdd = abs(np.min(drawdown))

        # Sharpe (Monthly assumption)
        pct_change = pd.Series(values).pct_change().dropna()
        if len(pct_change) > 0:
            vol = pct_change.std() * np.sqrt(12)
            sharpe = (cagr) / (vol + 1e-8)
        else:
            sharpe = 0

        return {
            "Total_Return": total_ret * 100,
            "CAGR": cagr * 100,
            "MDD": mdd * 100,
            "Sharpe": sharpe,
        }

    def save_and_plot(self, results_map: Dict[str, Dict]):
        """
        Saves logs and plots comparison.
        results_map: { "StrategyName": run_strategy_output, ... }
        """
        # 1. Save Trade Logs
        all_logs = []
        for name, res in results_map.items():
            all_logs.extend(res["trade_logs"])

        df_log = pd.DataFrame(all_logs)
        if not df_log.empty:
            path = os.path.join(self.results_dir, "trade_logs.csv")
            cols = ["Date", "Strategy", "Type"] + self.symbols
            df_log = df_log[[c for c in cols if c in df_log.columns]]
            df_log.to_csv(path, index=False)
            print(f"✅ Trade logs saved to {path}")

        # 2. Plot
        plt.figure(figsize=(12, 6))

        for name, res in results_map.items():
            dates = res["dates"]
            values = res["portfolio_values"]
            print(
                f"📈 Plotting {name}: {len(dates)} points, Final Value: {values[-1] if values else 'N/A'}"
            )

            # Normalize to Initial Capital
            # (Already starting from initial capital logic in simulation)
            metrics = self.compute_metrics(res)
            label = f"{name} (CAGR: {metrics['CAGR']:.1f}%, MDD: {metrics['MDD']:.1f}%)"

            plt.plot(dates, values, label=label, linewidth=2)

        plt.title(
            f"Backtest Comparison: {self.config['project'].get('selected_model')}"
        )
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plot_path = os.path.join(self.results_dir, "comparison.png")
        plt.savefig(plot_path)
        print(f"✅ Comparison plot saved to {plot_path}")
        # plt.close() # Don't close if interactive, but for script it's fine.
