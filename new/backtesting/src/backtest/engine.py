import numpy as np
import pandas as pd
import torch
import os
from typing import Dict, Any, List

from .strategy import StrategyHandler
from .visualization import Visualizer
from .metrics import compute_metrics


class Backtester:
    """
    Refactored Backtester Engine.
    Orchestrates simulation, strategy execution, and result aggregation.
    """

    def __init__(self, config: Dict[str, Any], model, dataset):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.device = getattr(model, "device", "cpu")
        self.symbols = dataset.symbols
        self.n_stocks = len(self.symbols)

        # Output Setup
        model_name = config["project"].get("selected_model", "default")
        results_dir = os.path.join(config["paths"]["results_dir"], model_name)
        self.visualizer = Visualizer(results_dir)

        # Strategy Setup
        self.strategy_handler = StrategyHandler(self.n_stocks, self.device)

        # Simulation Params
        self.initial_capital = 1_000_000

        # Prepare Data
        self.test_windows = self._get_test_windows()
        # Shift returns to t+1 to align with execution at t (Close)
        self.daily_returns_df = self._precalculate_returns().shift(-1).fillna(0.0)

        # Subset Masking Setup
        self.valid_mask = None
        if "test_valid_subset" in config["data"]:
            subset_symbols = config["data"]["test_valid_subset"]
            self.valid_mask = np.zeros(self.n_stocks)
            mapped_count = 0
            for sym in subset_symbols:
                if sym in self.symbols:
                    idx = self.symbols.index(sym)
                    self.valid_mask[idx] = 1.0
                    mapped_count += 1
            print(
                f"      [Backtester] Subset Masking Active: {mapped_count}/{self.n_stocks} stocks valid."
            )

    def _get_test_windows(self):
        test_split_date = pd.Timestamp(
            self.config["training"].get("test_split_date", "2021-01-01")
        )
        windows = [
            w
            for w in self.dataset.windows
            if pd.to_datetime(w["date"]) >= test_split_date
        ]

        if not windows:
            print(
                f"⚠️ Warning: No test windows found after {test_split_date}. Using last 20%."
            )
            split_idx = int(len(self.dataset.windows) * 0.8)
            windows = self.dataset.windows[split_idx:]

        return windows

    def _precalculate_returns(self):
        print("      [Init] Pre-calculating Daily Returns...")
        # Create empty DF with all unique dates
        dates = self.dataset.df.index.unique().sort_values()
        daily_returns = pd.DataFrame(index=dates)

        for sym in self.symbols:
            # Filter symbol data
            sym_df = self.dataset.df[self.dataset.df["Symbol"] == sym]
            if "Close" in sym_df.columns:
                # Calculate pct_change properly
                # Note: pct_change on filtered DF preserves index (Date)
                daily_returns[sym] = sym_df["Close"].pct_change()
            else:
                daily_returns[sym] = 0.0

        daily_returns.fillna(0.0, inplace=True)
        return daily_returns

    def run_strategy(
        self,
        strategy_type: str = "model",
        target_head: str = "Momentum1M",
        rebalance_interval: int = 1,
    ) -> Dict[str, Any]:
        """
        Runs strategy with specified rebalancing interval (in trading days).
        Default is 1 (Daily Rebalancing).
        """
        print(
            f"\n🔄 Running Strategy: {strategy_type.upper()} ({target_head}) | Interval: {rebalance_interval}d"
        )

        capital = self.initial_capital
        portfolio_values = [capital]
        dates = []
        trade_logs = []

        # Initial Weights
        current_weights = np.ones(self.n_stocks) / self.n_stocks

        # Apply initial mask if exists
        if self.valid_mask is not None:
            current_weights = current_weights * self.valid_mask
            current_weights /= np.sum(current_weights) + 1e-8

        for i, window in enumerate(self.test_windows):
            target_date = window["date"]

            # 1. Update Weights (Rebalance)
            # Only rebalance if interval is met OR it's the first step
            trade_type = "Hold"
            if i % rebalance_interval == 0:
                trade_type = "Rebalance"
                try:
                    prev_weights = current_weights.copy()
                    current_weights = self.strategy_handler.get_weights(
                        strategy_type, self.model, window, target_head
                    )

                    # Apply Subset Masking (Zero out missing stocks)
                    if self.valid_mask is not None:
                        current_weights = current_weights * self.valid_mask

                    # Force Normalize to prevent Cash Drag (Model might output sum < 1.0 due to float prec)
                    current_weights /= np.sum(current_weights) + 1e-8

                    if i > 0:
                        turnover = np.sum(np.abs(current_weights - prev_weights)) / 2
                        # if turnover >= 1e-4:
                        #     print(
                        #         f"      [Rebalance] Step {i}: Turnover {turnover:.4f}"
                        #     )

                except Exception as e:
                    print(f"      [Error] Weight calc failed at step {i}: {e}")
                    # On error, we might fallback to equal weights or keep previous
                    # For safety/research parity, fallback to equal is mostly standard or hold
                    current_weights = np.ones(self.n_stocks) / self.n_stocks

            # 2. Log Trade
            log_entry = {
                "Date": target_date,
                "Strategy": f"{strategy_type}_{target_head}"
                if strategy_type == "model"
                else strategy_type,
                "Type": trade_type,
            }

            # Determine symbols to log
            if (
                self.valid_mask is not None
                and "test_valid_subset" in self.config["data"]
            ):
                logging_symbols = self.config["data"]["test_valid_subset"]
            else:
                logging_symbols = self.symbols

            # Add weights to log
            for sym in logging_symbols:
                if sym in self.symbols:
                    idx = self.symbols.index(sym)
                    if idx < len(current_weights):
                        log_entry[sym] = round(float(current_weights[idx]), 4)

            trade_logs.append(log_entry)

            # 3. Calculate Return
            # Get daily return vector for this date (already shifted to t+1)
            try:
                if target_date in self.daily_returns_df.index:
                    daily_returns = self.daily_returns_df.loc[
                        target_date, self.symbols
                    ].values
                else:
                    daily_returns = np.zeros(self.n_stocks)
            except Exception:
                daily_returns = np.zeros(self.n_stocks)

            # PnL Update
            port_ret = np.dot(current_weights, daily_returns)
            capital *= 1 + port_ret

            portfolio_values.append(capital)
            dates.append(target_date)

            # 4. Drift Weights (Price Impact)
            # w_i_new = w_i_old * (1 + r_i) / (1 + port_ret)
            if (1 + port_ret) != 0:
                current_weights = current_weights * (1 + daily_returns) / (1 + port_ret)

            # Normalize just in case of float drift
            current_weights /= np.sum(current_weights)

        return {
            "dates": dates,
            "portfolio_values": portfolio_values[1:],  # align with dates
            "trade_logs": trade_logs,
            "final_capital": capital,
        }

    def compute_metrics(self, results):
        return compute_metrics(results["portfolio_values"], self.initial_capital)

    def save_and_plot(self, results):
        self.visualizer.save_logs(results, self.symbols)
        self.visualizer.plot_comparison(results)
