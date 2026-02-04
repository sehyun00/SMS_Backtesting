import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FinancialDataset(Dataset):
    """
    Dataset for Temporal Graph Neural Networks (Context-Aware).
    Separates Local (Price) and Global (Macro) features to prevent signal dilution.
    """

    def __init__(self, config: Dict[str, Any], data: pd.DataFrame, mode: str = "train"):
        """
        Args:
            config: Global configuration.
            data: Preprocessed DataFrame (must contain 'Date', 'Symbol', 'Sector', and features).
            mode: 'train' or 'test'.
        """
        self.config = config
        self.df = data.copy()

        # Ensure Date is datetime
        if not np.issubdtype(self.df.index.dtype, np.datetime64):
            self.df.index = pd.to_datetime(self.df.index)

        self.window_size = config["data"]["window_size"]

        # 1. Feature Separation
        # Local Features (Price/Volume) - Stock Specific
        self.price_cols = list(config["data"]["features"])

        # Global Features (Macro) - Shared across market
        self.macro_cols = []
        if "factors" in config["data"] and config["data"]["factors"]:
            self.macro_cols = [
                "Mkt_RF",  # 시장 초과수익률
                "SMB",  # Size (규모 효과)
                "HML",  # Value (가치 효과)
                "RMW",  # Profitability (수익성)
                "CMA",  # Investment (투자 보수성)
            ]

        # Combined columns for data extraction
        self.all_features = self.price_cols + self.macro_cols

        # Enforce symbols from config to match Model architecture
        self.symbols = config["data"]["stock_universes"]
        self.num_stocks = len(self.symbols)

        # Create Windows
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict[str, Any]]:
        windows = []
        # Group by Date to get snapshots
        # Use simple sliding window on unique dates
        dates = sorted(self.df.index.unique())

        for i in range(len(dates) - self.window_size - 1):
            window_dates = dates[i : i + self.window_size]
            next_window_dates = dates[i + 1 : i + 1 + self.window_size]
            target_date = dates[i + self.window_size]  # Predict next step (label)

            # 1. Extract All Features first
            # [N, T, F_total]
            all_features_array = self._get_features_for_window(
                window_dates, self.all_features
            )

            # 2. Next State (for RL)
            next_all_features_array = self._get_features_for_window(
                next_window_dates, self.all_features
            )

            # 3. Create Graph (Adjacency Matrix)
            last_date = window_dates[-1]
            adj_matrix = self._create_adj_matrix(self.df.loc[str(last_date)])

            # 4. Extract Labels
            target_df = self.df.loc[str(target_date)]
            labels_list = []
            target_cols = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]

            for symbol in self.symbols:
                row = target_df[target_df["Symbol"] == symbol]
                if not row.empty:
                    vals = [
                        row[col].values[0] if col in row else 0.0 for col in target_cols
                    ]
                    labels_list.append(vals)
                else:
                    labels_list.append([0.0] * len(target_cols))

            # 5. Split Features (Context-Aware)
            num_price = len(self.price_cols)

            # [N, T, F_price]
            prices = all_features_array[:, :, :num_price]
            next_prices = next_all_features_array[:, :, :num_price]

            # [N, T, F_macro] -> Macro is same for all stocks, but we keep structure for batching
            # Ideally [T, F_macro], but for compatibility [N, T, F_macro] is fine (Model will slice/pool)
            macro = all_features_array[:, :, num_price:]
            next_macro = next_all_features_array[:, :, num_price:]

            windows.append(
                {
                    "prices": torch.FloatTensor(prices),
                    "macro": torch.FloatTensor(macro),
                    "next_prices": torch.FloatTensor(next_prices),
                    "next_macro": torch.FloatTensor(next_macro),
                    "adj_matrix": torch.FloatTensor(adj_matrix),
                    "labels": torch.FloatTensor(labels_list),  # [N, 4]
                    "date": str(target_date),
                    # Deprecated legacy keys for backward compat if needed (but we updated trainer)
                    "features": torch.FloatTensor(all_features_array),
                    "next_features": torch.FloatTensor(next_all_features_array),
                }
            )

        return windows

    def _get_features_for_window(self, window_dates, div_features):
        current_window_df = self.df.loc[window_dates]
        batch_features = []

        for symbol in self.symbols:
            stock_df = current_window_df[current_window_df["Symbol"] == symbol]

            # Handle missing data (e.g. not listed yet)
            if len(stock_df) < self.window_size:
                pad_len = self.window_size - len(stock_df)
                vals = stock_df[div_features].values
                vals = np.pad(vals, ((pad_len, 0), (0, 0)), mode="constant")
                batch_features.append(vals)
            else:
                batch_features.append(stock_df[div_features].values)

        # [N, T, F]
        features_array = np.array(batch_features)

        # Robust Z-Score Normalization per Window
        # (Handling Price and Macro together effectively scales them relatively)
        mean = np.mean(features_array, axis=(0, 1), keepdims=True)
        std = np.std(features_array, axis=(0, 1), keepdims=True)
        features_array = (features_array - mean) / (std + 1e-8)

        return features_array

    def _create_adj_matrix(self, snapshot_df: pd.DataFrame) -> np.ndarray:
        # Simple correlation based on Sector
        n = self.num_stocks
        adj = np.eye(n)

        # Sector map
        sector_map = {}
        for sym in self.symbols:
            row = snapshot_df[snapshot_df["Symbol"] == sym]
            if not row.empty:
                sector_map[sym] = row["Sector"].values[0]
            else:
                sector_map[sym] = "Unknown"

        for i in range(n):
            for j in range(i + 1, n):
                s1 = self.symbols[i]
                s2 = self.symbols[j]
                if sector_map[s1] == sector_map[s2] and sector_map[s1] != "Unknown":
                    val = 1.0
                else:
                    val = 0.5  # Weak connection
                adj[i, j] = val
                adj[j, i] = val

        return adj

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        return self.windows[idx]
