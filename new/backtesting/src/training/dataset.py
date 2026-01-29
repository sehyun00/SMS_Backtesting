import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Any


class FinancialDataset(Dataset):
    """
    Dataset for Temporal Graph Neural Networks.
    Creates sliding windows of (features, adjacency_matrix, label).
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
        # ReseachCodeGuide Fix: Create a COPY of features list to prevent in-place mutation of config
        self.features = list(config["data"]["features"])

        if "factors" in config["data"]:
            # Add factor columns if they exist in df
            factor_cols = [
                "Beta_Factor",
                "Value_Factor",
                "Momentum_Factor",
                "Volatility_Factor",
            ]
            self.features += factor_cols

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

        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = dates[i + self.window_size]  # Predict next step

            # Extract features for all stocks in this window
            # Shape: [N, T, F]
            # We need to ensure order of stocks is consistent (sorted symbols)

            # Let's iterate symbols for safety first.
            current_window_df = self.df.loc[window_dates]

            batch_features = []

            for symbol in self.symbols:
                stock_df = current_window_df[current_window_df["Symbol"] == symbol]

                # Handle missing data (e.g. not listed yet)
                if len(stock_df) < self.window_size:
                    # Pad with zeros or skip?
                    # ResearchCodeGuide suggests robustness.
                    # Zero padding is safer for graph.
                    pad_len = self.window_size - len(stock_df)
                    vals = stock_df[self.features].values
                    vals = np.pad(vals, ((pad_len, 0), (0, 0)), mode="constant")
                    batch_features.append(vals)
                else:
                    batch_features.append(stock_df[self.features].values)

            # [N, T, F]
            features_array = np.array(batch_features)

            # Robust Z-Score Normalization per Window
            # Normalize each feature across (Nodes, Time)
            # This handles scale disparity (e.g. Volume vs Price vs Factors)
            mean = np.mean(features_array, axis=(0, 1), keepdims=True)
            std = np.std(features_array, axis=(0, 1), keepdims=True)
            features_array = (features_array - mean) / (std + 1e-8)

            # Create Graph (Adjacency Matrix)
            # Using last date of window for correlation/structure
            last_date = window_dates[-1]
            adj_matrix = self._create_adj_matrix(self.df.loc[str(last_date)])

            target_df = self.df.loc[str(target_date)]
            labels_list = []
            target_cols = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]

            for symbol in self.symbols:
                row = target_df[target_df["Symbol"] == symbol]
                if not row.empty:
                    # Collect all targets
                    vals = [
                        row[col].values[0] if col in row else 0.0 for col in target_cols
                    ]
                    labels_list.append(vals)
                else:
                    labels_list.append([0.0] * len(target_cols))

            windows.append(
                {
                    "features": torch.FloatTensor(features_array),  # [N, T, F]
                    "adj_matrix": torch.FloatTensor(adj_matrix),
                    "labels": torch.FloatTensor(labels_list),  # [N, 4]
                    "date": str(target_date),
                }
            )

        return windows

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
