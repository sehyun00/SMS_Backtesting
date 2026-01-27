import pandas as pd
import numpy as np
from typing import Dict, Any


class FactorCalculator:
    """
    Calculates Factor Scores and weighted signals.
    """

    @staticmethod
    def calculate_factors(df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates Percentile Rank for each factor using technical proxies.
        """
        df = df.copy()

        # 1. Beta Factor (Proxy: Inverse Volatility)
        if "Volatility" in df.columns:
            df["Beta_Factor"] = df["Volatility"].rank(pct=True, ascending=False)
        else:
            df["Beta_Factor"] = 0.5

        # 2. Value Factor (Proxy: Inverse RSI)
        if "RSI" in df.columns:
            df["Value_Factor"] = df["RSI"].rank(pct=True, ascending=False)
        else:
            df["Value_Factor"] = 0.5

        # 3. Momentum Factor (12M Momentum)
        if "Momentum12M" in df.columns:
            df["Momentum_Factor"] = df["Momentum12M"].rank(pct=True)
        else:
            df["Momentum_Factor"] = 0.5

        # 4. Volatility Factor (Inverse Volatility)
        if "Volatility" in df.columns:
            df["Volatility_Factor"] = df["Volatility"].rank(pct=True, ascending=False)
        else:
            df["Volatility_Factor"] = 0.5

        return df

    @staticmethod
    def calculate_weighted_score(
        df: pd.DataFrame, custom_weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Calculates final weighted score and generates Smart Signal.
        """
        df = df.copy()

        # Default Weights
        weights = (
            custom_weights
            if custom_weights
            else {
                "Value_Factor": 0.3,
                "Momentum_Factor": 0.3,
                "Volatility_Factor": 0.2,
                "Beta_Factor": 0.2,
            }
        )

        # Calculate Weighted Score
        df["weighted_score"] = (
            df["Value_Factor"] * weights.get("Value_Factor", 0.25)
            + df["Momentum_Factor"] * weights.get("Momentum_Factor", 0.25)
            + df["Volatility_Factor"] * weights.get("Volatility_Factor", 0.25)
            + df["Beta_Factor"] * weights.get("Beta_Factor", 0.25)
        ) * 100  # Scale to 0-100

        # Generate Signal
        conditions = [
            (df["weighted_score"] >= 80),
            (df["weighted_score"] >= 60),
            (df["weighted_score"] <= 40),
            (df["weighted_score"] <= 20),
        ]
        choices = ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]

        df["smart_signal"] = np.select(conditions, choices, default="NEUTRAL")

        return df
