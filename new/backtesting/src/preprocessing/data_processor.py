import pandas as pd
from typing import Dict, Any, Optional
from .indicators import TechnicalIndicators
from .factors import FactorCalculator
from .fama_french_loader import FamaFrenchLoader


class DataProcessor:
    """
    Processes raw stock data into 5-Factor model data.
    Acts as a Facade for Indicators, Factors, and External Data.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DataProcessor with configuration.
        """
        self.config = config
        self.factor_weights = config["data"]["factors"]["weights"]
        self.ff_loader = FamaFrenchLoader()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Adds technical indicators using TechnicalIndicators module.
        """
        return TechnicalIndicators.add_all_indicators(df)

    def calculate_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates 5-Factor scores using FactorCalculator module.
        """
        return FactorCalculator.calculate_factors(df)

    def calculate_weighted_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates specific weighted score using FactorCalculator module.
        """
        # Map config keys to FactorCalculator expectations if needed
        # FactorCalculator expects specific logic, passing config weights
        weights = {
            "Value_Factor": self.factor_weights.get("value", 0.3),
            "Momentum_Factor": self.factor_weights.get("momentum", 0.3),
            "Volatility_Factor": self.factor_weights.get("volatility", 0.2),
            "Beta_Factor": self.factor_weights.get("beta", 0.2),
        }
        return FactorCalculator.calculate_weighted_score(df, custom_weights=weights)

    def merge_fama_french_factors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downloads and merges Fama-French 5 Factors.
        """
        # Download if not already valid
        if self.ff_loader.ff_data is None:
            self.ff_loader.download_factors()

        return self.ff_loader.merge_with_stock_data(df)
