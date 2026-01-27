import sys
import os
import pandas as pd
import numpy as np
import yaml

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.preprocessing.data_processor import DataProcessor


def test_data_processor():
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "../config/config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. Create Dummy Data (1 year daily data)
    dates = pd.date_range(start="2023-01-01", periods=300, freq="D")
    # Random walk price
    close = np.random.normal(100, 1, 300).cumsum()
    close = close - close.min() + 100  # Ensure positive

    df = pd.DataFrame(
        {"Date": dates, "Close": close, "Volume": np.random.randint(1000, 10000, 300)}
    )
    df.set_index("Date", inplace=True)

    print(f"Testing with DF shape: {df.shape}")

    # 3. Initialize Processor
    processor = DataProcessor(config)

    # 4. Run Pipeline
    print("Adding Technical Indicators...")
    df = processor.add_technical_indicators(df)

    expected_cols = ["Momentum1M", "Momentum12M", "Volatility", "RSI", "MACD"]
    for col in expected_cols:
        assert col in df.columns, f"Missing column: {col}"
    print("✅ Technical Indicators added.")

    print("Calculating Factors...")
    df = processor.calculate_factors(df)

    factor_cols = [
        "Beta_Factor",
        "Value_Factor",
        "Momentum_Factor",
        "Volatility_Factor",
    ]
    for col in factor_cols:
        assert col in df.columns, f"Missing factor: {col}"
        assert df[col].max() <= 1.0, f"Factor {col} > 1.0"
        assert df[col].min() >= 0.0, f"Factor {col} < 0.0"
    print("✅ Factors calculated correctly (0-1 range).")

    print("Calculating Weighted Score...")
    df = processor.calculate_weighted_score(df)
    assert "weighted_score" in df.columns
    print(f"✅ Weighted Score calculated. Sample: {df['weighted_score'].iloc[-1]:.2f}")

    print("\n🎉 DataProcessor Verification SUCCESS!")


if __name__ == "__main__":
    test_data_processor()
