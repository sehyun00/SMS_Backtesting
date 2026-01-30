import pandas as pd
import json


def check_overlap():
    # Load Trained Universe
    try:
        with open("results/ddpg/trained_universe.json", "r") as f:
            trained = json.load(f)
        print(f"Trained Universe ({len(trained)}): {trained[:5]}...")
    except Exception as e:
        print(f"Error loading trained_universe: {e}")
        return

    # Load Test Data
    try:
        df = pd.read_csv("data/test_data.csv")
        test_symbols = sorted(df["Symbol"].unique().tolist())
        print(f"Test Symbols ({len(test_symbols)}): {test_symbols}")
    except Exception as e:
        print(f"Error loading test_data: {e}")
        return

    # Check Subset
    test_set = set(test_symbols)
    train_set = set(trained)

    is_subset = test_set.issubset(train_set)
    print(f"\nIs Subset: {is_subset}")

    if not is_subset:
        missing = test_set - train_set
        print(f"Symbols in Test but NOT in Train: {missing}")


if __name__ == "__main__":
    check_overlap()
