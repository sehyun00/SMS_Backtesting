
import torch
import pandas as pd
import numpy as np
import sys
from pathlib import Path
from model import DDPGAgent
from run_comparison import DDPGDataset

ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = ROOT_DIR / "data" / "processed_daily_5factor_model_10stocks_10years_20251127.csv"
MODEL_PATH = Path(__file__).parent / "best_ddpg.pth"

def debug_model():
    # 1. Load Data
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        "Beta", "MarketCap", "Momentum1M", "Momentum6M", "Volatility", "RSI",
        "Beta_Factor", "Value_Factor", "Size_Factor", "Momentum_Factor", "Volatility_Factor",
    ]
    dataset = DDPGDataset(df=df, window_size=12, feature_cols=feature_cols)
    
    num_stocks = len(dataset.symbols)
    num_features = len(feature_cols)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 2. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    agent = DDPGAgent(num_stocks, num_features, device=device)
    if MODEL_PATH.exists():
        try:
            agent.actor.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}")
            return
    else:
        print("Model file not found!")
        return

    # 3. Inspect Logits for Test Windows
    print("\nInpsecting Logits for Test Windows...")
    test_windows = dataset.get_test_windows()
    start_idx = dataset.test_start_idx
    
    # Check first 5 test steps
    for i in range(5):
        global_idx = start_idx + i
        state = dataset.get_state(global_idx)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Reproduce forward pass steps
            batch_size = state_tensor.shape[0]
            x = state_tensor.reshape(batch_size, num_stocks, num_features)
            x = agent.actor.encoder(x)
            x = x.reshape(batch_size, -1)
            scores = agent.actor.global_net(x) # Logits
            weights = torch.nn.functional.softmax(scores, dim=-1)
            
        print(f"\nStep {i} (Date: {test_windows[i]['date']})")
        print(f"Logits: {scores.cpu().numpy()[0]}")
        print(f"Weights: {weights.cpu().numpy()[0]}")
        print(f"Max Logit: {scores.max().item():.4f}, Min Logit: {scores.min().item():.4f}")
        
    # Check if weights are identical across steps
    print("\nChecking State Variation...")
    state0 = dataset.get_state(start_idx)
    state1 = dataset.get_state(start_idx + 10)
    print(f"State 0 vs State 10 L2 Diff: {np.linalg.norm(state0 - state1):.4f}")

if __name__ == "__main__":
    debug_model()
