"""
Momentum 1M, 3M, 6M, 12M 모델별 학습 및 비교 백테스팅
- 각 기간별로 별도 모델 학습 (총 4개 모델)
- 각 모델에 맞는 리밸런싱 주기 적용
- 지표: Cumulative Return, CAGR, MDD, Avg Annual DD
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from models.TGNN.model import TGNNModel, TGNNDataset, train_model
    from models.TGNN.backtester import Backtester, BacktestConfig, create_metrics_summary_table
except ImportError:
    try:
        from model import TGNNModel, TGNNDataset, train_model
        from backtester import Backtester, BacktestConfig, create_metrics_summary_table
    except ImportError:
         print("❌ 모듈 Import 실패. models 패키지가 PYTHONPATH에 포함되어 있는지 확인하세요.")
         sys.exit(1)

# 기본 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "TGNN_Comparison"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트
import platform
font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

def custom_collate(batch):
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "adj_matrix": torch.stack([item["adj_matrix"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "raw_labels": torch.stack([item["raw_labels"] for item in batch]),
        "active_mask": torch.stack([item["active_mask"] for item in batch]),
    }

def plot_performance(strategies, save_dir):
    plt.figure(figsize=(12, 6))
    for name, res in strategies.items():
        if not res['history']: continue
        df = pd.DataFrame(res['history'])
        if 'date' not in df.columns or 'cumulative_return' not in df.columns: continue
        dates = pd.to_datetime(df['date'])
        vals = df['cumulative_return'] * 100
        plt.plot(dates, vals, label=name)
        
    plt.title("Performance Comparison: 1M vs 3M vs 6M vs 12M")
    plt.ylabel("Cumulative Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_graph.png")
    plt.close()

def run_experiment(target_col, rebalance_freq, model_name):
    """
    특정 Target(예: Momentum1M)에 대해 학습하고 백테스팅 수행
    """
    print(f"\n{'='*50}")
    print(f"🚀 Experiment Start: {model_name} (Target: {target_col})")
    print(f"{'='*50}")

    feature_cols = [
        "Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M",
        "Volatility", "RSI", "MACD", "Signal", "MACD_Hist",
        "Beta_Factor", "Value_Factor", "Momentum_Factor", "Volatility_Factor",
        "weighted_score", "Mkt_RF", "SMB", "HML", "RMW", "CMA",
    ]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_path = RESULTS_DIR / f"model_{target_col}.pth"

    # ================= 1. 학습 =================
    if not TRAIN_DATA_PATH.exists(): 
        print(f"❌ 학습 데이터 없음: {TRAIN_DATA_PATH}")
        return None, None, None
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scaling
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = (train_df[col] - train_df[col].mean()) / (train_df[col].std() + 1e-8)
    
    # Target Clipping
    if target_col in train_df.columns:
        train_df[target_col] = train_df[target_col].clip(-1.0, 1.0)
    else:
        print(f"❌ Target Column '{target_col}' not found in train data.")
        return None, None, None

    # Dataset (TGNNDataset 내부에서 target_col의 '다음 시점 값'을 라벨로 사용)
    train_dataset = TGNNDataset(train_df, window_size=12, feature_cols=feature_cols, target_col=target_col)
    
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=custom_collate)
    
    model = TGNNModel(len(feature_cols), [64, 64, 32], 4, 0.4)
    print("   Training model...")
    train_model(model, train_loader, val_loader, num_epochs=50, lr=1e-4, save_path=str(model_save_path))

    # ================= 2. 백테스팅 =================
    if not TEST_DATA_PATH.exists(): 
        print(f"❌ 테스트 데이터 없음: {TEST_DATA_PATH}")
        return None, None, None
    
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Scaling (Feature Only)
    for col in feature_cols:
        if col in test_df.columns:
            test_df[col] = (test_df[col] - test_df[col].mean()) / (test_df[col].std() + 1e-8)
    
    # Target Column 결측치 처리 (원본 값 유지)
    if target_col in test_df.columns:
        test_df[target_col] = test_df[target_col].fillna(0.0)

    test_dataset = TGNNDataset(test_df, window_size=12, feature_cols=feature_cols, target_col=target_col)
    
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.to(device)
    
    config = BacktestConfig(initial_capital=10_000_000, cost_bps=10.0, top_k=5, weighting_method='equal')
    bt = Backtester(model, test_dataset, config, model_name)
    
    print(f"   Running Backtest ({rebalance_freq})...")
    result = bt.run(rebalance_freq)
    return result, bt.metrics, bt

def main():
    # 실험 목록 정의: (타겟 컬럼, 리밸런싱 주기, 표시 이름)
    experiments = [
        ("Momentum1M", "monthly", "TGNN 1M"),
        ("Momentum3M", "quarterly", "TGNN 3M"),
        ("Momentum6M", "semiannual", "TGNN 6M"),
        ("Momentum12M", "annual", "TGNN 12M")
    ]
    
    all_strategies = {}
    all_metrics = {}
    
    # 1. Buy & Hold (Benchmark) - 한 번만 실행하면 됨 (타겟은 1M으로 임시 설정하여 백테스터 생성)
    print("\n🚀 Calculating Benchmark: Buy & Hold...")
    
    # Buy & Hold를 위한 임시 실행 (학습 없이 백테스터 객체만 필요하므로 가장 짧은 1M 사용)
    # 실제로는 run_experiment가 학습을 포함하므로 시간이 걸리지만, 구조상 재활용
    # 만약 이미 학습된 모델이 있다면 로딩만 해도 되지만, 여기선 안전하게 첫 실험 결과 활용
    res_1m, metrics_1m, bt_1m = run_experiment("Momentum1M", "monthly", "TGNN 1M")
    
    if res_1m:
        # 1M 결과 저장
        all_strategies["TGNN 1M"] = res_1m
        all_metrics["TGNN 1M"] = metrics_1m
        
        # Buy & Hold 실행 (bt_1m 객체 재활용)
        bh_res = bt_1m.run_buy_and_hold()
        all_strategies["Buy & Hold"] = bh_res
        all_metrics["Buy & Hold"] = bt_1m.metrics
    else:
        print("❌ TGNN 1M 실험 실패로 Buy & Hold도 실행되지 않았습니다.")
        return

    # 2. 나머지 모멘텀 전략 실행 (3M, 6M, 12M)
    for target, freq, name in experiments[1:]: # 1M은 이미 위에서 했으므로 제외
        res, metrics, _ = run_experiment(target, freq, name)
        if res:
            all_strategies[name] = res
            all_metrics[name] = metrics
            
    # 3. 결과 집계 및 저장
    if all_metrics:
        df_res = create_metrics_summary_table(all_metrics)
        
        print("\n" + "="*60)
        print("📊 Final Comparison Results")
        print("="*60)
        print(df_res.to_string(index=False))
        
        df_res.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
        plot_performance(all_strategies, RESULTS_DIR)
        print(f"\n✅ All experiments finished. Results saved to {RESULTS_DIR}")
    else:
        print("\n❌ No results generated.")

if __name__ == "__main__":
    main()
