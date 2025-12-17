"""
train_data.csv로 학습하고 test_data.csv로 백테스팅
- RobustScaler 적용 (Feature만! Target 제외)
- Range Collapse 진단 및 백테스팅 오류 수정
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import sys
import os

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import (상대 경로 문제 시 수정 필요)
try:
    from models.TGNN.model import TGNNModel, TGNNDataset, train_model
    from models.TGNN.backtester import Backtester, BacktestConfig, create_metrics_summary_table
except ImportError:
    from model import TGNNModel, TGNNDataset, train_model
    from backtester import Backtester, BacktestConfig, create_metrics_summary_table

# 기본 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "TGNN_Sector"
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

def evaluate_model_performance(model, dataset, device, num_samples=50):
    """모델 예측값 분포 및 상관계수 확인"""
    print("\n" + "="*40)
    print("📊 모델 예측 성능 상세 진단")
    print("="*40)
    
    model.eval()
    all_preds, all_actuals = [], []
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in indices:
            batch = dataset[idx]
            feat = batch["features"].unsqueeze(0).to(device)
            adj = batch["adj_matrix"].unsqueeze(0).to(device)
            mask = batch["active_mask"]
            
            pred, _ = model(feat, adj)
            pred = pred.squeeze(0).cpu().numpy()
            
            # 학습 시 사용한 Target(labels)과 비교 (스케일링 된 값일 수 있음)
            actual = batch["labels"].numpy()
            
            all_preds.extend(pred[mask])
            all_actuals.extend(actual[mask])
            
    all_preds = np.array(all_preds)
    all_actuals = np.array(all_actuals)
    
    if len(all_preds) > 1:
        corr = np.corrcoef(all_preds, all_actuals)[0, 1]
    else:
        corr = 0
        
    print(f"✅ 샘플 수: {len(all_preds)}")
    print(f"📈 상관계수: {corr:.4f}")
    print(f"🔍 예측 범위: [{all_preds.min():.4f}, {all_preds.max():.4f}] (Std: {all_preds.std():.4f})")
    print(f"🔍 실제 범위: [{all_actuals.min():.4f}, {all_actuals.max():.4f}] (Std: {all_actuals.std():.4f})")
    
    if all_preds.std() < all_actuals.std() * 0.1:
        print("⚠️ 경고: Range Collapse 의심됨 (예측값이 평균에 몰려있음)")

def plot_performance(strategies, save_dir):
    plt.figure(figsize=(12, 6))
    for name, res in strategies.items():
        df = pd.DataFrame(res['history'])
        dates = pd.to_datetime(df['date'])
        vals = df['cumulative_return'] * 100
        plt.plot(dates, vals, label=name)
        
    plt.title("Cumulative Return Comparison")
    plt.ylabel("Return (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(save_dir / "performance_graph.png")
    plt.close()

def main(mode="all"):
    # 사용 Feature
    feature_cols = [
        "Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M",
        "Volatility", "RSI", "MACD", "Signal", "MACD_Hist",
        "Beta_Factor", "Value_Factor", "Momentum_Factor", "Volatility_Factor",
        "weighted_score", "Mkt_RF", "SMB", "HML", "RMW", "CMA",
    ]
    # Target Column
    target_col = "Momentum1M" # 다음달 수익률 (또는 별도 Return_1M 컬럼)
    
    model_path = RESULTS_DIR / "best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ============================
    # 1. 학습 모드
    # ============================
    if mode in ["train", "all"]:
        print("\n🚀 [Train] 데이터 로드 및 전처리...")
        train_df = pd.read_csv(TRAIN_DATA_PATH)
        train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # [중요] Feature Scaling (Robust)
        # Target(Momentum1M)은 스케일링 할지 말지 결정해야 함.
        # 학습 안정성을 위해 Target도 스케일링(Clipping) 하는 것이 좋음.
        # 단, 백테스팅 때는 원본 값을 써야 하므로 TGNNDataset이 raw_labels를 따로 챙김.
        
        for col in feature_cols:
            if col in train_df.columns:
                median = train_df[col].median()
                q1 = train_df[col].quantile(0.25)
                q3 = train_df[col].quantile(0.75)
                iqr = q3 - q1
                if iqr > 0:
                    train_df[col] = train_df[col].clip(median - 5*iqr, median + 5*iqr)
                train_df[col] = (train_df[col] - train_df[col].mean()) / (train_df[col].std() + 1e-8)

        # Target Clipping (너무 큰 수익률은 노이즈로 간주)
        if target_col in train_df.columns:
            train_df[target_col] = train_df[target_col].clip(-1.0, 1.0) # -100% ~ +100%

        train_dataset = TGNNDataset(train_df, window_size=12, feature_cols=feature_cols)
        
        # Split
        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size
        train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True, collate_fn=custom_collate)
        val_loader = DataLoader(val_data, batch_size=64, collate_fn=custom_collate)
        
        # Model
        model = TGNNModel(
            num_features=len(feature_cols),
            hidden_dims=[64, 64, 32],
            num_heads=4,
            dropout=0.4
        )
        
        print("\n🚀 [Train] 학습 시작...")
        train_model(model, train_loader, val_loader, num_epochs=150, lr=1e-4, save_path=str(model_path))
        
        # 진단
        model.load_state_dict(torch.load(model_path))
        model.to(device)
        evaluate_model_performance(model, train_dataset, device)

    # ============================
    # 2. 테스트(백테스팅) 모드
    # ============================
    if mode in ["test", "all"]:
        print("\n🚀 [Test] 백테스팅 시작...")
        if not model_path.exists():
            print("❌ 모델 파일 없음")
            return
            
        test_df = pd.read_csv(TEST_DATA_PATH)
        test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # [핵심] Feature만 스케일링! Target은 건드리지 않음!
        for col in feature_cols:
            if col in test_df.columns:
                median = test_df[col].median()
                iqr = test_df[col].quantile(0.75) - test_df[col].quantile(0.25)
                if iqr > 0:
                    test_df[col] = test_df[col].clip(median - 5*iqr, median + 5*iqr)
                test_df[col] = (test_df[col] - test_df[col].mean()) / (test_df[col].std() + 1e-8)
        
        # Target(Momentum1M)은 원본 그대로 둠 (수익률 계산용)
        # 단, 결측치 0 처리는 함
        if target_col in test_df.columns:
            test_df[target_col] = test_df[target_col].fillna(0.0)
            
        test_dataset = TGNNDataset(test_df, window_size=12, feature_cols=feature_cols)
        
        model = TGNNModel(num_features=len(feature_cols), hidden_dims=[64, 64, 32], num_heads=4)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        
        # 백테스팅 설정
        config = BacktestConfig(
            initial_capital=10_000_000, 
            cost_bps=5.0, 
            top_k=5, 
            weighting_method='equal' # Equal로 하면 더 안정적일 수 있음
        )
        
        strategies = {}
        metrics_all = {}
        
        # 1. Buy & Hold
        bt = Backtester(model, test_dataset, config, "Buy & Hold")
        strategies["Buy & Hold"] = bt.run_buy_and_hold()
        metrics_all["Buy & Hold"] = bt.metrics
        
        # 2. TGNN Monthly
        bt = Backtester(model, test_dataset, config, "TGNN Monthly")
        strategies["TGNN Monthly"] = bt.run("monthly")
        metrics_all["TGNN Monthly"] = bt.metrics
        
        # 3. TGNN Quarterly
        bt = Backtester(model, test_dataset, config, "TGNN Quarterly")
        strategies["TGNN Quarterly"] = bt.run("quarterly")
        metrics_all["TGNN Quarterly"] = bt.metrics

        # 결과 출력
        df_res = create_metrics_summary_table(metrics_all)
        print("\n[백테스팅 결과]")
        print(df_res.to_string(index=False))
        
        df_res.to_csv(RESULTS_DIR / "final_results.csv", index=False, encoding="utf-8-sig")
        plot_performance(strategies, RESULTS_DIR)
        print(f"\n✅ 완료! 결과 파일: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
