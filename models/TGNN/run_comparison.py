"""
TGNN 학습 & 리밸런싱 빈도별 백테스팅 비교
- 하나의 모델이 Momentum 1M/3M/6M/12M 모두 예측
- 백테스팅 시 리밸런싱 주기에 맞는 헤드 선택
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
    from model import TGNNModel, TGNNDataset
    from backtester import Backtester, BacktestConfig, create_metrics_summary_table
except ImportError:
    print("❌ 모듈 Import 실패")
    sys.exit(1)

# 기본 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "01_TGNN_Only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트
import platform
font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# ============ TGNN Dataset ============

class TGNN_Dataset(TGNNDataset):
    """TGNN 학습을 위한 Dataset"""
    
    def _create_windows(self):
        """윈도우 생성"""
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]
            
            if self.start_date and next_date < self.start_date:
                continue
            if self.end_date and next_date > self.end_date:
                continue

            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]

            # Features & Active Mask
            features = []
            active_mask = []

            for symbol in self.symbols:
                stock_data = window_df[window_df["Symbol"] == symbol]
                is_active = not stock_data[stock_data["Date"] == target_date].empty

                if is_active:
                    vals = stock_data[self.feature_cols].values
                    if len(vals) < self.window_size:
                        pad = np.zeros((self.window_size - len(vals), len(self.feature_cols)))
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    features.append(np.zeros((self.window_size, len(self.feature_cols))))
                    active_mask.append(False)

            # Graph
            adj_matrix = self._create_masked_graph(
                window_df[window_df["Date"] == target_date], 
                np.array(active_mask)
            )

            # 🔥 모든 Momentum 라벨 생성
            labels_1m = []
            labels_3m = []
            labels_6m = []
            labels_12m = []
            
            for symbol in self.symbols:
                next_stock = next_df[next_df["Symbol"] == symbol]
                labels_1m.append(next_stock["Momentum1M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_3m.append(next_stock["Momentum3M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_6m.append(next_stock["Momentum6M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_12m.append(next_stock["Momentum12M"].values[0] if len(next_stock) > 0 else 0.0)

            windows.append({
                "features": torch.FloatTensor(np.array(features)),
                "adj_matrix": torch.FloatTensor(adj_matrix),
                "Momentum1M": torch.FloatTensor(np.array(labels_1m)),
                "Momentum3M": torch.FloatTensor(np.array(labels_3m)),
                "Momentum6M": torch.FloatTensor(np.array(labels_6m)),
                "Momentum12M": torch.FloatTensor(np.array(labels_12m)),
                "date": target_date,
                "active_mask": np.array(active_mask),
            })

        return windows
    
    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            "features": w["features"],
            "adj_matrix": w["adj_matrix"],
            "Momentum1M": w["Momentum1M"],
            "Momentum3M": w["Momentum3M"],
            "Momentum6M": w["Momentum6M"],
            "Momentum12M": w["Momentum12M"],
            "active_mask": torch.BoolTensor(w["active_mask"]),
        }

# ============ TGNN Model ============

import torch.nn as nn
import torch.nn.functional as F

class TGNNModel(TGNNModel):
    """TGNN: 4개의 독립적인 예측 헤드"""
    
    def __init__(self, num_features, hidden_dims=[128, 128, 64], num_heads=8, num_stocks=10):
        # 부모 클래스 초기화 (predictor 제외)
        super(TGNNModel, self).__init__()
        
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        
        from model import GraphConvLayer, TemporalAttention
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
            for i in range(len(hidden_dims) - 1)
        ])
        
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)
        
        # 🔥 4개의 독립적인 헤드
        self.predictors = nn.ModuleDict({
            'Momentum1M': self._make_predictor(hidden_dims[-1]),
            'Momentum3M': self._make_predictor(hidden_dims[-1]),
            'Momentum6M': self._make_predictor(hidden_dims[-1]),
            'Momentum12M': self._make_predictor(hidden_dims[-1]),
        })
    
    def _make_predictor(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
    
    def forward(self, features, adj_matrix, target_type='Momentum1M'):
        batch, N, T, F = features.shape

        # GCN + Temporal Attention (공통)
        gcn_outputs = []
        for t in range(T):
            x_t = features[:, :, t, :]
            h = self.input_proj(x_t)
            for gcn in self.gcn_layers:
                h = gcn(h, adj_matrix)
            gcn_outputs.append(h)

        temporal_features = torch.stack(gcn_outputs, dim=1)
        node_embeddings = self.temporal_attn(temporal_features)
        
        # 선택된 헤드로 예측
        predictions = self.predictors[target_type](node_embeddings).squeeze(-1)
        
        return predictions, node_embeddings

# ============ 학습 함수 ============

def custom_collate(batch):
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "adj_matrix": torch.stack([item["adj_matrix"] for item in batch]),
        "Momentum1M": torch.stack([item["Momentum1M"] for item in batch]),
        "Momentum3M": torch.stack([item["Momentum3M"] for item in batch]),
        "Momentum6M": torch.stack([item["Momentum6M"] for item in batch]),
        "Momentum12M": torch.stack([item["Momentum12M"] for item in batch]),
        "active_mask": torch.stack([item["active_mask"] for item in batch]),
    }

def train_multitask_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, save_path="best_tgnn_multi.pth"):
    """Multi-Task Learning: 모든 Momentum을 동시에 학습"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  디바이스: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.HuberLoss(delta=0.1)
    
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            
            loss = 0
            # 🔥 4개 타겟 모두 학습
            for target in ['Momentum1M', 'Momentum3M', 'Momentum6M', 'Momentum12M']:
                predictions, _ = model(batch["features"], batch["adj_matrix"], target_type=target)
                loss += criterion(predictions, batch[target])
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                
                loss = 0
                for target in ['Momentum1M', 'Momentum3M', 'Momentum6M', 'Momentum12M']:
                    predictions, _ = model(batch["features"], batch["adj_matrix"], target_type=target)
                    loss += criterion(predictions, batch[target])
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n⏹️ Early stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, Best={best_val_loss:.6f}")

    print(f"\n✅ 최적 모델 저장: {save_path} (Best Val Loss: {best_val_loss:.6f})")

# ============ 시각화 ============

def plot_performance(strategies, save_dir):
    plt.figure(figsize=(14, 7))
    for name, res in strategies.items():
        if not res or not res.get('history'): 
            continue
        df = pd.DataFrame(res['history'])
        if 'date' not in df.columns or 'cumulative_return' not in df.columns: 
            continue
        dates = pd.to_datetime(df['date'])
        vals = df['cumulative_return']
        plt.plot(dates, vals, label=name, linewidth=2)
        
    plt.title("Performance Comparison: TGNN", fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    
    # 🔥 x축 범위를 2022년 1월부터 시작하도록 설정
    plt.xlim(left=pd.Timestamp('2022-01-01'))
    
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_graph.png", dpi=300)
    plt.close()
    print(f"✅ 그래프 저장: {save_dir / 'comparison_graph.png'}")

# ============ 메인 실행 ============

def main():
    print("="*60)
    print("🚀 TGNN 학습 & 백테스팅")
    print("="*60)
    
    # 🔥 Feature 컬럼들 (Momentum 제외 - 타겟 변수이므로 정규화하지 않음)
    feature_cols = [
        "Volatility", "RSI", "MACD", "Signal", "MACD_Hist",
        "Beta_Factor", "Value_Factor", "Momentum_Factor", "Volatility_Factor",
        "weighted_score", "Mkt_RF", "SMB", "HML", "RMW", "CMA",
    ]
    
    # Momentum 컬럼들 (타겟 변수 - 원본 값 유지)
    momentum_cols = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
    
    # 모든 컬럼 (모델 입력용)
    all_feature_cols = momentum_cols + feature_cols
    
    # ================= 1. 학습 데이터 로드 =================
    print("\n📂 학습 데이터 로드 중...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 🔥 Feature만 스케일링 (Momentum은 제외)
    train_mean = {}
    train_std = {}
    for col in feature_cols:
        if col in train_df.columns:
            train_mean[col] = train_df[col].mean()
            train_std[col] = train_df[col].std()
            train_df[col] = (train_df[col] - train_mean[col]) / (train_std[col] + 1e-8)
    
    # 🔥 Momentum은 정규화 없이 원본 값 사용 (극단값만 제거)
    # 일반적으로 Momentum은 -100% ~ +수백% 범위이므로 합리적인 범위로 제한
    for mom_col in momentum_cols:
        if mom_col in train_df.columns:
            # 극단값 제거 (-200% ~ +500% 범위로 제한)
            train_df[mom_col] = train_df[mom_col].clip(-2.0, 5.0)
    
    symbols = sorted(train_df["Symbol"].unique())
    print(f"종목 수: {len(symbols)}")
    
    train_dataset = MultiOutputDataset(
        train_df, 
        window_size=12, 
        feature_cols=all_feature_cols,  # 🔥 Momentum 포함한 전체 feature
        symbols=symbols
    )
    
    print(f"학습 윈도우 수: {len(train_dataset)}")
    
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_data, batch_size=32, collate_fn=custom_collate)
    
    # ================= 2. 모델 학습 =================
    print("\n🧠 TGNN 모델 학습 중...")
    model = MultiOutputTGNN(len(all_feature_cols), [128, 128, 64], 8, len(symbols))  # 🔥 Momentum 포함한 feature 개수
    model_path = RESULTS_DIR / "best_tgnn_multi.pth"
    
    train_multitask_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, save_path=str(model_path))
    
    # ================= 3. 테스트 데이터 로드 =================
    print("\n📂 테스트 데이터 로드 중...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 🔥 Feature만 train 스케일링 파라미터 적용 (Momentum은 제외)
    for col in feature_cols:
        if col in test_df.columns and col in train_mean:
            test_df[col] = (test_df[col] - train_mean[col]) / (train_std[col] + 1e-8)
    
    # 🔥 Momentum은 정규화 없이 원본 값 사용 (극단값만 제거)
    for mom_col in momentum_cols:
        if mom_col in test_df.columns:
            # 극단값 제거 (-200% ~ +500% 범위로 제한)
            test_df[mom_col] = test_df[mom_col].clip(-2.0, 5.0)
    
    test_dataset = MultiOutputDataset(
        test_df, 
        window_size=12, 
        feature_cols=all_feature_cols,  # 🔥 Momentum 포함한 전체 feature
        symbols=symbols
    )
    
    print(f"테스트 윈도우 수: {len(test_dataset)}")
    
    # 모델 로드
    model.load_state_dict(torch.load(model_path))
    
    # ================= 4. 백테스팅 =================
    print("\n📊 백테스팅 실행 중...")
    config = BacktestConfig(initial_capital=10_000_000, cost_bps=10.0, top_k=5, weighting_method='equal')
    
    all_strategies = {}
    all_metrics = {}
    all_backtesters = {}  # 🔥 Backtester 객체 저장 (시계열 데이터 추출용)
    
    # Buy & Hold
    print("[1/5] Buy & Hold...")
    bt_buyhold = Backtester(model, test_dataset, config, "Buy_Hold")
    bh_res = bt_buyhold.run_buy_and_hold()
    all_strategies["Buy & Hold"] = bh_res
    all_metrics["Buy & Hold"] = bt_buyhold.metrics
    all_backtesters["Buy & Hold"] = bt_buyhold
    
    # TGNN 각 주기별
    experiments = [
        ("monthly", "Momentum1M", "TGNN 1M"),
        ("quarterly", "Momentum3M", "TGNN 3M"),
        ("semiannual", "Momentum6M", "TGNN 6M"),
        ("annual", "Momentum12M", "TGNN 12M"),
    ]
    
    for i, (freq, target, name) in enumerate(experiments, 2):
        print(f"[{i}/5] {name}...")
        bt = Backtester(model, test_dataset, config, name)
        bt.target_type = target  # 🔥 어떤 헤드를 사용할지 지정
        res = bt.run(freq)
        all_strategies[name] = res
        all_metrics[name] = bt.metrics
        all_backtesters[name] = bt  # 🔥 Backtester 객체 저장
    
    # ================= 5. 결과 출력 및 저장 =================
    print("\n" + "="*60)
    print("📊 Final Results")
    print("="*60)
    
    df_res = create_metrics_summary_table(all_metrics)
    print(df_res.to_string(index=False))
    
    # 🔥 1. 간단한 비교 테이블 저장
    df_res.to_csv(RESULTS_DIR / "comparison_metrics.csv", index=False, encoding="utf-8-sig")
    print(f"\n✅ 저장: comparison_metrics.csv")
    
    # 🔥 2. 상세한 메트릭 비교 테이블 생성 (백업 스타일)
    detailed_rows = []
    for strategy, metrics in all_metrics.items():
        if not metrics:
            continue
        row = {
            "전략": strategy,
            "누적수익률": f"{metrics.get('total_return', 0):.2f}%",
            "CAGR": f"{metrics.get('cagr', 0):.2f}%",
            "변동성": f"{metrics.get('volatility', 0):.2f}%",
            "MDD": f"{metrics.get('max_drawdown', 0):.2f}%",
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "Sortino": f"{metrics.get('sortino_ratio', 0):.2f}",
            "Calmar": f"{metrics.get('calmar_ratio', 0):.2f}",
            "VaR(95%)": f"{metrics.get('var_95', 0):.2f}%",
            "CVaR(95%)": f"{metrics.get('cvar_95', 0):.2f}%",
            "Info Ratio": f"{metrics.get('information_ratio', 0):.2f}",
            "Avg Turnover": f"{metrics.get('avg_turnover', 0):.2f}%",
            "총 거래비용": f"{int(metrics.get('total_transaction_cost', 0)):,}원",
        }
        detailed_rows.append(row)
    
    df_detailed = pd.DataFrame(detailed_rows)
    df_detailed.to_csv(RESULTS_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"✅ 저장: metrics_comparison.csv (상세 메트릭)")
    
    # 🔥 3. 각 전략별 시계열 데이터 저장
    freq_map = {
        "Buy & Hold": "buyhold",
        "TGNN 1M": "monthly",
        "TGNN 3M": "quarterly",
        "TGNN 6M": "semiannual",
        "TGNN 12M": "annual",
    }
    
    for strategy_name, bt in all_backtesters.items():
        freq_name = freq_map.get(strategy_name, strategy_name.lower().replace(" ", "_"))
        filename = f"timeseries_{freq_name}.csv"
        filepath = RESULTS_DIR / filename
        bt.save_timeseries_csv(filepath)
    
    # 🔥 4. 그래프 저장
    plot_performance(all_strategies, RESULTS_DIR)
    
    print(f"\n{'='*60}")
    print(f"✅ 완료! 결과 폴더: {RESULTS_DIR}")
    print(f"{'='*60}")
    print("\n📁 생성된 파일들:")
    print("  - comparison_metrics.csv (요약)")
    print("  - metrics_comparison.csv (상세)")
    print("  - timeseries_buyhold.csv")
    print("  - timeseries_monthly.csv")
    print("  - timeseries_quarterly.csv")
    print("  - timeseries_semiannual.csv")
    print("  - timeseries_annual.csv")
    print("  - comparison_graph.png")

if __name__ == "__main__":
    main()
