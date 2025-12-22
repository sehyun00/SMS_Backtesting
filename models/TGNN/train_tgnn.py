"""
TGNN 학습 스크립트 (Multi-head: Momentum 1M/3M/6M/12M)
- train_data.csv 기준
- 2006-01-01 ~ 2018-12-31 : train
- 2019-01-01 ~ 2020-12-31 : val
- 학습 완료 후: best_tgnn_multi.pth + scaler_params.npz 저장
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import sys
import os
import platform
import matplotlib.pyplot as plt

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import TGNNDataset  # 원본 단일 타겟용 Dataset
    from model import TGNNModel as BaseTGNNModel  # 원본 단일 헤드 모델
except ImportError:
    print("❌ 모듈 Import 실패")
    sys.exit(1)

# 기본 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "01_TGNN_Only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트 (그래프는 여기선 거의 안 쓰지만 기존 코드 유지)
font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# ============ TGNN Dataset (Multi-output) ============

class TGNN_Dataset(TGNNDataset):
    """TGNN 학습을 위한 Dataset (Momentum 4개 라벨)"""

    def _create_windows(self):
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]

            # 기간 필터링
            if self.start_date and next_date < self.start_date:
                continue
            if self.end_date and next_date > self.end_date:
                continue

            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]

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
                np.array(active_mask),
            )

            # 모든 Momentum 라벨
            labels_1m, labels_3m, labels_6m, labels_12m = [], [], [], []
            for symbol in self.symbols:
                next_stock = next_df[next_df["Symbol"] == symbol]
                labels_1m.append(next_stock["Momentum1M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_3m.append(next_stock["Momentum3M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_6m.append(next_stock["Momentum6M"].values[0] if len(next_stock) > 0 else 0.0)
                labels_12m.append(next_stock["Momentum12M"].values[0] if len(next_stock) > 0 else 0.0)

            windows.append(
                {
                    "features": torch.FloatTensor(np.array(features)),
                    "adj_matrix": torch.FloatTensor(adj_matrix),
                    "Momentum1M": torch.FloatTensor(np.array(labels_1m)),
                    "Momentum3M": torch.FloatTensor(np.array(labels_3m)),
                    "Momentum6M": torch.FloatTensor(np.array(labels_6m)),
                    "Momentum12M": torch.FloatTensor(np.array(labels_12m)),
                    "date": target_date,
                    "active_mask": np.array(active_mask),
                }
            )

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

# ============ TGNN Model (Multi-head) ============

import torch.nn as nn
import torch.nn.functional as F

class TGNNModel(BaseTGNNModel):
    """TGNN: 4개의 독립적인 예측 헤드"""

    def __init__(self, num_features, hidden_dims=[128, 128, 64], num_heads=8, num_stocks=10):
        super().__init__(
            num_features=num_features,
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_stocks=num_stocks,
        )

        from model import GraphConvLayer, TemporalAttention

        self.input_proj = nn.Linear(num_features, hidden_dims[0])

        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)

        self.predictors = nn.ModuleDict(
            {
                "Momentum1M": self._make_predictor(hidden_dims[-1]),
                "Momentum3M": self._make_predictor(hidden_dims[-1]),
                "Momentum6M": self._make_predictor(hidden_dims[-1]),
                "Momentum12M": self._make_predictor(hidden_dims[-1]),
            }
        )

    def _make_predictor(self, hidden_dim):
        return nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1),
        )

    def forward(self, features, adj_matrix, target_type="Momentum1M"):
        batch, N, T, F = features.shape

        gcn_outputs = []
        for t in range(T):
            x_t = features[:, :, t, :]
            h = self.input_proj(x_t)
            for gcn in self.gcn_layers:
                h = gcn(h, adj_matrix)
            gcn_outputs.append(h)

        temporal_features = torch.stack(gcn_outputs, dim=1)
        node_embeddings = self.temporal_attn(temporal_features)

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
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            loss = 0.0
            for target in ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]:
                predictions, _ = model(batch["features"], batch["adj_matrix"], target_type=target)
                loss = loss + criterion(predictions, batch[target])

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                loss = 0.0
                for target in ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]:
                    predictions, _ = model(batch["features"], batch["adj_matrix"], target_type=target)
                    loss = loss + criterion(predictions, batch[target])

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
            print(
                f"Epoch {epoch}: Train={avg_train_loss:.6f}, "
                f"Val={avg_val_loss:.6f}, Best={best_val_loss:.6f}"
            )

    print(f"\n✅ 최적 모델 저장: {save_path} (Best Val Loss: {best_val_loss:.6f})")

# ============ main (학습만) ============

def main():
    print("=" * 60)
    print("🚀 TGNN 학습 전용 스크립트")
    print("=" * 60)

    feature_cols = [
        "Volatility",
        "RSI",
        "MACD",
        "Signal",
        "MACD_Hist",
        "Beta_Factor",
        "Value_Factor",
        "Momentum_Factor",
        "Volatility_Factor",
        "weighted_score",
        "Mkt_RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
    ]
    momentum_cols = ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]
    all_feature_cols = momentum_cols + feature_cols

    # 1. train_data.csv 로드
    print("\n📂 학습 데이터 로드 중...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_df["Date"] = pd.to_datetime(train_df["Date"])
    train_df = train_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Feature 스케일링 (Momentum 제외)
    train_mean = {}
    train_std = {}
    # 스케일은 2006-01-01 ~ 2020-12-31 기준으로 계산
    mask_scale = (train_df["Date"] >= "2006-01-01") & (train_df["Date"] <= "2020-12-31")
    df_scale = train_df[mask_scale].copy()

    for col in feature_cols:
        if col in train_df.columns:
            m = df_scale[col].mean()
            s = df_scale[col].std()
            train_mean[col] = m
            train_std[col] = s
            train_df[col] = (train_df[col] - m) / (s + 1e-8)

    # Momentum 극단값 클리핑
    for mom_col in momentum_cols:
        if mom_col in train_df.columns:
            train_df[mom_col] = train_df[mom_col].clip(-0.4, 0.5)

    symbols = sorted(train_df["Symbol"].unique())
    print(f"종목 수: {len(symbols)}")

    # 2. train/val Dataset 생성
    train_dataset = TGNN_Dataset(
        train_df,
        window_size=12,
        feature_cols=all_feature_cols,
        symbols=symbols,
        start_date="2006-01-01",
        end_date="2018-12-31",
    )

    val_dataset = TGNN_Dataset(
        train_df,
        window_size=12,
        feature_cols=all_feature_cols,
        symbols=symbols,
        start_date="2019-01-01",
        end_date="2020-12-31",
    )

    print(f"Train 윈도우 수: {len(train_dataset)}")
    print(f"Val 윈도우 수: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=custom_collate)

    # 3. 모델 학습
    model = TGNNModel(
        num_features=len(all_feature_cols),
        hidden_dims=[128, 128, 64],
        num_heads=8,
        num_stocks=len(symbols),
    )
    model_path = RESULTS_DIR / "best_tgnn_multi.pth"
    train_multitask_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, save_path=str(model_path))

    # 4. 스케일링 파라미터 저장
    scaler_path = RESULTS_DIR / "scaler_params.npz"
    np.savez(scaler_path, mean=train_mean, std=train_std, feature_cols=np.array(feature_cols))
    print(f"✅ 스케일러 저장: {scaler_path}")
    print(f"✅ 학습 완료. 모델: {model_path}")

if __name__ == "__main__":
    main()
