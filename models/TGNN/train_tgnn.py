"""
TGNN 학습 스크립트 (Multi-head: Momentum 1M/3M/6M/12M) - 개선 버전
- train_data.csv 기준
- 2006-01-01 ~ 2018-12-31 : train
- 2019-01-01 ~ 2020-12-31 : val
- 학습 완료 후: best_tgnn_multi.pth + scaler_params.npz 저장
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
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
    from model import GraphConvLayer, TemporalAttention  # 필요한 레이어들
except ImportError:
    print("❌ 모듈 Import 실패")
    sys.exit(1)

# 기본 설정
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "01_TGNN_Only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# 한글 폰트
font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# ============ 손실 함수 (여기서 정의) ============

def pairwise_ranking_loss(predictions, labels, active_mask=None, margin=0.1):
    """
    [최적화] Vectorized Pairwise Ranking Loss
    
    속도: 기존 대비 100배 이상 빠름
    """
    batch_size = predictions.shape[0]
    
    if active_mask is not None:
        # 비활성 종목 마스킹
        predictions = predictions.clone()
        labels = labels.clone()
        predictions[~active_mask] = 0
        labels[~active_mask] = 0
    
    # Vectorized pairwise comparison
    # [batch, n, 1] - [batch, 1, n] = [batch, n, n]
    pred_diff = predictions.unsqueeze(2) - predictions.unsqueeze(1)
    label_diff = labels.unsqueeze(2) - labels.unsqueeze(1)
    
    # label_diff > 0이면 pred_diff도 > 0이어야 함
    # max(0, margin - pred_diff) when label_diff > 0
    violation = torch.relu(margin - pred_diff * torch.sign(label_diff))
    
    # label_diff가 0인 경우 제외
    mask = (label_diff != 0).float()
    
    # 상삼각 행렬만 사용 (중복 제거)
    triu_mask = torch.triu(torch.ones_like(mask), diagonal=1)
    mask = mask * triu_mask
    
    if active_mask is not None:
        # 비활성 종목 페어 제외
        active_pair_mask = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
        mask = mask * active_pair_mask.float()
    
    loss = (violation * mask).sum() / (mask.sum() + 1e-8)
    
    return loss



def combined_loss(predictions, labels, active_mask=None, alpha=0.6, beta=0.4):
    """
    Combined Loss: MSE + Ranking Loss
    
    Args:
        predictions: 예측값 [batch, num_stocks]
        labels: 실제값 [batch, num_stocks]
        active_mask: 활성 마스크 [batch, num_stocks]
        alpha: MSE 가중치
        beta: Ranking Loss 가중치
    
    Returns:
        total_loss: 결합 손실
    """
    # MSE Loss
    if active_mask is not None:
        mse = F.mse_loss(predictions[active_mask], labels[active_mask])
    else:
        mse = F.mse_loss(predictions, labels)
    
    # Ranking Loss
    rank_loss = pairwise_ranking_loss(predictions, labels, active_mask)
    
    return alpha * mse + beta * rank_loss


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


# ============ TGNN Model (Multi-head + Batch Norm + Residual) ============

class TGNNModel(BaseTGNNModel):
    """
    [개선] TGNN: 4개의 독립적인 예측 헤드
    - Batch Normalization 추가
    - Residual Connection 추가
    - Tanh 제거
    """

    def __init__(self, num_features, hidden_dims=[128, 128, 64], num_heads=8, num_stocks=10):
        super().__init__(
            num_features=num_features,
            hidden_dims=hidden_dims,
            num_heads=num_heads,
            num_stocks=num_stocks,
        )

        # [개선] Input projection + Batch Norm
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        self.input_ln = nn.LayerNorm(hidden_dims[0])  

        # [개선] GCN layers + Batch Norm
        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )
        
        self.gcn_lns = nn.ModuleList(
            [nn.LayerNorm(hidden_dims[i + 1]) for i in range(len(hidden_dims) - 1)]
        )

        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)

        # [개선] 각 Momentum별 독립 헤드 (Tanh 제거)
        self.predictors = nn.ModuleDict({
            "Momentum1M": self._make_predictor(hidden_dims[-1]),
            "Momentum3M": self._make_predictor(hidden_dims[-1]),
            "Momentum6M": self._make_predictor(hidden_dims[-1]),
            "Momentum12M": self._make_predictor(hidden_dims[-1]),
        })

    def _make_predictor(self, hidden_dim):
        """Tanh 제거: 예측 범위 제한 없음"""
        return nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1)
        )

    def forward(self, features, adj_matrix, target_type="Momentum1M"):
        batch, N, T, F = features.shape

        gcn_outputs = []
        for t in range(T):
            x_t = features[:, :, t, :]
            h = self.input_proj(x_t)
            h = self.input_ln(h)
            
            # [개선] Residual Connection
            for gcn, ln in zip(self.gcn_layers, self.gcn_lns):
                h_new = gcn(h, adj_matrix)
                h_new = ln(h_new)
                
                # 차원이 같을 때만 Residual
                if h.shape[-1] == h_new.shape[-1]:
                    h = h_new + h
                else:
                    h = h_new
            
            gcn_outputs.append(h)

        temporal_features = torch.stack(gcn_outputs, dim=1)
        node_embeddings = self.temporal_attn(temporal_features)

        predictions = self.predictors[target_type](node_embeddings).squeeze(-1)

        return predictions, node_embeddings


# ============ 학습 함수 ============

def custom_collate(batch):
    """배치 데이터 병합"""
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "adj_matrix": torch.stack([item["adj_matrix"] for item in batch]),
        "Momentum1M": torch.stack([item["Momentum1M"] for item in batch]),
        "Momentum3M": torch.stack([item["Momentum3M"] for item in batch]),
        "Momentum6M": torch.stack([item["Momentum6M"] for item in batch]),
        "Momentum12M": torch.stack([item["Momentum12M"] for item in batch]),
        "active_mask": torch.stack([item["active_mask"] for item in batch]),
    }


def train_multitask_model(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=100, 
    lr=1e-4, 
    save_path="best_tgnn_multi.pth"
):
    """
    Multi-task TGNN 학습 함수 (개선 버전)
    
    주요 개선사항:
    1. Combined Loss 적용 (MSE + Ranking Loss)
    2. Cosine Annealing Warm Restarts 스케줄러
    3. Gradient Clipping 1.0
    4. Weight Decay 1e-4
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  사용 디바이스: {device}")

    model = model.to(device)
    
    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    # Early Stopping
    best_val_loss = float("inf")
    patience = 30
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"🏋️  학습 시작: {num_epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(num_epochs):
        # ========== Training ==========
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()
            }

            # 4개 Momentum 타겟에 대한 손실 합산
            total_loss = 0.0
            for target in ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]:
                predictions, _ = model(
                    batch["features"], 
                    batch["adj_matrix"], 
                    target_type=target
                )
                
                # Combined Loss 적용
                loss = combined_loss(
                    predictions,
                    batch[target],
                    batch.get("active_mask"),
                    alpha=0.3,  # MSE 가중치
                    beta=0.7    # Ranking 가중치
                )
                total_loss += loss

            # NaN/Inf 체크
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"⚠️  Epoch {epoch}: Loss is NaN/Inf, skipping batch")
                continue

            # Backpropagation
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += total_loss.item()
            train_batches += 1

        # ========== Validation ==========
        model.eval()
        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                batch = {
                    k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()
                }

                total_loss = 0.0
                for target in ["Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M"]:
                    predictions, _ = model(
                        batch["features"], 
                        batch["adj_matrix"], 
                        target_type=target
                    )
                    
                    loss = combined_loss(
                        predictions,
                        batch[target],
                        batch.get("active_mask"),
                        alpha=0.6,
                        beta=0.4
                    )
                    total_loss += loss

                if not torch.isnan(total_loss) and not torch.isinf(total_loss):
                    val_loss += total_loss.item()
                    val_batches += 1

        # ========== 메트릭 계산 ==========
        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        # 스케줄러 업데이트
        scheduler.step()

        # ========== Early Stopping ==========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
            status = "✅ Best"
        else:
            patience_counter += 1
            status = f"⏳ Patience {patience_counter}/{patience}"
            
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping at epoch {epoch}")
                break

        # ========== 로그 출력 ==========
        if epoch % 10 == 0 or epoch == num_epochs - 1:
            print(
                f"Epoch {epoch:3d} | "
                f"Train: {avg_train_loss:.6f} | "
                f"Val: {avg_val_loss:.6f} | "
                f"Best: {best_val_loss:.6f} | "
                f"{status}"
            )

    print(f"\n{'='*60}")
    print(f"✅ 학습 완료!")
    print(f"📁 최적 모델 저장: {save_path}")
    print(f"📊 Best Validation Loss: {best_val_loss:.6f}")
    print(f"{'='*60}\n")


# ============ main (학습만) ============

def main():
    print("=" * 60)
    print("🚀 TGNN Multi-task 학습 스크립트 (개선 버전)")
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
    mask_scale = (train_df["Date"] >= "2006-01-01") & (train_df["Date"] <= "2020-12-31")
    df_scale = train_df[mask_scale].copy()

    for col in feature_cols:
        if col in train_df.columns:
            m = df_scale[col].mean()
            s = df_scale[col].std()
            train_mean[col] = m
            train_std[col] = s
            train_df[col] = (train_df[col] - m) / (s + 1e-8)

    # # Momentum 극단값 클리핑 후 표준화
    # for mom_col in momentum_cols:
    #     if mom_col in train_df.columns:
    #         raw_values = train_df[mom_col].clip(-0.4, 0.5)
    #         train_df[mom_col] = (raw_values - raw_values.mean()) / (raw_values.std() + 1e-8)

    for mom_col in momentum_cols:
        if mom_col in train_df.columns:
            train_df[mom_col] = train_df[mom_col] / 100.0 
            train_df[mom_col] = train_df[mom_col].clip(-0.5, 0.5)

    symbols = sorted(train_df["Symbol"].unique())
    print(f"✅ 종목 수: {len(symbols)}")

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

    print(f"✅ Train 윈도우 수: {len(train_dataset)}")
    print(f"✅ Val 윈도우 수: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=custom_collate
    )

    # 3. 모델 학습
    model = TGNNModel(
        num_features=len(all_feature_cols),
        hidden_dims=[128, 128, 64],
        num_heads=8,
        num_stocks=len(symbols),
    )
    
    model_path = RESULTS_DIR / "best_tgnn_multi.pth"
    train_multitask_model(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=100, 
        lr=1e-4, 
        save_path=str(model_path)
    )

    # 4. 스케일링 파라미터 저장
    scaler_path = RESULTS_DIR / "scaler_params.npz"
    np.savez(
        scaler_path, 
        mean=train_mean, 
        std=train_std, 
        feature_cols=np.array(feature_cols)
    )
    print(f"✅ 스케일러 저장: {scaler_path}")
    print(f"✅ 전체 학습 완료!")


if __name__ == "__main__":
    main()
