"""
TGNN 모델, 데이터셋, 학습 함수 통합
- Range Collapse 문제 해결 (LeakyReLU, Residual, LayerNorm)
- Backtesting 오류 해결 (TGNNDataset에서 raw_labels 반환)
- Hybrid Loss 도입 (MSE + Correlation + Sign)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

# ============ 1. 모델 정의 (개선된 버전) ============

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        
        # 차원이 다를 경우를 대비한 Residual Projection
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()
            
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: [Batch, N, F]
        # adj: [Batch, N, N]
        
        # GCN Propagation: D^-0.5 * A * D^-0.5 * X * W
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        
        # Normalized Adjacency Matrix
        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        
        # Residual Connection & Normalization & Activation
        res = self.residual(x)
        output = self.norm(output + res)
        
        return F.leaky_relu(output, negative_slope=0.2)

class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [Batch, N, T, D] -> [Batch*N, T, D]로 변환하여 처리
        batch, N, T, D = x.shape
        x_reshaped = x.reshape(batch * N, T, D)
        
        # Self-Attention
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        
        # Add & Norm (Residual)
        x_reshaped = self.norm(x_reshaped + self.dropout(attn_out))
        
        # 마지막 시점(Last Timestep)의 Feature만 사용
        # (Global Pooling 대신 최근 정보를 중시)
        return x_reshaped[:, -1, :].reshape(batch, N, D)

class TGNNModel(nn.Module):
    def __init__(self, num_features: int, hidden_dims: List[int] = [64, 64, 32], num_heads: int = 4, dropout: float = 0.3):
        super().__init__()
        
        # 1. Input Projection
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        
        # 2. Spatial Learning (GCN Layers)
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dims[i], hidden_dims[i+1], dropout) 
            for i in range(len(hidden_dims)-1)
        ])
        
        # 3. Temporal Learning (Attention)
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads, dropout)
        
        # 4. Final Prediction Head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 1) # Output: Scalar Score (Raw Return Prediction)
        )

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        # features: [Batch, N, T, F]
        # adj_matrix: [Batch, N, N] (Static or Dynamic)
        
        batch, N, T, F = features.shape
        
        # [Batch, N, T, Hidden]
        h = self.input_proj(features)
        
        # Time-Distributed GCN (각 타임스텝별로 GCN 적용)
        gcn_outputs = []
        for t in range(T):
            h_t = h[:, :, t, :] # [Batch, N, Hidden]
            for gcn in self.gcn_layers:
                h_t = gcn(h_t, adj_matrix)
            gcn_outputs.append(h_t)
            
        # [Batch, N, T, Hidden] -> 원래 차원으로 복구
        temporal_features = torch.stack(gcn_outputs, dim=2) 
        # (주의: stack dim=2여야 [Batch, N, T, Hidden]이 됨. 이전 코드의 dim=1은 [Batch, T, N, H] 였을 수 있음)
        # TemporalAttention은 [Batch, N, T, D] 입력을 받도록 설계됨.
        
        # Temporal Attention -> [Batch, N, Hidden]
        node_embeddings = self.temporal_attn(temporal_features)
        
        # Final Prediction -> [Batch, N]
        predictions = self.predictor(node_embeddings).squeeze(-1)
        
        return predictions, node_embeddings

# ============ 2. 데이터셋 (수정됨) ============

class TGNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 12, feature_cols: List[str] = None, symbols: List[str] = None, target_col: str = "Momentum1M"):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.symbols = symbols if symbols else sorted(df["Symbol"].unique())
        
        # 월별 리샘플링 (ME = Month End)
        # 데이터가 이미 월별이면 이 과정은 중복제거 효과
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")]) 
            .last()
            .reset_index()
        )
        
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict]:
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []
        
        # Window Sliding
        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1] # Feature의 마지막 시점 (현재)
            next_date = dates[i + self.window_size] # 예측 대상 시점 (다음 달)
            
            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]
            
            features = []
            active_mask = []
            labels = [] # 학습용 Label (스케일링 됨)
            raw_labels = [] # 백테스팅용 원본 수익률
            
            # 현재 시점 기준 상장된 종목 확인
            snapshot_df = window_df[window_df["Date"] == target_date]
            
            for symbol in self.symbols:
                # 1. Feature 구성
                stock_data = window_df[window_df["Symbol"] == symbol]
                
                # 현재 시점에 데이터가 존재하는지 확인
                is_active = not stock_data[stock_data["Date"] == target_date].empty
                
                if is_active:
                    vals = stock_data[self.feature_cols].values
                    # 데이터 길이가 부족하면 Zero Padding (상장 초기 등)
                    if len(vals) < self.window_size:
                        pad = np.zeros((self.window_size - len(vals), len(self.feature_cols)))
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    # 데이터가 없으면 전체 Zero Padding
                    features.append(np.zeros((self.window_size, len(self.feature_cols))))
                    active_mask.append(False)
                
                # 2. Label 구성
                if not next_df[next_df["Symbol"] == symbol].empty:
                    # 학습용 Label (여기서는 이미 전처리된 값을 사용한다고 가정)
                    val = next_df[next_df["Symbol"] == symbol][self.target_col].values[0]
                    labels.append(val)
                    
                    # 백테스팅용 Raw Label (여기서는 동일하게 사용하지만, 별도 컬럼이 있다면 교체)
                    # 예: 'Raw_Return' 컬럼이 있다면 그것을 사용
                    raw_val = val 
                    if "Close" in next_df.columns and "Close" in snapshot_df.columns:
                         # 가능하다면 직접 수익률 계산 (Close_t+1 - Close_t) / Close_t
                         pass
                    
                    raw_labels.append(raw_val) 
                else:
                    labels.append(0.0)
                    raw_labels.append(0.0)

            # 3. Graph Construction
            adj_matrix = self._create_masked_graph(snapshot_df, np.array(active_mask))
            
            windows.append({
                "features": torch.FloatTensor(np.array(features)), # [N, T, F]
                "adj_matrix": torch.FloatTensor(adj_matrix),       # [N, N]
                "labels": torch.FloatTensor(np.array(labels)),     # [N]
                "raw_labels": torch.FloatTensor(np.array(raw_labels)), # [N]
                "date": target_date,
                "active_mask": torch.BoolTensor(np.array(active_mask)),
            })
            
        return windows

    def _create_masked_graph(self, snapshot_df, active_mask):
        n = len(self.symbols)
        if snapshot_df.empty: return np.zeros((n, n))
        
        # 기본: Identity Matrix (Self-loop)
        # 고급: Sector 기반 연결 or Correlation Matrix 사용 가능
        adj = np.eye(n) 
        
        # 활성화된 종목끼리만 연결되도록 마스킹
        # (죽은 종목이 살아있는 종목에 영향을 주지 않도록)
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def __getitem__(self, idx):
        return self.windows[idx]

    def __len__(self):
        return len(self.windows)

# ============ 3. 학습 함수 (수정됨) ============

class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 1. MSE Loss (절대값 오차)
        mse_loss = self.mse(pred, target)
        
        # 2. Correlation Loss (방향성 일치 유도)
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        corr_loss = 1 - cost 
        
        # 3. Sign Loss (부호 불일치 패널티)
        # 부호가 다르면 1, 같으면 0에 가까움
        sign_loss = torch.mean(torch.relu(-torch.sign(pred) * torch.sign(target)))

        return (self.alpha * mse_loss) + (self.beta * corr_loss) + ((1 - self.alpha - self.beta) * sign_loss)

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, save_path="best_model.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = HybridLoss(alpha=0.5, beta=0.3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    patience = 20 # Early Stopping patience 증가
    counter = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            feat = batch["features"].to(device)
            adj = batch["adj_matrix"].to(device)
            label = batch["labels"].to(device)
            mask = batch["active_mask"].to(device)
            
            optimizer.zero_grad()
            pred, _ = model(feat, adj)
            
            # Masking: 상장된 종목에 대해서만 Loss 계산
            pred = pred[mask]
            label = label[mask]
            
            if len(pred) > 0:
                loss = criterion(pred, label)
                loss.backward()
                
                # Gradient Clipping (폭주 방지)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                feat = batch["features"].to(device)
                adj = batch["adj_matrix"].to(device)
                label = batch["labels"].to(device)
                mask = batch["active_mask"].to(device)
                
                pred, _ = model(feat, adj)
                pred = pred[mask]
                label = label[mask]
                
                if len(pred) > 0:
                    loss = criterion(pred, label)
                    val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_val = val_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train:.4f}, Val={avg_val:.4f} (Best) ✅")
        else:
            counter += 1
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}: Train={avg_train:.4f}, Val={avg_val:.4f}")
            
            if counter >= patience:
                print(f"⏹️ Early Stopping at Epoch {epoch+1}")
                break
                
    print("🏁 Training Finished.")
