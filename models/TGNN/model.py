"""
TGNN 모델, 데이터셋, 학습 함수 통합
- Range Collapse 문제 해결 (LeakyReLU, Residual)
- Backtesting 오류 해결 (TGNNDataset에서 raw_labels 반환)
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple, Optional

# ============ 모델 정의 (개선된 버전) ============

class GraphConvLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        if in_features != out_features:
            self.residual = nn.Linear(in_features, out_features)
        else:
            self.residual = nn.Identity()
        self.norm = nn.LayerNorm(out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj shape: [Batch, N, N]
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0
        
        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)
        
        # Residual & Norm & Activation
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
        batch, T, N, D = x.shape
        # [Batch, T, N, D] -> [Batch*N, T, D]
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)
        
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        x_reshaped = self.norm(x_reshaped + self.dropout(attn_out))
        
        # Last timestep
        return x_reshaped[:, -1, :].reshape(batch, N, D)


class TGNNModel(nn.Module):
    def __init__(self, num_features: int, hidden_dims: List[int] = [64, 64, 32], num_heads: int = 4, dropout: float = 0.3, num_stocks: int = 10):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dims[0])
        
        self.gcn_layers = nn.ModuleList([
            GraphConvLayer(hidden_dims[i], hidden_dims[i+1], dropout) 
            for i in range(len(hidden_dims)-1)
        ])
        
        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads, dropout)
        
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
        # features: [Batch, N, T, F]
        batch, N, T, F = features.shape
        
        # [Batch, N, T, Hidden]
        h = self.input_proj(features)
        
        # Time-Distributed GCN
        gcn_outputs = []
        for t in range(T):
            h_t = h[:, :, t, :]
            for gcn in self.gcn_layers:
                h_t = gcn(h_t, adj_matrix)
            gcn_outputs.append(h_t)
            
        temporal_features = torch.stack(gcn_outputs, dim=1) # [Batch, T, N, Hidden]
        
        # Temporal Attention -> [Batch, N, Hidden]
        # (주의: TemporalAttention 내부에서 [Batch, T, N, D] 입력을 [Batch*N, T, D]로 처리)
        # stack 결과는 [Batch, T, N, D] 순서이므로 바로 넘김
        node_embeddings = self.temporal_attn(temporal_features)
        
        predictions = self.predictor(node_embeddings).squeeze(-1)
        return predictions, node_embeddings


# ============ 데이터셋 (수정됨) ============

class TGNNDataset(Dataset):
    def __init__(self, df: pd.DataFrame, window_size: int = 12, feature_cols: List[str] = None, symbols: List[str] = None):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.symbols = symbols if symbols else sorted(df["Symbol"].unique())
        
        # 월별 리샘플링
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")]) # ME = Month End
            .last()
            .reset_index()
        )
        
        # [중요] 원본 수익률 보존 (스케일링 전)
        # 만약 df가 이미 스케일링 된 상태라면, 원본 df를 따로 받거나 해야 함.
        # 여기서는 run_train_test.py에서 feature만 스케일링하고 target은 남겨둔다고 가정.
        # Momentum1M 컬럼이 타겟이라고 가정.
        
        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict]:
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []
        
        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1] # 현재 시점 (Features의 끝)
            next_date = dates[i + self.window_size] # 예측 대상 (Label)
            
            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]
            
            features = []
            active_mask = []
            labels = [] # 학습용 Label (스케일링 여부는 run_train_test에 달림)
            raw_labels = [] # 백테스팅용 원본 수익률
            
            snapshot_df = window_df[window_df["Date"] == target_date]
            
            for symbol in self.symbols:
                # 1. Feature 구성
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
                
                # 2. Label 구성
                if not next_df[next_df["Symbol"] == symbol].empty:
                    val = next_df[next_df["Symbol"] == symbol]["Momentum1M"].values[0]
                    labels.append(val)
                    raw_labels.append(val) 
                else:
                    labels.append(0.0)
                    raw_labels.append(0.0)

            # 3. Graph
            adj_matrix = self._create_masked_graph(snapshot_df, np.array(active_mask))
            
            windows.append({
                "features": torch.FloatTensor(np.array(features)), # [N, T, F]
                "adj_matrix": torch.FloatTensor(adj_matrix),
                "labels": torch.FloatTensor(np.array(labels)), # [N]
                "raw_labels": torch.FloatTensor(np.array(raw_labels)), # [N] (백테스팅용)
                "date": target_date,
                "active_mask": np.array(active_mask),
            })
            
        return windows

    def _create_masked_graph(self, snapshot_df, active_mask):
        n = len(self.symbols)
        if snapshot_df.empty: return np.zeros((n, n))
        
        # 간단한 Correlation Graph or Identity
        # (실제로는 섹터/상관관계 로직 들어가야 함)
        adj = np.eye(n) 
        
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def __getitem__(self, idx):
        w = self.windows[idx]
        return {
            "features": w["features"], # [N, T, F]
            "adj_matrix": w["adj_matrix"], # [N, N]
            "labels": w["labels"], # [N]
            "raw_labels": w["raw_labels"], # [N]
            "active_mask": torch.BoolTensor(w["active_mask"]),
        }

    def __len__(self):
        return len(self.windows)


# ============ 학습 함수 (수정됨) ============

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, save_path="best.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")
    
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # [중요] Hybrid Loss 사용 (Range Collapse 방지)
    # 여기서는 간단히 MSE만 쓰지 않고 Correlation Loss 추가해야 함
    from models.TGNN.backtester import HybridLoss # 순환 참조 주의, 파일 분리 권장
    criterion = HybridLoss(alpha=0.5, beta=0.3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    best_val_loss = float('inf')
    patience = 30
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
            
            # Masking
            pred = pred[mask]
            label = label[mask]
            
            if len(pred) > 0:
                loss = criterion(pred, label)
                loss.backward()
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
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), save_path)
            counter = 0
            print(f"Epoch {epoch}: Train={avg_train:.4f}, Val={avg_val:.4f} (Best)")
        else:
            counter += 1
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train={avg_train:.4f}, Val={avg_val:.4f}")
            if counter >= patience:
                print("Early Stopping")
                break
