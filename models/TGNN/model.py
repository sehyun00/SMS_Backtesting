"""
TGNN 모델, 데이터셋, 학습 함수 통합
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Tuple

# ============ 모델 정의 ============


class GraphConvLayer(nn.Module):
    """
    그래프 합성곱 레이어 (Graph Convolutional Layer)
    
    주요 기능:
    - 인접 행렬(adj)을 정규화하여 노드 간 연결 강도를 반영
    - 선형 변환(Linear)을 통해 노드 특성을 새로운 차원으로 투영
    - 정규화된 인접 행렬과 투영된 특성을 곱하여 이웃 노드 정보를 집계
    - ReLU 활성화 함수를 적용하여 비선형성 추가
    
    Args:
        in_features: 입력 특성 차원
        out_features: 출력 특성 차원
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        D = torch.sum(adj, dim=-1)
        D_inv_sqrt = torch.pow(D + 1e-6, -0.5)
        D_inv_sqrt[torch.isinf(D_inv_sqrt)] = 0.0

        norm_adj = D_inv_sqrt.unsqueeze(-1) * adj * D_inv_sqrt.unsqueeze(-2)
        support = self.linear(x)
        output = torch.matmul(norm_adj, support)

        return F.relu(output)

#DD
class TemporalAttention(nn.Module):
    """
    시간적 어텐션 레이어 (Temporal Attention Layer)
    
    주요 기능:
    - 시계열 데이터에서 각 시점(timestep)의 중요도를 학습
    - Multi-head Self-Attention을 사용하여 시간적 패턴 포착
    - 여러 시점의 정보를 종합하여 최종 시점의 임베딩 반환
    
    Args:
        hidden_dim: 히든 레이어 차원
        num_heads: 어텐션 헤드 수 (기본값: 8)
    """
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, N, D = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        return attn_out[:, -1, :].reshape(batch, N, D)


class TGNNModel(nn.Module):
    """
    시공간 그래프 신경망 모델 (Temporal Graph Neural Network Model)
    
    주요 기능:
    - 주식 데이터의 공간적 관계(종목 간 상관관계)와 시간적 패턴을 동시에 학습
    - 각 시점별로 GCN 레이어를 적용하여 그래프 구조 정보 반영
    - Temporal Attention으로 시계열 내 중요 시점 가중치 학습
    - 최종 Predictor를 통해 다음 기간 수익률 예측
    
    구조:
        1. Input Projection: 입력 특성을 히든 차원으로 변환
        2. GCN Layers: 그래프 합성곱으로 종목 간 정보 전파
        3. Temporal Attention: 시간축 정보 통합
        4. Predictor: 최종 수익률 예측 (MLP)
    
    Args:
        num_features: 입력 특성 수
        hidden_dims: 각 GCN 레이어의 히든 차원 리스트
        num_heads: 어텐션 헤드 수
        num_stocks: 종목 수
    """
    def __init__(
        self,
        num_features: int,
        hidden_dims: List[int] = [128, 128, 64],
        num_heads: int = 8,
        num_stocks: int = 10,
    ):
        super().__init__()
        self.input_proj = nn.Linear(num_features, hidden_dims[0])

        self.gcn_layers = nn.ModuleList(
            [
                GraphConvLayer(hidden_dims[i], hidden_dims[i + 1])
                for i in range(len(hidden_dims) - 1)
            ]
        )

        self.temporal_attn = TemporalAttention(hidden_dims[-1], num_heads)

        self.predictor = nn.Sequential(
            nn.Linear(hidden_dims[-1], 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, features: torch.Tensor, adj_matrix: torch.Tensor):
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
        predictions = self.predictor(node_embeddings).squeeze(-1)

        return predictions, node_embeddings


# ============ 데이터셋 ============


class TGNNDataset(Dataset):
    """
    TGNN 모델 학습용 데이터셋 클래스
    
    주요 기능:
    - 일별 데이터를 월별로 리샘플링하여 월간 수익률 예측에 활용
    - 슬라이딩 윈도우 방식으로 학습 데이터 생성 (window_size 개월치 데이터 → 다음 달 예측)
    - 동적 유니버스 지원: 상장/상폐된 종목에 대한 마스킹 처리
    - 상관계수 × 산업 유사도 기반의 그래프(인접 행렬) 자동 생성
    
    데이터 처리 흐름:
        1. 월별 리샘플링 (각 월 마지막 거래일 데이터 사용)
        2. 슬라이딩 윈도우로 학습 샘플 생성
        3. 각 윈도우마다 종목별 활성화 상태(active_mask) 계산
        4. 활성 종목 기반 그래프 구성 및 라벨(Momentum1M) 생성
    
    Args:
        df: 전체 주가 데이터 (Date, Symbol, feature_cols, Sector 등 포함)
        window_size: 입력 윈도우 크기 (기본값: 12개월)
        feature_cols: 사용할 특성 컬럼 리스트
        symbols: 분석 대상 종목 리스트
    """
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 12,
        feature_cols: List[str] = None,
        symbols: List[str] = None,
        start_date: str = None,
        end_date: str = None,
    ):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.symbols = symbols if symbols else sorted(df["Symbol"].unique())
        self.feature_cols = feature_cols
        self.start_date = pd.to_datetime(start_date) if start_date else None
        self.end_date = pd.to_datetime(end_date) if end_date else None

        # ✅ 정규화 제거 (run_train_test.py에서 이미 처리함)
        
        # 월별 리샘플링
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )

        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict]:
        """[수정] 동적 유니버스를 위한 윈도우 생성 함수 (기간 필터링 지원)"""
        # 전체 기간의 날짜 목록
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            # 1. 기간 설정
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]
            
            # 기간 필터링: next_date 기준으로 필터링 (예측 대상 날짜)
            if self.start_date and next_date < self.start_date:
                continue
            if self.end_date and next_date > self.end_date:
                continue

            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]

            # 2. Feature 및 Mask 생성
            features = []
            active_mask = []

            for symbol in self.symbols:
                stock_data = window_df[window_df["Symbol"] == symbol]

                # 현재 시점(target_date)에 데이터가 있는지 확인
                is_active = not stock_data[stock_data["Date"] == target_date].empty

                if is_active:
                    # 데이터가 있으면 채워넣기 (상장 직후 패딩 포함)
                    vals = stock_data[self.feature_cols].values
                    if len(vals) < self.window_size:
                        pad = np.zeros(
                            (self.window_size - len(vals), len(self.feature_cols))
                        )
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    # 데이터 없으면(상장 전) 0으로 채움
                    features.append(
                        np.zeros((self.window_size, len(self.feature_cols)))
                    )
                    active_mask.append(False)

            # 3. 그래프 생성 (Masking 적용)
            adj_matrix = self._create_masked_graph(
                window_df[window_df["Date"] == target_date], np.array(active_mask)
            )

            # 4. 라벨 생성 (없는 종목은 수익률 0)
            labels = []
            for symbol in self.symbols:
                val = next_df[next_df["Symbol"] == symbol]["Momentum1M"].values
                labels.append(val[0] if len(val) > 0 else 0.0)

            windows.append(
                {
                    "features": torch.FloatTensor(np.array(features)),
                    "adj_matrix": torch.FloatTensor(adj_matrix),
                    "labels": torch.FloatTensor(np.array(labels)),
                    "date": target_date,
                    "active_mask": np.array(active_mask),  # 백테스팅에 사용할 마스크
                }
            )

        return windows

    def _create_masked_graph(
        self, snapshot_df: pd.DataFrame, active_mask: np.ndarray
    ) -> np.ndarray:
        """[신규] 마스킹을 적용한 그래프 생성 함수"""
        n = len(self.symbols)
        # 먼저 기존 _create_graph 호출
        adj = (
            self._create_graph(snapshot_df)
            if not snapshot_df.empty
            else np.zeros((n, n))
        )

        # active_mask를 이용해 존재하지 않는 종목의 연결을 모두 제거
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    # [수정] __getitem__에서 active_mask도 반환하도록 변경
    def __getitem__(self, idx):
        w = self.windows[idx]
        # date는 DataLoader에서 문제를 일으키므로 제거
        return {
            "features": w["features"],
            "adj_matrix": w["adj_matrix"],
            "labels": w["labels"],
            "active_mask": torch.BoolTensor(w["active_mask"]),  # BoolTensor로 변환
        }

    def _create_graph(self, snapshot_df: pd.DataFrame) -> np.ndarray:
        """상관계수 × 산업 유사도"""
        n = len(self.symbols)
        corr_matrix = np.eye(n)

        for i, sym1 in enumerate(self.symbols):
            data1 = snapshot_df[snapshot_df["Symbol"] == sym1][
                self.feature_cols
            ].values.flatten()

            for j, sym2 in enumerate(self.symbols):
                if i >= j:
                    continue

                data2 = snapshot_df[snapshot_df["Symbol"] == sym2][
                    self.feature_cols
                ].values.flatten()

                if len(data1) > 0 and len(data2) > 0:
                    corr = np.corrcoef(data1, data2)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

        # 산업 유사도
        sector_map = snapshot_df.set_index("Symbol")["Sector"].to_dict()
        industry_sim = np.zeros((n, n))

        for i, sym1 in enumerate(self.symbols):
            for j, sym2 in enumerate(self.symbols):
                if sym1 in sector_map and sym2 in sector_map:
                    industry_sim[i, j] = (
                        1.0 if sector_map[sym1] == sector_map[sym2] else 0.5
                    )

        edge_weights = corr_matrix * industry_sim
        adj_matrix = (edge_weights >= 0.35).astype(float)

        return adj_matrix

    def __len__(self):
        return len(self.windows)


# ============ 학습 함수 ============

def train_model(
    model: TGNNModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 300,
    lr: float = 1e-5,  # ✅ 기본값 낮춤
    save_path: str = "best_tgnn.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # ✅ Gradient Clipping 값
    max_grad_norm = 0.5

    best_val_loss = float("inf")
    patience = 20
    patience_counter = 0

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_batches = 0

        for batch in train_loader:
            predictions, _ = model(batch["features"], batch["adj_matrix"])
            loss = criterion(predictions, batch["labels"])
            
            # ✅ NaN/Inf 체크
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️  Epoch {epoch}: Loss is NaN/Inf, skipping batch")
                continue

            optimizer.zero_grad()
            loss.backward()
            
            # ✅ Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1

        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                predictions, _ = model(batch["features"], batch["adj_matrix"])
                loss = criterion(predictions, batch["labels"])
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    val_loss += loss.item()
                    val_batches += 1

        avg_train_loss = train_loss / max(train_batches, 1)
        avg_val_loss = val_loss / max(val_batches, 1)

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={avg_train_loss:.6f}, Val={avg_val_loss:.6f}, Best Val={best_val_loss:.6f}")

    print(f"\n최적 모델 저장: {save_path} (Best Val Loss: {best_val_loss:.6f})")
