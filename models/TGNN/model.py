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


class TemporalAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, T, N, D = x.shape
        x_reshaped = x.permute(0, 2, 1, 3).reshape(batch * N, T, D)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        return attn_out[:, -1, :].reshape(batch, N, D)


class TGNNModel(nn.Module):
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
    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 12,
        feature_cols: List[str] = None,
        symbols: List[str] = None,
    ):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.symbols = symbols if symbols else sorted(df["Symbol"].unique())
        self.feature_cols = feature_cols

        # 월별 리샘플링
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )

        self.windows = self._create_windows()

    def _create_windows(self) -> List[Dict]:
        """[수정] 동적 유니버스를 위한 윈도우 생성 함수"""
        # 전체 기간의 날짜 목록
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            # 1. 기간 설정
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]

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
    lr: float = 1e-4,
    save_path: str = "best_tgnn.pth",
):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0

        for batch in train_loader:
            predictions, _ = model(batch["features"], batch["adj_matrix"])
            loss = criterion(predictions, batch["labels"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in val_loader:
                predictions, _ = model(batch["features"], batch["adj_matrix"])
                loss = criterion(predictions, batch["labels"])
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train={train_loss:.6f}, Val={val_loss:.6f}")

    print(f"\n최적 모델 저장: {save_path}")
