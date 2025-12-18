"""
Hybrid Dataset for TGNN-DDPG
그래프 구조와 시계열 윈도우 데이터 제공
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler


class HybridDataset:
    """
    Hybrid 모델용 Dataset 클래스
    - 주식 간 관계 그래프(인접 행렬) 생성
    - 시계열 윈도우 데이터 제공
    - train_data.csv와 test_data.csv를 별도로 로드
    """

    def __init__(self, train_df, test_df, window_size=12, feature_cols=None):
        """
        Args:
            train_df: 학습 데이터 DataFrame (2006-2020)
            test_df: 테스트 데이터 DataFrame (2021-2025)
            window_size: 시계열 윈도우 크기 (개월)
            feature_cols: 사용할 특성 컬럼 리스트
        """
        self.train_df = train_df.copy()
        self.test_df = test_df.copy()
        self.window_size = window_size
        self.feature_cols = feature_cols

        # 날짜 변환
        self.train_df["Date"] = pd.to_datetime(self.train_df["Date"])
        self.test_df["Date"] = pd.to_datetime(self.test_df["Date"])

        # 종목 리스트 (Train과 Test 별도)
        self.train_symbols = sorted(self.train_df["Symbol"].unique())
        self.test_symbols = sorted(self.test_df["Symbol"].unique())

        print(f"   📊 Train 종목 ({len(self.train_symbols)}개): {self.train_symbols}")
        print(f"   📊 Test 종목 ({len(self.test_symbols)}개): {self.test_symbols}")

        # 일별 → 월말 변환
        self.train_monthly = self._convert_to_monthly(self.train_df)
        self.test_monthly = self._convert_to_monthly(self.test_df)

        # 타겟 수익률 (다음 달 모멘텀)
        self.train_monthly["ReturnRaw"] = self.train_monthly["Momentum1M"].copy()
        self.test_monthly["ReturnRaw"] = self.test_monthly["Momentum1M"].copy()

        # 스케일링 (Train 데이터로만 피팅)
        self.fit_scaler_on_train()

        # 윈도우 생성
        self.train_windows = self.create_windows(self.train_monthly, self.train_symbols)
        self.test_windows = self.create_windows(self.test_monthly, self.test_symbols)

        print(f"   📈 Train windows: {len(self.train_windows)} months")
        print(f"   📉 Test windows: {len(self.test_windows)} months")

    def _convert_to_monthly(self, df):
        """일별 데이터를 월말 데이터로 변환"""
        monthly = (
            df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )
        return monthly

    def fit_scaler_on_train(self):
        """Train 데이터로만 StandardScaler 피팅 (5-Factor 제외)"""
        # 5-Factor 제외 (이미 % 단위)
        scale_cols = [
            col
            for col in self.feature_cols
            if col not in ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
        ]

        if not scale_cols:
            print("   ⚠️  스케일링할 컬럼이 없습니다!")
            return

        self.scaler = StandardScaler()
        self.scaler.fit(self.train_monthly[scale_cols].values)

        # Train과 Test 모두에 스케일링 적용
        self.train_monthly[scale_cols] = self.scaler.transform(
            self.train_monthly[scale_cols].values
        )
        self.test_monthly[scale_cols] = self.scaler.transform(
            self.test_monthly[scale_cols].values
        )

        print(f"   ✅ StandardScaler fitted on {len(scale_cols)} features (Train only)")

    def create_graph(self, snapshot_df, symbols):
        """
        상관관계 + 산업 유사도 기반 인접 행렬 생성

        Args:
            snapshot_df: 특정 시점의 데이터
            symbols: 종목 리스트

        Returns:
            adj: (N, N) 인접 행렬
        """
        n = len(symbols)

        # 🔥 최적화: 한 번에 모든 데이터 추출
        feature_data = []
        for sym in symbols:
            data = snapshot_df[snapshot_df["Symbol"] == sym][self.feature_cols].values
            if len(data) > 0:
                feature_data.append(data.flatten())
            else:
                feature_data.append(np.zeros(len(self.feature_cols)))

        feature_matrix = np.array(feature_data)  # (N, features)

        # 🔥 최적화: 벡터화된 상관계수 계산
        if feature_matrix.shape[0] > 1:
            corr_matrix = np.corrcoef(feature_matrix)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        else:
            corr_matrix = np.eye(n)

        # 산업 유사도 반영
        if "Sector" in snapshot_df.columns:
            sector_map = snapshot_df.set_index("Symbol")["Sector"].to_dict()
            industry_sim = np.zeros((n, n))
            for i, sym1 in enumerate(symbols):
                for j, sym2 in enumerate(symbols):
                    if sym1 in sector_map and sym2 in sector_map:
                        industry_sim[i, j] = (
                            1.0 if sector_map[sym1] == sector_map[sym2] else 0.5
                        )
            edge_weights = corr_matrix + industry_sim
        else:
            edge_weights = corr_matrix

        # 임계값 이상만 연결
        adj = (edge_weights > 0.35).astype(float)
        return adj

    def create_masked_graph(self, snapshot_df, symbols, active_mask):
        """비활성 종목의 연결 제거"""
        n = len(symbols)
        adj = (
            self.create_graph(snapshot_df, symbols)
            if not snapshot_df.empty
            else np.zeros((n, n))
        )
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def create_windows(self, monthly_df, symbols):
        """
        전체 기간의 윈도우 데이터 생성

        Args:
            monthly_df: 월별 데이터
            symbols: 종목 리스트

        Returns:
            windows: 윈도우 데이터 리스트
        """
        dates = sorted(monthly_df["Date"].unique())
        windows = []

        total_windows = len(dates) - self.window_size
        print(f"   🔄 Creating {total_windows} windows...")

        for i in range(len(dates) - self.window_size):
            if i % max(1, total_windows // 10) == 0:
                print(
                    f"      Progress: {i}/{total_windows} ({i * 100 // total_windows}%)"
                )
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]

            window_df = monthly_df[monthly_df["Date"].isin(window_dates)]
            next_df = monthly_df[monthly_df["Date"] == next_date]

            features = []
            active_mask = []

            for symbol in symbols:
                stock_data = window_df[window_df["Symbol"] == symbol]
                is_active = not stock_data[stock_data["Date"] == target_date].empty

                if is_active:
                    vals = stock_data[self.feature_cols].values
                    if len(vals) < self.window_size:
                        pad = np.zeros(
                            (self.window_size - len(vals), len(self.feature_cols))
                        )
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    features.append(
                        np.zeros((self.window_size, len(self.feature_cols)))
                    )
                    active_mask.append(False)

            active_mask = np.array(active_mask)
            adj = self.create_masked_graph(
                window_df[window_df["Date"] == target_date], symbols, active_mask
            )

            labels = []
            for symbol in symbols:
                val = next_df[next_df["Symbol"] == symbol]["ReturnRaw"].values
                labels.append(val[0] if len(val) > 0 else 0.0)

            windows.append(
                {
                    "features": np.array(features),
                    "adjmatrix": adj,
                    "labels": np.array(labels),
                    "date": target_date,
                    "activemask": active_mask,
                }
            )

        return windows

    def get_state(self, windows, idx):
        """
        RL 에이전트 입력용 상태 벡터 반환

        Args:
            windows: 윈도우 리스트 (train_windows 또는 test_windows)
            idx: 윈도우 인덱스

        Returns:
            state: (state_dim,) numpy array
        """
        w = windows[idx]
        features = w["features"]
        adj = w["adjmatrix"]

        # NaN 제거
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        adj = np.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=0.0)

        state = np.concatenate([features.flatten(), adj.flatten()])
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        return state.astype(np.float32)

    def get_train_windows(self):
        """학습용 윈도우 반환"""
        return self.train_windows

    def get_test_windows(self):
        """테스트용 윈도우 반환"""
        return self.test_windows

    def __len__(self):
        """전체 윈도우 개수 (Train + Test)"""
        return len(self.train_windows) + len(self.test_windows)
