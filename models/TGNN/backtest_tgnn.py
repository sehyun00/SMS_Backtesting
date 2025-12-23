"""
TGNN 백테스트 전용 스크립트
- train_tgnn.py에서 학습한 best_tgnn_multi.pth 사용
- test_data.csv (2021-01-01 ~ 2025-12-31) 구간 백테스트
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os
import platform

# 경로 설정
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from model import TGNNDataset
    from model import TGNNModel as BaseTGNNModel
    from backtester import Backtester, BacktestConfig, create_metrics_summary_table
except ImportError:
    print("❌ 모듈 Import 실패")
    sys.exit(1)

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"
RESULTS_DIR = ROOT_DIR / "results" / "01_TGNN_Only"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

font_name = "Malgun Gothic" if platform.system() == "Windows" else "AppleGothic"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

# ============ Dataset & Model 재사용 ============


class TGNN_Dataset(TGNNDataset):
    def _create_windows(self):
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

            features = []
            active_mask = []

            for symbol in self.symbols:
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

            adj_matrix = self._create_masked_graph(
                window_df[window_df["Date"] == target_date],
                np.array(active_mask),
            )

            labels_1m, labels_3m, labels_6m, labels_12m = [], [], [], []
            for symbol in self.symbols:
                next_stock = next_df[next_df["Symbol"] == symbol]
                labels_1m.append(
                    next_stock["Momentum1M"].values[0] if len(next_stock) > 0 else 0.0
                )
                labels_3m.append(
                    next_stock["Momentum3M"].values[0] if len(next_stock) > 0 else 0.0
                )
                labels_6m.append(
                    next_stock["Momentum6M"].values[0] if len(next_stock) > 0 else 0.0
                )
                labels_12m.append(
                    next_stock["Momentum12M"].values[0] if len(next_stock) > 0 else 0.0
                )

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


import torch.nn as nn


class TGNNModel(BaseTGNNModel):
    def __init__(
        self, num_features, hidden_dims=[128, 128, 64], num_heads=8, num_stocks=10
    ):
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


# ============ 시각화 ============


def plot_performance(strategies, save_dir):
    plt.figure(figsize=(14, 7))
    for name, res in strategies.items():
        if not res or not res.get("history"):
            continue
        df = pd.DataFrame(res["history"])
        if "date" not in df.columns or "cumulative_return" not in df.columns:
            continue
        dates = pd.to_datetime(df["date"])
        vals = df["cumulative_return"]
        plt.plot(dates, vals, label=name, linewidth=2)

    plt.title("Performance Comparison: TGNN", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Cumulative Return (%)", fontsize=12)
    plt.xlim(left=pd.Timestamp("2022-01-01"))
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best", fontsize=10)
    plt.tight_layout()
    plt.savefig(save_dir / "comparison_graph.png", dpi=300)
    plt.close()
    print(f"✅ 그래프 저장: {save_dir / 'comparison_graph.png'}")


# ============ main (백테스트만) ============


def main():
    print("=" * 60)
    print("📊 TGNN 백테스트 전용 스크립트")
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

    # 스케일러 파라미터 로드
    scaler_path = RESULTS_DIR / "scaler_params.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(f"스케일러 파일이 없습니다: {scaler_path}")
    scaler = np.load(scaler_path, allow_pickle=True)
    train_mean = scaler["mean"].item()
    train_std = scaler["std"].item()

    # 1. test_data.csv 로드
    print("\n📂 테스트 데이터 로드 중...")
    test_df = pd.read_csv(TEST_DATA_PATH)
    test_df["Date"] = pd.to_datetime(test_df["Date"])
    test_df = test_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    for col in feature_cols:
        if col in test_df.columns and col in train_mean:
            test_df[col] = (test_df[col] - train_mean[col]) / (train_std[col] + 1e-8)

    for mom_col in momentum_cols:
        if mom_col in test_df.columns:
            test_df[mom_col] = test_df[mom_col].clip(-0.4, 0.5)

    symbols = sorted(test_df["Symbol"].unique())
    print(f"종목 수(테스트): {len(symbols)}")

    test_dataset = TGNN_Dataset(
        test_df,
        window_size=12,
        feature_cols=all_feature_cols,
        symbols=symbols,
        start_date="2021-01-01",
        end_date="2025-12-31",
    )

    print(f"테스트 윈도우 수: {len(test_dataset)}")

    # 모델 로드
    model_path = RESULTS_DIR / "best_tgnn_multi.pth"
    model = TGNNModel(
        num_features=len(all_feature_cols),
        hidden_dims=[128, 128, 64],
        num_heads=8,
        num_stocks=len(symbols),
    )
    model.load_state_dict(torch.load(model_path, map_location="cpu"))

    # 백테스트
    config = BacktestConfig(
        initial_capital=10_000_000, cost_bps=10.0, top_k=5, weighting_method="equal"
    )

    all_strategies = {}
    all_metrics = {}
    all_backtesters = {}

    print("[1/5] Buy & Hold...")
    bt_buyhold = Backtester(model, test_dataset, config, "Buy_Hold")
    bh_res = bt_buyhold.run_buy_and_hold()
    all_strategies["Buy & Hold"] = bh_res
    all_metrics["Buy & Hold"] = bt_buyhold.metrics
    all_backtesters["Buy & Hold"] = bt_buyhold

    experiments = [
        ("monthly", "Momentum1M", "TGNN 1M"),
        ("quarterly", "Momentum3M", "TGNN 3M"),
        ("semiannual", "Momentum6M", "TGNN 6M"),
        ("annual", "Momentum12M", "TGNN 12M"),
    ]

    for i, (freq, target, name) in enumerate(experiments, 2):
        print(f"[{i}/5] {name}...")
        bt = Backtester(model, test_dataset, config, name)
        bt.target_type = target
        res = bt.run(freq)
        all_strategies[name] = res
        all_metrics[name] = bt.metrics
        all_backtesters[name] = bt

    # 메트릭 저장
    df_res = create_metrics_summary_table(all_metrics)
    print("\n" + "=" * 60)
    print("📊 Final Results")
    print("=" * 60)
    print(df_res.to_string(index=False))

    df_res.to_csv(
        RESULTS_DIR / "comparison_metrics.csv", index=False, encoding="utf-8-sig"
    )
    print(f"\n✅ 저장: comparison_metrics.csv")

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
    df_detailed.to_csv(
        RESULTS_DIR / "metrics_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print("✅ 저장: metrics_comparison.csv (상세 메트릭)")

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

    plot_performance(all_strategies, RESULTS_DIR)

    print(f"\n{'=' * 60}")
    print(f"✅ 완료! 결과 폴더: {RESULTS_DIR}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
