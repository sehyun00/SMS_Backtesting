"""
TGNN 학습 & 리밸런싱 빈도별 백테스팅 비교
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import TGNNModel, TGNNDataset, train_model

# 프로젝트 루트
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = (
    ROOT_DIR / "data" / "processed_daily_5factor_model_10stocks_10years_20251127.csv"
)

# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============ 백테스팅 함수들 ============


def softmax(x):
    """배열을 확률 분포(합 1)로 변환"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def custom_collate(batch):
    """학습 데이터 로더용 콜레이트 함수"""
    return {
        "features": torch.stack([item["features"] for item in batch]),
        "adj_matrix": torch.stack([item["adj_matrix"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        # active_mask는 학습에서 제외
    }


def run_buy_and_hold(dataset):
    """1/N 매수 후 보유"""
    initial_capital = 1000000
    n_stocks = len(dataset.symbols)
    weights = np.ones(n_stocks) / n_stocks

    portfolio_values = [initial_capital]
    dates = []

    for window in dataset.windows:
        actual_returns = window["labels"]
        portfolio_return = np.dot(weights, actual_returns)
        new_value = portfolio_values[-1] * (1 + portfolio_return / 100)

        portfolio_values.append(new_value)
        dates.append(window["date"])

    return {
        "dates": dates,
        "portfolio_values": portfolio_values[1:],
        "final_capital": portfolio_values[-1],
        "cumulative_return": (portfolio_values[-1] / initial_capital - 1) * 100,
    }


def run_tgnn_rebalancing(model, dataset, rebalance_freq="monthly"):
    """[수정] Dynamic Universe를 위한 TGNN 리밸런싱"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[rebalance_freq]

    model.eval()
    initial_capital = 1000000
    capital = initial_capital

    portfolio_values = []
    dates = []
    # [수정] 초기 웨이트는 0으로 시작
    current_weights = np.zeros(len(dataset.symbols))

    with torch.no_grad():
        # [수정] DataLoader 대신 dataset을 직접 순회
        for idx in range(len(dataset)):
            # DataLoader의 collate와 같은 역할을 직접 수행
            batch = dataset[idx]

            # 현재 시점의 날짜와 거래 가능 종목 마스크
            date = dataset.windows[idx]["date"]
            active_mask = batch["active_mask"].numpy()

            # 리밸런싱 시점
            if idx % interval == 0:
                features = batch["features"].unsqueeze(0)
                adj = batch["adj_matrix"].unsqueeze(0)

                predictions, _ = model(features, adj)
                pred_returns = predictions.squeeze(0).numpy()

                # [핵심] 상장 전 종목의 예측값을 -무한대로 설정하여 선택 방지
                pred_returns[~active_mask] = -np.inf

                # 거래 가능한 종목 수
                n_active = np.sum(active_mask)
                # Top-K에서 k는 5와 거래 가능 종목 수 중 작은 값
                k = min(5, n_active)

                new_weights = np.zeros(len(dataset.symbols))
                if k > 0:
                    top_k_idx = np.argsort(pred_returns)[-k:]
                    # new_weights[top_k_idx] = 1.0 / k

                    top_scores = pred_returns[top_k_idx]
                    new_weights[top_k_idx] = softmax(top_scores)

                current_weights = new_weights

            # 수익 계산
            actual_returns = batch["labels"].numpy()
            portfolio_return = np.dot(current_weights, actual_returns)
            capital *= 1 + portfolio_return / 100

            portfolio_values.append(capital)
            dates.append(date)

    return {
        "dates": dates,
        "portfolio_values": portfolio_values,
        "final_capital": capital,
        "cumulative_return": (capital / initial_capital - 1) * 100,
    }


# ============ 시각화 ============


def plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir):
    """비교 그래프 생성 (GridSpec 사용으로 축 겹침 해결)"""

    # 1. 레이아웃 설정 (GridSpec)
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)  # 2행 2열 그리드

    # ax1: 윗줄 전체 (0행, 모든 열)
    ax1 = fig.add_subplot(gs[0, :])
    # ax2: 아랫줄 왼쪽 (1행, 0열)
    ax2 = fig.add_subplot(gs[1, 0])
    # ax3: 아랫줄 오른쪽 (1행, 1열)
    ax3 = fig.add_subplot(gs[1, 1])

    strategies = {
        "1/N 매수 후 보유": buy_and_hold,
        "TGNN (월간)": monthly,
        "TGNN (분기)": quarterly,
        "TGNN (반기)": semiannual,
        "TGNN (연간)": annual,
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    # ==========================================
    # 1. 누적 수익률 그래프 (Top)
    # ==========================================
    all_returns = []  # Y축 범위 설정을 위해 수집

    for (name, data), color in zip(strategies.items(), colors):
        dates = pd.to_datetime(data["dates"])
        initial_value = data["portfolio_values"][0]

        # 수익률 계산 (%)
        returns = [(v / initial_value - 1) * 100 for v in data["portfolio_values"]]
        all_returns.extend(returns)

        ax1.plot(dates, returns, label=name, linewidth=2.5, color=color)

    ax1.set_title(
        "리밸런싱 빈도별 누적 수익률 (2015-2025)",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax1.set_ylabel("누적 수익률 (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("연도", fontsize=12, fontweight="bold")

    # X축 날짜 포맷
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    # Y축 범위 자동 조정 (여유 있게)
    y_min, y_max = min(all_returns), max(all_returns)
    ax1.set_ylim(y_min - 10, y_max * 1.1)

    ax1.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9)
    ax1.grid(True, which="major", alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # ==========================================
    # 2. CAGR 비교 (Bottom Left)
    # ==========================================
    cagr_values = []
    labels = ["Buy&Hold", "월간", "분기", "반기", "연간"]

    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        dates_list = pd.to_datetime(data["dates"])
        days = (dates_list.max() - dates_list.min()).days
        years = days / 365.25
        cagr = (pow(data["final_capital"] / 1000000, 1 / years) - 1) * 100
        cagr_values.append(cagr)

    bars = ax2.bar(
        range(5), cagr_values, color=colors, alpha=0.85, edgecolor="black", width=0.6
    )
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_title("연평균 수익률 (CAGR)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("수익률 (%)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # 값 표시
    for bar, value in zip(bars, cagr_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # ==========================================
    # 3. MDD 비교 (Bottom Right)
    # ==========================================
    mdd_values = []
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        portfolio = np.array(data["portfolio_values"])
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max * 100
        mdd_values.append(abs(drawdown.min()))

    bars = ax3.bar(
        range(5), mdd_values, color=colors, alpha=0.85, edgecolor="black", width=0.6
    )
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(labels, fontsize=10)
    ax3.set_title("최대 낙폭 (MDD)", fontsize=14, fontweight="bold")
    ax3.set_ylabel("낙폭 (%)", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)

    # 값 표시
    for bar, value in zip(bars, mdd_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 1,
            f"-{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#D32F2F",
        )

    # 레이아웃 마무리
    plt.tight_layout()

    save_path = save_dir / "rebalancing_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.show()


# ============ 메인 실행 ============


def main(mode="compare"):
    """
    메인 실행 함수

    Args:
        mode: 'train' (학습만) 또는 'compare' (백테스팅 비교)
    """
    # 데이터 로드
    df = pd.read_csv(DATA_PATH)

    feature_cols = [
        "Beta",
        "MarketCap",
        "Momentum1M",
        "Momentum6M",
        "Volatility",
        "RSI",
        "Beta_Factor",
        "Value_Factor",
        "Size_Factor",
        "Momentum_Factor",
        "Volatility_Factor",
    ]

    # 데이터셋
    dataset = TGNNDataset(df=df, window_size=12, feature_cols=feature_cols)

    # 모델
    model = TGNNModel(
        num_features=len(feature_cols),
        hidden_dims=[128, 128, 64],
        num_heads=8,
        num_stocks=10,
    )

    model_path = Path(__file__).parent / "best_tgnn.pth"

    if mode == "train":
        # ========== 학습 모드 ==========
        print("=" * 60)
        print("TGNN 모델 학습 시작")
        print("=" * 60)

        train_size = int(len(dataset) * 0.7)
        val_size = int(len(dataset) * 0.15)

        train_data = torch.utils.data.Subset(dataset, range(train_size))
        val_data = torch.utils.data.Subset(
            dataset, range(train_size, train_size + val_size)
        )

        train_loader = DataLoader(
            train_data,
            batch_size=32,
            shuffle=True,
            collate_fn=custom_collate,
        )
        val_loader = DataLoader(
            val_data,
            batch_size=32,
            collate_fn=custom_collate,
        )

        train_model(
            model, train_loader, val_loader, num_epochs=300, save_path=str(model_path)
        )
        print("\n✅ 학습 완료!")

    elif mode == "compare":
        # ========== 백테스팅 비교 모드 ==========
        if not model_path.exists():
            print("❌ 모델 파일이 없습니다. 먼저 학습하세요:")
            print("   python run_comparison.py train")
            return

        model.load_state_dict(torch.load(model_path))

        print("=" * 60)
        print("리밸런싱 빈도별 백테스팅 비교")
        print("=" * 60)

        # 실행
        print("\n[1/5] 1/N Buy & Hold...")
        buy_and_hold = run_buy_and_hold(dataset)

        print("[2/5] TGNN 월간 리밸런싱...")
        monthly = run_tgnn_rebalancing(model, dataset, "monthly")

        print("[3/5] TGNN 분기 리밸런싱...")
        quarterly = run_tgnn_rebalancing(model, dataset, "quarterly")

        print("[4/5] TGNN 반기 리밸런싱...")
        semiannual = run_tgnn_rebalancing(model, dataset, "semiannual")

        print("[5/5] TGNN 연간 리밸런싱...")
        annual = run_tgnn_rebalancing(model, dataset, "annual")

        # 결과 요약
        print("\n" + "=" * 60)
        print("결과 요약")
        print("=" * 60)

        results_df = pd.DataFrame(
            {
                "전략": [
                    "1/N Buy & Hold",
                    "TGNN (월간)",
                    "TGNN (분기)",
                    "TGNN (반기)",
                    "TGNN (연간)",
                ],
                "최종 자산 (원)": [
                    f"{buy_and_hold['final_capital']:,.0f}",
                    f"{monthly['final_capital']:,.0f}",
                    f"{quarterly['final_capital']:,.0f}",
                    f"{semiannual['final_capital']:,.0f}",
                    f"{annual['final_capital']:,.0f}",
                ],
                "누적 수익률": [
                    f"{buy_and_hold['cumulative_return']:.2f}%",
                    f"{monthly['cumulative_return']:.2f}%",
                    f"{quarterly['cumulative_return']:.2f}%",
                    f"{semiannual['cumulative_return']:.2f}%",
                    f"{annual['cumulative_return']:.2f}%",
                ],
            }
        )

        print(results_df.to_string(index=False))

        # 시각화
        print("\n그래프 생성 중...")
        save_dir = ROOT_DIR / "results" / "01_TGNN_Only"
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir)

        # CSV 저장
        results_df.to_csv(save_dir / "comparison_results.csv", index=False)
        print(f"\n✅ 완료! 결과는 {save_dir}/ 에 저장되었습니다.")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
