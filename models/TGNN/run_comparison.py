"""
TGNN 학습 & 리밸런싱 빈도별 백테스팅 비교 (연평균 낙폭 추가)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from model import TGNNModel, TGNNDataset, train_model
from backtester import Backtester, BacktestConfig, create_metrics_summary_table

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
    }


def calculate_annual_drawdown(portfolio_values, dates):
    """
    연평균 낙폭 계산
    
    Args:
        portfolio_values: 포트폴리오 가치 배열
        dates: 날짜 배열 (pandas datetime)
    
    Returns:
        float: 연평균 낙폭 (%)
    """
    portfolio = np.array(portfolio_values)
    dates = pd.to_datetime(dates)
    
    # 연도별로 그룹화
    df = pd.DataFrame({
        'date': dates,
        'value': portfolio
    })
    df['year'] = df['date'].dt.year
    
    annual_drawdowns = []
    
    for year in df['year'].unique():
        year_data = df[df['year'] == year]['value'].values
        
        if len(year_data) < 2:
            continue
        
        # 해당 연도의 MDD 계산
        running_max = np.maximum.accumulate(year_data)
        drawdown = (year_data - running_max) / running_max * 100
        annual_mdd = abs(drawdown.min())
        
        annual_drawdowns.append(annual_mdd)
    
    # 연평균 낙폭
    return np.mean(annual_drawdowns) if annual_drawdowns else 0.0


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
    """Dynamic Universe를 위한 TGNN 리밸런싱"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[rebalance_freq]

    model.eval()
    initial_capital = 1000000
    capital = initial_capital

    portfolio_values = []
    dates = []
    current_weights = np.zeros(len(dataset.symbols))

    with torch.no_grad():
        for idx in range(len(dataset)):
            batch = dataset[idx]
            date = dataset.windows[idx]["date"]
            active_mask = batch["active_mask"].numpy()

            # 리밸런싱 시점
            if idx % interval == 0:
                features = batch["features"].unsqueeze(0)
                adj = batch["adj_matrix"].unsqueeze(0)

                predictions, _ = model(features, adj)
                pred_returns = predictions.squeeze(0).numpy()

                pred_returns[~active_mask] = -np.inf

                n_active = np.sum(active_mask)
                k = min(5, n_active)

                new_weights = np.zeros(len(dataset.symbols))
                if k > 0:
                    top_k_idx = np.argsort(pred_returns)[-k:]
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


# ============ 시각화 (4개 subplot 버전) ============

def plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir):
    """비교 그래프 생성 (연평균 낙폭 추가, 개선된 레이아웃)"""

    # 한글 폰트 재설정 (스타일 적용 시 덮어씌워질 수 있으므로)
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    
    # 레이아웃 설정 - 상단 누적수익률, 하단 3개 균등 배치
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.25)

    # ax1: 윗줄 전체 (누적 수익률)
    ax1 = fig.add_subplot(gs[0, :])
    # ax2, ax3, ax4: 아랫줄 균등 배치
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    strategies = {
        "1/N 매수 후 보유": buy_and_hold,
        "TGNN (월간)": monthly,
        "TGNN (분기)": quarterly,
        "TGNN (반기)": semiannual,
        "TGNN (연간)": annual,
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    labels = ["B&H", "월간", "분기", "반기", "연간"]

    # ==========================================
    # 1. 누적 수익률 그래프
    # ==========================================
    all_returns = []

    for (name, data), color in zip(strategies.items(), colors):
        dates = pd.to_datetime(data["dates"])
        initial_value = data["portfolio_values"][0]
        returns = [(v / initial_value - 1) * 100 for v in data["portfolio_values"]]
        all_returns.extend(returns)
        ax1.plot(dates, returns, label=name, linewidth=2.5, color=color)

    ax1.set_title(
        "리밸런싱 빈도별 누적 수익률 (2018-2025)",
        fontsize=16,
        fontweight="bold",
        pad=15,
    )
    ax1.set_ylabel("누적 수익률 (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("연도", fontsize=12, fontweight="bold")

    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())

    y_min, y_max = min(all_returns), max(all_returns)
    ax1.set_ylim(y_min - 10, y_max * 1.1)

    ax1.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9, ncol=5)
    ax1.grid(True, which="major", alpha=0.3, linestyle="--")
    ax1.axhline(y=0, color="black", linestyle="--", linewidth=1)

    # ==========================================
    # 2. CAGR 비교
    # ==========================================
    cagr_values = []

    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        dates_list = pd.to_datetime(data["dates"])
        days = (dates_list.max() - dates_list.min()).days
        years = days / 365.25
        cagr = (pow(data["final_capital"] / 1000000, 1 / years) - 1) * 100
        cagr_values.append(cagr)

    bars = ax2.bar(range(5), cagr_values, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax2.set_title("연평균 수익률 (CAGR)", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("수익률 (%)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(cagr_values) * 1.2)

    for bar, value in zip(bars, cagr_values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(cagr_values) * 0.02,
            f"{value:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    # ==========================================
    # 3. MDD 비교
    # ==========================================
    mdd_values = []
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        portfolio = np.array(data["portfolio_values"])
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max * 100
        mdd_values.append(abs(drawdown.min()))

    bars = ax3.bar(range(5), mdd_values, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax3.set_title("최대 낙폭 (MDD)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("낙폭 (%)", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, max(mdd_values) * 1.2)

    for bar, value in zip(bars, mdd_values):
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(mdd_values) * 0.02,
            f"-{value:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#D32F2F",
        )

    # ==========================================
    # 4. 연평균 낙폭 비교
    # ==========================================
    avg_dd_values = []
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        avg_dd = calculate_annual_drawdown(data["portfolio_values"], data["dates"])
        avg_dd_values.append(avg_dd)

    bars = ax4.bar(range(5), avg_dd_values, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax4.set_title("연평균 낙폭 (Avg DD)", fontsize=13, fontweight="bold", pad=10)
    ax4.set_ylabel("낙폭 (%)", fontsize=11)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(0, max(avg_dd_values) * 1.2)

    for bar, value in zip(bars, avg_dd_values):
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + max(avg_dd_values) * 0.02,
            f"-{value:.1f}%",
            ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1565C0",
        )

    # 레이아웃 마무리
    plt.tight_layout()

    save_path = save_dir / "rebalancing_comparison_with_avgdd.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.show()


# ============ 메인 실행 ============

def main(mode="compare"):
    """
    메인 실행 함수
    
    Args:
        mode: 'train' (학습만) 또는 'compare' (백테스팅 비교)
    
    기간 설정:
        - 학습 기간: 2015-01-01 ~ 2017-12-31
        - 테스트 기간: 2018-01-01 ~ 2025-12-31
    """
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

    TRAIN_START = "2015-01-01"
    TRAIN_END = "2017-12-31"
    TEST_START = "2018-01-01"
    TEST_END = "2025-12-31"

    model = TGNNModel(
        num_features=len(feature_cols),
        hidden_dims=[128, 128, 64],
        num_heads=8,
        num_stocks=10,
    )

    model_path = Path(__file__).parent / "best_tgnn.pth"

    if mode == "train":
        print("=" * 60)
        print("TGNN 모델 학습 시작")
        print(f"학습 기간: {TRAIN_START} ~ {TRAIN_END}")
        print("=" * 60)

        train_dataset = TGNNDataset(
            df=df,
            window_size=12,
            feature_cols=feature_cols,
            start_date=TRAIN_START,
            end_date=TRAIN_END,
        )

        print(f"학습 데이터 윈도우 수: {len(train_dataset)}")

        train_size = int(len(train_dataset) * 0.8)
        val_size = len(train_dataset) - train_size

        train_data = torch.utils.data.Subset(train_dataset, range(train_size))
        val_data = torch.utils.data.Subset(
            train_dataset, range(train_size, train_size + val_size)
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
        if not model_path.exists():
            print("❌ 모델 파일이 없습니다. 먼저 학습하세요:")
            print("   python run_comparison.py train")
            return

        model.load_state_dict(torch.load(model_path))

        test_dataset = TGNNDataset(
            df=df,
            window_size=12,
            feature_cols=feature_cols,
            start_date=TEST_START,
            end_date=TEST_END,
        )

        print("=" * 60)
        print("리밸런싱 빈도별 백테스팅 비교")
        print(f"테스트 기간: {TEST_START} ~ {TEST_END}")
        print(f"테스트 데이터 윈도우 수: {len(test_dataset)}")
        print("=" * 60)

        print("\n[1/5] 1/N Buy & Hold...")
        buy_and_hold = run_buy_and_hold(test_dataset)

        print("[2/5] TGNN 월간 리밸런싱...")
        monthly = run_tgnn_rebalancing(model, test_dataset, "monthly")

        print("[3/5] TGNN 분기 리밸런싱...")
        quarterly = run_tgnn_rebalancing(model, test_dataset, "quarterly")

        print("[4/5] TGNN 반기 리밸런싱...")
        semiannual = run_tgnn_rebalancing(model, test_dataset, "semiannual")

        print("[5/5] TGNN 연간 리밸런싱...")
        annual = run_tgnn_rebalancing(model, test_dataset, "annual")

        # 결과 요약 (연평균 낙폭 추가)
        print("\n" + "=" * 60)
        print("결과 요약")
        print("=" * 60)

        strategies_data = [buy_and_hold, monthly, quarterly, semiannual, annual]
        strategy_names = [
            "1/N Buy & Hold",
            "TGNN (월간)",
            "TGNN (분기)",
            "TGNN (반기)",
            "TGNN (연간)",
        ]

        # CAGR 계산
        cagr_list = []
        for data in strategies_data:
            dates_list = pd.to_datetime(data["dates"])
            days = (dates_list.max() - dates_list.min()).days
            years = days / 365.25
            cagr = (pow(data["final_capital"] / 1000000, 1 / years) - 1) * 100
            cagr_list.append(f"{cagr:.2f}%")

        # MDD 계산
        mdd_list = []
        for data in strategies_data:
            portfolio = np.array(data["portfolio_values"])
            running_max = np.maximum.accumulate(portfolio)
            drawdown = (portfolio - running_max) / running_max * 100
            mdd = abs(drawdown.min())
            mdd_list.append(f"-{mdd:.2f}%")

        # 연평균 낙폭 계산
        avg_dd_list = []
        for data in strategies_data:
            avg_dd = calculate_annual_drawdown(data["portfolio_values"], data["dates"])
            avg_dd_list.append(f"-{avg_dd:.2f}%")

        results_df = pd.DataFrame(
            {
                "전략": strategy_names,
                "최종 자산 (원)": [
                    f"{data['final_capital']:,.0f}" for data in strategies_data
                ],
                "누적 수익률": [
                    f"{data['cumulative_return']:.2f}%" for data in strategies_data
                ],
                "CAGR": cagr_list,
                "MDD": mdd_list,
                "연평균 낙폭": avg_dd_list,
            }
        )

        print(results_df.to_string(index=False))

        # 시각화
        print("\n그래프 생성 중...")
        save_dir = ROOT_DIR / "results" / "01_TGNN_Only"
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir)

        # CSV 저장
        results_df.to_csv(save_dir / "comparison_results_with_avgdd.csv", index=False)
        
        # ========== 상세 백테스팅 (Backtester 사용) ==========
        print("\n" + "=" * 60)
        print("상세 백테스팅 분석 진행 중...")
        print("=" * 60)
        
        config = BacktestConfig(initial_capital=1000000, cost_bps=5.0, risk_free_rate=0.03)
        all_metrics = {}
        
        # Buy & Hold
        print("[1/5] Buy & Hold 상세 분석...")
        bt_buyhold = Backtester(model, test_dataset, config, "Buy_Hold")
        bt_buyhold.run_buy_and_hold()
        bt_buyhold.save_timeseries_csv(save_dir / "timeseries_buyhold.csv")
        all_metrics["Buy_Hold"] = bt_buyhold.metrics
        
        # TGNN 월간
        print("[2/5] TGNN 월간 상세 분석...")
        bt_monthly = Backtester(model, test_dataset, config, "TGNN_Monthly")
        bt_monthly.run("monthly")
        bt_monthly.save_timeseries_csv(save_dir / "timeseries_monthly.csv")
        all_metrics["TGNN_Monthly"] = bt_monthly.metrics
        
        # TGNN 분기
        print("[3/5] TGNN 분기 상세 분석...")
        bt_quarterly = Backtester(model, test_dataset, config, "TGNN_Quarterly")
        bt_quarterly.run("quarterly")
        bt_quarterly.save_timeseries_csv(save_dir / "timeseries_quarterly.csv")
        all_metrics["TGNN_Quarterly"] = bt_quarterly.metrics
        
        # TGNN 반기
        print("[4/5] TGNN 반기 상세 분석...")
        bt_semiannual = Backtester(model, test_dataset, config, "TGNN_Semiannual")
        bt_semiannual.run("semiannual")
        bt_semiannual.save_timeseries_csv(save_dir / "timeseries_semiannual.csv")
        all_metrics["TGNN_Semiannual"] = bt_semiannual.metrics
        
        # TGNN 연간
        print("[5/5] TGNN 연간 상세 분석...")
        bt_annual = Backtester(model, test_dataset, config, "TGNN_Annual")
        bt_annual.run("annual")
        bt_annual.save_timeseries_csv(save_dir / "timeseries_annual.csv")
        all_metrics["TGNN_Annual"] = bt_annual.metrics
        
        # 집계 지표 JSON 저장
        bt_monthly.save_metrics_json(save_dir / "metrics_summary.json", all_metrics)
        
        # 상세 지표 테이블 출력
        print("\n" + "=" * 60)
        print("상세 지표 요약")
        print("=" * 60)
        metrics_table = create_metrics_summary_table(all_metrics)
        print(metrics_table.to_string(index=False))
        
        # 상세 지표 CSV 저장
        metrics_table.to_csv(save_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
        
        print(f"\n✅ 완료! 결과는 {save_dir}/ 에 저장되었습니다.")
        print("저장된 파일:")
        print("  - comparison_results_with_avgdd.csv (요약)")
        print("  - timeseries_*.csv (시계열 데이터)")
        print("  - metrics_summary.json (집계 지표)")
        print("  - metrics_comparison.csv (지표 비교 테이블)")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
