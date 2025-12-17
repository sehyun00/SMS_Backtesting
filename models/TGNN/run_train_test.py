"""
train_data.csv로 학습하고 test_data.csv로 백테스팅
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
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"

# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


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
    """연평균 낙폭 계산"""
    portfolio = np.array(portfolio_values)
    dates = pd.to_datetime(dates)
    
    df = pd.DataFrame({'date': dates, 'value': portfolio})
    df['year'] = df['date'].dt.year
    
    annual_drawdowns = []
    for year in df['year'].unique():
        year_data = df[df['year'] == year]['value'].values
        if len(year_data) < 2:
            continue
        
        running_max = np.maximum.accumulate(year_data)
        drawdown = (year_data - running_max) / running_max * 100
        annual_mdd = abs(drawdown.min())
        annual_drawdowns.append(annual_mdd)
    
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
    """TGNN 리밸런싱 백테스팅"""
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


def plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir, test_period):
    """비교 그래프 생성"""
    plt.rcParams["font.family"] = "Malgun Gothic"
    plt.rcParams["axes.unicode_minus"] = False
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.5, 1], hspace=0.3, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[1, 2])

    strategies = {
        "1/N Buy & Hold": buy_and_hold,
        "TGNN (Monthly)": monthly,
        "TGNN (Quarterly)": quarterly,
        "TGNN (Semiannual)": semiannual,
        "TGNN (Annual)": annual,
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]
    labels = ["B&H", "Monthly", "Quarterly", "Semiannual", "Annual"]

    
    # ==========================================
    # 1. 누적 수익률 그래프 (절댓값 사용)
    # ==========================================
    all_returns = []

    for (name, data), color in zip(strategies.items(), colors):
        dates = pd.to_datetime(data["dates"])
        initial_value = data["portfolio_values"][0]
        
        returns = [abs((v / initial_value - 1) * 100) for v in data["portfolio_values"]]
        
        all_returns.extend(returns)
        ax1.plot(dates, returns, label=name, linewidth=2.5, color=color)

    ax1.set_title(f"Rebalancing Frequency Comparison ({test_period})", fontsize=16, fontweight="bold", pad=15)
    ax1.set_ylabel("Cumulative Return (%)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Year", fontsize=12, fontweight="bold")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.xaxis.set_major_locator(mdates.YearLocator())
    
    ax1.set_ylim(0, max(all_returns) * 1.1)
    
    ax1.legend(loc="upper left", fontsize=10, frameon=True, framealpha=0.9, ncol=5)
    ax1.grid(True, which="major", alpha=0.3, linestyle="--")
    
    # 2. CAGR
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
    ax2.set_title("CAGR", fontsize=13, fontweight="bold", pad=10)
    ax2.set_ylabel("Return (%)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(cagr_values) * 1.2)

    for bar, value in zip(bars, cagr_values):
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(cagr_values) * 0.02,
                 f"{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # 3. MDD
    mdd_values = []
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        portfolio = np.array(data["portfolio_values"])
        running_max = np.maximum.accumulate(portfolio)
        drawdown = (portfolio - running_max) / running_max * 100
        mdd_values.append(abs(drawdown.min()))

    bars = ax3.bar(range(5), mdd_values, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    ax3.set_xticks(range(5))
    ax3.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax3.set_title("Maximum Drawdown (MDD)", fontsize=13, fontweight="bold", pad=10)
    ax3.set_ylabel("Drawdown (%)", fontsize=11)
    ax3.grid(axis="y", alpha=0.3)
    ax3.set_ylim(0, max(mdd_values) * 1.2)

    for bar, value in zip(bars, mdd_values):
        ax3.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(mdd_values) * 0.02,
                 f"-{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#D32F2F")

    # 4. 연평균 낙폭
    avg_dd_values = []
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        avg_dd = calculate_annual_drawdown(data["portfolio_values"], data["dates"])
        avg_dd_values.append(avg_dd)

    bars = ax4.bar(range(5), avg_dd_values, color=colors, alpha=0.85, edgecolor="black", width=0.7)
    ax4.set_xticks(range(5))
    ax4.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax4.set_title("Average Annual Drawdown", fontsize=13, fontweight="bold", pad=10)
    ax4.set_ylabel("Drawdown (%)", fontsize=11)
    ax4.grid(axis="y", alpha=0.3)
    ax4.set_ylim(0, max(avg_dd_values) * 1.2)

    for bar, value in zip(bars, avg_dd_values):
        ax4.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + max(avg_dd_values) * 0.02,
                 f"-{value:.1f}%", ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1565C0")

    plt.tight_layout()
    save_path = save_dir / "rebalancing_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor='white')
    print(f"✅ 그래프 저장: {save_path}")
    plt.show()

def main(mode="train"):
    """
    메인 실행 함수
    
    Args:
        mode: 'train' (학습), 'test' (테스트), 'all' (학습+테스트)
    """
    
    # 특성 컬럼 정의
    feature_cols = [
        "Momentum1M", "Momentum3M", "Momentum6M", "Momentum12M",
        "Volatility", "RSI", "MACD", "Signal", "MACD_Hist",
        "Beta_Factor", "Value_Factor", "Momentum_Factor", "Volatility_Factor",
        "weighted_score", "Mkt_RF", "SMB", "HML", "RMW", "CMA",
    ]
    
    model_path = Path(__file__).parent / "best_tgnn_sector.pth"
    
    MODEL_CONFIG = {
        "num_features": len(feature_cols),
        "hidden_dims": [128, 128, 64],
        "num_heads": 8,
    }
    
    # ========== 학습 모드 ==========
    if mode in ["train", "all"]:
        print("=" * 60)
        print("TGNN 모델 학습 시작 (train_data.csv)")
        print("=" * 60)
        
        # 학습 데이터 로드
        train_df = pd.read_csv(TRAIN_DATA_PATH)
    
        # 데이터 정제
    print("\n🔧 데이터 정제 중...")
    train_df = train_df.replace([np.inf, -np.inf], np.nan)
    
    for col in feature_cols:
        if col in train_df.columns:
            train_df[col] = train_df[col].fillna(0)
            mean = train_df[col].mean()
            std = train_df[col].std()
            if std > 0:
                train_df[col] = train_df[col].clip(mean - 10*std, mean + 10*std)
    
    train_df['Momentum1M'] = train_df['Momentum1M'].fillna(0).clip(-1.0, 1.0)
        
    print(f"✅ 데이터 정제 완료")
    print(f"   총 행 수: {len(train_df)}")
    print(f"   기간: {train_df['Date'].min()} ~ {train_df['Date'].max()}")
    print(f"   종목 수: {train_df['Symbol'].nunique()}")
        
    # 데이터 통계 출력
    print(f"\n📊 Feature 통계:")
    for col in feature_cols[:5]:  # 처음 5개만 출력
        if col in train_df.columns:
            print(f"   {col}: mean={train_df[col].mean():.4f}, std={train_df[col].std():.4f}")
        
    # 학습 데이터셋 생성
    train_dataset = TGNNDataset(
        df=train_df,
        window_size=6,
        feature_cols=feature_cols,
    )
        
    print(f"\n생성된 윈도우 수: {len(train_dataset)}")
        
        # ✅ 데이터셋 샘플 확인
    if len(train_dataset) > 0:
        sample = train_dataset[0]
        print(f"   샘플 features shape: {sample['features'].shape}")
        print(f"   샘플 labels shape: {sample['labels'].shape}")
        print(f"   샘플 labels 범위: [{sample['labels'].min():.2f}, {sample['labels'].max():.2f}]")
        
    # Train/Val 분할 (80/20)
    train_size = int(len(train_dataset) * 0.8)
    val_size = len(train_dataset) - train_size
    
    train_data = torch.utils.data.Subset(train_dataset, range(train_size))
    val_data = torch.utils.data.Subset(train_dataset, range(train_size, train_size + val_size))
        
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=custom_collate)  # ✅ batch_size 감소
    val_loader = DataLoader(val_data, batch_size=16, collate_fn=custom_collate)
        
    # 모델 생성
    num_stocks = train_df['Symbol'].nunique()
    model = TGNNModel(
        num_features=len(feature_cols),
        hidden_dims=[64, 64, 32], 
        num_heads=4,  
        num_stocks=num_stocks,
    )
        
    print(f"\n🤖 모델 구조:")
    print(f"   입력 특성 수: {len(feature_cols)}")
    print(f"   히든 차원: [64, 64, 32]")
    print(f"   어텐션 헤드: 4")
    print(f"   종목 수: {num_stocks}")
        
    # 학습 실행
    train_model(model, train_loader, val_loader, num_epochs=300, lr=1e-5, save_path=str(model_path))
    print(f"\n✅ 학습 완료! 모델 저장: {model_path}")

    print("\n" + "=" * 60)
    print("학습 데이터 예측 정확도 테스트")
    print("=" * 60)
        
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
        
    all_predictions = []
    all_actuals = []
        
    with torch.no_grad():
        for idx in range(min(50, len(train_dataset))):  # 처음 50개 윈도우
            batch = train_dataset[idx]
            features = batch["features"].unsqueeze(0).to(device)
            adj = batch["adj_matrix"].unsqueeze(0).to(device)
                
            predictions, _ = model(features, adj)
            pred = predictions.squeeze(0).cpu().numpy()
            actual = batch["labels"].numpy()
                
            all_predictions.extend(pred)
            all_actuals.extend(actual)
        
    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)
        
    # 상관계수 계산
    correlation = np.corrcoef(all_predictions, all_actuals)[0, 1]
        
        # MSE 계산
    mse = np.mean((all_predictions - all_actuals) ** 2)
        
        # MAE 계산
    mae = np.mean(np.abs(all_predictions - all_actuals))
        
    print(f"\n📊 예측 정확도:")
    print(f"   상관계수 (Correlation): {correlation:.4f}")
    print(f"   MSE: {mse:.6f}")
    print(f"   MAE: {mae:.6f}")
    print(f"\n   예측값 범위: [{all_predictions.min():.4f}, {all_predictions.max():.4f}]")
    print(f"   실제값 범위: [{all_actuals.min():.4f}, {all_actuals.max():.4f}]")
        
    # 방향성 정확도 (부호가 일치하는 비율)
    direction_correct = np.mean(np.sign(all_predictions) == np.sign(all_actuals))
    print(f"   방향성 정확도: {direction_correct:.2%}")
    

    # ========== 테스트 모드 ==========
    if mode in ["test", "all"]:
        print("\n" + "=" * 60)
        print("TGNN 모델 테스트 시작 (test_data.csv)")
        print("=" * 60)
        
        # 모델 존재 확인
        if not model_path.exists():
            print(f"❌ 모델 파일이 없습니다: {model_path}")
            print("   먼저 학습을 실행하세요: python run_train_test.py train")
            return
        
        # 테스트 데이터 로드
        test_df = pd.read_csv(TEST_DATA_PATH)
        
        # ✅ 테스트 데이터도 정제
        print("\n🔧 데이터 정제 중...")
        test_df = test_df.replace([np.inf, -np.inf], np.nan)
        
        for col in feature_cols:
            if col in test_df.columns:
                test_df[col] = test_df[col].fillna(0)
                mean = test_df[col].mean()
                std = test_df[col].std()
                if std > 0:
                    lower = mean - 5 * std
                    upper = mean + 5 * std
                    test_df[col] = test_df[col].clip(lower, upper)
        
        if 'Momentum1M' in test_df.columns:
            test_df['Momentum1M'] = test_df['Momentum1M'].fillna(0)
            test_df['Momentum1M'] = test_df['Momentum1M'].clip(-50, 50)
        
        print(f"✅ 데이터 정제 완료")
        print(f"테스트 데이터 로드 완료: {len(test_df)}행")
        print(f"기간: {test_df['Date'].min()} ~ {test_df['Date'].max()}")
        print(f"종목 수: {test_df['Symbol'].nunique()}")
        
        # 테스트 데이터셋 생성
        test_dataset = TGNNDataset(
            df=test_df,
            window_size=12,
            feature_cols=feature_cols,
        )
        
        print(f"생성된 윈도우 수: {len(test_dataset)}")
        
        num_stocks = test_df['Symbol'].nunique()
        model = TGNNModel(
            num_features=MODEL_CONFIG["num_features"],    
            hidden_dims=MODEL_CONFIG["hidden_dims"],      
            num_heads=MODEL_CONFIG["num_heads"],         
            num_stocks=num_stocks,
        )
        
        # 모델 로드
        model.load_state_dict(torch.load(model_path))
        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"   모델 구조: hidden_dims={MODEL_CONFIG['hidden_dims']}, num_heads={MODEL_CONFIG['num_heads']}")
        
        # 백테스팅 실행
        print("\n" + "=" * 60)
        print("리밸런싱 빈도별 백테스팅 비교")
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
        
        # 결과 요약
        print("\n" + "=" * 60)
        print("결과 요약")
        print("=" * 60)
        
        strategies_data = [buy_and_hold, monthly, quarterly, semiannual, annual]
        strategy_names = ["1/N Buy & Hold", "TGNN (월간)", "TGNN (분기)", "TGNN (반기)", "TGNN (연간)"]
        
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
        
        # 연평균 낙폭
        avg_dd_list = []
        for data in strategies_data:
            avg_dd = calculate_annual_drawdown(data["portfolio_values"], data["dates"])
            avg_dd_list.append(f"-{avg_dd:.2f}%")
        
        results_df = pd.DataFrame({
            "전략": strategy_names,
            "최종 자산 (원)": [f"{data['final_capital']:,.0f}" for data in strategies_data],
            "누적 수익률": [f"{data['cumulative_return']:.2f}%" for data in strategies_data],
            "CAGR": cagr_list,
            "MDD": mdd_list,
            "연평균 낙폭": avg_dd_list,
        })
        
        print(results_df.to_string(index=False))
        
        # 결과 저장
        save_dir = ROOT_DIR / "results" / "TGNN_Sector"
        save_dir.mkdir(parents=True, exist_ok=True)
        
        results_df.to_csv(save_dir / "comparison_results.csv", index=False, encoding="utf-8-sig")
        
        # 시각화
        test_period = f"{test_df['Date'].min()[:4]}-{test_df['Date'].max()[:4]}"
        plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir, test_period)
        
        # 상세 백테스팅
        print("\n" + "=" * 60)
        print("상세 백테스팅 분석")
        print("=" * 60)
        
        config = BacktestConfig(initial_capital=1000000, cost_bps=5.0, risk_free_rate=0.03)
        all_metrics = {}
        
        print("[1/5] Buy & Hold...")
        bt_buyhold = Backtester(model, test_dataset, config, "Buy_Hold")
        bt_buyhold.run_buy_and_hold()
        bt_buyhold.save_timeseries_csv(save_dir / "timeseries_buyhold.csv")
        all_metrics["Buy_Hold"] = bt_buyhold.metrics
        
        print("[2/5] TGNN 월간...")
        bt_monthly = Backtester(model, test_dataset, config, "TGNN_Monthly")
        bt_monthly.run("monthly")
        bt_monthly.save_timeseries_csv(save_dir / "timeseries_monthly.csv")
        all_metrics["TGNN_Monthly"] = bt_monthly.metrics
        
        print("[3/5] TGNN 분기...")
        bt_quarterly = Backtester(model, test_dataset, config, "TGNN_Quarterly")
        bt_quarterly.run("quarterly")
        bt_quarterly.save_timeseries_csv(save_dir / "timeseries_quarterly.csv")
        all_metrics["TGNN_Quarterly"] = bt_quarterly.metrics
        
        print("[4/5] TGNN 반기...")
        bt_semiannual = Backtester(model, test_dataset, config, "TGNN_Semiannual")
        bt_semiannual.run("semiannual")
        bt_semiannual.save_timeseries_csv(save_dir / "timeseries_semiannual.csv")
        all_metrics["TGNN_Semiannual"] = bt_semiannual.metrics
        
        print("[5/5] TGNN 연간...")
        bt_annual = Backtester(model, test_dataset, config, "TGNN_Annual")
        bt_annual.run("annual")
        bt_annual.save_timeseries_csv(save_dir / "timeseries_annual.csv")
        all_metrics["TGNN_Annual"] = bt_annual.metrics
        
        # 집계 지표 저장
        bt_monthly.save_metrics_json(save_dir / "metrics_summary.json", all_metrics)
        
        metrics_table = create_metrics_summary_table(all_metrics)
        print("\n" + "=" * 60)
        print("상세 지표 요약")
        print("=" * 60)
        print(metrics_table.to_string(index=False))
        
        metrics_table.to_csv(save_dir / "metrics_comparison.csv", index=False, encoding="utf-8-sig")
        
        print(f"\n✅ 완료! 결과 저장 위치: {save_dir}/")
        print("저장된 파일:")
        print("  - comparison_results.csv (요약)")
        print("  - timeseries_*.csv (시계열)")
        print("  - metrics_summary.json (집계 지표)")
        print("  - metrics_comparison.csv (지표 비교)")
        print("  - rebalancing_comparison.png (그래프)")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    main(mode=mode)
