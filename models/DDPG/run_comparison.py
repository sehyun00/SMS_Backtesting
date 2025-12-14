"""
DDPG 학습 & 리밸런싱 빈도별 백테스팅 비교
- 3년 학습 (2015~2017) / 8년 실전 투자 테스트 (2018~2025) 분리 적용
- 학습 데이터(2015-2017) 기준으로 정규화하여 Look-ahead Bias 방지
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

from model import DDPGAgent

# 프로젝트 루트
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_PATH = (
    ROOT_DIR / "data" / "processed_daily_5factor_model_10stocks_10years_20251127.csv"
)

# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============ DDPG 전용 데이터셋 ============


class DDPGDataset:
    """
    DDPG 전용 데이터셋
    - 2015-2017: 학습 데이터 (Train)
    - 2018-2025: 테스트 데이터 (Test)
    """

    def __init__(self, df, window_size=12, feature_cols=None):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.symbols = sorted(df["Symbol"].unique())
        self.feature_cols = feature_cols

        # 1. 월별 리샘플링
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )

        # 2. 원본 수익률 보존
        self.monthly_df["Return_Raw"] = self.monthly_df["Momentum1M"].copy()

        # 3. 학습/테스트 분할 시점 설정 (2017년 12월 31일 기준)
        self.split_date = pd.Timestamp("2017-12-31")

        # 4. 학습 데이터(2015-2017) 기준으로 정규화
        self._fit_scaler_on_train_data()

        self.windows = self._create_windows()

        # 5. 분할 인덱스 찾기
        self.test_start_idx = 0
        for i, w in enumerate(self.windows):
            if w["date"] > self.split_date:
                self.test_start_idx = i
                break

        print(f"   📊 전체: {len(self.windows)}개월")
        print(f"   📈 학습: {self.test_start_idx}개월 (~2017)")
        print(f"   📉 테스트: {len(self.windows) - self.test_start_idx}개월 (2018~)")

    def _fit_scaler_on_train_data(self):
        """2017년까지 데이터로만 Scaler fit"""
        train_data = self.monthly_df[self.monthly_df["Date"] <= self.split_date]

        self.scaler = StandardScaler()
        self.scaler.fit(train_data[self.feature_cols].values)

        self.monthly_df[self.feature_cols] = self.scaler.transform(
            self.monthly_df[self.feature_cols].values
        )
        print(f"   ✅ 정규화 완료 (2017년 이전 데이터 기준)")

    def _create_windows(self):
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]
            next_date = dates[i + self.window_size]

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

            labels = []
            for symbol in self.symbols:
                val = next_df[next_df["Symbol"] == symbol]["Return_Raw"].values
                labels.append(val[0] if len(val) > 0 else 0.0)

            windows.append(
                {
                    "features": np.array(features),
                    "labels": np.array(labels),
                    "date": target_date,
                    "active_mask": np.array(active_mask),
                }
            )

        return windows

    def get_state(self, idx):
        w = self.windows[idx]
        state = w["features"][:, -1, :].flatten()
        return state.astype(np.float32)

    def get_train_windows(self):
        """2017년까지 데이터 반환"""
        return self.windows[: self.test_start_idx]

    def get_test_windows(self):
        """2018년 이후 데이터 반환 (실전 투자용)"""
        return self.windows[self.test_start_idx :]

    def __len__(self):
        return len(self.windows)


# ============ 포트폴리오 환경 ============


class PortfolioEnv:
    def __init__(self, dataset, windows=None, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows if windows else dataset.windows
        self.initial_cash = initial_cash
        self.symbols = dataset.symbols
        self.features = dataset.feature_cols
        self.n_steps = len(self.windows)
        self.current_step = 0
        self.portfolio_value = initial_cash

        # for Turnover Calculation
        self.n_stocks = len(self.symbols)
        self.prev_weights = np.zeros(self.n_stocks)

        # Academic Parameters
        self.gamma = 2.0  # Risk Aversion Coefficient (2.0: Moderate)
        self.cost_bps = 0.0005  # 5bps (0.05%) Transaction Cost - Encourage more trading

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)  # 초기화
        return self._get_state(0)

    def _get_state(self, idx):
        w = self.windows[idx]
        state = w["features"][:, -1, :].flatten()
        return state.astype(np.float32)

    def step(self, action):
        window = self.windows[self.current_step]
        returns = window["labels"]  # Percentage return (e.g., 5.0 for 5%)

        # 1. 포트폴리오 수익률 (Gross Return)
        portfolio_return_pct = np.dot(action, returns)
        portfolio_return = portfolio_return_pct / 100.0  # Decimal로 변환

        # 2. 거래비용 (Transaction Cost)
        # Turnover: sum(|w_t - w_{t-1}|)
        # 첫 Step은 이전 비중 0이므로 Turnover 계산에서 제외하거나 포함할 수 있음.
        # 여기서는 첫 진입 비용도 고려함.
        turnover = np.sum(np.abs(action - self.prev_weights))
        transaction_cost = turnover * self.cost_bps

        # 3. 순수익률 (Net Return)
        net_return = portfolio_return - transaction_cost

        # 4. 포트폴리오 가치 업데이트
        self.portfolio_value *= 1 + net_return

        self.current_step += 1
        done = self.current_step >= self.n_steps

        # ============ 보상 함수 (Theoretic CRRA Utility) ============
        # U(R) = ((1 + R)^(1 - gamma)) / (1 - gamma)
        # R이 -1(전액 손실)에 가까우면 -inf로 발산하여 매우 강력한 페널티

        # 안전장치: R이 -1보다 작으면 계산 불가하므로 클리핑
        safe_return = max(net_return, -0.99)

        exponent = 1.0 - self.gamma
        utility = ((1.0 + safe_return) ** exponent) / exponent

        reward = utility

        # =========================================================

        # 다음 스텝을 위해 가중치 저장
        self.prev_weights = action

        next_state = (
            self._get_state(self.current_step)
            if not done
            else np.zeros(len(self.symbols) * len(self.features))
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "date": window["date"],
            "turnover": turnover,
            "cost": transaction_cost,
        }

        return next_state, reward, done, info


# ============ 백테스팅 함수들 ============


# ============ 지표 계산 헬퍼 ============


def calculate_metrics(returns_dict, dates, strategy_name):
    """수익률 및 위험 지표 계산"""
    df = pd.DataFrame(returns_dict)
    df["date"] = pd.to_datetime(dates)
    df.set_index("date", inplace=True)

    # 일별/월별 수익률 (여기서는 월별 데이터임)
    r = df["return"] / 100.0

    # 1. CAGR
    days = (df.index.max() - df.index.min()).days
    years = days / 365.25
    total_return = (df["portfolio_value"].iloc[-1] / df["portfolio_value"].iloc[0]) - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # 2. MDD
    peak = df["portfolio_value"].cummax()
    drawdown = (df["portfolio_value"] - peak) / peak
    mdd = abs(drawdown.min())

    # 3. Volatility (Annualized)
    volatility = r.std() * np.sqrt(12)

    # 4. Sharpe Ratio (Risk Free = 0 가정)
    sharpe = cagr / (volatility + 1e-8)

    # 5. Sortino Ratio
    downside_std = r[r < 0].std() * np.sqrt(12)
    sortino = cagr / (downside_std + 1e-8)

    # 6. Turnover (Average Annual)
    avg_turnover = df["turnover"].mean() * 12  # 월간 턴오버 -> 연간 환산

    return {
        "Strategy": strategy_name,
        "CAGR": cagr * 100,
        "MDD": mdd * 100,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "Volatility": volatility * 100,
        "Avg_Turnover": avg_turnover,
        "Final_Value": df["portfolio_value"].iloc[-1],
    }


def run_buy_and_hold(dataset):
    """1/N 매수 후 보유 (테스트 기간만)"""
    initial_capital = 1_000_000
    n_stocks = len(dataset.symbols)
    weights = np.ones(n_stocks) / n_stocks

    portfolio_values = [initial_capital]
    dates = []
    trade_logs = []

    # 시계열 데이터 저장용
    ts_data = {"return": [], "portfolio_value": [], "drawdown": [], "turnover": []}

    # 테스트 기간만 사용
    test_windows = dataset.get_test_windows()

    if len(test_windows) > 0:
        first_date = test_windows[0]["date"]
        log_entry = {
            "Date": first_date,
            "Strategy": "1/N Buy & Hold",
            "Type": "Initial Buy",
        }
        for sym, w in zip(dataset.symbols, weights):
            log_entry[sym] = w
        trade_logs.append(log_entry)

    for i, window in enumerate(test_windows):
        actual_returns = window["labels"]
        portfolio_return = np.dot(weights, actual_returns)
        new_value = portfolio_values[-1] * (1 + portfolio_return / 100)

        # Drawdown 계산
        current_peak = (
            max(portfolio_values) if len(portfolio_values) > 0 else initial_capital
        )
        current_peak = max(current_peak, new_value)
        dd = (new_value - current_peak) / current_peak

        portfolio_values.append(new_value)
        dates.append(window["date"])

        ts_data["return"].append(portfolio_return)
        ts_data["portfolio_value"].append(new_value)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0.0)  # Buy & Hold는 턴오버 0

    metrics = calculate_metrics(ts_data, dates, "1/N Buy & Hold")

    return {
        "dates": dates,
        "portfolio_values": portfolio_values[1:],  # 첫 초기값 제외
        "final_capital": portfolio_values[-1],
        "cumulative_return": (portfolio_values[-1] / initial_capital - 1) * 100,
        "trade_logs": trade_logs,
        "ts_data": ts_data,
        "metrics": metrics,
    }


def run_ddpg_rebalancing(agent, dataset, rebalance_freq="monthly"):
    """DDPG 리밸런싱 (테스트 기간 2018~2025)"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[rebalance_freq]
    strategy_name = f"DDPG ({rebalance_freq})"

    initial_capital = 1_000_000
    capital = initial_capital

    portfolio_values = []
    dates = []
    trade_logs = []
    n_stocks = len(dataset.symbols)
    current_weights = np.ones(n_stocks) / n_stocks
    prev_weights = np.zeros(n_stocks)  # 턴오버 계산용

    # 시계열 데이터 저장용
    ts_data = {"return": [], "portfolio_value": [], "drawdown": [], "turnover": []}

    peak = initial_capital

    # 테스트 기간만 사용
    test_windows = dataset.get_test_windows()
    start_idx = dataset.test_start_idx

    for i, window in enumerate(test_windows):
        global_idx = start_idx + i
        turnover = 0.0

        # 리밸런싱
        if i % interval == 0:
            state = dataset.get_state(global_idx)
            action = agent.select_action(state, noise_std=0.0)

            # Turnover 계산 (|New - Old|)
            if i > 0:
                turnover = np.sum(np.abs(action - current_weights))

            current_weights = action

            # 로그 기록
            log_entry = {
                "Date": window["date"],
                "Strategy": strategy_name,
                "Type": "Rebalance",
            }
            for sym, w in zip(dataset.symbols, current_weights):
                log_entry[sym] = round(float(w), 4)
            trade_logs.append(log_entry)

        actual_returns = window["labels"]
        portfolio_return = np.dot(current_weights, actual_returns)
        capital *= 1 + portfolio_return / 100

        peak = max(peak, capital)
        dd = (capital - peak) / peak

        portfolio_values.append(capital)
        dates.append(window["date"])

        ts_data["return"].append(portfolio_return)
        ts_data["portfolio_value"].append(capital)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(turnover)

    metrics = calculate_metrics(ts_data, dates, strategy_name)

    return {
        "dates": dates,
        "portfolio_values": portfolio_values,
        "final_capital": capital,
        "cumulative_return": (capital / initial_capital - 1) * 100,
        "trade_logs": trade_logs,
        "ts_data": ts_data,
        "metrics": metrics,
    }


# ============ 학습 함수 ============


def train_ddpg(agent, env, num_episodes=100):
    """DDPG 학습 (2015-2017 데이터만)"""
    episode_rewards = []

    print(f"\n총 {num_episodes} 에피소드 학습 시작...")
    print(f"각 에피소드 = {env.n_steps}개월 거래 (2015~2017)\n")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        noise_std = max(0.01, 0.2 - episode * 0.002)

        while True:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > 256:
                agent.train(batch_size=64)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"[{episode + 1:3d}/{num_episodes}] Reward: {episode_reward:+8.2f} | Avg10: {avg_reward:+8.2f}"
            )

    return episode_rewards


# ============ 시각화 ============


def plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir):
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    strategies = {
        "1/N Buy & Hold": buy_and_hold,
        "DDPG (월간)": monthly,
        "DDPG (분기)": quarterly,
        "DDPG (반기)": semiannual,
        "DDPG (연간)": annual,
    }

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#6A994E"]

    # 1. 누적 수익률 그래프
    all_returns = []
    for (name, data), color in zip(strategies.items(), colors):
        dates = pd.to_datetime(data["dates"])
        initial_value = data["portfolio_values"][0]
        returns = [(v / initial_value - 1) * 100 for v in data["portfolio_values"]]
        all_returns.extend(returns)
        ax1.plot(dates, returns, label=name, linewidth=2.5, color=color)

    ax1.set_title(
        "실전 투자 수익률 비교 (2018~2025)", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_ylabel("누적 수익률 (%)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="major", alpha=0.3, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9)

    # 2. CAGR (연평균 수익률)
    cagr_values = []
    labels = ["Buy&Hold", "월간", "분기", "반기", "연간"]
    for data in [buy_and_hold, monthly, quarterly, semiannual, annual]:
        dates_list = pd.to_datetime(data["dates"])
        days = (dates_list.max() - dates_list.min()).days
        years = days / 365.25
        if years > 0:
            cagr = (pow(data["final_capital"] / 1_000_000, 1 / years) - 1) * 100
        else:
            cagr = 0
        cagr_values.append(cagr)

    bars = ax2.bar(
        range(5), cagr_values, color=colors, alpha=0.85, edgecolor="black", width=0.6
    )
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(labels, fontsize=10)
    ax2.set_title("연평균 수익률 (CAGR)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("수익률 (%)", fontsize=11)
    ax2.grid(axis="y", alpha=0.3)

    # CAGR 텍스트 표시
    for bar, value in zip(bars, cagr_values):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 3. MDD (최대 낙폭)
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
    ax3.invert_yaxis()  # MDD는 아래로 향하게

    # MDD 텍스트 표시
    for bar, value in zip(bars, mdd_values):
        height = bar.get_height()
        ax3.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"-{value:.1f}%",
            ha="center",
            va="top",
            fontsize=10,
            fontweight="bold",
            color="red",
        )

    plt.tight_layout()
    save_path = save_dir / "rebalancing_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 그래프 저장 완료: {save_path}")
    plt.show()


# ============ 메인 ============


def main(mode="compare"):
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

    print("=" * 60)
    print("DDPG 데이터셋 준비 (3년 학습 / 8년 테스트)")
    print("=" * 60)
    dataset = DDPGDataset(df=df, window_size=12, feature_cols=feature_cols)

    num_stocks = len(dataset.symbols)
    num_features = len(feature_cols)
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # 새로운 아키텍처 (Factor-Aware)에 맞는 인자 전달
    agent = DDPGAgent(
        num_stocks,
        num_features,
        lr_actor=1e-4,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        entropy_coef=0.01,
        device=DEVICE,
    )
    model_path = Path(__file__).parent / "best_ddpg.pth"

    # 모델 구조가 변경되었으므로 기존 모델 파일 삭제 (호환되지 않음)
    if mode == "train":
        if model_path.exists():
            print("⚠️ 기존 모델 삭제 (아키텍처 변경)")
            model_path.unlink()

        print("\n=== 학습 모드 (2015~2017 데이터) ===")
        train_windows = dataset.get_train_windows()
        train_env = PortfolioEnv(dataset, windows=train_windows)

        # 새 아키텍처로 학습 (Factor-Aware Network)
        # 팩터 간의 관계를 학습하여 더 '스마트한' 투자 유도
        train_ddpg(agent, train_env, num_episodes=100)
        torch.save(agent.actor.state_dict(), model_path)
        print(f"✅ 모델 저장 완료: {model_path}")

    elif mode == "compare":
        if model_path.exists():
            try:
                agent.actor.load_state_dict(torch.load(model_path))
            except RuntimeError:
                print("⚠️ 모델 구조 불일치. 기존 모델 삭제 후 재학습 시작...")
                model_path.unlink()
                train_windows = dataset.get_train_windows()
                train_env = PortfolioEnv(dataset, windows=train_windows)
                train_ddpg(agent, train_env, num_episodes=50)
                torch.save(agent.actor.state_dict(), model_path)
        else:
            print("⚠️ 학습된 모델 없음. 자동 학습 시작...")
            train_windows = dataset.get_train_windows()
            train_env = PortfolioEnv(dataset, windows=train_windows)
            train_ddpg(agent, train_env, num_episodes=50)
            torch.save(agent.actor.state_dict(), model_path)

        print("\n=== 실전 백테스팅 (2018~2025 데이터) ===")

        buy_and_hold = run_buy_and_hold(dataset)
        monthly = run_ddpg_rebalancing(agent, dataset, "monthly")
        quarterly = run_ddpg_rebalancing(agent, dataset, "quarterly")
        semiannual = run_ddpg_rebalancing(agent, dataset, "semiannual")
        annual = run_ddpg_rebalancing(agent, dataset, "annual")

        save_dir = ROOT_DIR / "results" / "02_DDPG_Only"
        save_dir.mkdir(parents=True, exist_ok=True)

        plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir)

        # 로그 저장 (기존 trade_logs.csv)
        all_logs = []
        all_logs.extend(buy_and_hold["trade_logs"])
        all_logs.extend(monthly["trade_logs"])
        all_logs.extend(quarterly["trade_logs"])
        all_logs.extend(semiannual["trade_logs"])
        all_logs.extend(annual["trade_logs"])

        logs_df = pd.DataFrame(all_logs)
        if not logs_df.empty:
            cols = ["Date", "Strategy", "Type"] + sorted(dataset.symbols)
            logs_df = logs_df[cols]
            log_path = save_dir / "trade_logs.csv"
            logs_df.to_csv(log_path, index=False)
            print(f"✅ 투자 로그 저장 완료: {log_path}")

        # === 1. 시계열 데이터 저장 (New) ===
        all_ts = []
        for res in [buy_and_hold, monthly, quarterly, semiannual, annual]:
            strategy = res["metrics"]["Strategy"]
            dates = res["dates"]
            ts = res["ts_data"]

            temp_df = pd.DataFrame(ts)
            temp_df["Date"] = dates
            temp_df["Strategy"] = strategy
            all_ts.append(temp_df)

        final_ts_df = pd.concat(all_ts, ignore_index=True)
        ts_path = save_dir / "results_ddpg_timeseries.csv"
        # 컬럼 순서 정리
        cols_order = [
            "Date",
            "Strategy",
            "portfolio_value",
            "return",
            "drawdown",
            "turnover",
        ]
        final_ts_df = final_ts_df[cols_order]
        final_ts_df.to_csv(ts_path, index=False)
        print(f"✅ 시계열 데이터 저장 완료: {ts_path}")

        # === 2. 집계 지표 출력 및 저장 (New) ===
        summary_metrics = [
            buy_and_hold["metrics"],
            monthly["metrics"],
            quarterly["metrics"],
            semiannual["metrics"],
            annual["metrics"],
        ]
        summary_df = pd.DataFrame(summary_metrics)
        # 컬럼 포맷팅을 위해 순서 조정
        summary_cols = [
            "Strategy",
            "Final_Value",
            "CAGR",
            "MDD",
            "Sharpe",
            "Sortino",
            "Volatility",
            "Avg_Turnover",
        ]
        summary_df = summary_df[summary_cols]

        json_path = save_dir / "results_summary.json"
        summary_df.to_json(json_path, orient="records", indent=4)
        print(f"✅ 집계 지표 저장 완료: {json_path}")

        print("\n" + "=" * 80)
        print(
            f"{'Strategy':<20} | {'CAGR':>8} | {'MDD':>8} | {'Sharpe':>8} | {'Sortino':>8} | {'Turnover':>8}"
        )
        print("-" * 80)
        for _, row in summary_df.iterrows():
            print(
                f"{row['Strategy']:<20} | {row['CAGR']:>7.1f}% | {row['MDD']:>7.1f}% | {row['Sharpe']:>8.2f} | {row['Sortino']:>8.2f} | {row['Avg_Turnover']:>8.2f}"
            )
        print("=" * 80 + "\n")

        print(f"✅ 결과 저장 완료: {save_dir}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
