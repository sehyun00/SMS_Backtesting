"""
DDPG 학습 & 리밸런싱 빈도별 백테스팅 비교
- train_data.csv / test_data.csv 사용 (Hybrid 모델과 동일)
- Hybrid TGNN-DDPG 방식 적용: Early Stopping + Fine-tuning
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

from model import DDPGAgent, ReplayBuffer

# 프로젝트 루트
ROOT_DIR = Path(__file__).parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"

# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============ DDPG 전용 데이터셋 ============


class DDPGDataset:
    def __init__(self, train_df, test_df, window_size=12, feature_cols=None):
        self.window_size = window_size
        self.feature_cols = feature_cols

        # ✅ 결측치 처리
        train_nan_count = train_df.isna().sum().sum()
        test_nan_count = test_df.isna().sum().sum()

        if train_nan_count > 0:
            print(
                f"⚠️ [DDPGDataset] Train Data NaN Found: {train_nan_count} -> Fill with 0"
            )
            train_df = train_df.fillna(0)

        if test_nan_count > 0:
            print(
                f"⚠️ [DDPGDataset] Test Data NaN Found: {test_nan_count} -> Fill with 0"
            )
            test_df = test_df.fillna(0)

        train_df["Date"] = pd.to_datetime(train_df["Date"])
        test_df["Date"] = pd.to_datetime(test_df["Date"])

        self.train_symbols = sorted(train_df["Symbol"].unique())
        self.test_symbols = sorted(test_df["Symbol"].unique())

        self.train_monthly = (
            train_df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )
        self.test_monthly = (
            test_df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )

        # ✅ Momentum1M을 레이블로만 사용 (정규화 제외)
        self.train_monthly["Return_Raw"] = self.train_monthly["Momentum1M"].copy()
        self.test_monthly["Return_Raw"] = self.test_monthly["Momentum1M"].copy()

        # ✅ 정규화할 특성 (Momentum1M 제외) - 이것이 실제 state 특성!
        self.norm_cols = [c for c in self.feature_cols if c != "Momentum1M"]

        # ✅ 학습 데이터 기준으로 정규화
        self._fit_scaler_on_train_data()

        # 윈도우 생성
        self.train_windows = self._create_windows(
            self.train_monthly, self.train_symbols
        )
        self.test_windows = self._create_windows(self.test_monthly, self.test_symbols)

        print(f"   📊 학습: {len(self.train_windows)}개월")
        print(f"   📉 테스트: {len(self.test_windows)}개월")

    def _fit_scaler_on_train_data(self):
        """학습 데이터로만 Scaler fit (Momentum1M 제외)"""
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_monthly[self.norm_cols].values)

        self.train_monthly[self.norm_cols] = self.scaler.transform(
            self.train_monthly[self.norm_cols].values
        )
        self.test_monthly[self.norm_cols] = self.scaler.transform(
            self.test_monthly[self.norm_cols].values
        )
        print(f"   ✅ 정규화 완료 (학습 데이터 기준)")

    def _create_windows(self, monthly_df, symbols):
        dates = sorted(monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
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
                    # ✅ 정규화된 특성만 사용 (Momentum1M 제외!)
                    vals = stock_data[self.norm_cols].values

                    if len(vals) < self.window_size:
                        pad = np.zeros(
                            (self.window_size - len(vals), len(self.norm_cols))
                        )
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    features.append(np.zeros((self.window_size, len(self.norm_cols))))
                    active_mask.append(False)

            # ✅ 레이블은 Return_Raw 사용
            labels = []
            for symbol in symbols:
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

    def get_state(self, windows, idx):
        w = windows[idx]
        state = w["features"][:, -1, :].flatten()
        return state.astype(np.float32)

    def get_train_windows(self):
        return self.train_windows

    def get_test_windows(self):
        return self.test_windows


# ============ 포트폴리오 환경 ============


class PortfolioEnv:
    def __init__(self, dataset, windows, symbols, features, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.features = features
        self.n_steps = len(self.windows)
        self.current_step = 0
        self.portfolio_value = initial_cash

        # for Turnover Calculation
        self.n_stocks = len(self.symbols)
        self.prev_weights = np.zeros(self.n_stocks)

        self.state_dim = self.windows[0]["features"][:, -1, :].flatten().shape[0]

        # Academic Parameters
        self.gamma = 2.0  # Risk Aversion Coefficient (2.0: Moderate)
        self.cost_bps = 0.0005  # 5bps (0.05%) Transaction Cost

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        return self._get_state(0)

    def _get_state(self, idx):
        w = self.windows[idx]
        state = w["features"][:, -1, :].flatten()
        return state.astype(np.float32)

    def step(self, action):
        window = self.windows[self.current_step]
        returns = window["labels"]

        # 1. 포트폴리오 수익률
        portfolio_return_pct = np.dot(action, returns)
        portfolio_return = portfolio_return_pct / 100.0

        # 2. 거래비용
        turnover = np.sum(np.abs(action - self.prev_weights))
        transaction_cost = turnover * self.cost_bps

        # 3. 순수익률
        net_return = portfolio_return - transaction_cost

        # 4. 포트폴리오 가치 업데이트
        self.portfolio_value *= 1 + net_return

        self.current_step += 1
        done = self.current_step >= self.n_steps

        # ✅ 새로운 보상 함수: 단순하고 직관적
        # 기본 보상: 월간 수익률 (%)
        base_reward = portfolio_return_pct

        # 거래 비용 페널티 (bps -> % 변환)
        cost_penalty = turnover * self.cost_bps * 10000  # 5bps = 0.5%

        # 최종 보상
        reward = base_reward - cost_penalty

        # 극단값 방지
        reward = np.clip(reward, -50, 50)

        self.prev_weights = action

        next_state = (
            self._get_state(self.current_step)
            if not done
            else np.zeros(self.state_dim, dtype=np.float32)
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "date": window["date"],
            "turnover": turnover,
            "cost": transaction_cost,
            "raw_return": portfolio_return_pct,  # 디버깅용
        }

        return next_state, reward, done, info


# ============ 백테스팅 함수들 ============


def calculate_metrics(returns_dict, dates, strategy_name):
    """수익률 및 위험 지표 계산"""
    df = pd.DataFrame(returns_dict)
    df["date"] = pd.to_datetime(dates)
    df.set_index("date", inplace=True)

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

    # 4. Sharpe Ratio
    sharpe = cagr / (volatility + 1e-8)

    # 5. Sortino Ratio
    downside_std = r[r < 0].std() * np.sqrt(12)
    sortino = cagr / (downside_std + 1e-8)

    # 6. Turnover (Average Annual)
    avg_turnover = df["turnover"].mean() * 12

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


def run_buy_and_hold(test_windows, test_symbols):
    """1/N 매수 후 보유 - 수정 버전"""
    initial_capital = 1_000_000
    n_stocks = len(test_symbols)
    weights = np.ones(n_stocks) / n_stocks

    portfolio_values = []  # ✅ 빈 리스트로 시작
    dates = []
    trade_logs = []

    ts_data = {"return": [], "portfolio_value": [], "drawdown": [], "turnover": []}

    # 초기 매수 로그
    if len(test_windows) > 0:
        first_date = test_windows[0]["date"]
        log_entry = {
            "Date": first_date,
            "Strategy": "1/N Buy & Hold",
            "Type": "Initial Buy",
        }
        for sym, w in zip(test_symbols, weights):
            log_entry[sym] = w
        trade_logs.append(log_entry)

    current_value = initial_capital  # ✅ 현재 포트폴리오 가치
    peak = initial_capital  # ✅ 누적 최고점 초기화

    for i, window in enumerate(test_windows):
        actual_returns = window["labels"]
        portfolio_return = np.dot(weights, actual_returns)

        # ✅ 포트폴리오 가치 업데이트
        current_value *= 1 + portfolio_return / 100

        # ✅ 누적 최고점 업데이트
        peak = max(peak, current_value)

        # ✅ Drawdown 계산 (peak 대비 하락률)
        dd = (current_value - peak) / peak

        portfolio_values.append(current_value)
        dates.append(window["date"])

        ts_data["return"].append(portfolio_return)
        ts_data["portfolio_value"].append(current_value)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0.0)  # Buy & Hold는 거래비용 없음

    metrics = calculate_metrics(ts_data, dates, "1/N Buy & Hold")

    return {
        "dates": dates,
        "portfolio_values": portfolio_values,  # ✅ 슬라이싱 제거
        "final_capital": current_value,  # ✅ 마지막 값 직접 사용
        "cumulative_return": (current_value / initial_capital - 1) * 100,
        "trade_logs": trade_logs,
        "ts_data": ts_data,
        "metrics": metrics,
    }


def run_ddpg_rebalancing(
    agent, dataset, test_windows, test_symbols, rebalance_freq="monthly"
):
    """DDPG 리밸런싱"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[rebalance_freq]
    strategy_name = f"DDPG ({rebalance_freq})"

    initial_capital = 1_000_000
    capital = initial_capital

    portfolio_values = []
    dates = []
    trade_logs = []
    n_stocks = len(test_symbols)
    current_weights = np.ones(n_stocks) / n_stocks

    ts_data = {"return": [], "portfolio_value": [], "drawdown": [], "turnover": []}

    peak = initial_capital

    for i, window in enumerate(test_windows):
        turnover = 0.0

        # 리밸런싱
        if i % interval == 0:
            state = dataset.get_state(test_windows, i)
            action = agent.select_action(state, noise_std=0.0)

            if i > 0:
                turnover = np.sum(np.abs(action - current_weights))

            current_weights = action

            log_entry = {
                "Date": window["date"],
                "Strategy": strategy_name,
                "Type": "Rebalance",
            }
            for sym, w in zip(test_symbols, current_weights):
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


# ============ 학습 함수 (Hybrid 방식 적용) ============


def train_ddpg(agent, env, num_episodes=800, patience=100, episode_length=24):
    """
    episode_length: 각 에피소드당 step 수 (기본 24개월 = 2년)
    """
    episode_rewards = []
    best_reward = -float("inf")
    no_improve_count = 0
    best_model_state = None

    noise_start = 0.3
    noise_end = 0.05
    noise_decay = (noise_start - noise_end) / num_episodes

    print(f"\n총 {num_episodes} 에피소드 학습 시작...")
    print(f"각 에피소드 = {episode_length}개월 거래")
    print(f"전체 학습 데이터: {env.n_steps}개월")
    print(f"Early Stopping: Patience={patience}\n")

    for episode in range(num_episodes):
        state = env.reset()  # ✅ 랜덤 시작점
        episode_reward = 0
        noise_std = max(noise_end, noise_start - episode * noise_decay)

        # ✅ episode_length만큼만 진행
        steps_taken = 0
        while steps_taken < episode_length and env.current_step < env.n_steps:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) > 256:
                agent.train(batch_size=64)

            episode_reward += reward
            state = next_state
            steps_taken += 1

            if done:
                break

        episode_rewards.append(episode_reward)

        # 최고 성능 모델 저장
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model_state = agent.actor.state_dict()
            no_improve_count = 0
        else:
            no_improve_count += 1

        # 로깅
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"[{episode + 1:3d}/{num_episodes}] "
                f"Reward: {episode_reward:+8.2f} | "
                f"Avg10: {avg_reward:+8.2f} | "
                f"Best: {best_reward:+8.2f} | "
                f"Noise: {noise_std:.3f} | "
                f"No Improve: {no_improve_count}"
            )

        # Early Stopping
        if no_improve_count >= patience:
            print(f"\n⚠️ Early Stopping at Episode {episode + 1}")
            print(f"   No improvement for {patience} episodes")
            break

    # 최고 성능 모델 로드
    if best_model_state is not None:
        agent.actor.load_state_dict(best_model_state)
        print(f"\n✅ Best model loaded (Reward: {best_reward:+.2f})")

    return episode_rewards, best_reward


def fine_tune_ddpg(agent, env, num_episodes=50):
    """
    Fine-tuning Phase
    - 작은 배치로 추가 학습
    - 낮은 노이즈로 exploitation 강화
    """
    print(f"\n=== Fine-tuning 시작 ({num_episodes} episodes) ===")

    episode_rewards = []
    noise_std = 0.1  # 낮은 탐색 노이즈

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Fine-tuning 전용 작은 배치
            if len(agent.replay_buffer) > 128:
                agent.train(batch_size=32)

            episode_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(
                f"[Fine-tune {episode + 1:2d}/{num_episodes}] "
                f"Reward: {episode_reward:+8.2f} | Avg: {avg_reward:+8.2f}"
            )

    print("✅ Fine-tuning 완료\n")
    return episode_rewards


def plot_training_curve(train_rewards, finetune_rewards, save_dir):
    """학습 진행 상황 시각화"""
    plt.figure(figsize=(12, 6))

    # Main Training
    plt.plot(
        range(1, len(train_rewards) + 1),
        train_rewards,
        label="Training",
        color="#2E86AB",
        alpha=0.6,
    )

    # Moving Average
    window = 20
    if len(train_rewards) >= window:
        ma = np.convolve(train_rewards, np.ones(window) / window, mode="valid")
        plt.plot(
            range(window, len(train_rewards) + 1),
            ma,
            label="MA(20)",
            color="#A23B72",
            linewidth=2,
        )

    # Fine-tuning
    if finetune_rewards:
        ft_start = len(train_rewards) + 1
        ft_x = range(ft_start, ft_start + len(finetune_rewards))
        plt.plot(
            ft_x, finetune_rewards, label="Fine-tuning", color="#F18F01", linewidth=2
        )

        plt.axvline(
            x=len(train_rewards),
            color="red",
            linestyle="--",
            label="Fine-tuning Start",
            alpha=0.7,
        )

    plt.xlabel("Episode", fontsize=12)
    plt.ylabel("Episode Reward", fontsize=12)
    plt.title(
        "DDPG Training Progress (with Early Stopping)", fontsize=14, fontweight="bold"
    )
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)

    save_path = save_dir / "training_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 학습 곡선 저장: {save_path}")
    plt.close()


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
    for (name, data), color in zip(strategies.items(), colors):
        dates = pd.to_datetime(data["dates"])
        initial_value = data["portfolio_values"][0]
        returns = [(v / initial_value - 1) * 100 for v in data["portfolio_values"]]
        ax1.plot(dates, returns, label=name, linewidth=2.5, color=color)

    ax1.set_title(
        "실전 투자 수익률 비교 (Test Period)", fontsize=16, fontweight="bold", pad=20
    )
    ax1.set_ylabel("누적 수익률 (%)", fontsize=12, fontweight="bold")
    ax1.grid(True, which="major", alpha=0.3, linestyle="--")
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax1.legend(loc="upper left", fontsize=11, frameon=True, framealpha=0.9)

    # 2. CAGR
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

    # 3. MDD
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
    ax3.invert_yaxis()

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
    plt.close()


# ============ 메인 ============


def main(mode="compare"):
    print("=" * 60)
    print("DDPG 데이터셋 준비 (train_data.csv / test_data.csv)")
    print("=" * 60)

    # 데이터 로드
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    # 특성 컬럼 정의
    feature_cols = [
        "Close",
        "Volume",
        "Momentum1M",
        "Momentum3M",
        "Momentum6M",
        "Momentum12M",
        "Volatility",
        "RSI",
        "MACD",
        "Signal",
        "MACD_Hist",
        "Beta_Factor",
        "Value_Factor",
        "Momentum_Factor",
        "Volatility_Factor",
        "Mkt_RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
    ]

    # 데이터셋 생성
    dataset = DDPGDataset(
        train_df=train_df, test_df=test_df, window_size=12, feature_cols=feature_cols
    )

    num_stocks = len(dataset.train_symbols)
    num_features = len([c for c in feature_cols if c != "Momentum1M"])
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Hybrid 방식과 동일한 학습률 적용
    agent = DDPGAgent(
        num_stocks,
        num_features,
        lr_actor=5e-5,  # Hybrid와 동일
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        entropy_coef=0.01,
        device=DEVICE,
    )
    model_path = Path(__file__).parent / "best_ddpg.pth"
    save_dir = ROOT_DIR / "results" / "02_DDPG_Only"
    save_dir.mkdir(parents=True, exist_ok=True)

    if mode == "train":
        if model_path.exists():
            print("⚠️ 기존 모델 삭제")
            model_path.unlink()

        print("\n=== 학습 모드 (Train Data) ===")
        train_windows = dataset.get_train_windows()
        train_env = PortfolioEnv(
            dataset, train_windows, dataset.train_symbols, feature_cols
        )

        # 1. 메인 학습 (Early Stopping 적용)
        train_rewards, best_reward = train_ddpg(
            agent,
            train_env,
            num_episodes=800,
            patience=100,  # Hybrid와 동일
        )

        # 2. Fine-tuning
        finetune_rewards = fine_tune_ddpg(agent, train_env, num_episodes=50)

        # 3. 최종 모델 저장
        torch.save(agent.actor.state_dict(), model_path)
        print(f"✅ 모델 저장 완료: {model_path}")

        # 4. 학습 곡선 시각화
        plot_training_curve(train_rewards, finetune_rewards, save_dir)

    elif mode == "compare":
        # ===== 1. 기존 학습된 모델 존재 여부 확인 =====
        if model_path.exists():
            print("✅ 기존 학습 모델 발견")
        else:
            print("⚠️ 학습된 모델 없음. 자동 학습 시작...")
            train_windows = dataset.get_train_windows()
            train_env = PortfolioEnv(
                dataset, train_windows, dataset.train_symbols, feature_cols
            )
            train_rewards, _ = train_ddpg(
                agent, train_env, num_episodes=800, patience=100
            )
            finetune_rewards = fine_tune_ddpg(agent, train_env, num_episodes=50)
            torch.save(agent.actor.state_dict(), model_path)
            plot_training_curve(train_rewards, finetune_rewards, save_dir)

        print("\n=== 실전 백테스팅 (Test Data) ===")

        test_windows = dataset.get_test_windows()
        test_symbols = dataset.test_symbols

        # ===== 2. 🔥 테스트 데이터용 새 에이전트 생성 (Hybrid 방식) =====
        num_stocks_test = len(test_symbols)
        print(f"📊 Train 종목: {num_stocks}개 -> Test 종목: {num_stocks_test}개")

        test_agent = DDPGAgent(
            num_stocks_test,
            num_features,
            lr_actor=5e-5,
            lr_critic=1e-3,
            gamma=0.99,
            tau=0.001,
            entropy_coef=0.01,
            device=DEVICE,
        )

        # ===== 3. 🔥 Encoder만 전이학습 (선택사항) =====
        if model_path.exists() and num_stocks_test != num_stocks:
            print("⚠️ 종목 수 불일치. Encoder만 전이학습 시도...")
            try:
                trained_state = torch.load(model_path, map_location=DEVICE)
                test_state = test_agent.actor.state_dict()

                # Encoder 파라미터만 복사
                encoder_keys = [k for k in trained_state.keys() if "encoder" in k]
                loaded_count = 0

                for key in encoder_keys:
                    if (
                        key in test_state
                        and trained_state[key].shape == test_state[key].shape
                    ):
                        test_state[key] = trained_state[key]
                        loaded_count += 1

                test_agent.actor.load_state_dict(test_state)
                print(f"✅ Encoder 전이학습 완료: {loaded_count}개 레이어")

            except Exception as e:
                print(f"⚠️ 전이학습 실패: {e}")
                print("→ 테스트 데이터로 처음부터 학습...")

        elif model_path.exists() and num_stocks_test == num_stocks:
            # 종목 수 동일하면 전체 로드
            test_agent.actor.load_state_dict(
                torch.load(model_path, map_location=DEVICE)
            )
            print("✅ 전체 모델 로드 완료")

        # ===== 4. 🔥 Fine-tuning (Test 데이터) =====
        print("\n[Fine-tuning] Test 데이터로 출력 레이어 조정 중...")
        test_env = PortfolioEnv(dataset, test_windows, test_symbols, feature_cols)

        finetune_buffer = ReplayBuffer(capacity=512)
        test_agent.replay_buffer = finetune_buffer  # 작은 버퍼 사용

        for episode in range(50):  # Fine-tune episodes
            state = test_env.reset()
            episode_reward = 0

            while True:
                action = test_agent.select_action(state, noise_std=0.1)
                next_state, reward, done, info = test_env.step(action)

                test_agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(test_agent.replay_buffer) >= 32:
                    test_agent.train(batch_size=32)

                episode_reward += reward
                state = next_state

                if done:
                    break

            if (episode + 1) % 10 == 0:
                print(f"  Episode {episode + 1}/50: Reward = {episode_reward:.2f}")

        print("✅ Fine-tuning 완료\n")
        test_agent.actor.eval()

        # ===== 5. 백테스팅 실행 (test_agent 사용) =====
        buy_and_hold = run_buy_and_hold(test_windows, test_symbols)
        monthly = run_ddpg_rebalancing(
            test_agent,
            dataset,
            test_windows,
            test_symbols,
            "monthly",  # ← test_agent!
        )
        quarterly = run_ddpg_rebalancing(
            test_agent, dataset, test_windows, test_symbols, "quarterly"
        )
        semiannual = run_ddpg_rebalancing(
            test_agent, dataset, test_windows, test_symbols, "semiannual"
        )
        annual = run_ddpg_rebalancing(
            test_agent, dataset, test_windows, test_symbols, "annual"
        )

        plot_comparison(buy_and_hold, monthly, quarterly, semiannual, annual, save_dir)

        # ===== 나머지 로그 저장 코드 동일 =====
        all_logs = []
        all_logs.extend(buy_and_hold["trade_logs"])
        all_logs.extend(monthly["trade_logs"])
        all_logs.extend(quarterly["trade_logs"])
        all_logs.extend(semiannual["trade_logs"])
        all_logs.extend(annual["trade_logs"])

        logs_df = pd.DataFrame(all_logs)
        if not logs_df.empty:
            cols = ["Date", "Strategy", "Type"] + sorted(test_symbols)
            logs_df = logs_df[cols]
            log_path = save_dir / "trade_logs.csv"
            logs_df.to_csv(log_path, index=False)
            print(f"✅ 투자 로그 저장 완료: {log_path}")

        # 시계열 데이터 저장
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

        # 집계 지표 저장
        summary_metrics = [
            buy_and_hold["metrics"],
            monthly["metrics"],
            quarterly["metrics"],
            semiannual["metrics"],
            annual["metrics"],
        ]
        summary_df = pd.DataFrame(summary_metrics)
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
