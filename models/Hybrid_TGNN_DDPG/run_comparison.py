import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
from model import HybridAgent
from visualization import BacktestVisualizer

# ==========================================
# 프로젝트 루트 및 데이터 경로 설정
# ==========================================
ROOT_DIR = Path(__file__).parent.parent.parent
data_files = sorted(
    ROOT_DIR.glob("data/processed_daily_5factor_model_10stocks_*years_*.csv")
)
if not data_files:
    raise FileNotFoundError("Data file not found. Please run preprocessing first.")
DATA_PATH = data_files[-1]  # 가장 최근 파일
print(f"📂 Using data file: {DATA_PATH.name}\n")


# ==========================================
# Hybrid Dataset (TGNN용 그래프 + 윈도우 데이터)
# ==========================================
class HybridDataset:
    """
    Hybrid 모델용 Dataset 클래스
    - 주식 간 관계 그래프(인접 행렬) 생성
    - 시계열 윈도우 데이터 제공
    """

    def __init__(self, df, window_size=12, feature_cols=None):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.symbols = sorted(df["Symbol"].unique())

        # 1. 일별 → 월말 변환
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )

        # 2. 타겟 수익률 (다음 달 모멘텀)
        self.monthly_df["ReturnRaw"] = self.monthly_df["Momentum1M"].copy()

        # 3. Train/Test 분할 날짜
        self.split_date = pd.Timestamp("2015-12-31")

        # 4. 🔥 스케일링 (5-Factor 제외)
        self.fit_scaler_on_train_data()

        # 5. 윈도우 생성
        self.windows = self.create_windows()

        # 6. 테스트 시작 인덱스 찾기
        self.test_start_idx = 0
        for i, w in enumerate(self.windows):
            if w["date"] > self.split_date:
                self.test_start_idx = i
                break

        print(f"   📊 Hybrid dataset created: Total {len(self.windows)} months")
        print(f"   📈 Training data: {self.test_start_idx} months (~2017)")
        print(
            f"   📉 Test data: {len(self.windows) - self.test_start_idx} months (2018~)"
        )

    def fit_scaler_on_train_data(self):
        """🔥 학습 데이터로만 StandardScaler 피팅 (5-Factor 제외)"""
        train_data = self.monthly_df[self.monthly_df["Date"] <= self.split_date]

        # 🔥 5-Factor 제외 (이미 % 단위)
        scale_cols = [
            col
            for col in self.feature_cols
            if col not in ["Mkt_RF", "SMB", "HML", "RMW", "CMA"]
        ]

        if not scale_cols:
            print("⚠️  No columns to scale!")
            return

        self.scaler = StandardScaler()
        self.scaler.fit(train_data[scale_cols].values)

        # 스케일링 적용 (5-Factor 제외)
        self.monthly_df[scale_cols] = self.scaler.transform(
            self.monthly_df[scale_cols].values
        )

        print(
            f"   ✅ StandardScaler fitted on {len(scale_cols)} columns (5-Factor raw)"
        )

    def create_graph(self, snapshot_df):
        """
        TGNN 로직: 상관관계 + 산업 유사도 기반 인접 행렬 생성
        """
        n = len(self.symbols)
        corr_matrix = np.eye(n)

        # 1. 상관계수 계산
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

        # 2. 산업 유사도 반영
        if "Sector" in snapshot_df.columns:
            sector_map = snapshot_df.set_index("Symbol")["Sector"].to_dict()
            industry_sim = np.zeros((n, n))
            for i, sym1 in enumerate(self.symbols):
                for j, sym2 in enumerate(self.symbols):
                    if sym1 in sector_map and sym2 in sector_map:
                        industry_sim[i, j] = (
                            1.0 if sector_map[sym1] == sector_map[sym2] else 0.5
                        )
            edge_weights = corr_matrix + industry_sim
        else:
            edge_weights = corr_matrix

        # 3. 임계값 이상만 연결
        adj = (edge_weights > 0.35).astype(float)
        return adj

    def create_masked_graph(self, snapshot_df, active_mask):
        """비활성 종목의 연결 제거"""
        n = len(self.symbols)
        adj = (
            self.create_graph(snapshot_df)
            if not snapshot_df.empty
            else np.zeros((n, n))
        )
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def create_windows(self):
        """전체 기간의 윈도우 데이터 생성"""
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

            active_mask = np.array(active_mask)
            adj = self.create_masked_graph(
                window_df[window_df["Date"] == target_date], active_mask
            )

            labels = []
            for symbol in self.symbols:
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

    def get_state(self, idx):
        """RL 에이전트 입력용 상태 벡터 반환"""
        w = self.windows[idx]
        features = w["features"]
        adj = w["adjmatrix"]

        # 🔥 NaN 제거
        features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
        adj = np.nan_to_num(adj, nan=0.0, posinf=1.0, neginf=0.0)

        state = np.concatenate([features.flatten(), adj.flatten()])
        state = np.nan_to_num(state, nan=0.0, posinf=10.0, neginf=-10.0)
        return state.astype(np.float32)

    def get_train_windows(self):
        return self.windows[: self.test_start_idx]

    def get_test_windows(self):
        return self.windows[self.test_start_idx :]

    def __len__(self):
        return len(self.windows)


# ==========================================
# Hybrid Portfolio Environment
# ==========================================
class HybridPortfolioEnv:
    """강화학습 환경"""

    def __init__(self, dataset, windows=None, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows if windows else dataset.windows
        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.current_step = 0
        self.n_steps = len(self.windows)
        self.gamma = 2.0
        self.cost_bps = 0.0005
        self.n_stocks = len(dataset.symbols)
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []
        return self.get_state(0)

    def get_state(self, idx):
        w = self.windows[idx]
        features = w["features"]
        adj = w["adjmatrix"]
        state = np.concatenate([features.flatten(), adj.flatten()])
        return state.astype(np.float32)

    def step(self, action):
        w = self.windows[self.current_step]
        returns = w["labels"]  # 이미 % 단위 (Momentum1M)
        features = w["features"]

        # 🔥 수정: *100 제거
        portfolio_return_pct = np.dot(action, returns)  # % 단위
        portfolio_return = portfolio_return_pct  # 🔥 그대로 사용!

        # 거래 비용
        turnover = np.sum(np.abs(action - self.prev_weights))
        cost = turnover * self.cost_bps * 100  # bp → % 변환

        net_return = portfolio_return - cost

        # 🔥 % → 비율 변환 (포트폴리오 가치 계산용)
        self.portfolio_value *= 1 + net_return / 100

        self.current_step += 1
        done = self.current_step >= self.n_steps
        self.return_history.append(net_return)  # % 단위로 저장

        # MDD 계산
        if len(self.return_history) >= 12:
            cumulative_returns = np.cumprod(
                1 + np.array(self.return_history[-12:]) / 100
            )
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / peak
            self.current_mdd = abs(min(drawdowns))
        else:
            self.current_mdd = 0.0

        # 리워드 계산
        if len(self.return_history) >= 6:
            returns_array = np.array(self.return_history[-12:])  # % 단위
            mean_return = np.mean(returns_array)

            negative_returns = returns_array[returns_array < 0]
            downside_std = (
                np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            )
            concentration = np.sum(action**2)
            volatility = np.std(returns_array)

            # 5-Factor (정상)
            mkt_rf = features[:, -1, 13] * 100
            smb = features[:, -1, 14] * 100
            hml = features[:, -1, 15] * 100
            rmw = features[:, -1, 16] * 100
            cma = features[:, -1, 17] * 100

            mkt_rf = np.nan_to_num(mkt_rf, nan=0.0, posinf=10.0, neginf=-10.0)
            smb = np.nan_to_num(smb, nan=0.0, posinf=10.0, neginf=-10.0)
            hml = np.nan_to_num(hml, nan=0.0, posinf=10.0, neginf=-10.0)
            rmw = np.nan_to_num(rmw, nan=0.0, posinf=10.0, neginf=-10.0)
            cma = np.nan_to_num(cma, nan=0.0, posinf=10.0, neginf=-10.0)

            # 기존 리워드 (스케일 조정)
            return_reward = mean_return * 5.0  # 50 → 5
            risk_adjusted_return = mean_return / (volatility + 1e-8)
            sharpe_bonus = risk_adjusted_return * 3.0  # 30 → 3
            downside_penalty = 5.0 * downside_std  # 50 → 5

            # MDD Penalty
            if len(self.return_history) >= 12:
                cumulative_returns = np.cumprod(
                    1 + np.array(self.return_history[-12:]) / 100
                )
                peak = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - peak) / peak
                current_mdd = abs(min(drawdowns))

                if current_mdd > 0.25:
                    mdd_penalty = 300.0 * (current_mdd - 0.25) ** 2
                elif current_mdd > 0.20:
                    mdd_penalty = 150.0 * (current_mdd - 0.20) ** 2
                elif current_mdd > 0.15:
                    mdd_penalty = 50.0 * (current_mdd - 0.15) ** 2
                else:
                    mdd_penalty = 0

                current_value = cumulative_returns[-1]
                current_peak = np.max(cumulative_returns)
                if current_value < current_peak:
                    recovery_ratio = current_value / current_peak
                    if recovery_ratio > 0.95:
                        recovery_bonus = 15.0 * (1 - recovery_ratio)
                    else:
                        recovery_bonus = 0
                else:
                    mdd_penalty = 0
                    recovery_bonus = 0
            else:
                mdd_penalty = 0
                recovery_bonus = 0

            # Volatility Penalty
            if volatility > 5.0:
                volatility_penalty = 15.0 * (volatility - 5.0) ** 2
            elif volatility > 3.5:
                volatility_penalty = 5.0 * (volatility - 3.5) ** 2
            else:
                volatility_penalty = 0

            # Concentration Penalty
            if concentration > 0.12:
                concentration_penalty = 500.0 * (concentration - 0.12) ** 4
            else:
                concentration_penalty = 0

            # Diversity Bonus
            entropy = -np.sum(action * np.log(action + 1e-10))
            max_entropy = np.log(len(action))
            normalized_entropy = entropy / max_entropy

            if normalized_entropy > 0.85:
                diversity_bonus = 6.0 * normalized_entropy
            elif normalized_entropy > 0.75:
                diversity_bonus = 4.0 * normalized_entropy
            else:
                diversity_bonus = 2.0 * normalized_entropy

            # Turnover Penalty
            recent_return = (
                np.mean(self.return_history[-3:])
                if len(self.return_history) >= 3
                else 0
            )

            if recent_return < -3.0:
                turnover_penalty = turnover * 1.0
            elif recent_return > 5.0:
                turnover_penalty = turnover * 1.0
            else:
                turnover_penalty = turnover * 2.5

            # 5-Factor 리워드
            market_exposure = np.dot(action, mkt_rf)
            market_exposure = np.clip(market_exposure, -10, 10)
            market_bonus = 0.4 * np.maximum(market_exposure, 0)

            value_exposure = np.dot(action, hml)
            value_exposure = np.clip(value_exposure, -10, 10)
            value_bonus = 0.35 * np.maximum(value_exposure, 0)

            quality_exposure = np.dot(action, rmw)
            quality_exposure = np.clip(quality_exposure, -10, 10)
            quality_bonus = 0.35 * np.maximum(quality_exposure, 0)

            factor_exposures = np.array(
                [
                    np.dot(action, smb),
                    np.dot(action, hml),
                    np.dot(action, rmw),
                    np.dot(action, cma),
                ]
            )
            factor_exposures = np.nan_to_num(factor_exposures, nan=0.0)
            factor_std = np.std(factor_exposures)

            if factor_std < 0.2:
                factor_diversity_bonus = 0.25 * (0.2 - factor_std)
            else:
                factor_diversity_bonus = 0

            # Final Reward
            reward = (
                return_reward
                + sharpe_bonus
                - downside_penalty
                + market_bonus
                + value_bonus
                + quality_bonus
                + factor_diversity_bonus
                - mdd_penalty
                - volatility_penalty
                - concentration_penalty
                + diversity_bonus
                - turnover_penalty
                + recovery_bonus
            )

            # NaN 체크
            if np.isnan(reward) or np.isinf(reward):
                reward = net_return * 10 - turnover * 5.0
        else:
            reward = net_return * 10 - turnover * 5.0

        self.prev_weights = action

        next_state = (
            self.get_state(self.current_step)
            if not done
            else np.zeros_like(self.get_state(0))
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "date": w["date"],
            "turnover": turnover,
            "cost": cost,
            "concentration": np.sum(action**2),
            "downside_risk": downside_std if len(self.return_history) >= 6 else 0,
            "current_mdd": self.current_mdd,
        }

        return next_state, reward, done, info


# ==========================================
# 학습 및 백테스팅 로직
# ==========================================
def train_hybrid(agent, env, num_episodes=200):
    """Hybrid 에이전트 학습 루프"""
    print(f"🚀 Starting Hybrid model training: Total {num_episodes} episodes")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        noise_std = max(0.01, 0.2 - episode * 0.002)

        while True:
            action, alpha_value = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)

            agent.actor.current_mdd = info.get("current_mdd", 0.0)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= 256:
                agent.train(batch_size=64)

            episode_reward += reward
            state = next_state

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(
                f"{episode + 1:3d}/{num_episodes} | Reward: {episode_reward:7.2f} | Noise: {noise_std:.3f} | Alpha: {alpha_value:.3f}"
            )


def calculate_metrics(ts_data, dates, strategy_name):
    """성과 지표 계산"""
    df = pd.DataFrame(ts_data)
    df["date"] = pd.to_datetime(dates)
    initial = df["portfolio_value"].iloc[0]
    final = df["portfolio_value"].iloc[-1]
    days = (df["date"].max() - df["date"].min()).days
    years = days / 365.25
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0
    mdd = abs(min(df["drawdown"])) * 100
    r = df["return"] * 100
    vol = r.std() * np.sqrt(12)
    sharpe = (cagr / 100) / (vol + 1e-8)

    return {
        "Strategy": strategy_name,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "FinalValue": final,
    }


def run_hybrid_rebalancing(agent, dataset, freq="monthly"):
    """Hybrid 리밸런싱 백테스트"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[freq]
    test_windows = dataset.get_test_windows()
    start_idx = dataset.test_start_idx

    capital = 1_000_000
    peak = capital
    current_weights = np.ones(10) / 10
    portfolio_history = [capital]

    ts_data = {
        "portfolio_value": [],
        "return": [],
        "drawdown": [],
        "turnover": [],
        "alpha": [],
    }
    dates = []
    trade_logs = []

    for i, w in enumerate(test_windows):
        state = dataset.get_state(start_idx + i)

        # MDD 업데이트
        if len(portfolio_history) >= 12:
            recent_values = np.array(portfolio_history[-12:])
            peak_value = np.maximum.accumulate(recent_values)
            drawdowns = (recent_values - peak_value) / peak_value
            current_mdd = abs(min(drawdowns))
        else:
            current_mdd = 0.0

        agent.actor.current_mdd = current_mdd

        # 리밸런싱
        if i % interval == 0:
            action, alpha_value = agent.select_action(state, noise_std=0.0)
            current_weights = action

            log = {
                "Date": w["date"],
                "Strategy": f"Hybrid_{freq}",
                "Type": "Rebalance",
                "Alpha": round(alpha_value, 3),
            }
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)
        else:
            alpha_value = 0.5
            log = {
                "Date": w["date"],
                "Strategy": f"Hybrid_{freq}",
                "Type": "Hold",
                "Alpha": round(alpha_value, 3),
            }
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)

        # 수익률 계산
        ret = np.dot(current_weights, w["labels"])
        capital *= 1 + ret / 100
        portfolio_history.append(capital)
        peak = max(peak, capital)
        dd = (capital - peak) / peak

        dates.append(w["date"])
        ts_data["portfolio_value"].append(capital)
        ts_data["return"].append(ret)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0)
        ts_data["alpha"].append(alpha_value)

    metrics = calculate_metrics(ts_data, dates, f"Hybrid_{freq}")

    return {
        "dates": dates,
        "portfolio_values": ts_data["portfolio_value"],
        "alphas": ts_data["alpha"],
        "metrics": metrics,
        "trade_logs": trade_logs,
    }


def run_fixed_weights(dataset, strategy_name="1/N Buy & Hold"):
    """고정 비중 전략"""
    test_windows = dataset.get_test_windows()
    capital = 1_000_000
    peak = capital
    num_stocks = len(dataset.symbols)

    ts_data = {
        "portfolio_value": [],
        "return": [],
        "drawdown": [],
        "turnover": [],
    }
    dates = []
    trade_logs = []

    for i, w in enumerate(test_windows):
        current_weights = np.ones(num_stocks) / num_stocks

        if i == 0:
            log = {"Date": w["date"], "Strategy": strategy_name, "Type": "Init"}
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)

        ret = np.dot(current_weights, w["labels"])
        capital *= 1 + ret / 100
        peak = max(peak, capital)
        dd = (capital - peak) / peak

        dates.append(w["date"])
        ts_data["portfolio_value"].append(capital)
        ts_data["return"].append(ret)
        ts_data["drawdown"].append(dd)
        ts_data["turnover"].append(0)

    metrics = calculate_metrics(ts_data, dates, strategy_name)

    return {
        "dates": dates,
        "portfolio_values": ts_data["portfolio_value"],
        "metrics": metrics,
        "trade_logs": trade_logs,
    }


# ==========================================
# Main 함수
# ==========================================
def main(mode="compare"):
    df = pd.read_csv(DATA_PATH)

    # Feature Columns (5-Factor 포함)
    feature_cols = [
        "Close",
        "Volume",
        "Beta",
        "MarketCap",
        "Momentum1M",
        "Momentum3M",
        "Momentum6M",
        "Momentum12M",
        "Volatility",
        "RSI",
        "MACD",
        "Signal",
        "MACD_Hist",
        "Mkt_RF",
        "SMB",
        "HML",
        "RMW",
        "CMA",
    ]

    print("[Initialization] Preparing Hybrid dataset...")
    dataset = HybridDataset(df, feature_cols=feature_cols)

    num_stocks = 10
    window_size = 12
    num_features = len(feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Using device: {device}")

    agent = HybridAgent(num_stocks, window_size, num_features, device=device)
    model_path = Path(__file__).parent / "best_hybrid.pth"

    if mode == "train":
        print("\n[Training] Starting model training using 2010-2015 data...")
        if model_path.exists():
            print("⚠️  Existing model found. Deleting and retraining...")
            model_path.unlink()

        train_windows = dataset.get_train_windows()
        train_env = HybridPortfolioEnv(dataset, windows=train_windows)
        train_hybrid(agent, train_env, num_episodes=200)

        torch.save(agent.actor.state_dict(), model_path)
        print(f"✅ Model saved successfully: {model_path}")
        return

    elif mode == "compare":
        if not model_path.exists():
            print(
                "❌ No trained model found. Please run: python run_comparison.py train"
            )
            return

        print("\n[Testing] Loading saved model...")
        agent.actor.load_state_dict(torch.load(model_path))
        print("[Testing] Performing backtesting on 2018-2025 data...")

        # 벤치마크
        buy_and_hold = run_fixed_weights(dataset, "1/N Buy & Hold")

        # Hybrid 전략
        monthly = run_hybrid_rebalancing(agent, dataset, "monthly")
        quarterly = run_hybrid_rebalancing(agent, dataset, "quarterly")
        semiannual = run_hybrid_rebalancing(agent, dataset, "semiannual")
        annual = run_hybrid_rebalancing(agent, dataset, "annual")

        # 시각화
        save_dir = ROOT_DIR / "results" / "03_Hybrid_TGNN_DDPG"
        save_dir.mkdir(parents=True, exist_ok=True)

        visualizer = BacktestVisualizer(save_dir=save_dir)
        visualizer.plot_rebalancing_comparison(
            buy_and_hold, monthly, quarterly, semiannual, annual
        )

        # 성과 요약
        print("\n" + "=" * 70)
        print("📊 Hybrid Model Final Performance")
        print("=" * 70)

        results_list = [buy_and_hold, monthly, quarterly, semiannual, annual]
        summary_data = [res["metrics"] for res in results_list]
        pd.DataFrame(summary_data).to_csv(save_dir / "summary_metrics.csv", index=False)

        for res in results_list:
            m = res["metrics"]
            print(
                f"{m['Strategy']:15} | CAGR: {m['CAGR']:6.1f}% | MDD: {m['MDD']:6.1f}% | Final: ${m['FinalValue']:,.0f}"
            )

        # Trade Logs 저장
        all_logs = []
        for res in results_list:
            all_logs.extend(res["trade_logs"])
        pd.DataFrame(all_logs).to_csv(save_dir / "hybrid_trade_logs.csv", index=False)
        print(f"\n✅ Trade logs saved: {save_dir / 'hybrid_trade_logs.csv'}")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
