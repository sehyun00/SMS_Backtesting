"""
Hybrid TGNN-DDPG Training and Performance Comparison Script
- Combines TGNN's graph generation logic with DDPG's reinforcement learning logic for training and testing.
- Training Period: ~2017 (3 years)
- Test Period: 2018~2025 (8 years)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

from model import HybridAgent
from visualization import BacktestVisualizer

# Project root and data path settings
ROOT_DIR = Path(__file__).parent.parent.parent

# 🔥 Automatically select the most recent CSV file
data_files = sorted(
    ROOT_DIR.glob("data/processed_daily_5factor_model_10stocks_*years_*.csv")
)

if not data_files:
    raise FileNotFoundError("Data file not found. Please run preprocessing first.")

DATA_PATH = data_files[-1]  # Most recent file (last after sorting)
print(f"📂 Using data file: {DATA_PATH.name}")


# ============ Hybrid Dataset (Graph Structure + Window Data) ============


class HybridDataset:
    """
    Dataset class for Hybrid model
    - Generates and provides time series window data and relationship graphs (Adjacency Matrix) between stocks.
    """

    def __init__(self, df, window_size=12, feature_cols=None):
        self.df = df.copy()
        self.df["Date"] = pd.to_datetime(self.df["Date"])
        self.window_size = window_size
        self.feature_cols = feature_cols
        self.symbols = sorted(df["Symbol"].unique())

        # 1. Convert to monthly data (Resampling)
        # - Merge daily data based on month-end.
        self.monthly_df = (
            self.df.set_index("Date")
            .groupby(["Symbol", pd.Grouper(freq="ME")])
            .last()
            .reset_index()
        )
        # Use Momentum1M as target return (can be replaced if needed)
        self.monthly_df["Return_Raw"] = self.monthly_df["Momentum1M"].copy()

        # 2. Train/Test data split date (December 31, 2017)
        self.split_date = pd.Timestamp("2015-12-31")

        # 3. Fit scaler (based on Train data)
        # - Prevent information leakage (Look-ahead Bias) from Test data
        self._fit_scaler_on_train_data()

        # 4. Create window data (including graphs)
        self.windows = self._create_windows()

        # 5. Find test start index
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

    def _fit_scaler_on_train_data(self):
        """Fit StandardScaler only on training data."""
        train_data = self.monthly_df[self.monthly_df["Date"] <= self.split_date]
        self.scaler = StandardScaler()
        self.scaler.fit(train_data[self.feature_cols].values)

        # Transform all data
        self.monthly_df[self.feature_cols] = self.scaler.transform(
            self.monthly_df[self.feature_cols].values
        )

    def _create_graph(self, snapshot_df):
        """
        TGNN logic: Correlation * Industry Similarity
        - Define relationships between two stocks to generate adjacency matrix.
        """
        n = len(self.symbols)
        corr_matrix = np.eye(n)

        # 1. Calculate correlation coefficient
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

        # 2. Reflect industry similarity
        if "Sector" in snapshot_df.columns:
            sector_map = snapshot_df.set_index("Symbol")["Sector"].to_dict()
            industry_sim = np.zeros((n, n))
            for i, sym1 in enumerate(self.symbols):
                for j, sym2 in enumerate(self.symbols):
                    if sym1 in sector_map and sym2 in sector_map:
                        industry_sim[i, j] = (
                            1.0 if sector_map[sym1] == sector_map[sym2] else 0.5
                        )

            edge_weights = corr_matrix * industry_sim
        else:
            edge_weights = corr_matrix

        # Keep only connections above threshold (0.35) (Binary Adjacency Matrix)
        adj = (edge_weights >= 0.35).astype(float)
        return adj

    def _create_masked_graph(self, snapshot_df, active_mask):
        """Generate graph with removed connections for non-existent stocks (pre-listing/delisted, etc.)."""
        n = len(self.symbols)
        adj = (
            self._create_graph(snapshot_df)
            if not snapshot_df.empty
            else np.zeros((n, n))
        )

        # Remove connections of inactive stocks using mask matrix
        mask_matrix = np.outer(active_mask, active_mask)
        return adj * mask_matrix

    def _create_windows(self):
        """Generate window data for the entire period."""
        dates = sorted(self.monthly_df["Date"].unique())
        windows = []

        for i in range(len(dates) - self.window_size):
            window_dates = dates[i : i + self.window_size]
            target_date = window_dates[-1]  # Current time point (portfolio composition time)
            next_date = dates[i + self.window_size]  # Next time point (return check time)

            window_df = self.monthly_df[self.monthly_df["Date"].isin(window_dates)]
            next_df = self.monthly_df[self.monthly_df["Date"] == next_date]

            features = []
            active_mask = []

            for symbol in self.symbols:
                stock_data = window_df[window_df["Symbol"] == symbol]
                # Check if stock data exists at current time point
                is_active = not stock_data[stock_data["Date"] == target_date].empty

                if is_active:
                    vals = stock_data[self.feature_cols].values
                    # Pad with zeros if data length is insufficient (early listing, etc.)
                    if len(vals) < self.window_size:
                        pad = np.zeros(
                            (self.window_size - len(vals), len(self.feature_cols))
                        )
                        vals = np.vstack([pad, vals])
                    features.append(vals)
                    active_mask.append(True)
                else:
                    # Fill inactive stocks with zeros
                    features.append(
                        np.zeros((self.window_size, len(self.feature_cols)))
                    )
                    active_mask.append(False)

            # Build graph (apply Mask)
            active_mask = np.array(active_mask)
            adj = self._create_masked_graph(
                window_df[window_df["Date"] == target_date], active_mask
            )

            # Labels (next month's returns)
            labels = []
            for symbol in self.symbols:
                val = next_df[next_df["Symbol"] == symbol]["Return_Raw"].values
                labels.append(val[0] if len(val) > 0 else 0.0)

            windows.append(
                {
                    "features": np.array(features),  # (N, T, F)
                    "adj_matrix": adj,  # (N, N)
                    "labels": np.array(labels),  # (N,)
                    "date": target_date,
                    "active_mask": active_mask,
                }
            )

        return windows

    def get_state(self, idx):
        """Return state vector to be used as input for RL agent."""
        w = self.windows[idx]
        features = w["features"]  # (N, T, F)
        adj = w["adj_matrix"]  # (N, N)

        # Flatten and combine: [feature vector..., adjacency matrix vector...]
        state = np.concatenate([features.flatten(), adj.flatten()])
        return state.astype(np.float32)

    def get_train_windows(self):
        return self.windows[: self.test_start_idx]

    def get_test_windows(self):
        return self.windows[self.test_start_idx :]

    def __len__(self):
        return len(self.windows)


# ============ Portfolio Environment ============


class HybridPortfolioEnv:
    """
    Reinforcement Learning Environment
    - Defines State, Action, Reward interactions.
    """

    def __init__(self, dataset, windows=None, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows if windows else dataset.windows
        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.current_step = 0
        self.n_steps = len(self.windows)
        self.gamma = 2.0  # Risk Aversion
        self.cost_bps = 0.0005  # Transaction cost (5bp)

        self.n_stocks = len(dataset.symbols)
        self.prev_weights = np.zeros(self.n_stocks)

        # Return history for Sharpe Ratio calculation
        self.return_history = []

    def reset(self):
        """Initialize environment"""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []  # Initialize return history
        return self._get_state(0)

    def _get_state(self, idx):
        """Generate state vector for current step"""
        w = self.windows[idx]
        features = w["features"]
        adj = w["adj_matrix"]
        state = np.concatenate([features.flatten(), adj.flatten()])
        return state.astype(np.float32)

    def step(self, action):
        w = self.windows[self.current_step]
        returns = w["labels"]

        portfolio_return_pct = np.dot(action, returns)
        portfolio_return = portfolio_return_pct / 100.0

        turnover = np.sum(np.abs(action - self.prev_weights))
        cost = turnover * self.cost_bps
        net_return = portfolio_return - cost

        self.portfolio_value *= 1 + net_return
        self.current_step += 1
        done = self.current_step >= self.n_steps

        self.return_history.append(net_return)

        if len(self.return_history) >= 6:
            returns_array = np.array(self.return_history[-12:])
            mean_return = np.mean(returns_array)

            negative_returns = returns_array[returns_array < 0]
            downside_std = (
                np.std(negative_returns) if len(negative_returns) > 0 else 0.0
            )

            concentration = np.sum(action**2)  # HHI
            volatility = np.std(returns_array)

            # ============ 🔥 Sophisticated Reward Function Design ============

            # 1. Return component (basic)
            return_reward = mean_return * 100

            # 2. Risk adjustment (Sharpe-like)
            risk_adjusted_return = mean_return / (volatility + 1e-8)
            sharpe_bonus = risk_adjusted_return * 30.0

            # 3. Downside risk (Sortino-like)
            downside_penalty = 50.0 * downside_std

            # 4. MDD penalty (Core!)
            if len(self.return_history) >= 12:
                cumulative_returns = np.cumprod(1 + np.array(self.return_history[-12:]))
                peak = np.maximum.accumulate(cumulative_returns)
                drawdowns = (cumulative_returns - peak) / peak
                current_mdd = abs(min(drawdowns))

                # Nonlinear penalty according to MDD
                if current_mdd > 0.30:  # Over 30%: extreme penalty
                    mdd_penalty = 800.0 * (current_mdd - 0.30) ** 2
                elif current_mdd > 0.25:  # 25-30%: strong penalty
                    mdd_penalty = 400.0 * (current_mdd - 0.25) ** 2
                elif current_mdd > 0.20:  # 20-25%: moderate penalty
                    mdd_penalty = 150.0 * (current_mdd - 0.20) ** 2
                else:  # Below 20%: no penalty
                    mdd_penalty = 0
            else:
                mdd_penalty = 0

            # 5. Volatility penalty (smooth curve)
            # Target: monthly volatility below 4%
            if volatility > 0.06:  # Over 6%: strong penalty
                volatility_penalty = 100.0 * (volatility - 0.06) ** 2
            elif volatility > 0.04:  # 4-6%: weak penalty
                volatility_penalty = 30.0 * (volatility - 0.04) ** 2
            else:  # Below 4%: no penalty
                volatility_penalty = 0

            # 6. Concentration penalty (smooth curve)
            # Target: HHI below 0.20 (5 stocks equal = 0.20)
            if concentration > 0.30:  # Extreme concentration
                concentration_penalty = 200.0 * (concentration - 0.30) ** 2
            elif concentration > 0.25:  # High concentration
                concentration_penalty = 100.0 * (concentration - 0.25) ** 2
            elif concentration > 0.20:  # Slight concentration
                concentration_penalty = 40.0 * (concentration - 0.20) ** 2
            else:  # Appropriate diversification
                concentration_penalty = 0

            # 7. Diversity bonus (entropy-based)
            entropy = -np.sum(action * np.log(action + 1e-10))
            max_entropy = np.log(len(action))
            normalized_entropy = entropy / max_entropy

            # Higher entropy gives more bonus
            if normalized_entropy > 0.85:  # Very equal (8-9 stocks)
                diversity_bonus = 60.0 * normalized_entropy
            elif normalized_entropy > 0.75:  # Moderately equal (6-7 stocks)
                diversity_bonus = 40.0 * normalized_entropy
            else:  # Concentrated (4-5 stocks)
                diversity_bonus = 20.0 * normalized_entropy

            # ============ 🔥 Final Reward Function ============
            reward = (
                return_reward  # Return (basic)
                + sharpe_bonus  # Risk-adjusted return
                - downside_penalty  # Downside risk
                - mdd_penalty  # MDD (Core!)
                - volatility_penalty  # Volatility
                - concentration_penalty  # Concentration
                + diversity_bonus  # Diversity
            )

        else:
            # Initial few steps: simple reward
            reward = net_return * 100

        self.prev_weights = action
        next_state = (
            self._get_state(self.current_step)
            if not done
            else np.zeros_like(self._get_state(0))
        )

        info = {
            "portfolio_value": self.portfolio_value,
            "date": w["date"],
            "turnover": turnover,
            "cost": cost,
            "concentration": np.sum(action**2),
            "downside_risk": downside_std if len(self.return_history) >= 6 else 0,
        }

        return next_state, reward, done, info


# ============ Training and Execution Logic ============


def train_hybrid(agent, env, num_episodes=100):
    """Hybrid agent training loop"""
    print(f"\n🚀 Starting Hybrid model training: Total {num_episodes} episodes")

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        # Exploration noise reduction (Exploration scheduling)
        noise_std = max(0.01, 0.2 - episode * 0.002)

        while True:
            # Action selection and environment interaction
            action, alpha_value = agent.select_action(state, noise_std)  # Tuple unpacking
            next_state, reward, done, info = env.step(action)  # Pass action only

            # Store experience
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # Training (batch size 64)
            if len(agent.replay_buffer) > 256:
                agent.train(batch_size=64)

            episode_reward += reward
            state = next_state

            if done:
                break

        if (episode + 1) % 10 == 0:
            print(
                f"[{episode + 1:3d}/{num_episodes}] Reward: {episode_reward:.2f}, Noise: {noise_std:.3f}, Alpha: {alpha_value:.3f}"
            )


def calculate_metrics(ts_data, dates, strategy_name):
    """Calculate performance metrics (CAGR, MDD, Sharpe Ratio)"""
    df = pd.DataFrame(ts_data)
    df["date"] = pd.to_datetime(dates)

    initial = df["portfolio_value"].iloc[0]
    final = df["portfolio_value"].iloc[-1]

    days = (df["date"].max() - df["date"].min()).days
    years = days / 365.25
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    mdd = abs(min(df["drawdown"])) * 100

    # Sharpe Ratio (annualized)
    r = df["return"] / 100
    vol = r.std() * np.sqrt(12)
    sharpe = (cagr / 100) / (vol + 1e-8)

    return {
        "Strategy": strategy_name,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "Final_Value": final,
    }


def run_hybrid_rebalancing(agent, dataset, freq="monthly"):
    """Execute rebalancing (reflecting select_action return value change)"""
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[freq]

    test_windows = dataset.get_test_windows()
    start_idx = dataset.test_start_idx

    capital = 1_000_000
    peak = capital
    current_weights = np.ones(10) / 10

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

        if i % interval == 0:
            # select_action returns (action, alpha) tuple
            action, alpha_value = agent.select_action(state, noise_std=0.0)
            current_weights = action

            log = {
                "Date": w["date"],
                "Strategy": f"Hybrid({freq})",
                "Type": "Rebalance",
                "Alpha": round(alpha_value, 3),
            }
            for sym, val in zip(dataset.symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)
        else:
            alpha_value = 0.5  # Default value on Hold
            log = {
                "Date": w["date"],
                "Strategy": f"Hybrid({freq})",
                "Type": "Hold",
                "Alpha": round(alpha_value, 3),
            }
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
        ts_data["alpha"].append(alpha_value)  # Add Alpha recording

    metrics = calculate_metrics(ts_data, dates, f"Hybrid({freq})")

    return {
        "dates": dates,
        "portfolio_values": ts_data["portfolio_value"],
        "alphas": ts_data["alpha"],  # Alpha history for analysis
        "metrics": metrics,
        "trade_logs": trade_logs,
    }


def run_fixed_weights(dataset, strategy_name="1/N Buy & Hold"):
    """
    Benchmark: Fixed weight strategy (assuming Monthly Rebalancing)
    """
    test_windows = dataset.get_test_windows()

    capital = 1_000_000
    peak = capital
    num_stocks = len(dataset.symbols)

    ts_data = {"portfolio_value": [], "return": [], "drawdown": [], "turnover": []}
    dates = []
    trade_logs = []

    for i, w in enumerate(test_windows):
        # Assume rebalancing to 1/N every month (standard benchmark)
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


def main(mode="compare"):
    df = pd.read_csv(DATA_PATH)
    feature_cols = [
        # 🔥 Original indicators (dynamically changing)
        "Close",  # Stock price
        "Volume",  # Trading volume
        "Beta",
        "MarketCap",
        # 🔥 Momentum (rate of change)
        "Momentum1M",
        "Momentum3M",
        "Momentum6M",
        "Momentum12M",
        # 🔥 Technical indicators (dynamic)
        "Volatility",
        "RSI",
        "MACD",
        "Signal",
        "MACD_Hist",
    ]

    print("\n[Initialization] Preparing Hybrid dataset...")
    dataset = HybridDataset(df, feature_cols=feature_cols)

    num_stocks = 10
    window_size = 12
    num_features = len(feature_cols)

    # Automatic GPU detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Using device: {device}")

    agent = HybridAgent(num_stocks, window_size, num_features, device=device)

    model_path = Path(__file__).parent / "best_hybrid.pth"

    if mode == "train":
        print("\n[Training] Starting model training using 2010-2015 data...")
        if model_path.exists():
            print("⚠️ Existing model found. Deleting and retraining...")
            model_path.unlink()

        train_windows = dataset.get_train_windows()
        train_env = HybridPortfolioEnv(dataset, windows=train_windows)

        # Actual training
        train_hybrid(agent, train_env, num_episodes=100)

        # Save Actor model
        torch.save(agent.actor.state_dict(), model_path)
        print("✅ Model saved successfully ({})".format(model_path))
        return

    elif mode == "compare":
        if not model_path.exists():
            print(
                "⚠️ No trained model found. Please run 'python run_comparison.py train' first."
            )
            return

        print("\n[Testing] Loading saved model...")
        agent.actor.load_state_dict(torch.load(model_path))

        print("\n[Testing] Performing backtesting on 2018-2025 data...")

        # Execute backtesting by strategy
        buy_and_hold = run_fixed_weights(dataset, "1/N Buy & Hold")
        monthly = run_hybrid_rebalancing(agent, dataset, "monthly")
        quarterly = run_hybrid_rebalancing(agent, dataset, "quarterly")
        semiannual = run_hybrid_rebalancing(agent, dataset, "semiannual")
        annual = run_hybrid_rebalancing(agent, dataset, "annual")

        # Result save path
        save_dir = ROOT_DIR / "results" / "03_Hybrid_TGNN_DDPG"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Visualize results
        visualizer = BacktestVisualizer(save_dir=save_dir)
        visualizer.plot_rebalancing_comparison(
            buy_and_hold, monthly, quarterly, semiannual, annual
        )

    print("\n=== Hybrid Model Final Performance ===")
    results_list = [buy_and_hold, monthly, quarterly, semiannual, annual]

    # Save summary metrics
    summary_data = [res["metrics"] for res in results_list]
    pd.DataFrame(summary_data).to_csv(save_dir / "summary_metrics.csv", index=False)

    for res in results_list:
        m = res["metrics"]
        print(
            f"{m['Strategy']:<15} | CAGR: {m['CAGR']:>6.1f}% | MDD: {m['MDD']:>6.1f}% | Final: ${m['Final_Value']:,.0f}"
        )

    # Save trade logs
    all_logs = []
    for res in results_list:
        all_logs.extend(res["trade_logs"])
    pd.DataFrame(all_logs).to_csv(save_dir / "hybrid_trade_logs.csv", index=False)
    print(f"\n✅ Trade logs saved successfully: {save_dir / 'hybrid_trade_logs.csv'}")


if __name__ == "__main__":
    import sys

    # Use 'compare' as default if no argument provided
    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
