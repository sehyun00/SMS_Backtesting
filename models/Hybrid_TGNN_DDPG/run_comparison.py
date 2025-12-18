"""
Hybrid TGNN-DDPG Backtesting & Comparison
학습 및 백테스팅 메인 스크립트
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib
from training_monitor import TrainingMonitor

matplotlib.use("Agg")
import warnings

warnings.filterwarnings("ignore")

# 로컬 모듈 임포트
from agent import HybridAgent
from environment import HybridDataset, HybridPortfolioEnv
from utils import calculate_metrics
from visualization import BacktestVisualizer


# ==========================================
# 프로젝트 루트 및 데이터 경로 설정
# ==========================================
ROOT_DIR = Path(__file__).parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"

print(f"📂 Train data: {TRAIN_DATA_PATH.name}")
print(f"📂 Test data: {TEST_DATA_PATH.name}\n")


# ==========================================
# 학습 함수
# ==========================================
def train_hybrid(agent, env, num_episodes=200, save_dir=None):
    """
    Hybrid 에이전트 학습 루프 (모니터링 추가)

    Args:
        agent: HybridAgent 인스턴스
        env: HybridPortfolioEnv 인스턴스
        num_episodes: 학습 에피소드 수
        save_dir: 모니터링 결과 저장 디렉토리
    """
    print(f"🚀 Starting Hybrid model training: Total {num_episodes} episodes")

    # 학습 모니터 초기화
    if save_dir:
        monitor = TrainingMonitor(save_dir / "training_logs", save_interval=50)
    else:
        monitor = None

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_returns = []
        episode_values = []
        noise_std = max(0.01, 0.2 - episode * 0.002)

        while True:
            action, alpha_value = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)

            agent.actor.current_mdd = info.get("current_mdd", 0.0)
            agent.replay_buffer.push(state, action, reward, next_state, done)

            if len(agent.replay_buffer) >= 256:
                agent.train(batch_size=64)

            episode_reward += reward
            episode_returns.append(info.get("return", 0.0))
            episode_values.append(info.get("portfolio_value", 1000000))

            state = next_state

            if done:
                break

        # 에피소드 성과 계산
        avg_return = np.mean(episode_returns) if episode_returns else 0.0

        # MDD 계산
        values = np.array(episode_values)
        if len(values) > 0:
            peak = np.maximum.accumulate(values)
            drawdowns = (values - peak) / peak
            mdd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            mdd = 0.0

        # Sharpe 계산 (간단 버전)
        if len(episode_returns) > 1:
            sharpe = np.mean(episode_returns) / (np.std(episode_returns) + 1e-8)
        else:
            sharpe = 0.0

        # 모니터에 기록
        if monitor:
            monitor.record_episode(
                episode, episode_reward, avg_return, mdd, sharpe, alpha_value
            )

        if (episode + 1) % 10 == 0:
            print(
                f"{episode + 1:3d}/{num_episodes} | Reward: {episode_reward:7.2f} | "
                f"Return: {avg_return * 100:5.2f}% | MDD: {mdd * 100:5.2f}% | "
                f"Sharpe: {sharpe:.3f} | Alpha: {alpha_value:.3f}"
            )


# ==========================================
# 백테스팅 함수
# ==========================================
def run_hybrid_rebalancing(agent, dataset, freq="monthly"):
    """
    Hybrid 리밸런싱 백테스트

    Args:
        agent: HybridAgent 인스턴스
        dataset: HybridDataset 인스턴스
        freq: 리밸런싱 빈도 (monthly/quarterly/semiannual/annual)

    Returns:
        결과 dict (dates, portfolio_values, alphas, metrics, trade_logs)
    """
    freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
    interval = freq_map[freq]

    test_windows = dataset.get_test_windows()
    test_symbols = dataset.test_symbols

    capital = 1_000_000
    peak = capital
    current_weights = np.ones(len(test_symbols)) / len(test_symbols)
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
        state = dataset.get_state(test_windows, i)

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
            for sym, val in zip(test_symbols, current_weights):
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
            for sym, val in zip(test_symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)

        # 수익률 계산
        ret = np.dot(current_weights, w["labels"])
        capital *= 1 + ret
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
    """
    고정 비중 전략 (벤치마크)

    Args:
        dataset: HybridDataset 인스턴스
        strategy_name: 전략 이름

    Returns:
        결과 dict
    """
    test_windows = dataset.get_test_windows()
    test_symbols = dataset.test_symbols

    capital = 1_000_000
    peak = capital
    num_stocks = len(test_symbols)

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
            for sym, val in zip(test_symbols, current_weights):
                log[sym] = round(float(val), 4)
            trade_logs.append(log)

        ret = np.dot(current_weights, w["labels"])
        ret = float(ret)  # ✅ 명시적으로 float 변환
        capital *= 1.0 + ret

        if i < 3:
            print(
                f"[DEBUG] Month {i}: ret={ret:.4f}, capital before={capital:.0f}",
                end="",
            )
        if i < 3:
            print(f", after={capital:.0f}")

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
    """
    메인 실행 함수

    Args:
        mode: 'train' 또는 'compare'
    """
    # 데이터 로드
    print("[Initialization] Loading preprocessed data...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

    print(
        f"✅ Train data: {len(train_df):,} rows, {len(train_df['Symbol'].unique())} stocks"
    )
    print(
        f"✅ Test data: {len(test_df):,} rows, {len(test_df['Symbol'].unique())} stocks\n"
    )

    # Feature Columns
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

    # Sector One-Hot Encoding
    train_sectors = pd.get_dummies(train_df["Sector"], prefix="Sector")
    test_sectors = pd.get_dummies(test_df["Sector"], prefix="Sector")

    train_df = pd.concat([train_df, train_sectors], axis=1)
    test_df = pd.concat([test_df, test_sectors], axis=1)

    feature_cols.extend(train_sectors.columns.tolist())

    print("[Initialization] Preparing Hybrid dataset...")
    dataset = HybridDataset(
        train_df, test_df, window_size=12, feature_cols=feature_cols
    )

    # 모델 파라미터
    num_stocks_train = len(dataset.train_symbols)
    num_stocks_test = len(dataset.test_symbols)
    window_size = 12
    num_features = len(feature_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[System] Using device: {device}")
    print(
        f"[System] Train stocks: {num_stocks_train}, Test stocks: {num_stocks_test}\n"
    )

    # Agent 초기화 (Train 종목 수 기준)
    agent = HybridAgent(num_stocks_train, window_size, num_features, device=device)
    model_path = Path(__file__).parent / "best_hybrid.pth"

    if mode == "train":
        print("\n[Training] Starting model training on 2006-2020 data...")
        if model_path.exists():
            print("⚠️  Existing model found. Deleting and retraining...")
            model_path.unlink()

        train_windows = dataset.get_train_windows()
        train_env = HybridPortfolioEnv(dataset, windows=train_windows)

        # 저장 디렉토리 설정
        save_dir = ROOT_DIR / "results" / "03_Hybrid_TGNN_DDPG"
        save_dir.mkdir(parents=True, exist_ok=True)

        # 학습 실행 (save_dir 추가)
        train_hybrid(agent, train_env, num_episodes=200, save_dir=save_dir)

        torch.save(agent.actor.state_dict(), model_path)
        print(f"✅ Model saved successfully: {model_path}")
        return

    elif mode == "compare":
        print(f"\n🔍 Debug: model_path = {model_path}")
        print(f"🔍 Debug: exists = {model_path.exists()}")
        if not model_path.exists():
            print(
                "❌ No trained model found. Please run: python run_comparison.py train"
            )
            return

        print("\n[Testing] Loading saved model...")

        # ⚠️ 중요: Test 종목 수에 맞게 새로운 Agent 생성
        test_agent = HybridAgent(
            num_stocks_test, window_size, num_features, device=device
        )

        # ✅ 전이 학습 로직 구현
        trained_state_dict = torch.load(model_path, map_location=device)
        model_state = test_agent.actor.state_dict()

        # TGNN Encoder와 공통 레이어만 로드 (Output 레이어 제외)
        loaded_keys = []
        skipped_keys = []
        for key in trained_state_dict.keys():
            # Output 레이어와 종목 수 의존 레이어 제외
            if "output" not in key and "weight_net" not in key:
                if (
                    key in model_state
                    and trained_state_dict[key].shape == model_state[key].shape
                ):
                    model_state[key] = trained_state_dict[key]
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)

        test_agent.actor.load_state_dict(model_state)
        test_agent.actor.eval()
        print(
            f"✅ Loaded {len(loaded_keys)} layers, skipped {len(skipped_keys)} layers (size mismatch)"
        )

        print("[Testing] Performing backtesting on 2021-2025 data...")

        # 벤치마크
        buy_and_hold = run_fixed_weights(dataset, "1/N Buy & Hold")

        # Hybrid 전략
        monthly = run_hybrid_rebalancing(test_agent, dataset, "monthly")
        quarterly = run_hybrid_rebalancing(test_agent, dataset, "quarterly")
        semiannual = run_hybrid_rebalancing(test_agent, dataset, "semiannual")
        annual = run_hybrid_rebalancing(test_agent, dataset, "annual")

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
                f"{m['Strategy']:15} | CAGR: {m['CAGR']:6.1f}% | "
                f"MDD: {m['MDD']:6.1f}% | Final: ${m['FinalValue']:,.0f}"
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
