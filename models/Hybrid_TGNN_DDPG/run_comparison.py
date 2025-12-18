"""
Hybrid TGNN-DDPG Backtesting System
학습 및 백테스팅 메인 스크립트 (Clean Code Refactored)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
warnings.filterwarnings("ignore")

from agent import HybridAgent
from environment import HybridDataset, HybridPortfolioEnv
from utils import calculate_metrics
from visualization import BacktestVisualizer
from training_monitor import TrainingMonitor


# ==========================================
# Configuration
# ==========================================
@dataclass
class Config:
    """학습 및 백테스트 설정"""

    # 경로 설정
    ROOT_DIR: Path = Path(__file__).parent.parent.parent
    TRAIN_DATA_PATH: Path = ROOT_DIR / "data" / "train_data.csv"
    TEST_DATA_PATH: Path = ROOT_DIR / "data" / "test_data.csv"
    MODEL_SAVE_PATH: Path = Path(__file__).parent / "best_hybrid.pth"
    RESULTS_DIR: Path = ROOT_DIR / "results" / "03_Hybrid_TGNN_DDPG"

    # 학습 하이퍼파라미터
    MAX_EPISODES: int = 800
    EARLY_STOPPING_PATIENCE: int = 50
    BATCH_SIZE: int = 64
    MIN_BUFFER_SIZE: int = 256
    NOISE_DECAY_RATE: float = 0.002
    INITIAL_NOISE_STD: float = 0.2
    MIN_NOISE_STD: float = 0.01

    # Fine-tuning 설정
    FINETUNE_EPISODES: int = 50
    FINETUNE_BATCH_SIZE: int = 32
    FINETUNE_BUFFER_SIZE: int = 64
    FINETUNE_NOISE_STD: float = 0.1

    # 백테스트 설정
    INITIAL_CAPITAL: int = 1_000_000
    WINDOW_SIZE: int = 12

    # 리밸런싱 빈도
    REBALANCE_FREQUENCIES: Dict[str, int] = None

    def __post_init__(self):
        self.REBALANCE_FREQUENCIES = {
            "monthly": 1,
            "quarterly": 3,
            "semiannual": 6,
            "annual": 12,
        }
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# Data Management
# ==========================================
class DataManager:
    """데이터 로딩 및 전처리 관리"""

    FEATURE_COLUMNS = [
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

    def __init__(self, config: Config):
        self.config = config

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """학습 및 테스트 데이터 로드"""
        print("[DataManager] Loading preprocessed data...")

        train_df = pd.read_csv(self.config.TRAIN_DATA_PATH)
        test_df = pd.read_csv(self.config.TEST_DATA_PATH)

        self._validate_data(train_df, test_df)
        self._log_data_info(train_df, test_df)

        return train_df, test_df

    def prepare_features(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
        """특성 준비 (Sector One-Hot Encoding 포함)"""
        print("[DataManager] Preparing features...")

        # Sector 통일
        all_sectors = sorted(set(train_df["Sector"]) | set(test_df["Sector"]))
        train_df["Sector"] = pd.Categorical(train_df["Sector"], categories=all_sectors)
        test_df["Sector"] = pd.Categorical(test_df["Sector"], categories=all_sectors)

        # One-Hot Encoding
        train_sectors = pd.get_dummies(train_df["Sector"], prefix="Sector")
        test_sectors = pd.get_dummies(test_df["Sector"], prefix="Sector")

        # 차원 검증
        assert train_sectors.shape[1] == test_sectors.shape[1], (
            f"Sector dimension mismatch: Train={train_sectors.shape[1]}, Test={test_sectors.shape[1]}"
        )

        # 데이터프레임 결합
        train_df = pd.concat([train_df, train_sectors], axis=1)
        test_df = pd.concat([test_df, test_sectors], axis=1)

        # 전체 특성 목록
        feature_cols = self.FEATURE_COLUMNS + train_sectors.columns.tolist()

        print(
            f"✅ Total features: {len(feature_cols)} (Base: {len(self.FEATURE_COLUMNS)}, Sectors: {train_sectors.shape[1]})"
        )

        return train_df, test_df, feature_cols

    def create_dataset(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: List[str]
    ) -> HybridDataset:
        """HybridDataset 생성"""
        print("[DataManager] Creating HybridDataset...")

        dataset = HybridDataset(
            train_df=train_df,
            test_df=test_df,
            window_size=self.config.WINDOW_SIZE,
            feature_cols=feature_cols,
        )

        return dataset

    @staticmethod
    def _validate_data(train_df: pd.DataFrame, test_df: pd.DataFrame):
        """데이터 유효성 검증"""
        required_columns = ["Symbol", "Date", "Close", "Sector"]

        for col in required_columns:
            if col not in train_df.columns:
                raise ValueError(f"Missing column in train data: {col}")
            if col not in test_df.columns:
                raise ValueError(f"Missing column in test data: {col}")

    @staticmethod
    def _log_data_info(train_df: pd.DataFrame, test_df: pd.DataFrame):
        """데이터 정보 출력"""
        train_stocks = len(train_df["Symbol"].unique())
        test_stocks = len(test_df["Symbol"].unique())

        print(f"✅ Train: {len(train_df):,} rows, {train_stocks} stocks")
        print(f"✅ Test: {len(test_df):,} rows, {test_stocks} stocks\n")


# ==========================================
# Model Management
# ==========================================
class ModelManager:
    """모델 초기화 및 저장/로딩 관리"""

    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ModelManager] Using device: {self.device}")

    def create_agent(self, num_stocks: int, num_features: int) -> HybridAgent:
        """새로운 에이전트 생성"""
        agent = HybridAgent(
            num_stocks=num_stocks,
            window_size=self.config.WINDOW_SIZE,
            num_features=num_features,
            device=self.device,
        )
        return agent

    def save_model(self, agent: HybridAgent):
        """모델 저장"""
        torch.save(agent.actor.state_dict(), self.config.MODEL_SAVE_PATH)
        print(f"✅ Model saved: {self.config.MODEL_SAVE_PATH}")

    def load_model(self, agent: HybridAgent, encoder_only: bool = False) -> HybridAgent:
        """모델 로드 (전이 학습 지원)"""
        if not self.config.MODEL_SAVE_PATH.exists():
            raise FileNotFoundError(f"Model not found: {self.config.MODEL_SAVE_PATH}")

        print(f"[ModelManager] Loading model from {self.config.MODEL_SAVE_PATH}...")

        trained_state = torch.load(
            self.config.MODEL_SAVE_PATH, map_location=self.device
        )

        if encoder_only:
            self._load_encoder_only(agent, trained_state)
        else:
            agent.actor.load_state_dict(trained_state)
            print("✅ Full model loaded")

        return agent

    def _load_encoder_only(self, agent: HybridAgent, trained_state: dict):
        """인코더 레이어만 로드 (전이 학습용)"""
        model_state = agent.actor.state_dict()
        encoder_keys = ["tgnn_encoder", "ddpg_encoder", "gat_conv", "lstm"]

        loaded_keys = []
        skipped_keys = []

        for key, value in trained_state.items():
            if any(enc_key in key for enc_key in encoder_keys):
                if key in model_state and value.shape == model_state[key].shape:
                    model_state[key] = value
                    loaded_keys.append(key)
                else:
                    skipped_keys.append(key)
            else:
                skipped_keys.append(key)

        agent.actor.load_state_dict(model_state)
        print(f"✅ Encoder loaded: {len(loaded_keys)} layers")
        print(f"⚠️  Skipped: {len(skipped_keys)} layers (output/dimension mismatch)")


# ==========================================
# Training
# ==========================================
class Trainer:
    """모델 학습 관리"""

    def __init__(self, config: Config):
        self.config = config

    def train(
        self,
        agent: HybridAgent,
        env: HybridPortfolioEnv,
        monitor: Optional[TrainingMonitor] = None,
    ):
        """에이전트 학습 (Early Stopping 포함)"""
        print(f"🚀 Training started: Max {self.config.MAX_EPISODES} episodes")
        print(f"   Early Stopping patience: {self.config.EARLY_STOPPING_PATIENCE}")

        best_reward = -float("inf")
        best_episode = 0
        no_improve_count = 0

        for episode in range(self.config.MAX_EPISODES):
            episode_result = self._run_episode(agent, env, episode)

            # Early Stopping 체크
            if episode_result["reward"] > best_reward:
                best_reward = episode_result["reward"]
                best_episode = episode
                no_improve_count = 0
                self._save_best_model(agent)
            else:
                no_improve_count += 1

            if no_improve_count >= self.config.EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early Stopping at episode {episode + 1}")
                print(
                    f"   Best reward: {best_reward:.2f} at episode {best_episode + 1}"
                )
                break

            # 로깅
            if (episode + 1) % 10 == 0:
                self._log_episode(episode, episode_result, no_improve_count)

            # 모니터 기록
            if monitor:
                monitor.record_episode(
                    episode=episode,
                    reward=episode_result["reward"],
                    avg_return=episode_result["avg_return"],
                    mdd=episode_result["mdd"],
                    sharpe=episode_result["sharpe"],
                    alpha=episode_result["alpha"],
                )

        print(f"\n✅ Training completed. Best episode: {best_episode + 1}")

    def _run_episode(
        self, agent: HybridAgent, env: HybridPortfolioEnv, episode: int
    ) -> Dict:
        """단일 에피소드 실행"""
        state = env.reset()
        episode_reward = 0
        episode_returns = []
        episode_values = []

        # Noise 감소
        noise_std = max(
            self.config.MIN_NOISE_STD,
            self.config.INITIAL_NOISE_STD - episode * self.config.NOISE_DECAY_RATE,
        )

        while True:
            action, alpha_value = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)

            # MDD 업데이트
            agent.actor.current_mdd = info.get("current_mdd", 0.0)

            # Replay Buffer에 저장
            agent.replay_buffer.push(state, action, reward, next_state, done)

            # 학습
            if len(agent.replay_buffer) >= self.config.MIN_BUFFER_SIZE:
                agent.train(batch_size=self.config.BATCH_SIZE)

            episode_reward += reward
            episode_returns.append(info.get("return", 0.0))
            episode_values.append(
                info.get("portfolio_value", self.config.INITIAL_CAPITAL)
            )

            state = next_state

            if done:
                break

        # 🔥 추가: 디버깅
        if episode < 3:
            print(f"\n[DEBUG] Episode {episode + 1}:")
            print(f"  Values count: {len(episode_values)}")
            print(f"  First 5 values: {episode_values[:5]}")
            print(f"  Last 5 values: {episode_values[-5:]}")
            print(f"  Min value: {min(episode_values):.0f}")
            print(f"  Max value: {max(episode_values):.0f}")

            values_array = np.array(episode_values)
            peak = np.maximum.accumulate(values_array)
            drawdowns = (values_array - peak) / peak
            print(f"  Min drawdown: {min(drawdowns):.4f}")
            print(f"  MDD: {abs(min(drawdowns)) * 100:.2f}%\n")

        # 성과 계산
        metrics = self._calculate_episode_metrics(episode_returns, episode_values)
        metrics["reward"] = episode_reward
        metrics["alpha"] = alpha_value

        return metrics

    @staticmethod
    def _calculate_episode_metrics(returns: List[float], values: List[float]) -> Dict:
        """에피소드 성과 지표 계산"""
        returns_array = np.array(returns)
        returns_array = returns_array[~np.isnan(returns_array)]

        avg_return = np.mean(returns_array) if len(returns_array) > 0 else 0.0

        # Sharpe Ratio
        if len(returns_array) > 1:
            mean_ret = np.mean(returns_array)
            std_ret = np.std(returns_array)
            sharpe = mean_ret / std_ret if std_ret > 1e-8 else 0.0
        else:
            sharpe = 0.0

        # MDD
        values_array = np.array(values)
        if len(values_array) > 0:
            peak = np.maximum.accumulate(values_array)
            drawdowns = (values_array - peak) / peak
            mdd = abs(min(drawdowns)) if len(drawdowns) > 0 else 0.0
        else:
            mdd = 0.0

        return {
            "avg_return": float(avg_return),
            "sharpe": float(sharpe),
            "mdd": float(mdd),
        }

    def _save_best_model(self, agent: HybridAgent):
        """최고 성과 모델 저장"""
        torch.save(agent.actor.state_dict(), self.config.MODEL_SAVE_PATH)

    def _log_episode(self, episode: int, result: Dict, no_improve_count: int):
        """에피소드 로그 출력"""
        print(
            f"{episode + 1:3d}/{self.config.MAX_EPISODES} | "
            f"Reward: {result['reward']:7.2f} | "
            f"Return: {result['avg_return'] * 100:5.2f}% | "
            f"MDD: {result['mdd'] * 100:5.2f}% | "
            f"Sharpe: {result['sharpe']:.3f} | "
            f"Alpha: {result['alpha']:.3f} | "
            f"No Improve: {no_improve_count}/{self.config.EARLY_STOPPING_PATIENCE}"
        )

    def finetune(self, agent: HybridAgent, env: HybridPortfolioEnv):
        """출력 레이어 Fine-tuning"""
        print(f"\n[Fine-tuning] Training output layers on test data...")

        for episode in range(self.config.FINETUNE_EPISODES):
            state = env.reset()
            episode_reward = 0

            while True:
                action, _ = agent.select_action(
                    state, noise_std=self.config.FINETUNE_NOISE_STD
                )
                next_state, reward, done, info = env.step(action)

                agent.actor.current_mdd = info.get("current_mdd", 0.0)
                agent.replay_buffer.push(state, action, reward, next_state, done)

                if len(agent.replay_buffer) >= self.config.FINETUNE_BUFFER_SIZE:
                    agent.train(batch_size=self.config.FINETUNE_BATCH_SIZE)

                episode_reward += reward
                state = next_state

                if done:
                    break

            if (episode + 1) % 5 == 0:
                print(
                    f"  Episode {episode + 1}/{self.config.FINETUNE_EPISODES}: Reward = {episode_reward:.2f}"
                )

        print("✅ Fine-tuning completed\n")
        agent.actor.eval()


# ==========================================
# Backtesting
# ==========================================
class BacktestRunner:
    """백테스트 실행 관리"""

    def __init__(self, config: Config):
        self.config = config

    def run_hybrid_strategy(
        self, agent: HybridAgent, dataset: HybridDataset, frequency: str
    ) -> Dict:
        """Hybrid 리밸런싱 전략 실행"""
        interval = self.config.REBALANCE_FREQUENCIES[frequency]
        test_windows = dataset.get_test_windows()
        test_symbols = dataset.test_symbols

        portfolio = Portfolio(self.config.INITIAL_CAPITAL, len(test_symbols))
        trade_logs = []

        for i, window in enumerate(test_windows):
            state = dataset.get_state(test_windows, i)

            # MDD 계산 및 업데이트
            current_mdd = portfolio.calculate_mdd()
            agent.actor.current_mdd = current_mdd

            # 리밸런싱 또는 홀딩
            if i % interval == 0:
                action, alpha_value = agent.select_action(state, noise_std=0.0)
                portfolio.rebalance(action)
                trade_type = "Rebalance"
            else:
                action = portfolio.weights
                alpha_value = 0.5
                trade_type = "Hold"

            # 수익률 적용
            returns = window["labels"]
            portfolio.update(returns)

            # 로그 기록
            trade_logs.append(
                self._create_trade_log(
                    date=window["date"],
                    strategy=f"Hybrid_{frequency}",
                    trade_type=trade_type,
                    weights=action,
                    alpha=alpha_value,
                    symbols=test_symbols,
                )
            )

        # 성과 지표 계산
        metrics = calculate_metrics(
            portfolio.get_time_series(), portfolio.dates, f"Hybrid_{frequency}"
        )

        return {
            "dates": portfolio.dates,
            "portfolio_values": portfolio.values,
            "alphas": portfolio.alphas,
            "metrics": metrics,
            "trade_logs": trade_logs,
        }

    def run_fixed_strategy(
        self, dataset: HybridDataset, strategy_name: str = "1/N Buy & Hold"
    ) -> Dict:
        """고정 비중 전략 (벤치마크)"""
        test_windows = dataset.get_test_windows()
        test_symbols = dataset.test_symbols

        num_stocks = len(test_symbols)
        portfolio = Portfolio(self.config.INITIAL_CAPITAL, num_stocks)
        portfolio.rebalance(np.ones(num_stocks) / num_stocks)

        trade_logs = []

        for i, window in enumerate(test_windows):
            if i == 0:
                trade_logs.append(
                    self._create_trade_log(
                        date=window["date"],
                        strategy=strategy_name,
                        trade_type="Init",
                        weights=portfolio.weights,
                        alpha=None,
                        symbols=test_symbols,
                    )
                )

            returns = window["labels"]
            portfolio.update(returns)

        metrics = calculate_metrics(
            portfolio.get_time_series(), portfolio.dates, strategy_name
        )

        return {
            "dates": portfolio.dates,
            "portfolio_values": portfolio.values,
            "metrics": metrics,
            "trade_logs": trade_logs,
        }

    @staticmethod
    def _create_trade_log(
        date: str,
        strategy: str,
        trade_type: str,
        weights: np.ndarray,
        alpha: Optional[float],
        symbols: List[str],
    ) -> Dict:
        """거래 로그 생성"""
        log = {"Date": date, "Strategy": strategy, "Type": trade_type}

        if alpha is not None:
            log["Alpha"] = round(alpha, 3)

        for symbol, weight in zip(symbols, weights):
            log[symbol] = round(float(weight), 4)

        return log


# ==========================================
# Portfolio Helper
# ==========================================
class Portfolio:
    """포트폴리오 관리 헬퍼 클래스"""

    def __init__(self, initial_capital: float, num_stocks: int):
        self.capital = initial_capital
        self.peak = initial_capital
        self.num_stocks = num_stocks
        self.weights = np.ones(num_stocks) / num_stocks

        # 기록
        self.values = [initial_capital]
        self.returns = []
        self.drawdowns = []
        self.turnovers = []
        self.alphas = []
        self.dates = []

    def rebalance(self, new_weights: np.ndarray):
        """포트폴리오 리밸런싱"""
        self.weights = new_weights

    def update(self, asset_returns: np.ndarray):
        """자산 수익률 적용 및 포트폴리오 업데이트"""
        portfolio_return = np.dot(self.weights, asset_returns)
        self.capital *= 1 + portfolio_return

        self.peak = max(self.peak, self.capital)
        drawdown = (self.capital - self.peak) / self.peak

        self.values.append(self.capital)
        self.returns.append(float(portfolio_return))
        self.drawdowns.append(float(drawdown))
        self.turnovers.append(0)

    def calculate_mdd(self) -> float:
        """최근 12개월 MDD 계산"""
        if len(self.values) < 12:
            return 0.0

        recent_values = np.array(self.values[-12:])
        peak_values = np.maximum.accumulate(recent_values)
        drawdowns = (recent_values - peak_values) / peak_values

        return abs(min(drawdowns))

    def get_time_series(self) -> Dict:
        """시계열 데이터 반환"""
        return {
            "portfolio_value": self.values,
            "return": self.returns,
            "drawdown": self.drawdowns,
            "turnover": self.turnovers,
        }


# ==========================================
# Main Workflow
# ==========================================
class WorkflowManager:
    """전체 워크플로우 관리"""

    def __init__(self, config: Config):
        self.config = config
        self.data_manager = DataManager(config)
        self.model_manager = ModelManager(config)
        self.trainer = Trainer(config)
        self.backtest_runner = BacktestRunner(config)

    def run_training_mode(self):
        """학습 모드 실행"""
        print("\n" + "=" * 70)
        print("🎯 TRAINING MODE")
        print("=" * 70)

        # 데이터 준비
        train_df, test_df = self.data_manager.load_data()
        train_df, test_df, feature_cols = self.data_manager.prepare_features(
            train_df, test_df
        )
        dataset = self.data_manager.create_dataset(train_df, test_df, feature_cols)

        # 모델 생성
        num_stocks = len(dataset.train_symbols)
        num_features = len(feature_cols)
        agent = self.model_manager.create_agent(num_stocks, num_features)

        # 학습 환경 생성
        train_windows = dataset.get_train_windows()
        train_env = HybridPortfolioEnv(dataset, windows=train_windows)

        # 학습 실행
        monitor = TrainingMonitor(
            save_dir=self.config.RESULTS_DIR / "training_logs", save_interval=50
        )
        self.trainer.train(agent, train_env, monitor)

        # 모델 저장
        self.model_manager.save_model(agent)

        print("\n✅ Training completed successfully!")

    def run_comparison_mode(self):
        """비교 모드 실행 (백테스트)"""
        print("\n" + "=" * 70)
        print("📊 COMPARISON MODE")
        print("=" * 70)

        # 데이터 준비
        train_df, test_df = self.data_manager.load_data()
        train_df, test_df, feature_cols = self.data_manager.prepare_features(
            train_df, test_df
        )
        dataset = self.data_manager.create_dataset(train_df, test_df, feature_cols)

        # 테스트 에이전트 생성
        num_stocks_test = len(dataset.test_symbols)
        num_features = len(feature_cols)
        test_agent = self.model_manager.create_agent(num_stocks_test, num_features)

        # 모델 로드 (Encoder만)
        self.model_manager.load_model(test_agent, encoder_only=True)

        # Fine-tuning
        test_windows = dataset.get_test_windows()
        test_env = HybridPortfolioEnv(dataset, windows=test_windows)
        self.trainer.finetune(test_agent, test_env)

        # 백테스트 실행
        print("[Backtesting] Running strategies on 2021-2025 data...")
        results = self._run_all_strategies(test_agent, dataset)

        # 결과 시각화 및 저장
        self._save_and_visualize_results(results)

        print("\n✅ Comparison completed successfully!")

    def _run_all_strategies(self, agent: HybridAgent, dataset: HybridDataset) -> Dict:
        """모든 전략 실행"""
        return {
            "buy_and_hold": self.backtest_runner.run_fixed_strategy(dataset),
            "monthly": self.backtest_runner.run_hybrid_strategy(
                agent, dataset, "monthly"
            ),
            "quarterly": self.backtest_runner.run_hybrid_strategy(
                agent, dataset, "quarterly"
            ),
            "semiannual": self.backtest_runner.run_hybrid_strategy(
                agent, dataset, "semiannual"
            ),
            "annual": self.backtest_runner.run_hybrid_strategy(
                agent, dataset, "annual"
            ),
        }

    def _save_and_visualize_results(self, results: Dict):
        """결과 저장 및 시각화"""
        # 시각화
        visualizer = BacktestVisualizer(save_dir=self.config.RESULTS_DIR)
        visualizer.plot_rebalancing_comparison(
            results["buy_and_hold"],
            results["monthly"],
            results["quarterly"],
            results["semiannual"],
            results["annual"],
        )

        # 성과 요약 저장
        summary_data = [res["metrics"] for res in results.values()]
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(self.config.RESULTS_DIR / "summary_metrics.csv", index=False)

        # 거래 로그 저장
        all_logs = []
        for res in results.values():
            all_logs.extend(res["trade_logs"])
        pd.DataFrame(all_logs).to_csv(
            self.config.RESULTS_DIR / "hybrid_trade_logs.csv", index=False
        )

        # 콘솔 출력
        self._print_summary(results)

    @staticmethod
    def _print_summary(results: Dict):
        """성과 요약 출력"""
        print("\n" + "=" * 70)
        print("📊 Final Performance Summary")
        print("=" * 70)

        for key, res in results.items():
            m = res["metrics"]
            print(
                f"{m['Strategy']:20} | "
                f"CAGR: {m['CAGR']:6.1f}% | "
                f"MDD: {m['MDD']:6.1f}% | "
                f"Final: ${m['FinalValue']:,.0f}"
            )


# ==========================================
# Entry Point
# ==========================================
def main(mode: str = "compare"):
    """메인 실행 함수

    Args:
        mode: 'train' 또는 'compare'
    """
    config = Config()
    workflow = WorkflowManager(config)

    if mode == "train":
        workflow.run_training_mode()
    elif mode == "compare":
        workflow.run_comparison_mode()
    else:
        raise ValueError(f"Invalid mode: {mode}. Use 'train' or 'compare'")


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "compare"
    main(mode=mode)
