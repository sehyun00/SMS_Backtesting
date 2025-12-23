"""
DDPG Phase 3: Advanced Learning Strategies
- Prioritized Experience Replay (PER)
- Curriculum Learning (easy -> hard)
- Adaptive Noise Scheduling
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
from collections import deque

warnings.filterwarnings("ignore")

from model_phase3 import (
    DDPGAgent,
    CurriculumScheduler,
    AdaptiveNoiseScheduler,
)

# 프로젝트 루트
ROOT_DIR = Path(__file__).parent.parent.parent
TRAIN_DATA_PATH = ROOT_DIR / "data" / "train_data.csv"
TEST_DATA_PATH = ROOT_DIR / "data" / "test_data.csv"

# 한글 폰트
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False


# ============ DDPG 전용 데이터셋 (기존과 동일) ============


class DDPGDataset:
    def __init__(self, train_df, test_df, window_size=12, feature_cols=None):
        self.window_size = window_size
        self.feature_cols = feature_cols

        # 결측치 처리
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

        self.train_monthly["Return_Raw"] = self.train_monthly["Momentum1M"].copy()
        self.test_monthly["Return_Raw"] = self.test_monthly["Momentum1M"].copy()

        self.norm_cols = [c for c in self.feature_cols if c != "Momentum1M"]

        self._fit_scaler_on_train_data()

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


# ============ 포트폴리오 환경 (Phase 1 개선 적용) ============


class PortfolioEnv:
    """Phase 1 개선사항 포함"""

    def __init__(self, dataset, windows, symbols, features, initial_cash=1_000_000):
        self.dataset = dataset
        self.windows = windows
        self.initial_cash = initial_cash
        self.symbols = symbols
        self.features = features
        self.n_steps = len(self.windows)
        self.current_step = 0
        self.portfolio_value = initial_cash

        self.n_stocks = len(self.symbols)
        self.prev_weights = np.zeros(self.n_stocks)

        self.state_dim = self.windows[0]["features"][:, -1, :].flatten().shape[0]

        self.gamma = 2.0
        self.cost_bps = 0.0005

        self.return_history = deque(maxlen=12)
        self.volatility_window = 6

    def reset(self):
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history.clear()
        return self._get_state(0)

    def _get_state(self, idx):
        w = self.windows[idx]
        state = w["features"][:, -1, :].flatten()
        return state.astype(np.float32)

    def _calculate_portfolio_volatility(self):
        if len(self.return_history) < 2:
            return 0.0
        returns = np.array(list(self.return_history))
        return np.std(returns)

    def _calculate_concentration_penalty(self, weights):
        hhi = np.sum(weights**2)
        ideal_hhi = 1.0 / self.n_stocks
        concentration = max(0, hhi - ideal_hhi * 1.5)
        return concentration * 10

    def step(self, action):
        window = self.windows[self.current_step]
        returns = window["labels"]

        portfolio_return_pct = np.dot(action, returns)
        portfolio_return = portfolio_return_pct / 100.0

        self.return_history.append(portfolio_return_pct)

        turnover = np.sum(np.abs(action - self.prev_weights))
        transaction_cost = turnover * self.cost_bps

        net_return = portfolio_return - transaction_cost
        self.portfolio_value *= 1 + net_return

        self.current_step += 1
        done = self.current_step >= self.n_steps

        # Phase 1 개선된 보상 함수
        base_reward = portfolio_return_pct
        cost_penalty = turnover * self.cost_bps * 10000
        volatility = self._calculate_portfolio_volatility()
        risk_penalty = self.gamma * (volatility**2)
        concentration_penalty = self._calculate_concentration_penalty(action)

        reward = base_reward - cost_penalty - risk_penalty - concentration_penalty
        reward = np.clip(reward, -100, 100)

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
            "raw_return": portfolio_return_pct,
            "volatility": volatility,
            "risk_penalty": risk_penalty,
            "concentration": concentration_penalty,
        }

        return next_state, reward, done, info


# ============ 🔥 Phase 3: Curriculum Learning 적용 학습 ============


def train_ddpg_with_curriculum(
    agent,
    dataset,
    train_windows,
    symbols,
    features,
    curriculum_stages=5,
    episodes_per_stage=100,
    use_adaptive_noise=True,
):
    """
    Curriculum Learning + Adaptive Noise를 적용한 학습
    
    Args:
        curriculum_stages: 커리큘럼 단계 수 (쉬운 -> 어려운)
        episodes_per_stage: 각 단계별 에피소드 수
        use_adaptive_noise: 적응형 노이즈 사용 여부
    """
    print("\n" + "=" * 80)
    print("🔥 Phase 3: Curriculum Learning + Adaptive Noise")
    print("=" * 80)
    
    # 커리큘럼 스케줄러 생성
    curriculum = CurriculumScheduler(train_windows, stages=curriculum_stages)
    
    # 적응형 노이즈 스케줄러
    if use_adaptive_noise:
        noise_scheduler = AdaptiveNoiseScheduler(
            initial_noise=0.3,
            min_noise=0.05,
            max_noise=0.5,
            patience=20
        )
    
    all_rewards = []
    best_reward = -float('inf')
    best_model_state = None
    
    # 단계별 학습
    for stage in range(curriculum_stages):
        print(f"\n{'='*60}")
        print(f"📚 Curriculum Stage {stage + 1}/{curriculum_stages}")
        print(f"{'='*60}")
        
        # 현재 단계의 학습 윈도우 가져오기
        stage_windows = curriculum.get_windows_for_stage(stage)
        print(f"   학습 윈도우: {len(stage_windows)}개월")
        
        # 환경 생성
        env = PortfolioEnv(dataset, stage_windows, symbols, features)
        
        # 현재 단계 학습
        for episode in range(episodes_per_stage):
            state = env.reset()
            episode_reward = 0
            
            # 노이즈 결정
            if use_adaptive_noise:
                noise_std = noise_scheduler.get_noise()
            else:
                # 선형 감소
                progress = (stage * episodes_per_stage + episode) / \
                          (curriculum_stages * episodes_per_stage)
                noise_std = max(0.05, 0.3 - 0.25 * progress)
            
            # 에피소드 실행
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
            
            all_rewards.append(episode_reward)
            
            # 적응형 노이즈 업데이트
            if use_adaptive_noise:
                noise_scheduler.update(episode_reward)
            
            # 최고 성능 모델 저장
            if episode_reward > best_reward:
                best_reward = episode_reward
                best_model_state = agent.actor.state_dict()
            
            # 로깅
            global_episode = stage * episodes_per_stage + episode + 1
            total_episodes = curriculum_stages * episodes_per_stage
            
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(all_rewards[-20:])
                print(
                    f"  [{global_episode:3d}/{total_episodes}] "
                    f"Stage {stage+1} | Ep {episode+1:3d}/{episodes_per_stage} | "
                    f"Reward: {episode_reward:+8.2f} | "
                    f"Avg20: {avg_reward:+8.2f} | "
                    f"Noise: {noise_std:.3f} | "
                    f"Best: {best_reward:+8.2f}"
                )
    
    # 최고 모델 로드
    if best_model_state is not None:
        agent.actor.load_state_dict(best_model_state)
        print(f"\n✅ Best model loaded (Reward: {best_reward:+.2f})")
    
    return all_rewards, best_reward


# ============ Fine-tuning (기존과 동일) ============


def fine_tune_ddpg(agent, env, num_episodes=50):
    print(f"\n=== Fine-tuning 시작 ({num_episodes} episodes) ===")
    episode_rewards = []
    noise_std = 0.1

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0

        while True:
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, done, info = env.step(action)
            agent.replay_buffer.push(state, action, reward, next_state, done)

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


# ============ 학습 곡선 시각화 ============


def plot_phase3_training_curve(train_rewards, finetune_rewards, curriculum_stages, save_dir):
    """Phase 3 학습 진행 상황 시각화 (커리큘럼 단계 표시)"""
    plt.figure(figsize=(14, 7))
    
    episodes_per_stage = len(train_rewards) // curriculum_stages
    
    # Main Training
    plt.plot(
        range(1, len(train_rewards) + 1),
        train_rewards,
        label="Training (Curriculum)",
        color="#2E86AB",
        alpha=0.5,
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
            linewidth=2.5,
        )
    
    # 커리큘럼 단계 구분선
    for stage in range(1, curriculum_stages):
        stage_boundary = stage * episodes_per_stage
        plt.axvline(
            x=stage_boundary,
            color="gray",
            linestyle="--",
            alpha=0.5,
            label=f"Stage {stage}" if stage == 1 else ""
        )
    
    # Fine-tuning
    if finetune_rewards:
        ft_start = len(train_rewards) + 1
        ft_x = range(ft_start, ft_start + len(finetune_rewards))
        plt.plot(
            ft_x, finetune_rewards, label="Fine-tuning", color="#F18F01", linewidth=2.5
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
        "Phase 3: DDPG Training with Curriculum Learning + Adaptive Noise",
        fontsize=14,
        fontweight="bold"
    )
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    save_path = save_dir / "phase3_training_curve.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"✅ 학습 곡선 저장: {save_path}")
    plt.close()


# ============ 메인 ============


def main():
    print("=" * 60)
    print("Phase 3: Advanced Learning Strategies")
    print("=" * 60)

    # 데이터 로드
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)

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

    print(f"\n📊 데이터셋 정보:")
    print(f"   종목 수: {num_stocks}")
    print(f"   특성 수: {num_features}")
    print(f"   디바이스: {DEVICE}")

    # 🔥 Phase 3: PER 활성화된 에이전트 생성
    agent = DDPGAgent(
        num_stocks,
        num_features,
        lr_actor=5e-5,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.001,
        entropy_coef=0.01,
        use_per=True,  # 🔥 Prioritized Experience Replay
        device=DEVICE,
    )

    model_path = Path(__file__).parent / "best_ddpg_phase3.pth"
    save_dir = ROOT_DIR / "results" / "03_DDPG_Phase3"
    save_dir.mkdir(parents=True, exist_ok=True)

    # 기존 모델 삭제
    if model_path.exists():
        print("⚠️ 기존 모델 삭제")
        model_path.unlink()

    train_windows = dataset.get_train_windows()
    train_symbols = dataset.train_symbols

    # 🔥 Phase 3: Curriculum Learning + Adaptive Noise
    train_rewards, best_reward = train_ddpg_with_curriculum(
        agent,
        dataset,
        train_windows,
        train_symbols,
        feature_cols,
        curriculum_stages=5,
        episodes_per_stage=80,  # 총 400 에피소드
        use_adaptive_noise=True,
    )

    # Fine-tuning (전체 데이터)
    train_env = PortfolioEnv(dataset, train_windows, train_symbols, feature_cols)
    finetune_rewards = fine_tune_ddpg(agent, train_env, num_episodes=50)

    # 최종 모델 저장
    torch.save(agent.actor.state_dict(), model_path)
    print(f"✅ 모델 저장 완료: {model_path}")

    # 학습 곡선 시각화
    plot_phase3_training_curve(train_rewards, finetune_rewards, 5, save_dir)

    print("\n" + "=" * 60)
    print("✅ Phase 3 학습 완료!")
    print(f"   Best Reward: {best_reward:+.2f}")
    print(f"   총 에피소드: {len(train_rewards) + len(finetune_rewards)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
