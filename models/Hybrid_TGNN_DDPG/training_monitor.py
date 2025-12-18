"""
Training Progress Monitor
학습 진행 상황을 실시간으로 추적하고 시각화
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class TrainingMonitor:
    def __init__(self, save_dir, save_interval=50):
        """
        Args:
            save_dir: 결과 저장 디렉토리
            save_interval: 시각화 저장 간격 (에피소드)
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval

        # 학습 기록 저장
        self.episode_rewards = []
        self.episode_returns = []
        self.episode_mdds = []
        self.episode_sharpes = []
        self.episode_alphas = []

    def record_episode(self, episode, reward, avg_return, mdd, sharpe, alpha):
        """에피소드 결과 기록"""
        self.episode_rewards.append(reward)
        self.episode_returns.append(avg_return)
        self.episode_mdds.append(mdd)
        self.episode_sharpes.append(sharpe)
        self.episode_alphas.append(alpha)

        # 저장 간격마다 시각화
        if (episode + 1) % self.save_interval == 0:
            self.plot_progress(episode + 1)

    def plot_progress(self, current_episode):
        """학습 진행 상황 시각화"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Hybrid TGNN-DDPG Training Progress", fontsize=16, fontweight="bold"
        )

        # 1. Episode Reward
        ax = axes[0, 0]
        ax.plot(self.episode_rewards, "b-", linewidth=1.5)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Episode Reward")
        ax.grid(True, alpha=0.3)

        # 2. Average Return (%)
        ax = axes[0, 1]
        returns_pct = [r * 100 for r in self.episode_returns]
        ax.plot(returns_pct, "g-", linewidth=1.5)
        ax.axhline(y=0, color="r", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Return (%)")
        ax.set_title("Average Return (%)")
        ax.grid(True, alpha=0.3)

        # 3. Maximum Drawdown (%)
        ax = axes[0, 2]
        mdds_pct = [m * 100 for m in self.episode_mdds]
        ax.plot(mdds_pct, "r-", linewidth=1.5)
        ax.axhline(y=10, color="orange", linestyle="--", alpha=0.7, label="10% Target")
        ax.set_xlabel("Episode")
        ax.set_ylabel("MDD (%)")
        ax.set_title("Maximum Drawdown (%)")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 4. Sharpe Ratio
        ax = axes[1, 0]
        ax.plot(self.episode_sharpes, "m-", linewidth=1.5)
        ax.axhline(y=0.5, color="orange", linestyle="--", alpha=0.7, label="0.5 Target")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Sharpe")
        ax.set_title("Sharpe Ratio")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 5. Ensemble Alpha
        ax = axes[1, 1]
        ax.plot(self.episode_alphas, "c-", linewidth=1.5)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, label="Balanced")
        ax.axhline(y=0.2, color="pink", linestyle="--", alpha=0.7, label="Min")
        ax.axhline(y=0.8, color="gray", linestyle="--", alpha=0.7, label="Max")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Alpha (TGNN Weight)")
        ax.set_title("Ensemble Alpha")
        ax.set_ylim(0, 1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 6. Summary Stats (텍스트 박스)
        ax = axes[1, 2]
        ax.axis("off")

        # 최근 10개 에피소드 평균
        recent_n = min(10, len(self.episode_rewards))
        recent_reward = np.mean(self.episode_rewards[-recent_n:])
        recent_return = np.mean(self.episode_returns[-recent_n:]) * 100
        recent_mdd = np.mean(self.episode_mdds[-recent_n:]) * 100
        recent_sharpe = np.mean(self.episode_sharpes[-recent_n:])
        recent_alpha = np.mean(self.episode_alphas[-recent_n:])

        # 최고 성과
        best_reward = max(self.episode_rewards)
        best_return = max(self.episode_returns) * 100
        best_mdd = min(self.episode_mdds) * 100
        best_sharpe = max(self.episode_sharpes)

        summary_text = f"""Training Progress (Episode {current_episode})

Reward:
  Last {recent_n} avg: {recent_reward:.2f}
  Max: {best_reward:.2f}

Return (%):
  Last {recent_n} avg: {recent_return:.2f}%
  Best: {best_return:.2f}%

MDD (%):
  Last {recent_n} avg: {recent_mdd:.2f}%
  Best: {best_mdd:.2f}%

Sharpe:
  Last {recent_n} avg: {recent_sharpe:.3f}
  Best: {best_sharpe:.3f}

Alpha:
  Last {recent_n} avg: {recent_alpha:.3f}
  Range: [{min(self.episode_alphas):.3f}, {max(self.episode_alphas):.3f}]
"""

        ax.text(
            0.1,
            0.5,
            summary_text,
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        plt.tight_layout()
        save_path = self.save_dir / f"training_progress_ep{current_episode}.jpg"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Training progress saved: {save_path.name}")
