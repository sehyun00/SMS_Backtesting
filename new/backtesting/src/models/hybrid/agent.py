"""
HybridAgent - Composition Based DDPG + TGNN Ensemble

기존 독립 구현에서 DDPG와 TGNN 인스턴스를 조합하는 방식으로 변경.
이를 통해 코드 중복 제거 및 공정한 Ablation Study 가능.
"""

import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from typing import Dict, Any, Tuple

from src.models.base_model import BaseModel
from src.models.ddpg.agent import DDPGAgent
from src.models.tgnn.model import TGNN
from src.models.layers import GlobalPoolHead


class HybridAgent(BaseModel):
    """
    Hybrid TGNN-DDPG Agent (Composition Based).

    구조:
    - self.ddpg: DDPGAgent 인스턴스 (Actor-Critic)
    - self.tgnn: TGNN 인스턴스 (Multi-Head Predictor)
    - self.ensemble_net: 앙상블 alpha 학습 네트워크

    출력:
    - final_weights = alpha * tgnn_weights + (1-alpha) * ddpg_weights
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)  # BaseModel에 config 전달
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters
        self.gamma = config["training"].get("gamma", 0.99)
        self.tau = config["training"].get("tau", 0.005)
        self.lr_actor = config["training"].get("lr_actor", 1e-4)
        self.lr_critic = config["training"].get("lr_critic", 1e-3)
        self.batch_size = config["training"].get("batch_size", 64)
        self.temperature = config["model"].get("softmax_temperature", 1.0)

        # Alpha 제약 (config에서 읽기)
        self.alpha_min = config["model"].get("hybrid_alpha_min", 0.2)
        self.alpha_max = config["model"].get("hybrid_alpha_max", 0.8)
        self.alpha_mode = config["model"].get(
            "hybrid_alpha_mode", "fixed"
        )  # "fixed" or "dynamic"
        self.horizon_dim = config["model"].get("hybrid_horizon_dim", 8)

        # ============================================================
        # Composition: DDPG와 TGNN 인스턴스 사용
        # ============================================================
        self.ddpg = DDPGAgent(config)
        self.tgnn = TGNN(config)

        # TGNN 체크포인트 로드 (Brain Transplant)
        # Random Weight로 초기화된 TGNN은 노이즈만 생성하므로, 사전 학습된 가중치를 필수적으로 로드해야 함.
        self._load_tgnn_checkpoint()

        # 리밸런싱 주기 임베딩 (동적 Alpha용)
        # 0: monthly, 1: quarterly, 2: semiannual, 3: annual
        self.horizon_embedding = nn.Embedding(4, self.horizon_dim)

        # 앙상블 레이어: Global Pooling으로 alpha 계산
        # Input: DDPG hidden (64) + TGNN embedding (64) + scores (2) + horizon (8) = 138
        hidden_dim = 128
        tgnn_emb_dim = 64  # TGNN Node Embedding (Fixed to 64 in TGNN)
        if "factors" in config["data"] and config["data"]["factors"]:
            tgnn_emb_dim += 32  # Add Macro Embedding Dimension
        ddpg_emb_dim = 64  # DDPG encoder 출력
        if self.alpha_mode == "dynamic":
            ensemble_input_dim = tgnn_emb_dim + ddpg_emb_dim + 2 + self.horizon_dim
        else:
            ensemble_input_dim = tgnn_emb_dim + ddpg_emb_dim + 2  # +2 for scores

        self.ensemble_net = GlobalPoolHead(
            input_dim=ensemble_input_dim, hidden_dim=hidden_dim, output_dim=1
        )

        # Ensemble optimizer (ensemble_net + horizon_embedding 함께 학습)
        ensemble_params = list(self.ensemble_net.parameters()) + list(
            self.horizon_embedding.parameters()
        )
        self.ensemble_optimizer = optim.Adam(ensemble_params, lr=self.lr_actor)

        # Replay Buffer는 DDPG의 buffer를 사용 (Composition 일관성)
        # self.buffer property로 접근

        self.to(self.device)

    @property
    def buffer(self):
        """DDPG Buffer에 대한 접근자 (학습 시 Trainer와 일치)"""
        return self.ddpg.buffer

    @property
    def actor(self):
        """DDPG Actor에 대한 접근자 (백테스팅 호환)"""
        return self.ddpg.actor

    def forward(
        self,
        x: torch.Tensor,
        adj: torch.Tensor,
        target_head: str = "Momentum1M",
        horizon: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for prediction/inference.

        Args:
            x: [Batch, N, T, F]
            adj: [Batch, N, N]
            target_head: TGNN 예측 헤드 (default: Momentum1M)
            horizon: 리밸런싱 주기 (0=monthly, 1=quarterly, 2=semiannual, 3=annual)

        Returns:
            final_weights: [Batch, N] 앙상블된 포트폴리오 비중
            alpha: [Batch, 1] TGNN 비중
        """
        batch = x.shape[0]
        N = x.shape[1]

        # --- DDPG Path ---
        ddpg_weights, _ = self.ddpg.actor(x)  # [Batch, N]

        # DDPG 중간 임베딩 가져오기 (encoder 출력)
        if x.dim() == 4:
            x_flat = x.reshape(batch, N, -1)
        else:
            x_flat = x
        ddpg_emb = self.ddpg.actor.encoder(x_flat)  # [Batch, N, 64]

        # --- TGNN Path ---
        # Context-Aware: Split features into Prices and Macro
        num_price = 5  # OHLCV
        prices = x[:, :, :, :num_price]  # [B, N, T, 5]
        macro = x[:, :, :, num_price:]  # [B, N, T, 5]

        tgnn_weights, tgnn_emb = self.tgnn.get_portfolio_weights(
            prices.to(self.tgnn.device),
            adj.to(self.tgnn.device),
            macro=macro.to(self.tgnn.device),
            target_head=target_head,
            temperature=self.temperature,
        )  # [Batch, N], [Batch, N, 64]

        # --- Ensemble ---
        if self.alpha_mode == "dynamic":
            # 동적 Alpha: horizon 임베딩 추가
            horizon_tensor = torch.tensor([horizon], device=self.device)
            horizon_emb = self.horizon_embedding(horizon_tensor)  # [1, horizon_dim]
            # horizon_emb를 [Batch, N, horizon_dim]으로 확장
            horizon_emb = horizon_emb.unsqueeze(0).expand(batch, N, -1)

            ensemble_in = torch.cat(
                [
                    ddpg_emb,
                    tgnn_emb,
                    ddpg_weights.unsqueeze(-1),
                    tgnn_weights.unsqueeze(-1),
                    horizon_emb,
                ],
                dim=-1,
            )  # [Batch, N, 64+64+2+horizon_dim]
        else:
            # 고정 Alpha: 기존 로직
            ensemble_in = torch.cat(
                [
                    ddpg_emb,
                    tgnn_emb,
                    ddpg_weights.unsqueeze(-1),
                    tgnn_weights.unsqueeze(-1),
                ],
                dim=-1,
            )  # [Batch, N, 64+64+2]

        # Global Pooling -> Alpha
        raw_alpha = self.ensemble_net(ensemble_in)  # [Batch, 1]
        alpha = torch.sigmoid(raw_alpha)
        alpha = torch.clamp(alpha, self.alpha_min, self.alpha_max)

        # Mixing
        final_weights = alpha * tgnn_weights + (1 - alpha) * ddpg_weights

        # 최종 정규화
        final_weights = final_weights / (final_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return final_weights, alpha

    def predict(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Prediction helper for evaluation.
        """
        self.eval()
        with torch.no_grad():
            x = batch["features"].to(self.device)
            adj = batch["adj_matrix"].to(self.device)
            weights, _ = self.forward(x, adj)
        return weights.cpu()

    def select_action(
        self, state_feat: np.ndarray, state_adj: np.ndarray, noise_std: float = 0.1
    ) -> np.ndarray:
        """
        Select action with Dirichlet noise for exploration.
        """
        self.eval()
        with torch.no_grad():
            feat_tensor = torch.FloatTensor(state_feat).unsqueeze(0).to(self.device)
            adj_tensor = torch.FloatTensor(state_adj).unsqueeze(0).to(self.device)

            weights, _ = self.forward(feat_tensor, adj_tensor)
            action = weights.cpu().numpy()[0]

        if noise_std > 0:
            concentration = action / (noise_std + 1e-8)
            concentration = np.clip(concentration, 0.1, 100.0)
            action = np.random.dirichlet(concentration)

        return action

    def update(self):
        """
        Perform one step of Actor-Critic update using Replay Buffer.
        1. DDPG 업데이트 (Actor + Critic)
        2. Ensemble 네트워크 업데이트 (alpha 학습)
        """
        # DDPG 업데이트
        ddpg_result = self.ddpg.update()

        if ddpg_result is None:
            return None

        # ============================================================
        # Ensemble 네트워크 업데이트 (alpha 학습)
        # 목표: 높은 Q-value를 받는 포트폴리오 비중을 학습
        # ============================================================
        if len(self.buffer) < self.batch_size:
            return ddpg_result

        # Sample Batch (DDPG와 동일한 버퍼 사용)
        states, _, _, _, _ = self.buffer.sample(self.batch_size)
        states = torch.FloatTensor(states).to(self.device)  # [B, N, T, F]

        # Adjacency Matrix 생성 (완전 연결 그래프)
        batch_size = states.shape[0]
        N = states.shape[1]
        adj = torch.ones(batch_size, N, N).to(self.device)

        # Hybrid Forward (alpha 계산 포함)
        self.train()
        final_weights, alpha = self.forward(states, adj)

        # DDPG Critic을 사용하여 Q-value 계산
        # 목표: Q-value를 최대화하는 alpha 학습
        q_value = self.ddpg.critic(states, final_weights)

        # Ensemble Loss: -Q (maximize Q)
        ensemble_loss = -q_value.mean()

        # Alpha 정규화: 극단적인 값 방지 (0.5 근처로 유도)
        alpha_reg = 0.01 * ((alpha - 0.5) ** 2).mean()
        total_ensemble_loss = ensemble_loss + alpha_reg

        # Optimizer Step
        self.ensemble_optimizer.zero_grad()
        total_ensemble_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ensemble_net.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(self.horizon_embedding.parameters(), 1.0)
        self.ensemble_optimizer.step()

        return {
            "critic_loss": ddpg_result["critic_loss"],
            "actor_loss": ddpg_result["actor_loss"],
            "ensemble_loss": total_ensemble_loss.item(),
            "alpha_mean": alpha.mean().item(),
        }

    def _load_tgnn_checkpoint(self):
        """
        TGNN 사전 학습 모델을 로드합니다.
        Config의 'tgnn_checkpoint'가 'auto'이면 최신 모델을 자동 탐색합니다.
        """
        import os
        import glob

        ckpt_path = (
            self.config.get("model", {})
            .get("hybrid", {})
            .get("tgnn_checkpoint", "auto")
        )

        if ckpt_path == "auto":
            # 자동 탐색: results/tgnn/checkpoints/best_model_*.pth
            # 주의: config 구조에 따라 results_dir 경로 추론 필요
            base_dir = self.config["paths"]["results_dir"]
            # tgnn 폴더가 하드코딩 되어있다고 가정 (Trainer에서 생성)
            ckpt_dir = os.path.join(base_dir, "tgnn", "checkpoints")

            if not os.path.exists(ckpt_dir):
                print(
                    f"⚠️ 경고: TGNN 체크포인트 디렉토리를 찾을 수 없습니다: {ckpt_dir}"
                )
                print(
                    "   TGNN이 'HybridAgent' 내부에서 초기화된 상태로 시작합니다 (Random Weights)."
                )
                return

            # 파일 리스트 (최신순 정렬)
            files = glob.glob(os.path.join(ckpt_dir, "best_model_*.pth"))
            if not files:
                print(f"⚠️ 경고: {ckpt_dir} 경로에 TGNN 체크포인트 파일이 없습니다.")
                return

            # 수정시간 기준 정렬 (최신 파일)
            latest_ckpt = max(files, key=os.path.getmtime)
            ckpt_path = latest_ckpt
            print(f"✅ TGNN 체크포인트 자동 탐색 성공: {ckpt_path}")

        # 로드 실행
        if ckpt_path and os.path.exists(ckpt_path):
            try:
                state_dict = torch.load(ckpt_path, map_location=self.device)
                self.tgnn.load_state_dict(state_dict)
                print(f"✅ TGNN 가중치 로드 완료: {ckpt_path}")

                # 가중치 동결 (Freeze) - Hybrid 학습 중 TGNN 변질 방지
                self.tgnn.eval()
                for param in self.tgnn.parameters():
                    param.requires_grad = False
                print("🔒 TGNN 파라미터 동결 (Freeze) 완료")

            except Exception as e:
                print(f"❌ TGNN 로드 중 오류 발생: {e}")
        else:
            print(
                f"⚠️ 경고: 지정된 TGNN 체크포인트 경로가 유효하지 않습니다: {ckpt_path}"
            )
            print("   TGNN이 Random Weights로 동작합니다 (성능 저하 위험).")

    def save(self, path: str):
        """
        Hybrid Agent 저장 (DDPG, Ensemble Net)
        TGNN은 저장하지 않습니다 (Freeze 상태이므로).
        """
        torch.save(
            {
                "ddpg_actor": self.ddpg.actor.state_dict(),
                "ddpg_critic": self.ddpg.critic.state_dict(),
                "ensemble": self.ensemble_net.state_dict(),
                "config": self.config,
            },
            path,
        )

    def load(self, path: str, strict: bool = True):
        """
        Hybrid Agent 로드
        Args:
            path: 체크포인트 경로
            strict: 엄격한 로딩 여부 (BaseModel 호환성 위해 추가, Hybrid는 수동 로드하므로 무시하거나 참조)
        """
        checkpoint = torch.load(path, map_location=self.device)
        # Hybrid는 각 컴포넌트별로 state_dict가 분리되어 있으므로 strict=strict로 전달
        self.ddpg.actor.load_state_dict(checkpoint["ddpg_actor"], strict=strict)
        self.ddpg.critic.load_state_dict(checkpoint["ddpg_critic"], strict=strict)
        self.ensemble_net.load_state_dict(checkpoint["ensemble"], strict=strict)
        print(f"✅ Hybrid Agent 로드 완료 (Strict={strict}): {path}")
