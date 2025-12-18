"""
Portfolio Environment for Reinforcement Learning
포트폴리오 관리 강화학습 환경
"""

import numpy as np


class HybridPortfolioEnv:
    """
    강화학습 환경
    - 포트폴리오 리밸런싱 시뮬레이션
    - 리워드 계산 (CRRA 효용, 리스크 조정)
    """

    def __init__(self, dataset, windows=None, initial_cash=1_000_000):
        """
        Args:
            dataset: HybridDataset 인스턴스
            windows: 사용할 윈도우 리스트 (None이면 전체 사용)
            initial_cash: 초기 자본
        """
        self.dataset = dataset
        self.windows = windows if windows else dataset.get_train_windows()
        self.initial_cash = initial_cash
        self.portfolio_value = initial_cash
        self.current_step = 0
        self.n_steps = len(self.windows)
        self.gamma = 2.0  # CRRA 위험 회피 계수
        self.cost_bps = 0.0005  # 거래 비용 (0.05%)
        self.n_stocks = len(self.windows[0]["features"])
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []

    def reset(self):
        """환경 초기화"""
        self.current_step = 0
        self.portfolio_value = self.initial_cash
        self.prev_weights = np.zeros(self.n_stocks)
        self.return_history = []
        return self.dataset.get_state(self.windows, 0)

    def step(self, action):
        """
        한 스텝 진행

        Args:
            action: (N,) 포트폴리오 가중치

        Returns:
            next_state: 다음 상태
            reward: 보상
            done: 에피소드 종료 여부
            info: 추가 정보
        """
        w = self.windows[self.current_step]
        returns = w["labels"]  # % 단위 (Momentum1M)
        features = w["features"]

        # 🔥 1. Action 정규화 및 검증
        action = np.array(action, dtype=np.float64)
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(len(action)) / len(action)

        # 🔥 2. Returns 처리 (nan = 상장폐지 = 0%)
        returns = np.array(returns, dtype=np.float64)

        # nan = 상장폐지/데이터 없음 = 0% (패턴 학습 가능)
        returns = np.nan_to_num(
            returns,
            nan=0.0,  # 상장폐지 = 0% (TGNN 패턴 학습 가능)
            posinf=0.5,  # 극단값 제한
            neginf=-0.5,  # 극단값 제한
        )

        # 데이터 오류 대비 추가 안전장치
        returns = np.clip(returns, -0.95, 2.0)

        # 포트폴리오 수익률 계산
        portfolio_return = float(np.dot(action, returns))
        portfolio_return = np.clip(portfolio_return, -0.8, 1.0)

        # 거래 비용
        turnover = np.sum(np.abs(action - self.prev_weights))
        cost = turnover * self.cost_bps  # 비율 단위 (0.0005 = 0.05%)
        net_return = portfolio_return - cost
        net_return = np.clip(net_return, -0.8, 1.0)

        # 🔥 3. 포트폴리오 가치 업데이트
        self.portfolio_value *= 1 + net_return

        # 안전장치: 최소값 보장 및 nan/inf 방지
        self.portfolio_value = max(self.portfolio_value, 1000.0)
        if np.isnan(self.portfolio_value) or np.isinf(self.portfolio_value):
            self.portfolio_value = self.initial_cash

        self.current_step += 1
        done = self.current_step >= self.n_steps
        self.return_history.append(net_return)

        # 🔥 4. MDD 계산 (전체 에피소드)
        if len(self.return_history) >= 12:
            cumulative_returns = np.cumprod(1 + np.array(self.return_history))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / peak
            self.current_mdd = abs(min(drawdowns))
        else:
            self.current_mdd = 0.0

        # 🔥 5. 리워드 계산
        reward = self._calculate_reward(action, returns, features, net_return, turnover)

        self.prev_weights = action

        next_state = (
            self.dataset.get_state(self.windows, self.current_step)
            if not done
            else np.zeros_like(self.dataset.get_state(self.windows, 0))
        )

        # 🔥 6. Info 딕셔너리
        info = {
            "portfolio_value": self.portfolio_value,
            "return": net_return,
            "date": w["date"],
            "turnover": turnover,
            "cost": cost,
            "concentration": np.sum(action**2),
            "downside_risk": self._calculate_downside_std(),
            "current_mdd": self.current_mdd,
        }

        return next_state, reward, done, info

    def _calculate_downside_std(self):
        """하방 표준편차 계산"""
        if len(self.return_history) < 6:
            return 0.0
        returns_array = np.array(self.return_history[-12:])
        negative_returns = returns_array[returns_array < 0]
        return np.std(negative_returns) if len(negative_returns) > 0 else 0.0

    def _calculate_reward(self, action, returns, features, net_return, turnover):
        """
        단순화된 리워드 계산 (NaN 방지)

        Args:
            action: 포트폴리오 가중치
            returns: 종목별 수익률
            features: 특성 데이터
            net_return: 순수익률
            turnover: 회전율

        Returns:
            reward: 보상값
        """
        # 🔥 입력 검증 및 정규화
        action = np.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
        action = np.clip(action, 0, 1)
        action_sum = np.sum(action)
        if action_sum > 0:
            action = action / action_sum
        else:
            action = np.ones(len(action)) / len(action)

        returns = np.nan_to_num(returns, nan=0.0, posinf=1.0, neginf=-1.0)
        net_return = np.clip(net_return, -1.0, 1.0)

        # 초기 단계: 단순 리워드
        if len(self.return_history) < 6:
            reward = net_return * 100 - turnover * 10.0
            return np.clip(reward, -100, 100)

        # 최근 수익률 배열
        returns_array = np.array(self.return_history[-12:])
        returns_array = np.nan_to_num(returns_array, nan=0.0)

        mean_return = np.mean(returns_array)
        std_return = np.std(returns_array) + 1e-8

        # ========== 1. 수익률 보상 ==========
        return_reward = mean_return * 50.0

        # ========== 2. 샤프 비율 보상 ==========
        sharpe = mean_return / std_return
        sharpe = np.clip(sharpe, -5, 5)
        sharpe_reward = sharpe * 20.0

        # ========== 3. MDD 페널티 ==========
        if len(returns_array) >= 12:
            cumulative = np.cumprod(1 + returns_array)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (cumulative - peak) / (peak + 1e-8)
            mdd = abs(np.min(drawdown))

            if mdd > 0.20:
                mdd_penalty = 100.0 * (mdd - 0.20) ** 2
            else:
                mdd_penalty = 0
        else:
            mdd_penalty = 0

        # ========== 4. 집중도 페널티 ==========
        concentration = np.sum(action**2)
        if concentration > 0.15:
            concentration_penalty = 200.0 * (concentration - 0.15) ** 2
        else:
            concentration_penalty = 0

        # ========== 5. 회전율 페널티 ==========
        turnover_penalty = turnover * 5.0

        # ========== 최종 리워드 ==========
        reward = (
            return_reward
            + sharpe_reward
            - mdd_penalty
            - concentration_penalty
            - turnover_penalty
        )

        # 🔥 최종 검증
        reward = np.nan_to_num(reward, nan=0.0, posinf=100.0, neginf=-100.0)
        reward = np.clip(reward, -100, 100)

        return float(reward)

    def _calculate_mdd_penalty(self):
        """MDD 페널티 및 회복 보너스"""
        if len(self.return_history) < 12:
            return 0, 0

        cumulative_returns = np.cumprod(1 + np.array(self.return_history[-12:]))
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
            recovery_bonus = 15.0 * (1 - recovery_ratio) if recovery_ratio > 0.95 else 0
        else:
            mdd_penalty = 0
            recovery_bonus = 0

        return mdd_penalty, recovery_bonus

    def _calculate_volatility_penalty(self, volatility):
        """변동성 페널티"""
        if volatility > 5.0:
            return 15.0 * (volatility - 5.0) ** 2
        elif volatility > 3.5:
            return 5.0 * (volatility - 3.5) ** 2
        return 0

    def _calculate_concentration_penalty(self, concentration):
        """집중도 페널티"""
        if concentration > 0.12:
            return 500.0 * (concentration - 0.12) ** 4
        return 0

    def _calculate_diversity_bonus(self, action):
        """다양성 보너스 (엔트로피 기반)"""
        entropy = -np.sum(action * np.log(action + 1e-10))
        max_entropy = np.log(len(action))
        normalized_entropy = entropy / max_entropy

        if normalized_entropy > 0.85:
            return 6.0 * normalized_entropy
        elif normalized_entropy > 0.75:
            return 4.0 * normalized_entropy
        return 2.0 * normalized_entropy

    def _calculate_turnover_penalty(self, turnover):
        """회전율 페널티 (최근 수익률 고려)"""
        recent_return = (
            np.mean(self.return_history[-3:]) if len(self.return_history) >= 3 else 0
        )

        if recent_return < -3.0:
            return turnover * 1.0
        elif recent_return > 5.0:
            return turnover * 1.0
        return turnover * 2.5

    def _calculate_factor_bonus(self, action, mkt_rf, hml, rmw, smb, cma):
        """5-Factor 보너스"""
        # 시장 노출
        market_exposure = np.dot(action, mkt_rf)
        market_exposure = np.clip(market_exposure, -10, 10)
        market_bonus = 0.4 * np.maximum(market_exposure, 0)

        # 가치 프리미엄
        value_exposure = np.dot(action, hml)
        value_exposure = np.clip(value_exposure, -10, 10)
        value_bonus = 0.35 * np.maximum(value_exposure, 0)

        # 품질 프리미엄
        quality_exposure = np.dot(action, rmw)
        quality_exposure = np.clip(quality_exposure, -10, 10)
        quality_bonus = 0.35 * np.maximum(quality_exposure, 0)

        # 팩터 다양성
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

        factor_diversity_bonus = 0.25 * (0.2 - factor_std) if factor_std < 0.2 else 0

        return market_bonus + value_bonus + quality_bonus + factor_diversity_bonus
