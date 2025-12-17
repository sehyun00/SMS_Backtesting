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

        # 포트폴리오 수익률 계산
        portfolio_return = np.dot(action, returns)

        # 거래 비용
        turnover = np.sum(np.abs(action - self.prev_weights))

        cost = turnover * self.cost_bps  # 비율 단위 (0.0005 = 0.05%)
        net_return = portfolio_return - cost

        # 포트폴리오 가치 업데이트
        self.portfolio_value *= 1 + net_return

        self.current_step += 1
        done = self.current_step >= self.n_steps
        self.return_history.append(net_return)

        # MDD 계산
        if len(self.return_history) >= 12:
            cumulative_returns = np.cumprod(1 + np.array(self.return_history[-12:]))
            peak = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - peak) / peak
            self.current_mdd = abs(min(drawdowns))
        else:
            self.current_mdd = 0.0

        # 리워드 계산
        reward = self._calculate_reward(action, returns, features, net_return, turnover)

        self.prev_weights = action

        next_state = (
            self.dataset.get_state(self.windows, self.current_step)
            if not done
            else np.zeros_like(self.dataset.get_state(self.windows, 0))
        )

        info = {
            "portfolio_value": self.portfolio_value,
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
        복합 리워드 계산

        포함 요소:
        - 수익률
        - 샤프 비율
        - 하방 리스크
        - MDD 페널티
        - 변동성 페널티
        - 집중도 페널티
        - 다양성 보너스
        - 회전율 페널티
        - 5-Factor 보너스
        """
        if len(self.return_history) < 6:
            return net_return * 10 - turnover * 5.0

        returns_array = np.array(self.return_history[-12:])
        mean_return = np.mean(returns_array)

        # 하방 리스크
        downside_std = self._calculate_downside_std()

        # 기타 메트릭
        concentration = np.sum(action**2)
        volatility = np.std(returns_array)

        # 5-Factor 추출
        mkt_rf = features[:, -1, -5] * 100
        smb = features[:, -1, -4] * 100
        hml = features[:, -1, -3] * 100
        rmw = features[:, -1, -2] * 100
        cma = features[:, -1, -1] * 100

        # NaN 제거
        mkt_rf = np.nan_to_num(mkt_rf, nan=0.0, posinf=10.0, neginf=-10.0)
        smb = np.nan_to_num(smb, nan=0.0, posinf=10.0, neginf=-10.0)
        hml = np.nan_to_num(hml, nan=0.0, posinf=10.0, neginf=-10.0)
        rmw = np.nan_to_num(rmw, nan=0.0, posinf=10.0, neginf=-10.0)
        cma = np.nan_to_num(cma, nan=0.0, posinf=10.0, neginf=-10.0)

        # 기본 리워드
        return_reward = mean_return * 5.0
        risk_adjusted_return = mean_return / (volatility + 1e-8)
        sharpe_bonus = risk_adjusted_return * 3.0
        downside_penalty = 5.0 * downside_std

        # MDD 페널티
        mdd_penalty, recovery_bonus = self._calculate_mdd_penalty()

        # 변동성 페널티
        volatility_penalty = self._calculate_volatility_penalty(volatility)

        # 집중도 페널티
        concentration_penalty = self._calculate_concentration_penalty(concentration)

        # 다양성 보너스
        diversity_bonus = self._calculate_diversity_bonus(action)

        # 회전율 페널티
        turnover_penalty = self._calculate_turnover_penalty(turnover)

        # 5-Factor 리워드
        factor_bonus = self._calculate_factor_bonus(action, mkt_rf, hml, rmw, smb, cma)

        # 최종 리워드
        reward = (
            return_reward
            + sharpe_bonus
            - downside_penalty
            + factor_bonus
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

        return reward

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
