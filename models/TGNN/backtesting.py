"""
TGNN 백테스팅 엔진
"""

import numpy as np
import torch
from typing import Dict


class TGNNBacktester:
    def __init__(
        self,
        model,
        dataset,
        top_k: int = 5,
        transaction_cost: float = 0.001,
        initial_capital: float = 1000000,
    ):
        self.model = model
        self.dataset = dataset
        self.top_k = top_k
        self.transaction_cost = transaction_cost
        self.initial_capital = initial_capital

        self.portfolio_history = []
        self.weights_history = []
        self.returns_history = []

    def run(self) -> dict:
        """백테스팅 실행"""
        self.model.eval()

        capital = self.initial_capital
        prev_weights = np.zeros(len(self.dataset.symbols))

        # 👇 날짜 기록 추가
        self.dates = []

        with torch.no_grad():
            for idx, data in enumerate(self.dataset):
                features = data["features"].unsqueeze(0)
                adj = data["adj_matrix"].unsqueeze(0)

                predictions, _ = self.model(features, adj)
                pred_returns = predictions.squeeze(0).numpy()

                # Top-K 전략
                new_weights = np.zeros(len(self.dataset.symbols))
                top_k_idx = np.argsort(pred_returns)[-self.top_k :]
                new_weights[top_k_idx] = 1.0 / self.top_k

                # 거래비용
                turnover = np.abs(new_weights - prev_weights).sum()
                cost = turnover * self.transaction_cost * capital

                # 수익
                actual_returns = data["labels"].numpy()
                portfolio_return = np.dot(new_weights, actual_returns)

                capital = capital * (1 + portfolio_return / 100) - cost

                # 👇 날짜 기록
                self.dates.append(self.dataset.windows[idx]["date"])

                self.portfolio_history.append(capital)
                self.weights_history.append(new_weights)
                self.returns_history.append(portfolio_return)

                prev_weights = new_weights

        metrics = self._compute_metrics()

        # 👇 시각화용 데이터 추가
        metrics["dates"] = self.dates
        metrics["portfolio_values"] = self.portfolio_history

        return metrics

    def _compute_metrics(self) -> Dict:
        returns = np.array(self.returns_history) / 100

        sharpe = (returns.mean() * 12) / (returns.std() * np.sqrt(12) + 1e-6)

        downside = returns[returns < 0]
        sortino = (
            (returns.mean() * 12) / (downside.std() * np.sqrt(12) + 1e-6)
            if len(downside) > 0
            else sharpe
        )

        cvar = np.percentile(returns, 5)

        gains = returns[returns > 0].sum()
        losses = -returns[returns < 0].sum()
        omega = gains / (losses + 1e-6)

        weights_array = np.array(self.weights_history)
        turnover = np.mean(np.abs(np.diff(weights_array, axis=0)).sum(axis=1))

        cum_return = (self.portfolio_history[-1] / self.initial_capital - 1) * 100

        return {
            "Sharpe Ratio": sharpe,
            "Sortino Ratio": sortino,
            "CVaR (95%)": cvar,
            "Omega Ratio": omega,
            "Turnover": turnover,
            "Transaction Cost (%)": turnover
            * self.transaction_cost
            * 100
            * len(returns),
            "Cumulative Return (%)": cum_return,
            "Final Capital": self.portfolio_history[-1],
        }
