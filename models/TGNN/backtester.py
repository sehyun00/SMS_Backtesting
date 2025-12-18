"""
논문용 백테스팅 시스템 (Multi-Output TGNN 지원)
- 시계열 데이터 수집
- 집계 지표 계산 (Sharpe, Sortino, Calmar, MDD, VaR, CVaR 등)
- CSV/JSON 저장
- Multi-Output 모델의 타겟 헤드 선택 지원
"""

import numpy as np
import pandas as pd
import torch
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class BacktestConfig:
    """백테스팅 설정"""
    initial_capital: float = 1_000_000
    cost_bps: float = 5.0  # 거래비용 (basis points)
    risk_free_rate: float = 0.03  # 연간 무위험 수익률 (3%)
    top_k: int = 5  # 상위 K개 종목 선택
    weighting_method: str = "softmax"  # "softmax" or "equal"


@dataclass
class TimeSeriesRecord:
    """시계열 기록"""
    date: pd.Timestamp
    portfolio_value: float
    period_return: float
    cumulative_return: float
    drawdown: float
    turnover: float
    transaction_cost: float
    weights: np.ndarray
    benchmark_return: float = 0.0
    excess_return: float = 0.0


class Backtester:
    """
    논문용 백테스팅 시스템
    
    Features:
    - 시계열 데이터 수집 (포트폴리오 가치, 수익률, 드로다운, 턴오버 등)
    - 집계 지표 계산 (Sharpe, Sortino, Calmar 등)
    - CSV/JSON 저장
    - Multi-Output TGNN 지원 (target_type 파라미터)
    """
    
    def __init__(self, model, dataset, config: BacktestConfig = None, strategy_name: str = "TGNN"):
        self.model = model
        self.dataset = dataset
        self.config = config or BacktestConfig()
        self.strategy_name = strategy_name
        
        self.history: List[Dict] = []
        self.metrics: Dict = {}
        self.target_type: str = "Momentum1M"  # 🔥 기본 타겟 (외부에서 변경 가능)
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """배열을 확률 분포로 변환"""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def run(self, rebalance_freq: str = "monthly", benchmark_weights: np.ndarray = None) -> Dict:
        """
        백테스팅 실행
        
        Args:
            rebalance_freq: 리밸런싱 빈도 ("monthly", "quarterly", "semiannual", "annual")
            benchmark_weights: 벤치마크 비중 (None이면 동일가중)
        
        Returns:
            Dict: 백테스팅 결과 (시계열 + 집계 지표)
        """
        freq_map = {"monthly": 1, "quarterly": 3, "semiannual": 6, "annual": 12}
        interval = freq_map[rebalance_freq]
        
        self.model.eval()
        
        capital = self.config.initial_capital
        peak = capital
        n_stocks = len(self.dataset.symbols)
        
        current_weights = np.zeros(n_stocks)
        prev_weights = np.zeros(n_stocks)
        
        # 벤치마크 (동일가중)
        if benchmark_weights is None:
            benchmark_weights = np.ones(n_stocks) / n_stocks
        
        self.history = []
        
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                batch = self.dataset[idx]
                date = self.dataset.windows[idx]["date"]
                active_mask = batch["active_mask"].numpy()
                
                # 🔥 실제 수익률은 target_type에 해당하는 라벨 사용
                if self.target_type in batch:
                    actual_returns = batch[self.target_type].numpy()
                else:
                    # 기존 방식 (단일 출력 모델 호환)
                    actual_returns = batch.get("labels", batch.get("Momentum1M")).numpy()
                
                # 리밸런싱 시점
                if idx % interval == 0:
                    features = batch["features"].unsqueeze(0)
                    adj = batch["adj_matrix"].unsqueeze(0)
                    
                    # 🔥 Multi-Output 모델: target_type 파라미터 전달
                    try:
                        predictions, _ = self.model(features, adj, target_type=self.target_type)
                    except TypeError:
                        # 단일 출력 모델 (기존 모델 호환)
                        predictions, _ = self.model(features, adj)
                    
                    pred_returns = predictions.squeeze(0).numpy()
                    
                    # 🔥 NaN/Inf 체크 및 클리핑
                    pred_returns = np.nan_to_num(pred_returns, nan=-np.inf, posinf=10.0, neginf=-10.0)
                    pred_returns = np.clip(pred_returns, -10.0, 10.0)
                    
                    # 상장 전 종목 제외
                    pred_returns[~active_mask] = -np.inf
                    
                    # Top-K 선택
                    n_active = np.sum(active_mask)
                    k = min(self.config.top_k, n_active)
                    
                    new_weights = np.zeros(n_stocks)
                    if k > 0:
                        top_k_idx = np.argsort(pred_returns)[-k:]
                        
                        if self.config.weighting_method == "softmax":
                            top_scores = pred_returns[top_k_idx]
                            new_weights[top_k_idx] = self.softmax(top_scores)
                        else:  # equal
                            new_weights[top_k_idx] = 1.0 / k
                    
                    prev_weights = current_weights.copy()
                    current_weights = new_weights
                
                # Turnover 계산
                turnover = np.abs(current_weights - prev_weights).sum() / 2
                
                # 거래비용
                transaction_cost = turnover * capital * (self.config.cost_bps / 10000)
                
                # 🔥 실제 수익률 클리핑 (극단값 방지)
                actual_returns = np.clip(actual_returns, -50.0, 50.0)
                
                # 포트폴리오 수익률
                portfolio_return = np.dot(current_weights, actual_returns)
                
                # 벤치마크 수익률
                benchmark_return = np.dot(benchmark_weights, actual_returns)
                
                # 🔥 수익률 검증
                if np.isnan(portfolio_return) or np.isinf(portfolio_return):
                    print(f"⚠️ Warning: Invalid portfolio_return at {date}: {portfolio_return}")
                    portfolio_return = 0.0
                
                # 자산 업데이트
                capital = capital * (1 + portfolio_return / 100) - transaction_cost
                
                # 드로다운
                peak = max(peak, capital)
                drawdown = (capital - peak) / peak * 100
                
                # 누적 수익률
                cumulative_return = (capital / self.config.initial_capital - 1) * 100
                
                # 초과 수익률
                excess_return = portfolio_return - benchmark_return
                
                # 기록 저장
                self.history.append({
                    "date": date,
                    "portfolio_value": capital,
                    "period_return": portfolio_return,
                    "cumulative_return": cumulative_return,
                    "drawdown": drawdown,
                    "turnover": turnover,
                    "transaction_cost": transaction_cost,
                    "benchmark_return": benchmark_return,
                    "excess_return": excess_return,
                    "weights": current_weights.copy(),
                    "active_stocks": int(np.sum(active_mask)),
                })
                
                prev_weights = current_weights.copy()
        
        # 집계 지표 계산
        self.metrics = self._calculate_metrics()
        
        return {
            "history": self.history,
            "metrics": self.metrics,
            "final_capital": capital,
            "cumulative_return": cumulative_return,
        }
    
    def run_buy_and_hold(self) -> Dict:
        """Buy & Hold (동일가중) 백테스팅"""
        capital = self.config.initial_capital
        peak = capital
        n_stocks = len(self.dataset.symbols)
        weights = np.ones(n_stocks) / n_stocks
        
        self.history = []
        
        for idx in range(len(self.dataset)):
            batch = self.dataset[idx]
            date = self.dataset.windows[idx]["date"]
            active_mask = batch["active_mask"].numpy()
            
            # 🔥 Momentum1M 사용 (월별 수익률)
            if "Momentum1M" in batch:
                actual_returns = batch["Momentum1M"].numpy()
            else:
                actual_returns = batch.get("labels", torch.zeros(n_stocks)).numpy()
            
            # 🔥 극단값 클리핑
            actual_returns = np.clip(actual_returns, -50.0, 50.0)
            
            # 활성 종목만 동일가중
            n_active = np.sum(active_mask)
            if n_active > 0:
                active_weights = np.zeros(n_stocks)
                active_weights[active_mask] = 1.0 / n_active
            else:
                active_weights = weights
            
            portfolio_return = np.dot(active_weights, actual_returns)
            
            # 🔥 검증
            if np.isnan(portfolio_return) or np.isinf(portfolio_return):
                portfolio_return = 0.0
            
            capital = capital * (1 + portfolio_return / 100)
            
            peak = max(peak, capital)
            drawdown = (capital - peak) / peak * 100
            cumulative_return = (capital / self.config.initial_capital - 1) * 100
            
            self.history.append({
                "date": date,
                "portfolio_value": capital,
                "period_return": portfolio_return,
                "cumulative_return": cumulative_return,
                "drawdown": drawdown,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "benchmark_return": portfolio_return,
                "excess_return": 0.0,
                "weights": active_weights.copy(),
                "active_stocks": int(n_active),
            })
        
        self.metrics = self._calculate_metrics()
        
        return {
            "history": self.history,
            "metrics": self.metrics,
            "final_capital": capital,
            "cumulative_return": cumulative_return,
        }
    
    def _calculate_metrics(self) -> Dict:
        """집계 지표 계산"""
        if not self.history:
            return {}
        
        df = pd.DataFrame(self.history)
        returns = df["period_return"].values / 100  # 퍼센트 → 소수
        excess_returns = df["excess_return"].values / 100
        
        n_periods = len(returns)
        periods_per_year = 12  # 월간 데이터
        
        # === 수익 지표 ===
        total_return = (df["portfolio_value"].iloc[-1] / self.config.initial_capital - 1)
        years = n_periods / periods_per_year
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # === 위험 지표 ===
        volatility = returns.std() * np.sqrt(periods_per_year) if len(returns) > 0 else 0
        max_drawdown = df["drawdown"].min() / 100  # 퍼센트 → 소수
        
        # Downside Deviation
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
        
        # VaR & CVaR (95%)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        # === 위험조정 수익 ===
        rf_monthly = self.config.risk_free_rate / periods_per_year
        
        sharpe_ratio = (returns.mean() - rf_monthly) / returns.std() * np.sqrt(periods_per_year) if returns.std() > 0 else 0
        sortino_ratio = (returns.mean() - rf_monthly) / downside_deviation if downside_deviation > 0 else 0
        calmar_ratio = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # === 벤치마크 대비 ===
        excess_return_annualized = excess_returns.mean() * periods_per_year
        tracking_error = excess_returns.std() * np.sqrt(periods_per_year) if len(excess_returns) > 0 else 0
        information_ratio = excess_return_annualized / tracking_error if tracking_error > 0 else 0
        
        # === 거래 특성 ===
        avg_turnover = df["turnover"].mean() if "turnover" in df.columns else 0
        total_transaction_cost = df["transaction_cost"].sum() if "transaction_cost" in df.columns else 0
        
        # 거래비용별 순수익률 (0, 5, 10 bps)
        net_returns = {}
        for bps in [0, 5, 10]:
            cost_impact = avg_turnover * bps / 10000 * periods_per_year
            net_returns[f"net_cagr_{bps}bps"] = cagr - cost_impact
        
        return {
            # 수익 지표
            "total_return": total_return * 100,
            "cagr": cagr * 100,
            "annualized_return": cagr * 100,
            
            # 위험 지표
            "volatility": volatility * 100,
            "max_drawdown": max_drawdown * 100,
            "downside_deviation": downside_deviation * 100,
            "var_95": var_95 * 100,
            "cvar_95": cvar_95 * 100,
            
            # 위험조정 수익
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            
            # 벤치마크 대비
            "excess_return": excess_return_annualized * 100,
            "tracking_error": tracking_error * 100,
            "information_ratio": information_ratio,
            
            # 거래 특성
            "avg_turnover": avg_turnover * 100,
            "total_transaction_cost": total_transaction_cost,
            **{k: v * 100 for k, v in net_returns.items()},
            
            # 기타
            "n_periods": n_periods,
            "years": years,
        }
    
    def get_timeseries_df(self) -> pd.DataFrame:
        """시계열 데이터를 DataFrame으로 반환"""
        if not self.history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.history)
        
        # weights 컬럼을 개별 종목 컬럼으로 분리
        if "weights" in self.history[0]:
            weights_df = pd.DataFrame(
                df["weights"].tolist(),
                columns=[f"weight_{sym}" for sym in self.dataset.symbols]
            )
            df = pd.concat([df.drop("weights", axis=1), weights_df], axis=1)
        
        return df
    
    def save_timeseries_csv(self, filepath: Path) -> None:
        """시계열 데이터 CSV 저장"""
        df = self.get_timeseries_df()
        if not df.empty:
            df.to_csv(filepath, index=False, encoding="utf-8-sig")
            print(f"📊 시계열 데이터 저장: {filepath}")
    
    def save_metrics_json(self, filepath: Path, all_strategies: Dict = None) -> None:
        """집계 지표 JSON 저장"""
        if all_strategies:
            data = all_strategies
        else:
            data = {self.strategy_name: self.metrics}
        
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        print(f"📈 집계 지표 저장: {filepath}")


def create_metrics_summary_table(all_metrics: Dict) -> pd.DataFrame:
    """
    모든 전략의 지표를 비교 테이블로 생성
    
    Args:
        all_metrics: {"전략명": {지표들}}
    
    Returns:
        pd.DataFrame: 비교 테이블
    """
    rows = []
    
    for strategy, metrics in all_metrics.items():
        if not metrics:
            continue
        
        row = {
            "Strategy": strategy,
            "Cumulative Return": f"{metrics.get('total_return', 0):.2f}%",
            "CAGR": f"{metrics.get('cagr', 0):.2f}%",
            "MDD": f"{metrics.get('max_drawdown', 0):.2f}%",
            "Avg Annual DD": f"{metrics.get('downside_deviation', 0):.2f}%",
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
        }
        rows.append(row)
    
    return pd.DataFrame(rows)
