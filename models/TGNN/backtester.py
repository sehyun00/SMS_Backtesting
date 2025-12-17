"""
논문용 백테스팅 시스템
- 시계열 데이터 수집
- 집계 지표 계산 (Sharpe, Sortino, Calmar, MDD, Avg Annual DD 등)
- CSV/JSON 저장
- [수정] 자산 계산 시 스케일링 된 값이 아닌 원본 수익률 사용
- [수정] Average Annual Drawdown 지표 추가
"""

import numpy as np
import pandas as pd
import torch
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
import torch.nn as nn

# --- 1. 손실 함수 정의 ---
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.2):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        mse_loss = self.mse(pred, target)
        
        vx = pred - torch.mean(pred)
        vy = target - torch.mean(target)
        cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        corr_loss = 1 - cost 
        
        sign_loss = torch.mean(torch.relu(-torch.sign(pred) * torch.sign(target)))

        return (self.alpha * mse_loss) + (self.beta * corr_loss) + ((1 - self.alpha - self.beta) * sign_loss)

def evaluate_metrics(y_true, y_pred):
    y_true = np.array(y_true).flatten()
    y_pred = np.array(y_pred).flatten()
    
    mse = np.mean((y_true - y_pred)**2)
    mae = np.mean(np.abs(y_true - y_pred))
    correlation = np.corrcoef(y_true, y_pred)[0, 1] if len(y_true) > 1 else 0
    
    true_sign = np.sign(y_true)
    pred_sign = np.sign(y_pred)
    direction_acc = np.mean(true_sign == pred_sign) * 100
    
    return {
        "MSE": mse, "MAE": mae, "Correlation": correlation, "Direction_Acc": direction_acc
    }

# --- 2. 백테스팅 설정 및 클래스 ---

@dataclass
class BacktestConfig:
    initial_capital: float = 1_000_000
    cost_bps: float = 10.0  # 거래비용 (basis points)
    risk_free_rate: float = 0.03  # 연간 무위험 수익률 (3%)
    top_k: int = 5
    weighting_method: str = "equal" # 'softmax', 'equal', 'rank'

@dataclass
class TimeSeriesRecord:
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
    def __init__(self, model, dataset, config: BacktestConfig = None, strategy_name: str = "TGNN"):
        self.model = model
        self.dataset = dataset
        self.config = config or BacktestConfig()
        self.strategy_name = strategy_name
        
        self.history: List[Dict] = []
        self.metrics: Dict = {}
        
    def softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    
    def run(self, rebalance_freq: str = "monthly", benchmark_weights: np.ndarray = None) -> Dict:
        freq_map = {
            "monthly": 1, 
            "quarterly": 3, 
            "semiannual": 6, 
            "annual": 12
        } 
        
        if isinstance(rebalance_freq, int):
            interval = rebalance_freq
        else:
            interval = freq_map.get(rebalance_freq, 1)
        
        self.model.eval()
        
        capital = self.config.initial_capital
        peak = capital
        
        # 종목 수 확인
        try:
            sample_batch = self.dataset[0]
            if "active_mask" in sample_batch:
                n_stocks = len(sample_batch["active_mask"])
            else:
                n_stocks = sample_batch["adj_matrix"].shape[0]
        except:
            n_stocks = 10 

        current_weights = np.zeros(n_stocks)
        prev_weights = np.zeros(n_stocks)
        
        if benchmark_weights is None:
            benchmark_weights = np.ones(n_stocks) / n_stocks
        
        self.history = []
        
        with torch.no_grad():
            for idx in range(len(self.dataset)):
                batch = self.dataset[idx]
                
                features = batch["features"].unsqueeze(0)
                adj = batch["adj_matrix"].unsqueeze(0)
                
                if "active_mask" in batch:
                    active_mask = batch["active_mask"].numpy()
                else:
                    active_mask = np.ones(n_stocks, dtype=bool)
                
                # [핵심] 실제 수익률 가져오기
                if "raw_labels" in batch:
                    actual_returns = batch["raw_labels"].numpy()
                else:
                    actual_returns = batch["labels"].numpy()
                    # 안전장치
                    if np.max(np.abs(actual_returns)) > 1.0:
                         actual_returns = actual_returns * 0.01

                if hasattr(self.dataset, 'windows'):
                    date = self.dataset.windows[idx]["date"]
                else:
                    date = idx

                # 리밸런싱
                if idx % interval == 0:
                    predictions, _ = self.model(features, adj)
                    pred_returns = predictions.squeeze(0).cpu().numpy()
                    
                    # 상장 전 종목 제외
                    pred_returns[~active_mask] = -np.inf
                    
                    n_active = np.sum(active_mask)
                    k = min(self.config.top_k, n_active)
                    
                    new_weights = np.zeros(n_stocks)
                    if k > 0:
                        # 유효한 값들만 필터링
                        valid_indices = np.where(pred_returns != -np.inf)[0]
                        
                        if len(valid_indices) >= k:
                            top_k_idx = np.argsort(pred_returns)[-k:]
                            
                            if self.config.weighting_method == 'softmax':
                                top_scores = pred_returns[top_k_idx]
                                new_weights[top_k_idx] = self.softmax(top_scores)
                            elif self.config.weighting_method == 'rank':
                                ranks = np.arange(1, k + 1)
                                new_weights[top_k_idx] = ranks / ranks.sum()
                            else: # equal
                                new_weights[top_k_idx] = 1.0 / k
                        elif len(valid_indices) > 0:
                            # k개보다 적으면 있는 것만이라도 매수
                            new_weights[valid_indices] = 1.0 / len(valid_indices)
                    
                    prev_weights = current_weights.copy()
                    current_weights = new_weights
                
                # Turnover
                if idx % interval == 0:
                    turnover = np.abs(current_weights - prev_weights).sum() / 2
                else:
                    turnover = 0.0
                
                # 비용
                transaction_cost = turnover * capital * (self.config.cost_bps / 10000)
                
                # 포트폴리오 수익률
                portfolio_return = np.dot(current_weights, actual_returns)
                benchmark_return = np.dot(benchmark_weights, actual_returns)
                
                # 자산 업데이트
                capital = capital * (1 + portfolio_return) - transaction_cost
                
                peak = max(peak, capital)
                if peak > 0:
                    drawdown = (capital - peak) / peak
                else:
                    drawdown = 0.0 # 초기값이거나 손실이 없는 경우
                
                cumulative_return = (capital / self.config.initial_capital - 1)
                excess_return = portfolio_return - benchmark_return
                
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
                    "active_stocks": int(np.sum(active_mask)),
                })
        
        self.metrics = self._calculate_metrics()
        
        return {
            "history": self.history,
            "metrics": self.metrics,
            "final_capital": capital,
            "cumulative_return": cumulative_return,
        }
    
    def run_buy_and_hold(self) -> Dict:
        capital = self.config.initial_capital
        peak = capital
        try:
            sample_batch = self.dataset[0]
            if "active_mask" in sample_batch:
                n_stocks = len(sample_batch["active_mask"])
            else:
                n_stocks = sample_batch["adj_matrix"].shape[0]
        except:
             n_stocks = 10

        history_bh = []
        
        for idx in range(len(self.dataset)):
            batch = self.dataset[idx]
            
            if "active_mask" in batch:
                active_mask = batch["active_mask"].numpy()
            else:
                active_mask = np.ones(n_stocks, dtype=bool)
            
            if "raw_labels" in batch:
                actual_returns = batch["raw_labels"].numpy()
            else:
                actual_returns = batch["labels"].numpy()

            if hasattr(self.dataset, 'windows'):
                date = self.dataset.windows[idx]["date"]
            else:
                date = idx
            
            n_active = np.sum(active_mask)
            if n_active > 0:
                active_weights = np.zeros(n_stocks)
                active_weights[active_mask] = 1.0 / n_active
            else:
                active_weights = np.zeros(n_stocks)
            
            portfolio_return = np.dot(active_weights, actual_returns)
            capital = capital * (1 + portfolio_return)
            if capital < 0: capital = 0
            
            peak = max(peak, capital)
            if peak > 0:
                drawdown = (capital - peak) / peak
            else:
                drawdown = 0.0
            
            cumulative_return = (capital / self.config.initial_capital - 1)
            
            history_bh.append({
                "date": date,
                "portfolio_value": capital,
                "period_return": portfolio_return,
                "cumulative_return": cumulative_return,
                "drawdown": drawdown,
                "turnover": 0.0,
                "transaction_cost": 0.0,
                "benchmark_return": portfolio_return,
                "excess_return": 0.0,
                "active_stocks": int(n_active),
            })
        
        temp_history = self.history
        self.history = history_bh
        metrics_bh = self._calculate_metrics()
        self.history = temp_history
        
        return {
            "history": history_bh,
            "metrics": metrics_bh,
            "final_capital": capital,
            "cumulative_return": cumulative_return,
        }
    
    def _calculate_metrics(self) -> Dict:
        df = pd.DataFrame(self.history)
        if df.empty: return {}
            
        returns = df["period_return"].values
        periods_per_year = 12 # 기본 월간 데이터 가정
        
        # CAGR
        total_return = (df["portfolio_value"].iloc[-1] / self.config.initial_capital - 1)
        
        if "date" in df.columns:
            days = (pd.to_datetime(df["date"].iloc[-1]) - pd.to_datetime(df["date"].iloc[0])).days
            years = days / 365.0
        else:
            years = len(df) / periods_per_year

        if years > 0.1 and total_return > -1:
            cagr = (1 + total_return) ** (1 / years) - 1
        else:
            cagr = 0.0 
        
        # Volatility
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # MDD
        max_drawdown = df["drawdown"].min()
        
        # [신규] Average Annual Drawdown
        # 연도별로 그룹화하여 각 연도의 MDD(최소 drawdown 값)를 구하고 평균을 냄
        df['year'] = pd.to_datetime(df['date']).dt.year
        yearly_mdd = df.groupby('year')['drawdown'].min()
        avg_annual_dd = yearly_mdd.mean()
        
        # Sharpe
        rf_period = self.config.risk_free_rate / periods_per_year
        if returns.std() > 1e-9:
            sharpe_ratio = (returns.mean() - rf_period) / returns.std() * np.sqrt(periods_per_year)
        else:
            sharpe_ratio = 0.0
            
        # Sortino
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 1e-8
        
        if downside_deviation > 1e-9:
            sortino_ratio = (returns.mean() - rf_period) / downside_deviation
        else:
            sortino_ratio = 0.0
        
        # Calmar
        if abs(max_drawdown) > 1e-9:
            calmar_ratio = cagr / abs(max_drawdown)
        else:
            calmar_ratio = 0.0
        
        avg_turnover = df["turnover"].mean()
        
        return {
            "total_return": total_return * 100,
            "cagr": cagr * 100,
            "volatility": volatility * 100,
            "max_drawdown": max_drawdown * 100,
            "avg_annual_dd": avg_annual_dd * 100, # 추가됨
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "calmar_ratio": calmar_ratio,
            "avg_turnover": avg_turnover * 100,
        }
    
    def save_timeseries_csv(self, filepath: Union[str, Path]) -> None:
        df = pd.DataFrame(self.history)
        if "weights" in df.columns:
             df = df.drop("weights", axis=1) # 저장 시 무거우니 제외
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(filepath, index=False, encoding="utf-8-sig")
    
    def save_metrics_json(self, filepath: Union[str, Path], all_strategies: Dict = None) -> None:
        if all_strategies:
            data = all_strategies
        else:
            data = {self.strategy_name: self.metrics}
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

def create_metrics_summary_table(all_metrics: Dict) -> pd.DataFrame:
    rows = []
    for strategy, metrics in all_metrics.items():
        row = {
            "Strategy": strategy,
            "Cumulative Return": f"{metrics.get('total_return', 0):.2f}%",
            "CAGR": f"{metrics.get('cagr', 0):.2f}%",
            "MDD": f"{metrics.get('max_drawdown', 0):.2f}%",
            "Avg Annual DD": f"{metrics.get('avg_annual_dd', 0):.2f}%", # 추가됨
            "Sharpe": f"{metrics.get('sharpe_ratio', 0):.2f}",
        }
        rows.append(row)
    return pd.DataFrame(rows)
