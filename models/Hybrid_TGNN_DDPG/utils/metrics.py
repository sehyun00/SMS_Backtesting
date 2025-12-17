"""
Performance Metrics Calculation
성과 지표 계산 유틸리티
"""

import numpy as np
import pandas as pd


def calculate_metrics(ts_data, dates, strategy_name):
    """
    백테스팅 성과 지표 계산

    Args:
        ts_data: 시계열 데이터 dict
            - portfolio_value: 포트폴리오 가치
            - return: 수익률
            - drawdown: 낙폭
            - turnover: 회전율
        dates: 날짜 리스트
        strategy_name: 전략 이름

    Returns:
        metrics: dict with CAGR, MDD, Sharpe, FinalValue
    """
    df = pd.DataFrame(ts_data)
    df["date"] = pd.to_datetime(dates)

    initial = df["portfolio_value"].iloc[0]
    final = df["portfolio_value"].iloc[-1]

    # 기간 계산
    days = (df["date"].max() - df["date"].min()).days
    years = days / 365.25

    # CAGR 계산
    cagr = ((final / initial) ** (1 / years) - 1) * 100 if years > 0 else 0

    # MDD 계산
    mdd = abs(min(df["drawdown"])) * 100

    # Sharpe Ratio 계산
    r = df["return"] * 100
    vol = r.std() * np.sqrt(12)  # 연율화 변동성
    sharpe = (cagr / 100) / (vol + 1e-8)

    return {
        "Strategy": strategy_name,
        "CAGR": cagr,
        "MDD": mdd,
        "Sharpe": sharpe,
        "FinalValue": final,
    }


def calculate_rolling_metrics(returns, window=12):
    """
    롤링 윈도우 기반 메트릭 계산

    Args:
        returns: 수익률 시계열 (% 단위)
        window: 롤링 윈도우 크기 (개월)

    Returns:
        dict with rolling_sharpe, rolling_volatility, rolling_mdd
    """
    returns_array = np.array(returns)
    n = len(returns_array)

    rolling_sharpe = []
    rolling_volatility = []
    rolling_mdd = []

    for i in range(window, n):
        window_returns = returns_array[i - window : i]

        # 샤프 비율
        mean_return = np.mean(window_returns)
        std_return = np.std(window_returns)
        sharpe = (mean_return * 12) / (std_return * np.sqrt(12) + 1e-8)
        rolling_sharpe.append(sharpe)

        # 변동성
        volatility = std_return * np.sqrt(12)
        rolling_volatility.append(volatility)

        # MDD
        cumulative = np.cumprod(1 + window_returns / 100)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - peak) / peak
        mdd = abs(min(drawdown))
        rolling_mdd.append(mdd)

    return {
        "rolling_sharpe": rolling_sharpe,
        "rolling_volatility": rolling_volatility,
        "rolling_mdd": rolling_mdd,
    }


def calculate_downside_metrics(returns, target_return=0):
    """
    하방 리스크 메트릭 계산

    Args:
        returns: 수익률 시계열 (% 단위)
        target_return: 목표 수익률 (% 단위)

    Returns:
        dict with downside_deviation, sortino_ratio
    """
    returns_array = np.array(returns)

    # 하방 편차
    downside_returns = returns_array[returns_array < target_return]
    downside_deviation = np.std(downside_returns) if len(downside_returns) > 0 else 0

    # Sortino Ratio
    mean_return = np.mean(returns_array)
    sortino = (mean_return - target_return) / (downside_deviation + 1e-8)

    return {
        "downside_deviation": downside_deviation,
        "sortino_ratio": sortino,
    }


def calculate_risk_adjusted_returns(returns, risk_free_rate=0):
    """
    위험 조정 수익률 계산

    Args:
        returns: 수익률 시계열 (% 단위)
        risk_free_rate: 무위험 수익률 (연율 %)

    Returns:
        dict with sharpe, sortino, calmar
    """
    returns_array = np.array(returns)

    # 평균 수익률
    mean_return = np.mean(returns_array) * 12  # 연율화

    # 변동성
    volatility = np.std(returns_array) * np.sqrt(12)

    # Sharpe Ratio
    sharpe = (mean_return - risk_free_rate) / (volatility + 1e-8)

    # Sortino Ratio
    downside_metrics = calculate_downside_metrics(
        returns_array, target_return=risk_free_rate / 12
    )
    sortino = downside_metrics["sortino_ratio"]

    # Calmar Ratio (CAGR / MDD)
    cumulative = np.cumprod(1 + returns_array / 100)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / peak
    mdd = abs(min(drawdown))
    calmar = mean_return / (mdd * 100 + 1e-8)

    return {
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
    }
