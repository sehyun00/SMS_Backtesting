import numpy as np
import pandas as pd
from typing import Dict, List, Union


def compute_metrics(
    portfolio_values: Union[List[float], np.ndarray], initial_capital: float = 1_000_000
) -> Dict[str, float]:
    """
    Computes financial metrics (Return, CAGR, MDD, Sharpe) from portfolio history.
    """
    values = np.array(portfolio_values)
    if len(values) == 0:
        return {"Total_Return": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sharpe": 0.0}

    # Returns
    total_ret = (values[-1] / initial_capital) - 1

    # CAGR (Annualized) - Assuming Daily Steps
    days = len(values)  # Trading Days
    years = days / 252.0
    if years > 0:
        cagr = (values[-1] / initial_capital) ** (1 / years) - 1
    else:
        cagr = 0

    # MDD
    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    mdd = abs(np.min(drawdown))

    # Sharpe (Daily assumption: 252)
    pct_change = pd.Series(values).pct_change().dropna()
    if len(pct_change) > 0:
        vol = pct_change.std() * np.sqrt(252)
        sharpe = (cagr) / (vol + 1e-8)
    else:
        sharpe = 0

    return {
        "Total_Return": total_ret * 100,
        "CAGR": cagr * 100,
        "MDD": mdd * 100,
        "Sharpe": sharpe,
    }
