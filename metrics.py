"""
metrics.py — Standard performance/risk metrics for a returns series.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def total_return(equity_curve: pd.Series) -> float:
    return equity_curve.iloc[-1] / equity_curve.iloc[0] - 1


def annualized_return(returns: pd.Series) -> float:
    cum = (1 + returns).prod()
    n_years = len(returns) / TRADING_DAYS
    if n_years <= 0:
        return 0.0
    return cum ** (1 / n_years) - 1


def annualized_vol(returns: pd.Series) -> float:
    return returns.std() * np.sqrt(TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    if excess.std() == 0:
        return 0.0
    return (excess.mean() / excess.std()) * np.sqrt(TRADING_DAYS)


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0 or np.isnan(downside_std):
        return 0.0
    return (excess.mean() / downside_std) * np.sqrt(TRADING_DAYS)


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1
    return drawdown.min()


def win_rate(returns: pd.Series) -> float:
    active = returns[returns != 0]
    if len(active) == 0:
        return 0.0
    return (active > 0).mean()


def calmar_ratio(returns: pd.Series, equity_curve: pd.Series) -> float:
    mdd = abs(max_drawdown(equity_curve))
    if mdd == 0:
        return 0.0
    return annualized_return(returns) / mdd


def summarize(equity_curve: pd.Series, returns: pd.Series) -> dict:
    return {
        "Total Return": f"{total_return(equity_curve):.2%}",
        "Annualized Return": f"{annualized_return(returns):.2%}",
        "Annualized Volatility": f"{annualized_vol(returns):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns):.2f}",
        "Sortino Ratio": f"{sortino_ratio(returns):.2f}",
        "Max Drawdown": f"{max_drawdown(equity_curve):.2%}",
        "Calmar Ratio": f"{calmar_ratio(returns, equity_curve):.2f}",
        "Win Rate": f"{win_rate(returns):.2%}",
    }
