from __future__ import annotations

import numpy as np
import pandas as pd


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple percentage returns."""
    return prices.pct_change().dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Continuously compounded returns."""
    return np.log(prices / prices.shift(1)).dropna(how="all")


def annualized_return(returns: pd.Series, periods: int = 252) -> float:
    """Annualized geometric return."""
    wealth = (1.0 + returns.dropna()).prod()
    years = len(returns.dropna()) / periods
    return wealth ** (1.0 / years) - 1.0 if years > 0 else np.nan


def annualized_volatility(returns: pd.Series, periods: int = 252) -> float:
    return returns.dropna().std() * np.sqrt(periods)


def sharpe_ratio(
    returns: pd.Series,
    annual_risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    rf_daily = (1.0 + annual_risk_free_rate) ** (1.0 / periods) - 1.0
    excess = returns.dropna() - rf_daily
    vol = excess.std()
    return np.sqrt(periods) * excess.mean() / vol if vol > 0 else np.nan


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns.fillna(0.0)).cumprod()
    drawdown = wealth / wealth.cummax() - 1.0
    return drawdown.min()


def performance_summary(
    returns: pd.DataFrame,
    annual_risk_free_rate: float = 0.0,
    periods: int = 252,
) -> pd.DataFrame:
    rows = {}
    for col in returns.columns:
        r = returns[col].dropna()
        rows[col] = {
            "Annual Return": annualized_return(r, periods),
            "Annual Volatility": annualized_volatility(r, periods),
            "Sharpe": sharpe_ratio(r, annual_risk_free_rate, periods),
            "Max Drawdown": max_drawdown(r),
        }
    return pd.DataFrame(rows).T
