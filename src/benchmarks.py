from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import turnover
from .momentum import rebalance_schedule


def buy_and_hold_single_asset(returns: pd.DataFrame, ticker: str) -> pd.Series:
    """
    Passive buy-and-hold on a single asset -- the simplest possible
    benchmark, and the one most people mean by "the market" (e.g. SPY).
    No rebalancing, no transaction costs beyond the single initial trade.
    """
    return returns[ticker].rename(f"Buy & Hold {ticker}")


def equal_weight_target_weights(
    prices: pd.DataFrame, rebalance_frequency: str = "ME"
) -> pd.DataFrame:
    """
    1/N target weights, reset to equal weight at each rebalance date and
    held in between (same rebalance-schedule convention as the momentum
    strategy in src/momentum.py, and the same 1-day execution lag), so
    the two are directly cost-comparable.
    """
    n_assets = prices.shape[1]
    target = np.full(n_assets, 1.0 / n_assets)
    rebalance_dates = set(rebalance_schedule(prices.index, rebalance_frequency))

    raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(target, index=prices.columns)

    for date in prices.index:
        if date in rebalance_dates:
            current[:] = target
        raw.loc[date] = current

    return raw.shift(1).fillna(0.0)


def equal_weight_returns(
    prices: pd.DataFrame,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "ME",
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Naive 1/N portfolio, periodically rebalanced back to equal weight at
    the same frequency and cost assumption as the momentum strategy. This
    isolates what the momentum SIGNAL adds on top of simple
    diversification across the same universe, separate from just holding
    everything equally.
    """
    asset_returns = prices.pct_change().fillna(0.0)
    weights = equal_weight_target_weights(prices, rebalance_frequency=rebalance_frequency)

    gross = (weights * asset_returns).sum(axis=1)
    costs = turnover(weights) * transaction_cost_bps / 10_000.0
    net = gross - costs

    return net, weights


def excess_return_stats(
    strategy_returns: pd.Series, benchmark_returns: pd.Series, periods: int = 252
) -> dict:
    """
    Tracking error, correlation and (annualized) information ratio of a
    strategy versus a benchmark, aligned on shared dates.
    """
    aligned = pd.concat(
        [strategy_returns.rename("strategy"), benchmark_returns.rename("benchmark")],
        axis=1,
    ).dropna()
    if aligned.empty:
        return {"correlation": np.nan, "tracking_error": np.nan, "information_ratio": np.nan}

    diff = aligned["strategy"] - aligned["benchmark"]
    tracking_error = diff.std() * np.sqrt(periods)
    info_ratio = (diff.mean() * periods) / tracking_error if tracking_error > 0 else np.nan
    correlation = aligned["strategy"].corr(aligned["benchmark"])
    return {
        "correlation": correlation,
        "tracking_error": tracking_error,
        "information_ratio": info_ratio,
    }
