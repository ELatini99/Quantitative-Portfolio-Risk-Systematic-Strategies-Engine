from __future__ import annotations

import numpy as np
import pandas as pd


def momentum_scores(
    prices: pd.DataFrame,
    lookback: int = 126,
) -> pd.DataFrame:
    """Trailing price momentum measured as percentage change."""
    return prices.pct_change(lookback)


def rebalance_schedule(
    index: pd.DatetimeIndex, frequency: str = "ME"
) -> pd.DatetimeIndex:
    """
    Actual trading days on which the portfolio rebalances: the last
    trading day observed within each calendar period (e.g. month-end),
    NOT the raw calendar period-end label.

    This distinction matters: `some_series.resample("ME").last().index`
    returns calendar period-end timestamps (e.g. 2024-01-31), which are
    frequently *not* trading days (weekends/holidays). Checking
    `trading_date in that_index` would then rarely match, silently
    breaking the rebalance loop. Building the schedule from the actual
    index (as done here) avoids that failure mode entirely.
    """
    grouped = pd.Series(index=index, data=index).groupby(pd.Grouper(freq=frequency))
    last_trading_day_per_period = grouped.last().dropna()
    return pd.DatetimeIndex(last_trading_day_per_period.values)


def cross_sectional_momentum_weights(
    prices: pd.DataFrame,
    lookback: int = 126,
    top_fraction: float = 0.3,
    rebalance_frequency: str = "ME",
) -> pd.DataFrame:
    """
    Long-only cross-sectional momentum portfolio.

    At each rebalance date (by default, month-end), invest equally in the
    top fraction of assets ranked by trailing momentum, and HOLD that
    allocation until the next rebalance date (rather than re-selecting
    every single day the ranking shifts, which would produce unrealistic,
    near-daily turnover). Weights are shifted by one day before being
    applied to returns, to avoid look-ahead bias.
    """
    scores = momentum_scores(prices, lookback)
    n_select = max(1, int(np.ceil(prices.shape[1] * top_fraction)))
    rebalance_dates = set(rebalance_schedule(prices.index, rebalance_frequency))

    raw = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)

    for date in prices.index:
        if date in rebalance_dates:
            row = scores.loc[date].dropna()
            if len(row) > 0:
                selected = row.nlargest(min(n_select, len(row))).index
                current[:] = 0.0
                current.loc[selected] = 1.0 / len(selected)
        raw.loc[date] = current

    # Signal generated from information available at t is held from t+1.
    return raw.shift(1).fillna(0.0)


def momentum_returns(
    prices: pd.DataFrame,
    lookback: int = 126,
    top_fraction: float = 0.3,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "ME",
) -> tuple[pd.Series, pd.DataFrame]:
    """Compute daily strategy returns including turnover-based transaction costs."""
    asset_returns = prices.pct_change().fillna(0.0)
    weights = cross_sectional_momentum_weights(
        prices,
        lookback=lookback,
        top_fraction=top_fraction,
        rebalance_frequency=rebalance_frequency,
    )

    gross = (weights * asset_returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    costs = turnover * transaction_cost_bps / 10_000.0
    net = gross - costs

    return net, weights
