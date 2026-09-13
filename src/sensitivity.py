from __future__ import annotations

import pandas as pd

from .backtest import turnover
from .momentum import momentum_returns
from .returns import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio
from .validation import Fold, walk_forward_momentum_validation


def walk_forward_grid_search(
    prices: pd.DataFrame,
    folds: list[Fold],
    lookbacks: list[int],
    rebalance_frequencies: list[str],
    top_fraction: float = 0.3,
    transaction_cost_bps: float = 5.0,
    trading_days: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """
    Sweep (lookback, rebalance_frequency) combinations and score each ONLY
    on out-of-sample walk-forward performance (never in-sample).

    This is deliberately NOT a parameter-selection tool: picking the best
    cell here and reporting only that number would just move the data
    snooping from "in-sample vs out-of-sample" to "which of many
    out-of-sample runs looked best by luck". The point is the opposite --
    to show how sensitive (or robust) the strategy's OOS Sharpe is to a
    parameter most people would treat as a free knob, and to make the
    turnover / cost trade-off of trading more or less often explicit.

    Returns one row per (lookback, frequency) combination with OOS
    return/vol/Sharpe/drawdown plus the annualized turnover implied by
    that rebalance frequency (turnover, not just the frequency label, is
    what actually drives transaction costs).
    """
    rows = []
    for lookback in lookbacks:
        for freq in rebalance_frequencies:
            oos_returns, fold_summary = walk_forward_momentum_validation(
                prices,
                folds,
                lookback=lookback,
                top_fraction=top_fraction,
                transaction_cost_bps=transaction_cost_bps,
                rebalance_frequency=freq,
                trading_days=trading_days,
                annual_risk_free_rate=annual_risk_free_rate,
            )
            if oos_returns.empty:
                continue

            # Turnover is measured on the full history at this configuration
            # (not per-fold) purely to get a stable, representative estimate
            # of how often this configuration actually trades.
            _, weights = momentum_returns(
                prices,
                lookback=lookback,
                top_fraction=top_fraction,
                transaction_cost_bps=transaction_cost_bps,
                rebalance_frequency=freq,
            )
            ann_turnover = turnover(weights).mean() * trading_days

            rows.append(
                {
                    "Lookback (days)": lookback,
                    "Rebalance Frequency": freq,
                    "OOS Annual Return (%)": annualized_return(oos_returns, trading_days) * 100,
                    "OOS Annual Volatility (%)": annualized_volatility(oos_returns, trading_days) * 100,
                    "OOS Sharpe Ratio": sharpe_ratio(oos_returns, annual_risk_free_rate, trading_days),
                    "OOS Max Drawdown (%)": max_drawdown(oos_returns) * 100,
                    "Annualized Turnover (x/yr)": ann_turnover,
                    "Folds": len(fold_summary),
                }
            )

    return pd.DataFrame(rows)
