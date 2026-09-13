from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .momentum import momentum_returns
from .returns import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio


@dataclass
class Fold:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def generate_folds(
    index: pd.DatetimeIndex,
    train_years: int = 3,
    test_years: int = 1,
    step_years: int = 1,
) -> list[Fold]:
    """Rolling schedule of non-overlapping (relative to each other) test windows."""
    start, end = index.min(), index.max()
    folds: list[Fold] = []

    train_start = start
    while True:
        train_end = train_start + pd.DateOffset(years=train_years)
        test_start = train_end
        test_end = test_start + pd.DateOffset(years=test_years)
        if test_end > end:
            break
        folds.append(Fold(train_start, train_end, test_start, test_end))
        train_start = train_start + pd.DateOffset(years=step_years)

    return folds


def walk_forward_momentum_validation(
    prices: pd.DataFrame,
    folds: list[Fold],
    lookback: int = 126,
    top_fraction: float = 0.3,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "ME",
    trading_days: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Rolling train/test walk-forward validation of the momentum strategy.

    The momentum rule itself has no free parameters fit on the training
    window (the lookback is fixed by design, matching the published
    12-1-style momentum convention). What the training window provides is
    warm-up history so that, by the time each test window begins, the
    signal and the rebalance schedule are already "live" -- exactly as
    they would be in production. Performance is measured ONLY on the
    out-of-sample test segment of each fold, so a strategy that only
    works in one historical regime cannot hide behind a single blended
    in-sample number.

    Returns:
        oos_returns: concatenated out-of-sample daily net returns across
            all folds (chronologically sorted, non-overlapping).
        fold_summary: per-fold out-of-sample performance table.
    """
    oos_segments = []
    fold_rows = []

    for i, fold in enumerate(folds):
        window_prices = prices.loc[fold.train_start: fold.test_end]
        net_returns, _ = momentum_returns(
            window_prices,
            lookback=lookback,
            top_fraction=top_fraction,
            transaction_cost_bps=transaction_cost_bps,
            rebalance_frequency=rebalance_frequency,
        )
        test_returns = net_returns.loc[fold.test_start: fold.test_end]
        if test_returns.empty:
            continue

        oos_segments.append(test_returns)
        fold_rows.append(
            {
                "Fold": i + 1,
                "Train Start": fold.train_start.date().isoformat(),
                "Train End": fold.test_start.date().isoformat(),
                "Test Start": fold.test_start.date().isoformat(),
                "Test End": test_returns.index.max().date().isoformat(),
                "Annual Return (%)": annualized_return(test_returns, trading_days) * 100,
                "Annual Volatility (%)": annualized_volatility(test_returns, trading_days) * 100,
                "Sharpe Ratio": sharpe_ratio(test_returns, annual_risk_free_rate, trading_days),
                "Max Drawdown (%)": max_drawdown(test_returns) * 100,
                "Observations": int(test_returns.notna().sum()),
            }
        )

    oos_returns = (
        pd.concat(oos_segments).sort_index() if oos_segments else pd.Series(dtype=float)
    )
    fold_summary = pd.DataFrame(fold_rows)
    return oos_returns, fold_summary
