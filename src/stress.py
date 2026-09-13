from __future__ import annotations

import pandas as pd


def historical_stress(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
    window: int = 5,
) -> pd.Series:
    """
    Rolling multi-day portfolio returns.

    The worst observations provide a simple historical stress test.
    """
    portfolio = asset_returns.loc[:, weights.index] @ weights
    return portfolio.rolling(window).sum().dropna().sort_values()


def scenario_stress(
    weights: pd.Series,
    shocks: pd.DataFrame,
) -> pd.Series:
    """
    Apply deterministic asset shocks.

    `shocks` should have scenarios as rows and assets as columns.
    """
    shocks = shocks.reindex(columns=weights.index).fillna(0.0)
    return shocks @ weights
