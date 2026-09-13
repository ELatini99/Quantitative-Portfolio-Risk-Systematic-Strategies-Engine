from __future__ import annotations

import pandas as pd


def buy_and_hold_returns(
    asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    return asset_returns.loc[:, weights.index] @ weights


def cumulative_wealth(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    return initial_value * (1.0 + returns.fillna(0.0)).cumprod()


def turnover(weights: pd.DataFrame) -> pd.Series:
    return weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
