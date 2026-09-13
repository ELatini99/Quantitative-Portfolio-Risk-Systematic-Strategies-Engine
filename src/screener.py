from __future__ import annotations

import numpy as np
import pandas as pd

from .momentum import momentum_scores

DISCLAIMER = (
    "NOT INVESTMENT ADVICE. This ranks the configured universe purely by "
    "trailing price momentum and realized volatility (see src/screener.py "
    "for the exact formula). It has no view on valuation, fundamentals, "
    "earnings, news, or anything else a real investment decision should "
    "weigh, and past momentum is not a guarantee of future performance. "
    "For informational/research purposes only."
)


def realized_volatility(prices: pd.DataFrame, window: int = 63) -> pd.DataFrame:
    """Trailing annualized realized volatility, per asset, as of each date."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def risk_adjusted_momentum(
    prices: pd.DataFrame,
    momentum_lookback: int = 126,
    vol_window: int = 63,
) -> pd.DataFrame:
    """
    Risk-adjusted momentum score per asset, for every date:

        score_t = momentum_return_t / realized_volatility_t

    A "momentum Sharpe": it rewards assets with strong trailing returns
    AND penalizes ones with high volatility, which is exactly the
    momentum/volatility trade-off a raw momentum score alone ignores --
    two stocks with identical trailing returns are NOT equally
    attractive if one got there with twice the volatility (and is
    correspondingly more likely to give it back in a sharp reversal).
    """
    momentum = momentum_scores(prices, momentum_lookback)
    vol = realized_volatility(prices, vol_window)
    return momentum / vol.replace(0.0, np.nan)


def current_model_portfolio(
    prices: pd.DataFrame,
    momentum_lookback: int = 126,
    vol_window: int = 63,
    top_n: int = 10,
    max_volatility: float | None = None,
) -> pd.DataFrame:
    """
    Snapshot of the model's current target portfolio: ranks the universe
    by risk-adjusted momentum AS OF THE LATEST AVAILABLE DATE, optionally
    drops anything above a volatility ceiling (`max_volatility`, as an
    annualized fraction, e.g. 0.45 = 45%/yr), and inverse-volatility
    weights the surviving top-N names -- the same construction convention
    used by the momentum backtest elsewhere in this project.

    This is a MODEL OUTPUT, not investment advice -- see `DISCLAIMER`
    above. It reflects only trailing price momentum and realized
    volatility on the configured universe.
    """
    momentum_row = momentum_scores(prices, momentum_lookback).iloc[-1]
    vol_row = realized_volatility(prices, vol_window).iloc[-1]
    score_row = momentum_row / vol_row.replace(0.0, np.nan)

    table = pd.DataFrame(
        {
            "Momentum Return (%)": momentum_row * 100,
            "Realized Volatility (%)": vol_row * 100,
            "Risk-Adjusted Momentum Score": score_row,
        }
    ).dropna()

    if max_volatility is not None:
        table = table[table["Realized Volatility (%)"] <= max_volatility * 100]

    # A "buy" screen should only ever surface assets with genuinely
    # positive trailing momentum -- ranking by score alone could
    # otherwise surface the "least negative" name once high-volatility
    # names are excluded, which is not a momentum buy signal at all.
    table = table[table["Momentum Return (%)"] > 0]

    table = table.sort_values(
        "Risk-Adjusted Momentum Score", ascending=False
    ).head(top_n)

    if table.empty:
        table["Target Weight (%)"] = []
        table.index.name = "Ticker"
        return table.reset_index()

    inverse_vol = 1.0 / table["Realized Volatility (%)"]
    table = table.copy()
    table["Target Weight (%)"] = (inverse_vol / inverse_vol.sum()) * 100

    table.index.name = "Ticker"
    return table.reset_index()
