from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf


def download_prices(
    tickers: list[str] | tuple[str, ...],
    start: str,
    end: str | None = None,
    use_synthetic_fallback: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Download adjusted close prices and return a clean price matrix.

    If live data cannot be retrieved (no internet, Yahoo Finance rate
    limiting / outage, delisted tickers, etc.) and `use_synthetic_fallback`
    is True, fall back to a reproducible synthetic multi-asset price panel
    so the rest of the pipeline remains runnable end to end.
    """
    try:
        raw = yf.download(
            list(tickers),
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )

        if raw.empty:
            raise RuntimeError("No market data were downloaded.")

        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" in raw.columns.get_level_values(0):
                prices = raw["Close"]
            else:
                prices = raw.xs("Close", axis=1, level=0)
        else:
            prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

        # Drop tickers that returned no data at all (delisted, mistyped,
        # or a transient per-symbol failure) rather than letting a single
        # bad ticker wipe out every date for every OTHER asset below: a
        # plain row-wise dropna() requires every column to have data on
        # a given day to keep that day, so one entirely-NaN column can
        # silently zero out the whole date range for a large universe.
        failed = prices.columns[prices.isna().all()].tolist()
        if failed:
            print(
                f"[data] {len(failed)} ticker(s) returned no data and were "
                f"dropped from the universe: {', '.join(failed)}"
            )
            prices = prices.drop(columns=failed)

        prices = prices.ffill().dropna(how="any")
        if prices.empty:
            raise RuntimeError("No usable price data after cleaning.")
        return prices

    except Exception as exc:  # noqa: BLE001 - any data/network issue triggers the fallback
        if not use_synthetic_fallback:
            raise
        print(
            f"[data] Live data unavailable ({exc.__class__.__name__}: {exc}). "
            f"Falling back to a synthetic price panel so the pipeline can still run."
        )
        return _synthetic_prices(tickers, start=start, end=end, seed=seed)


def _synthetic_prices(
    tickers: list[str] | tuple[str, ...],
    start: str,
    end: str | None = None,
    seed: int = 42,
    annual_vol_range: tuple[float, float] = (0.12, 0.35),
    annual_drift_range: tuple[float, float] = (0.01, 0.10),
    market_beta_range: tuple[float, float] = (0.3, 1.4),
) -> pd.DataFrame:
    """
    Generate a synthetic multi-asset price panel via GBM with a one-factor
    (market) structure, so the resulting covariance matrix is non-trivial
    (assets are correlated, not independent) -- this matters for testing
    covariance estimation, PCA and portfolio optimization downstream.

    NOT real market data. Only used when live data is unavailable, purely
    to keep the pipeline demoable offline.
    """
    tickers = list(tickers)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, end=end or pd.Timestamp.today())
    n_days, n_assets = len(dates), len(tickers)

    betas = rng.uniform(*market_beta_range, size=n_assets)
    idio_vol = rng.uniform(*annual_vol_range, size=n_assets) / np.sqrt(252)
    drift = rng.uniform(*annual_drift_range, size=n_assets) / 252
    market_vol = 0.15 / np.sqrt(252)

    market_shock = rng.normal(0, market_vol, size=n_days)
    idio_shocks = rng.normal(0, 1, size=(n_days, n_assets)) * idio_vol

    log_returns = drift + np.outer(market_shock, betas) + idio_shocks
    log_prices = np.cumsum(log_returns, axis=0)
    prices = 100 * np.exp(log_prices)

    return pd.DataFrame(prices, index=dates, columns=tickers)


def align_assets(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep dates for which all assets have observations."""
    return prices.dropna(how="any")
