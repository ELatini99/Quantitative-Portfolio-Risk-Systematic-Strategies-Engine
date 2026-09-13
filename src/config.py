from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class Config:
    tickers: Tuple[str, ...] = (
        "SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD"
    )
    start: str = "2015-01-01"
    end: str | None = None
    confidence: float = 0.99
    trading_days: int = 252
    lookback_cov: int = 252
    lookback_momentum: int = 126
    top_fraction: float = 0.30
    # NOTE: pandas >= 2.2 deprecates the "M" offset alias in favor of "ME"
    # (month-end); pandas 3.0 removes "M" entirely. "ME" is required here.
    rebalance_frequency: str = "ME"
    annual_risk_free_rate: float = 0.0
    transaction_cost_bps: float = 5.0
    monte_carlo_paths: int = 100_000
    random_seed: int = 42

    # Walk-forward *validation* (rolling train/test folds), distinct from
    # rebalance_frequency above (which controls how often the live
    # portfolio trades). This is used to check that the momentum signal's
    # performance is stable out-of-sample across different market regimes,
    # not just over one single in-sample backtest.
    walk_forward_train_years: int = 3
    walk_forward_test_years: int = 1
    walk_forward_step_years: int = 1

    # If live market data cannot be downloaded (no internet, rate limit,
    # temporary Yahoo Finance outage), fall back to a reproducible
    # synthetic multi-asset price panel so the pipeline is always
    # runnable end to end (useful for demos / interviews without a
    # guaranteed connection).
    use_synthetic_fallback: bool = True

    # Parameter sensitivity grid: sweeps momentum lookback x rebalance
    # frequency, scored strictly on out-of-sample walk-forward performance
    # (see src/sensitivity.py). "W" = weekly, "ME" = month-end,
    # "QE" = quarter-end. Kept small on purpose -- each cell re-runs the
    # full walk-forward schedule.
    sensitivity_lookbacks: Tuple[int, ...] = (63, 126, 189, 252)
    sensitivity_rebalance_frequencies: Tuple[str, ...] = ("W", "ME", "QE")

    # Passive benchmark for comparison (must be one of `tickers`, or the
    # single-asset buy-and-hold benchmark is skipped with a warning).
    benchmark_ticker: str = "SPY"

    # Regime-conditioned momentum exposure: scales the momentum sleeve
    # down when the market's variance becomes unusually concentrated in
    # its first principal component (rolling "Absorption Ratio" of
    # Kritzman, Li, Page & Rigobon, 2010 -- see src/factors.py and
    # src/regime.py). This connects the PCA factor model to the momentum
    # strategy: a rising absorption ratio signals a single-driver,
    # "risk-off" regime where diversification benefits shrink and
    # momentum crashes become more likely.
    absorption_ratio_window: int = 126
    absorption_ratio_n_components: int = 1
    regime_zscore_window: int = 252
    regime_min_exposure: float = 0.3
    regime_zscore_low: float = -1.0
    regime_zscore_high: float = 1.5

    # Current-model company screener: ranks a universe of individual
    # stocks by risk-adjusted momentum (momentum return / realized
    # volatility) as of the LATEST available date, to produce a
    # snapshot "model portfolio" -- NOT investment advice, purely the
    # mechanical output of the same momentum methodology used elsewhere
    # in this project, applied to single stocks instead of the ETF
    # universe above. Customize this list to your own investable
    # universe; this default is only a broad, diversified starting point.
    stock_universe: Tuple[str, ...] = (
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA",
        "JPM", "V", "MA", "BAC",
        "UNH", "JNJ", "PFE", "MRK",
        "PG", "KO", "PEP", "COST", "WMT", "HD", "MCD",
        "XOM", "CVX",
        "CAT", "GE", "BA",
        "DIS", "NFLX", "ADBE", "CRM",
    )
    screener_vol_window: int = 63
    screener_top_n: int = 10
    # Optional hard cap on annualized realized volatility (e.g. 0.45 =
    # 45%/yr); set to None to disable and rely only on the risk-adjusted
    # score itself.
    screener_max_volatility: float | None = 0.45
