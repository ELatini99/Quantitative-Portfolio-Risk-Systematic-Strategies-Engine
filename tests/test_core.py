import numpy as np
import pandas as pd
from unittest.mock import patch

from src.returns import max_drawdown
from src.optimization import minimum_variance_portfolio
from src.risk import historical_var_es
from src.covariance import ledoit_wolf_covariance
from src.momentum import rebalance_schedule, cross_sectional_momentum_weights
from src.validation import generate_folds
from src.sensitivity import walk_forward_grid_search
from src.benchmarks import equal_weight_target_weights, equal_weight_returns, excess_return_stats
from src.factors import rolling_absorption_ratio
from src.regime import regime_scaling_factor, regime_scaled_momentum_weights, regime_scaled_momentum_returns
from src.screener import current_model_portfolio, risk_adjusted_momentum
from src.data import download_prices


def test_max_drawdown():
    r = pd.Series([0.10, -0.20, 0.05])
    assert max_drawdown(r) < 0


def test_weights_sum_to_one():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        columns=["A", "B"],
        index=["A", "B"],
    )
    w = minimum_variance_portfolio(cov)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()


def test_historical_risk_positive():
    r = pd.Series([0.01, -0.02, -0.03, 0.005, 0.01])
    var, es = historical_var_es(r, 0.8)
    assert var >= 0
    assert es >= var


def test_ledoit_wolf_shrinkage_intensity_in_unit_interval():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame(rng.normal(0, 0.01, size=(100, 5)), columns=list("ABCDE"))
    cov, shrinkage = ledoit_wolf_covariance(returns, annualize=False)
    assert 0.0 <= shrinkage <= 1.0
    assert cov.shape == (5, 5)
    # covariance must be symmetric
    assert np.allclose(cov.values, cov.values.T)


def test_rebalance_schedule_only_returns_actual_trading_days():
    # Business-day index for two months: every date in the schedule must
    # be an actual trading day present in the index (this is exactly the
    # bug the "M" -> "ME" fix and this helper guard against).
    index = pd.bdate_range("2024-01-01", "2024-03-31")
    schedule = rebalance_schedule(index, frequency="ME")
    assert len(schedule) == 3
    assert all(date in index for date in schedule)


def test_momentum_weights_hold_between_rebalances():
    # With monthly rebalancing, weights should NOT change on most days --
    # only turnover should occur around rebalance dates plus the initial
    # ramp-up. This guards against the previous bug where the top-K
    # selection was silently re-evaluated every single day.
    rng = np.random.default_rng(1)
    index = pd.bdate_range("2020-01-01", "2021-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(index), 4)), axis=0)),
        index=index,
        columns=list("ABCD"),
    )
    weights = cross_sectional_momentum_weights(
        prices, lookback=60, top_fraction=0.5, rebalance_frequency="ME"
    )
    turnover_days = (weights.diff().abs().sum(axis=1) > 1e-9).sum()
    # ~24 months of history -> turnover should occur on a small number of
    # days (approximately one per rebalance), not on a large fraction of
    # all trading days.
    assert turnover_days < 40


def test_generate_folds_are_chronological_and_non_overlapping_tests():
    index = pd.bdate_range("2015-01-01", "2024-12-31")
    folds = generate_folds(index, train_years=3, test_years=1, step_years=1)
    assert len(folds) > 0
    for fold in folds:
        assert fold.train_start < fold.train_end == fold.test_start < fold.test_end
    for prev, nxt in zip(folds, folds[1:]):
        assert nxt.train_start > prev.train_start


def test_sensitivity_grid_is_scored_out_of_sample_and_covers_every_combination():
    rng = np.random.default_rng(2)
    index = pd.bdate_range("2012-01-01", "2024-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(len(index), 5)), axis=0)),
        index=index,
        columns=list("ABCDE"),
    )
    folds = generate_folds(index, train_years=3, test_years=1, step_years=1)
    grid = walk_forward_grid_search(
        prices,
        folds,
        lookbacks=[63, 126],
        rebalance_frequencies=["ME", "QE"],
        top_fraction=0.4,
    )
    # every (lookback, frequency) combination should produce a row
    assert len(grid) == 4
    assert set(grid["Lookback (days)"]) == {63, 126}
    assert set(grid["Rebalance Frequency"]) == {"ME", "QE"}
    # quarterly rebalancing must never trade more often than monthly
    for lookback in (63, 126):
        subset = grid[grid["Lookback (days)"] == lookback]
        me_turnover = subset.loc[subset["Rebalance Frequency"] == "ME", "Annualized Turnover (x/yr)"].iloc[0]
        qe_turnover = subset.loc[subset["Rebalance Frequency"] == "QE", "Annualized Turnover (x/yr)"].iloc[0]
        assert qe_turnover <= me_turnover + 1e-9


def test_equal_weight_weights_sum_to_one_after_first_rebalance():
    index = pd.bdate_range("2020-01-01", "2020-06-30")
    prices = pd.DataFrame(
        100.0, index=index, columns=list("ABC")
    )
    weights = equal_weight_target_weights(prices, rebalance_frequency="ME")
    # after the first rebalance date each row should sum to ~1 (fully invested)
    post_first_rebalance = weights.iloc[25:]
    assert np.allclose(post_first_rebalance.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(post_first_rebalance.nunique(axis=1), 1)  # exactly equal weight


def test_equal_weight_returns_are_cost_comparable_to_momentum_convention():
    rng = np.random.default_rng(3)
    index = pd.bdate_range("2018-01-01", "2020-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(index), 3)), axis=0)),
        index=index,
        columns=list("ABC"),
    )
    net, weights = equal_weight_returns(prices, transaction_cost_bps=5.0, rebalance_frequency="ME")
    assert len(net) == len(prices)
    assert weights.shape == prices.shape
    # turnover-driven costs mean net return should never exceed gross
    gross = (weights * prices.pct_change().fillna(0.0)).sum(axis=1)
    assert (net <= gross + 1e-12).all()


def test_excess_return_stats_zero_for_identical_series():
    idx = pd.bdate_range("2021-01-01", "2021-06-30")
    r = pd.Series(np.linspace(-0.01, 0.01, len(idx)), index=idx)
    stats = excess_return_stats(r, r)
    assert stats["tracking_error"] == 0.0
    assert np.isclose(stats["correlation"], 1.0)


def test_rolling_absorption_ratio_is_bounded_and_lagged_correctly():
    rng = np.random.default_rng(4)
    index = pd.bdate_range("2020-01-01", "2021-12-31")
    # Highly correlated panel (one common factor should dominate)
    common = rng.normal(0, 0.01, len(index))
    returns = pd.DataFrame(
        {c: common + rng.normal(0, 0.001, len(index)) for c in "ABCD"}, index=index
    )
    ar = rolling_absorption_ratio(returns, window=60, n_components=1)
    valid = ar.dropna()
    assert len(valid) > 0
    assert (valid >= 0).all() and (valid <= 1.0 + 1e-9).all()
    # a single dominant common factor should absorb most of the variance
    assert valid.mean() > 0.8
    # first (window - 1) values must be NaN (no look-ahead)
    assert ar.iloc[:59].isna().all()


def test_regime_scaling_factor_reduces_exposure_when_ar_spikes():
    idx = pd.bdate_range("2020-01-01", "2022-12-31")
    # Absorption ratio flat at a "normal" level, then a late spike
    ar = pd.Series(0.4, index=idx)
    ar.iloc[-5:] = 0.95
    scale = regime_scaling_factor(ar, zscore_window=252, min_scale=0.3, z_low=-1.0, z_high=1.5)
    assert (scale <= 1.0).all() and (scale >= 0.3 - 1e-9).all()
    # exposure right after the spike should be below full exposure
    assert scale.iloc[-1] < scale.iloc[-10]


def test_regime_scaled_momentum_never_exceeds_base_momentum_exposure():
    rng = np.random.default_rng(5)
    index = pd.bdate_range("2015-01-01", "2020-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(len(index), 5)), axis=0)),
        index=index,
        columns=list("ABCDE"),
    )
    scaled_weights, ar, scale = regime_scaled_momentum_weights(
        prices, lookback=63, top_fraction=0.4, rebalance_frequency="ME"
    )
    base_weights = cross_sectional_momentum_weights(
        prices, lookback=63, top_fraction=0.4, rebalance_frequency="ME"
    )
    # scaling can only shrink gross exposure, never amplify it
    assert (scaled_weights.abs().sum(axis=1) <= base_weights.abs().sum(axis=1) + 1e-9).all()


def test_regime_scaled_momentum_returns_runs_end_to_end():
    rng = np.random.default_rng(6)
    index = pd.bdate_range("2015-01-01", "2019-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, size=(len(index), 4)), axis=0)),
        index=index,
        columns=list("ABCD"),
    )
    net, weights, ar, scale = regime_scaled_momentum_returns(
        prices, lookback=63, top_fraction=0.5, transaction_cost_bps=5.0, rebalance_frequency="ME"
    )
    assert len(net) == len(prices)
    assert not net.isna().all()


def test_regime_exposure_only_updates_on_rebalance_dates_not_daily():
    # The exposure multiplier baked into the traded weights must only
    # change on rebalance dates -- a continuously-varying daily scale
    # would reintroduce high-frequency trading purely to track the
    # regime dial, defeating the whole point of fixed-frequency
    # rebalancing established for the base momentum strategy.
    rng = np.random.default_rng(7)
    index = pd.bdate_range("2015-01-01", "2021-12-31")
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.012, size=(len(index), 5)), axis=0)),
        index=index,
        columns=list("ABCDE"),
    )
    scaled_weights, _, _ = regime_scaled_momentum_weights(
        prices, lookback=63, top_fraction=0.4, rebalance_frequency="ME"
    )
    rebalance_dates = rebalance_schedule(prices.index, "ME")
    turnover_days = (scaled_weights.diff().abs().sum(axis=1) > 1e-9).sum()
    # turnover can only occur on (or the day after, given the 1-day lag)
    # a rebalance date -- generously allow +2 per rebalance for the lag
    # and edge effects, still far below "changes every day" territory.
    assert turnover_days <= len(rebalance_dates) * 2 + 5


def test_current_model_portfolio_only_includes_positive_momentum():
    rng = np.random.default_rng(8)
    index = pd.bdate_range("2020-01-01", "2023-12-31")
    tickers = [f"S{i}" for i in range(15)]
    drift = rng.uniform(-0.06, 0.18, len(tickers)) / 252
    noise = rng.normal(0, 0.015, size=(len(index), len(tickers)))
    logret = drift + noise
    prices = pd.DataFrame(100 * np.exp(np.cumsum(logret, axis=0)), index=index, columns=tickers)

    picks = current_model_portfolio(prices, momentum_lookback=126, vol_window=63, top_n=5)
    if not picks.empty:
        assert (picks["Momentum Return (%)"] > 0).all()
        assert np.isclose(picks["Target Weight (%)"].sum(), 100.0, atol=1e-6)
        # higher realized volatility should never earn more weight than a
        # lower-volatility name with a similar or better score (inverse-vol)
        assert picks["Target Weight (%)"].idxmax() != picks["Realized Volatility (%)"].idxmax() \
            or len(picks) == 1


def test_current_model_portfolio_respects_volatility_cap():
    rng = np.random.default_rng(9)
    index = pd.bdate_range("2020-01-01", "2023-12-31")
    tickers = [f"S{i}" for i in range(15)]
    drift = rng.uniform(0.02, 0.15, len(tickers)) / 252
    vol_scale = rng.uniform(0.1, 0.9, len(tickers))
    noise = rng.normal(0, 1, size=(len(index), len(tickers))) * vol_scale / np.sqrt(252)
    logret = drift + noise
    prices = pd.DataFrame(100 * np.exp(np.cumsum(logret, axis=0)), index=index, columns=tickers)

    picks = current_model_portfolio(
        prices, momentum_lookback=126, vol_window=63, top_n=10, max_volatility=0.30
    )
    if not picks.empty:
        assert (picks["Realized Volatility (%)"] <= 30.0 + 1e-6).all()


def test_current_model_portfolio_empty_when_nothing_qualifies():
    index = pd.bdate_range("2020-01-01", "2023-12-31")
    tickers = [f"S{i}" for i in range(5)]
    # deterministic, monotonic downward drift, no noise -> momentum is
    # guaranteed negative for every asset, every date
    drift = -0.20 / 252
    logret = np.full((len(index), len(tickers)), drift)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(logret, axis=0)), index=index, columns=tickers)

    picks = current_model_portfolio(prices, momentum_lookback=126, vol_window=63, top_n=5)
    assert picks.empty
    assert list(picks.columns) == [
        "Ticker", "Momentum Return (%)", "Realized Volatility (%)",
        "Risk-Adjusted Momentum Score", "Target Weight (%)",
    ]


def test_download_prices_drops_failed_tickers_without_zeroing_all_rows():
    # A single entirely-failed ticker (delisted, mistyped, transient
    # per-symbol error) must not silently wipe out every date for every
    # OTHER ticker -- this is the exact bug that broke the company
    # screener for larger universes: a plain row-wise dropna() requires
    # every column to have data on a given day to keep that day.
    dates = pd.bdate_range("2022-01-01", "2022-06-30")
    good_data = pd.DataFrame(
        {
            "GOOD1": np.linspace(100, 110, len(dates)),
            "GOOD2": np.linspace(50, 55, len(dates)),
            "BAD": [np.nan] * len(dates),
        },
        index=dates,
    )
    multi = pd.concat({"Close": good_data}, axis=1)

    with patch("src.data.yf.download", return_value=multi):
        prices = download_prices(
            ["GOOD1", "GOOD2", "BAD"], start="2022-01-01", use_synthetic_fallback=False
        )

    assert "BAD" not in prices.columns
    assert set(prices.columns) == {"GOOD1", "GOOD2"}
    assert len(prices) == len(dates)
