from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import turnover
from .factors import rolling_absorption_ratio
from .momentum import cross_sectional_momentum_weights, rebalance_schedule
from .returns import annualized_return, annualized_volatility, max_drawdown, sharpe_ratio
from .validation import Fold


def regime_scaling_factor(
    absorption_ratio: pd.Series,
    zscore_window: int = 252,
    min_scale: float = 0.3,
    z_low: float = -1.0,
    z_high: float = 1.5,
) -> pd.Series:
    """
    Map the absorption ratio (see `src.factors.rolling_absorption_ratio`)
    into a momentum exposure multiplier in [min_scale, 1.0].

    The absorption ratio is compared to its OWN trailing history (a
    rolling z-score) rather than to an arbitrary fixed threshold, since
    what counts as "unusually concentrated" is universe-specific. Full
    exposure (1.0) is kept while factor concentration is at or below its
    normal historical range (z <= z_low); exposure scales down linearly
    as concentration rises above that range, down to `min_scale` once
    the z-score reaches `z_high` -- a market unusually dominated by a
    single common driver, the textbook "risk-off" signature where
    diversification stops working and momentum crashes are more likely.
    """
    rolling_mean = absorption_ratio.rolling(zscore_window, min_periods=max(zscore_window // 4, 20)).mean()
    rolling_std = absorption_ratio.rolling(zscore_window, min_periods=max(zscore_window // 4, 20)).std()
    z = (absorption_ratio - rolling_mean) / rolling_std.replace(0.0, np.nan)

    scale = 1.0 - (z - z_low) / (z_high - z_low)
    scale = scale.clip(lower=min_scale, upper=1.0)
    # Default to full exposure while there isn't enough history yet to
    # compute a meaningful z-score (rather than silently zeroing out).
    return scale.fillna(1.0)


def regime_scaled_momentum_weights(
    prices: pd.DataFrame,
    lookback: int = 126,
    top_fraction: float = 0.3,
    rebalance_frequency: str = "ME",
    ar_window: int = 126,
    ar_n_components: int = 1,
    zscore_window: int = 252,
    min_scale: float = 0.3,
    z_low: float = -1.0,
    z_high: float = 1.5,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Cross-sectional momentum weights (see `src.momentum`), scaled down
    when the market's variance becomes unusually concentrated in its
    first principal component.

    Returns (scaled_weights, absorption_ratio, exposure_scale). `exposure_scale`
    is the continuous daily signal (useful for the diagnostic chart showing
    how the regime indicator moves day to day); the scale actually baked
    into `scaled_weights` is only RE-SAMPLED at the same rebalance dates as
    the momentum weights and held constant in between -- for exactly the
    same reason the momentum selection itself is held between rebalances:
    applying a continuously-varying daily scale to the traded weights
    would reintroduce high-frequency trading purely to track the exposure
    dial, on top of whatever turnover the momentum selection itself
    generates. The scale is computed from the raw (unshifted) absorption
    ratio and then shifted by one day, exactly mirroring the one-day
    execution lag already applied inside `cross_sectional_momentum_weights`
    -- so the regime signal uses the same "known as of yesterday's close"
    information cut as the momentum signal itself, no extra look-ahead
    advantage either way.
    """
    base_weights = cross_sectional_momentum_weights(
        prices,
        lookback=lookback,
        top_fraction=top_fraction,
        rebalance_frequency=rebalance_frequency,
    )
    asset_returns = prices.pct_change().fillna(0.0)
    absorption_ratio = rolling_absorption_ratio(
        asset_returns, window=ar_window, n_components=ar_n_components
    )
    continuous_scale = regime_scaling_factor(
        absorption_ratio,
        zscore_window=zscore_window,
        min_scale=min_scale,
        z_low=z_low,
        z_high=z_high,
    ).shift(1).fillna(1.0)

    rebalance_dates = set(rebalance_schedule(prices.index, rebalance_frequency))
    held_scale = pd.Series(index=prices.index, dtype=float)
    current = 1.0
    for date in prices.index:
        if date in rebalance_dates and date in continuous_scale.index:
            current = continuous_scale.loc[date]
        held_scale.loc[date] = current

    aligned_scale = held_scale.reindex(base_weights.index).fillna(1.0)
    scaled_weights = base_weights.mul(aligned_scale, axis=0)
    return scaled_weights, absorption_ratio, continuous_scale


def regime_scaled_momentum_returns(
    prices: pd.DataFrame,
    lookback: int = 126,
    top_fraction: float = 0.3,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "ME",
    ar_window: int = 126,
    ar_n_components: int = 1,
    zscore_window: int = 252,
    min_scale: float = 0.3,
    z_low: float = -1.0,
    z_high: float = 1.5,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, pd.Series]:
    """
    Net daily returns of the regime-scaled momentum strategy, plus the
    underlying absorption ratio and exposure-scale diagnostics.

    Transaction costs are computed from the SCALED weights' turnover, so
    the cost of actually delevering/relevering as the regime signal moves
    is captured too, not just the cost of the underlying momentum trades.
    """
    asset_returns = prices.pct_change().fillna(0.0)
    weights, absorption_ratio, scale = regime_scaled_momentum_weights(
        prices,
        lookback=lookback,
        top_fraction=top_fraction,
        rebalance_frequency=rebalance_frequency,
        ar_window=ar_window,
        ar_n_components=ar_n_components,
        zscore_window=zscore_window,
        min_scale=min_scale,
        z_low=z_low,
        z_high=z_high,
    )
    gross = (weights * asset_returns).sum(axis=1)
    costs = turnover(weights) * transaction_cost_bps / 10_000.0
    net = gross - costs
    return net, weights, absorption_ratio, scale


def walk_forward_regime_momentum_validation(
    prices: pd.DataFrame,
    folds: list[Fold],
    lookback: int = 126,
    top_fraction: float = 0.3,
    transaction_cost_bps: float = 5.0,
    rebalance_frequency: str = "ME",
    ar_window: int = 126,
    ar_n_components: int = 1,
    zscore_window: int = 252,
    min_scale: float = 0.3,
    z_low: float = -1.0,
    z_high: float = 1.5,
    trading_days: int = 252,
    annual_risk_free_rate: float = 0.0,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Same rolling train/test walk-forward schedule as
    `src.validation.walk_forward_momentum_validation`, applied to the
    regime-scaled momentum strategy instead of the plain one, so the two
    can be compared fold-by-fold on identical out-of-sample windows --
    the regime overlay must earn its place out-of-sample too, not just
    look better over one full-sample backtest.
    """
    oos_segments = []
    fold_rows = []

    for i, fold in enumerate(folds):
        window_prices = prices.loc[fold.train_start: fold.test_end]
        net_returns, _, _, _ = regime_scaled_momentum_returns(
            window_prices,
            lookback=lookback,
            top_fraction=top_fraction,
            transaction_cost_bps=transaction_cost_bps,
            rebalance_frequency=rebalance_frequency,
            ar_window=ar_window,
            ar_n_components=ar_n_components,
            zscore_window=zscore_window,
            min_scale=min_scale,
            z_low=z_low,
            z_high=z_high,
        )
        test_returns = net_returns.loc[fold.test_start: fold.test_end]
        if test_returns.empty:
            continue

        oos_segments.append(test_returns)
        fold_rows.append(
            {
                "Fold": i + 1,
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
