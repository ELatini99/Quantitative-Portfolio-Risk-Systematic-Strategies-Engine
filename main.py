from __future__ import annotations

import os
import sys
from pathlib import Path

# Force the non-interactive "Agg" backend BEFORE matplotlib is imported at
# all (both via the env var AND matplotlib.use(..., force=True)). This
# script only ever saves figures to disk (plt.savefig + plt.close), it
# never shows an interactive window -- letting matplotlib auto-select a
# GUI backend (e.g. TkAgg, common by default on Windows) causes spurious
# "Exception ignored ... main thread is not in main loop" / "Tcl_AsyncDelete"
# tkinter errors during interpreter shutdown as the many figures created
# in this batch run get garbage-collected. Setting MPLBACKEND covers cases
# where some other import path initializes matplotlib's backend before
# this module's own matplotlib.use() call would otherwise run.
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.data import align_assets, download_prices
from src.returns import (
    simple_returns,
    performance_summary,
    sharpe_ratio,
    max_drawdown,
    annualized_return,
    annualized_volatility,
)
from src.factors import pca_factors, factor_loadings
from src.covariance import (
    sample_covariance,
    ewma_covariance,
    ledoit_wolf_covariance,
    correlation_matrix,
)
from src.optimization import minimum_variance_portfolio, risk_parity_portfolio
from src.risk import (
    portfolio_returns,
    historical_var_es,
    parametric_var_es,
    monte_carlo_var_es,
)
from src.stress import historical_stress, scenario_stress
from src.backtest import cumulative_wealth
from src.momentum import momentum_returns, momentum_scores
from src.validation import generate_folds, walk_forward_momentum_validation
from src.sensitivity import walk_forward_grid_search
from src.benchmarks import buy_and_hold_single_asset, equal_weight_returns, excess_return_stats
from src.regime import regime_scaled_momentum_returns, walk_forward_regime_momentum_validation
from src.screener import current_model_portfolio, risk_adjusted_momentum, DISCLAIMER as SCREENER_DISCLAIMER
from src.reporting import write_table, create_excel_report


def pct(x: float) -> float:
    return float(x * 100.0)


def risk_contributions(weights: pd.Series, covariance: pd.DataFrame) -> pd.Series:
    """Percentage contribution of each asset to total portfolio volatility."""
    w = weights.loc[covariance.index].to_numpy()
    cov = covariance.to_numpy()
    sigma = np.sqrt(w @ cov @ w)
    marginal = cov @ w / max(sigma, 1e-16)
    contribution = w * marginal
    return pd.Series(contribution / max(sigma, 1e-16), index=covariance.index)


def main() -> None:
    cfg = Config()
    results_dir = ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    raw_dir = results_dir / "raw"
    figures_dir = results_dir / "figures"
    raw_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    print("=" * 72)
    print("QUANTITATIVE PORTFOLIO RISK & SYSTEMATIC STRATEGIES ENGINE")
    print("=" * 72)

    print("\n[1/14] Downloading market data...")
    prices = align_assets(
        download_prices(
            cfg.tickers,
            cfg.start,
            cfg.end,
            use_synthetic_fallback=cfg.use_synthetic_fallback,
            seed=cfg.random_seed,
        )
    )
    returns = simple_returns(prices)

    start_date = prices.index.min().date().isoformat()
    end_date = prices.index.max().date().isoformat()
    as_of = end_date

    print(f"Universe: {', '.join(cfg.tickers)}")
    print(f"Period:   {start_date} -> {end_date}")
    print(f"Observations: {len(prices):,}")

    # ------------------------------------------------------------------
    # 1. Asset performance
    # ------------------------------------------------------------------
    print("\n[2/14] Computing asset performance statistics...")
    summary = performance_summary(
        returns,
        annual_risk_free_rate=cfg.annual_risk_free_rate,
        periods=cfg.trading_days,
    )

    asset_report = summary.reset_index().rename(columns={"index": "Asset"})
    asset_report["Annual Return"] = asset_report["Annual Return"].map(pct)
    asset_report["Annual Volatility"] = asset_report["Annual Volatility"].map(pct)
    asset_report["Max Drawdown"] = asset_report["Max Drawdown"].map(pct)
    asset_report["Start Date"] = start_date
    asset_report["End Date"] = end_date
    asset_report["Observations"] = returns.notna().sum().reindex(asset_report["Asset"]).values

    asset_report = asset_report[
        [
            "Asset",
            "Annual Return",
            "Annual Volatility",
            "Sharpe",
            "Max Drawdown",
            "Start Date",
            "End Date",
            "Observations",
        ]
    ]
    write_table(
        asset_report,
        raw_dir / "asset_performance.csv",
        float_format="%.4f",
    )

    # ------------------------------------------------------------------
    # 2. PCA / factor structure
    # ------------------------------------------------------------------
    print("[3/14] Estimating PCA factor structure...")
    n_components = min(5, len(cfg.tickers))
    factors, explained = pca_factors(returns, n_components=n_components)
    loadings = factor_loadings(returns, n_components=min(3, len(cfg.tickers)))

    pca_report = pd.DataFrame(
        {
            "Component": [f"PC{i+1}" for i in range(len(explained))],
            "Explained Variance (%)": explained * 100,
            "Cumulative Variance (%)": np.cumsum(explained) * 100,
        }
    )
    write_table(
        pca_report,
        raw_dir / "pca_explained_variance.csv",
        float_format="%.4f",
    )

    pca_loadings = loadings.reset_index().rename(columns={"index": "Asset"})
    write_table(
        pca_loadings,
        raw_dir / "pca_loadings.csv",
        float_format="%.6f",
    )

    plt.figure(figsize=(9, 5))
    plt.bar(pca_report["Component"], pca_report["Explained Variance (%)"])
    plt.xlabel("Principal component")
    plt.ylabel("Explained variance (%)")
    plt.title("PCA Factor Structure")
    plt.tight_layout()
    plt.savefig(figures_dir / "pca_explained_variance.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------------
    # 3. Covariance and correlation
    # ------------------------------------------------------------------
    print("[4/14] Estimating covariance and correlation matrices...")
    cov_sample = sample_covariance(returns)
    cov_ewma = ewma_covariance(returns)
    cov_shrink, shrink_intensity = ledoit_wolf_covariance(returns)
    corr = correlation_matrix(returns)

    print(f"        Ledoit-Wolf shrinkage intensity: {shrink_intensity:.3f} "
          f"(0 = pure sample cov, 1 = pure structured target)")

    cov_sample.to_csv(
        raw_dir / "sample_covariance.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    cov_ewma.to_csv(
        raw_dir / "ewma_covariance.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    cov_shrink.to_csv(
        raw_dir / "ledoit_wolf_covariance.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    corr.to_csv(
        raw_dir / "correlation_matrix.csv",
        encoding="utf-8-sig",
        float_format="%.6f",
    )

    # A long-form comparison is easier to inspect/filter in Excel.
    covariance_comparison = []
    for i in cov_ewma.index:
        for j in cov_ewma.columns:
            covariance_comparison.append(
                {
                    "Asset A": i,
                    "Asset B": j,
                    "Sample Covariance": cov_sample.loc[i, j],
                    "EWMA Covariance": cov_ewma.loc[i, j],
                    "Ledoit-Wolf Covariance": cov_shrink.loc[i, j],
                    "Difference (EWMA - Sample)": cov_ewma.loc[i, j] - cov_sample.loc[i, j],
                    "Difference (%) (EWMA vs Sample)": (
                        100
                        * (cov_ewma.loc[i, j] - cov_sample.loc[i, j])
                        / abs(cov_sample.loc[i, j])
                        if cov_sample.loc[i, j] != 0
                        else np.nan
                    ),
                }
            )

    write_table(
        pd.DataFrame(covariance_comparison),
        raw_dir / "covariance_comparison.csv",
        float_format="%.6f",
    )

    # ------------------------------------------------------------------
    # 4. Portfolio construction
    # ------------------------------------------------------------------
    print("[5/14] Constructing portfolios...")
    # Ledoit-Wolf shrinkage is used for portfolio construction: it is
    # materially more stable than the raw sample covariance whenever T is
    # not much larger than N, which is the case for most trading
    # universes. Sample and EWMA covariances are still computed and
    # reported above for comparison.
    w_minvar = minimum_variance_portfolio(cov_shrink)
    w_riskparity = risk_parity_portfolio(cov_shrink)

    rc_minvar = risk_contributions(w_minvar, cov_shrink)
    rc_rp = risk_contributions(w_riskparity, cov_shrink)

    weights = pd.DataFrame(
        {
            "Asset": cov_shrink.index,
            "Min Variance Weight (%)": w_minvar.reindex(cov_shrink.index).values * 100,
            "Min Variance Risk Contribution (%)": rc_minvar.values * 100,
            "Risk Parity Weight (%)": w_riskparity.reindex(cov_shrink.index).values * 100,
            "Risk Parity Risk Contribution (%)": rc_rp.values * 100,
        }
    )

    write_table(
        weights,
        raw_dir / "portfolio_weights.csv",
        float_format="%.4f",
    )

    minvar_r = portfolio_returns(returns, w_minvar)
    rp_r = portfolio_returns(returns, w_riskparity)

    # ------------------------------------------------------------------
    # 5. Portfolio risk analytics
    # ------------------------------------------------------------------
    print("[6/14] Computing VaR, Expected Shortfall and performance...")
    risk_rows = []

    for name, series in {
        "Minimum Variance": minvar_r,
        "Risk Parity": rp_r,
    }.items():
        h_var, h_es = historical_var_es(series, cfg.confidence)
        p_var, p_es = parametric_var_es(series, cfg.confidence)
        mc_var, mc_es = monte_carlo_var_es(
            series,
            confidence=cfg.confidence,
            n_paths=cfg.monte_carlo_paths,
            seed=cfg.random_seed,
        )

        risk_rows.append(
            {
                "Portfolio": name,
                "Confidence Level (%)": cfg.confidence * 100,
                "Historical VaR (%)": h_var * 100,
                "Historical ES (%)": h_es * 100,
                "Gaussian VaR (%)": p_var * 100,
                "Gaussian ES (%)": p_es * 100,
                "Monte Carlo VaR (%)": mc_var * 100,
                "Monte Carlo ES (%)": mc_es * 100,
                "Annual Return (%)": annualized_return(series, cfg.trading_days) * 100,
                "Annual Volatility (%)": annualized_volatility(series, cfg.trading_days) * 100,
                "Sharpe Ratio": sharpe_ratio(
                    series, cfg.annual_risk_free_rate, cfg.trading_days
                ),
                "Max Drawdown (%)": max_drawdown(series) * 100,
                "Observations": series.notna().sum(),
            }
        )

    risk_table = pd.DataFrame(risk_rows)
    write_table(
        risk_table,
        raw_dir / "portfolio_risk_analytics.csv",
        float_format="%.4f",
    )

    # ------------------------------------------------------------------
    # 6. Stress testing
    # ------------------------------------------------------------------
    print("[7/14] Running stress tests...")
    stress = historical_stress(returns, w_minvar, window=5)

    historical_stress_report = pd.DataFrame(
        {
            "Rank": range(1, min(10, len(stress)) + 1),
            "Window": "5 trading days",
            "Portfolio": "Minimum Variance",
            "Cumulative Loss (%)": -stress.head(10).values * 100,
            "Period End": stress.head(10).index.strftime("%Y-%m-%d"),
        }
    )
    write_table(
        historical_stress_report,
        raw_dir / "worst_5day_stress.csv",
        float_format="%.4f",
    )

    scenarios = pd.DataFrame(
        {
            "SPY": [-0.20, -0.10, -0.10],
            "QQQ": [-0.25, -0.12, -0.15],
            "IWM": [-0.30, -0.15, -0.18],
            "EFA": [-0.15, -0.08, -0.08],
            "EEM": [-0.25, -0.10, -0.12],
            "TLT": [0.05, -0.15, 0.08],
            "GLD": [0.00, 0.00, 0.03],
        },
        index=["Equity Crash", "Rates Shock", "Risk-Off"],
    )

    scenario_results = scenario_stress(w_minvar, scenarios)
    scenario_report = pd.DataFrame(
        {
            "Scenario": scenario_results.index,
            "Portfolio Return (%)": scenario_results.values * 100,
            "Portfolio Loss (%)": -scenario_results.values * 100,
        }
    )
    write_table(
        scenario_report,
        raw_dir / "scenario_stress.csv",
        float_format="%.4f",
    )

    scenario_asset_report = scenarios.reset_index().rename(columns={"index": "Scenario"})
    scenario_asset_report.to_csv(
        raw_dir / "scenario_asset_shocks.csv",
        index=False,
        encoding="utf-8-sig",
        float_format="%.4f",
    )

    # ------------------------------------------------------------------
    # 7. Momentum strategy
    # ------------------------------------------------------------------
    print("[8/14] Backtesting momentum strategy...")
    momentum_r, momentum_weights = momentum_returns(
        prices,
        lookback=cfg.lookback_momentum,
        top_fraction=cfg.top_fraction,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance_frequency=cfg.rebalance_frequency,
    )

    momentum_hvar, momentum_hes = historical_var_es(
        momentum_r, cfg.confidence
    )
    momentum_summary = pd.DataFrame(
        [
            {
                "Strategy": "Cross-Sectional Momentum",
                "Momentum Lookback (days)": cfg.lookback_momentum,
                "Top Fraction (%)": cfg.top_fraction * 100,
                "Rebalance Frequency": cfg.rebalance_frequency,
                "Transaction Cost (bps)": cfg.transaction_cost_bps,
                "Annual Return (%)": annualized_return(momentum_r, cfg.trading_days) * 100,
                "Annual Volatility (%)": annualized_volatility(momentum_r, cfg.trading_days) * 100,
                "Sharpe Ratio": sharpe_ratio(
                    momentum_r, cfg.annual_risk_free_rate, cfg.trading_days
                ),
                "Max Drawdown (%)": max_drawdown(momentum_r) * 100,
                "Historical VaR (%)": momentum_hvar * 100,
                "Historical ES (%)": momentum_hes * 100,
                "Observations": momentum_r.notna().sum(),
            }
        ]
    )
    write_table(
        momentum_summary,
        raw_dir / "momentum_performance.csv",
        float_format="%.4f",
    )

    # Average weights are more informative than a huge daily weight CSV.
    avg_momentum_weights = (
        momentum_weights.mean()
        .sort_values(ascending=False)
        .rename("Average Weight (%)")
        .mul(100)
        .reset_index()
        .rename(columns={"index": "Asset"})
    )
    write_table(
        avg_momentum_weights,
        raw_dir / "momentum_average_weights.csv",
        float_format="%.4f",
    )

    # ------------------------------------------------------------------
    # 8b. Walk-forward validation
    # ------------------------------------------------------------------
    print(f"[9/14] Running walk-forward validation "
          f"({cfg.walk_forward_train_years}y train / {cfg.walk_forward_test_years}y test)...")
    folds = generate_folds(
        prices.index,
        train_years=cfg.walk_forward_train_years,
        test_years=cfg.walk_forward_test_years,
        step_years=cfg.walk_forward_step_years,
    )
    oos_returns, fold_summary = walk_forward_momentum_validation(
        prices,
        folds,
        lookback=cfg.lookback_momentum,
        top_fraction=cfg.top_fraction,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance_frequency=cfg.rebalance_frequency,
        trading_days=cfg.trading_days,
        annual_risk_free_rate=cfg.annual_risk_free_rate,
    )

    if fold_summary.empty:
        print("        Not enough history for a full walk-forward schedule with "
              "these window sizes -- skipping (shrink the windows or use more history).")
        wf_aggregate = pd.DataFrame()
    else:
        write_table(
            fold_summary,
            raw_dir / "walk_forward_folds.csv",
            float_format="%.4f",
        )
        wf_aggregate = pd.DataFrame(
            [
                {
                    "Metric": "Out-of-sample annual return (%)",
                    "Value": annualized_return(oos_returns, cfg.trading_days) * 100,
                },
                {
                    "Metric": "Out-of-sample annual volatility (%)",
                    "Value": annualized_volatility(oos_returns, cfg.trading_days) * 100,
                },
                {
                    "Metric": "Out-of-sample Sharpe ratio",
                    "Value": sharpe_ratio(oos_returns, cfg.annual_risk_free_rate, cfg.trading_days),
                },
                {
                    "Metric": "Out-of-sample max drawdown (%)",
                    "Value": max_drawdown(oos_returns) * 100,
                },
                {"Metric": "Number of folds", "Value": len(fold_summary)},
                {"Metric": "Out-of-sample observations", "Value": int(oos_returns.notna().sum())},
            ]
        )
        write_table(
            wf_aggregate,
            raw_dir / "walk_forward_aggregate.csv",
            float_format="%.4f",
        )

        plt.figure(figsize=(9, 5))
        colors = ["#4C72B0" if s >= 0 else "#C44E52" for s in fold_summary["Sharpe Ratio"]]
        plt.bar(fold_summary["Fold"].astype(str), fold_summary["Sharpe Ratio"], color=colors)
        plt.axhline(0, linewidth=1, color="black")
        plt.xlabel("Walk-forward fold")
        plt.ylabel("Out-of-sample Sharpe ratio")
        plt.title("Walk-Forward Validation — Out-of-Sample Sharpe by Fold")
        plt.tight_layout()
        wf_fig = figures_dir / "walk_forward_sharpe_by_fold.png"
        plt.savefig(wf_fig, dpi=200, bbox_inches="tight")
        plt.close()

    # ------------------------------------------------------------------
    # 8c. Parameter sensitivity: lookback x rebalance frequency (OOS only)
    # ------------------------------------------------------------------
    print(f"[10/14] Running parameter sensitivity grid "
          f"({len(cfg.sensitivity_lookbacks)} lookbacks x "
          f"{len(cfg.sensitivity_rebalance_frequencies)} frequencies, "
          f"scored out-of-sample)...")
    sensitivity = walk_forward_grid_search(
        prices,
        folds,
        lookbacks=list(cfg.sensitivity_lookbacks),
        rebalance_frequencies=list(cfg.sensitivity_rebalance_frequencies),
        top_fraction=cfg.top_fraction,
        transaction_cost_bps=cfg.transaction_cost_bps,
        trading_days=cfg.trading_days,
        annual_risk_free_rate=cfg.annual_risk_free_rate,
    )

    if sensitivity.empty:
        print("        Not enough history to run the sensitivity grid -- skipping.")
        freq_at_default_lookback = pd.DataFrame()
    else:
        write_table(
            sensitivity,
            raw_dir / "sensitivity_grid.csv",
            float_format="%.4f",
        )

        # Heatmap: lookback (rows) x rebalance frequency (columns) -> OOS Sharpe.
        pivot = sensitivity.pivot(
            index="Lookback (days)", columns="Rebalance Frequency", values="OOS Sharpe Ratio"
        ).reindex(columns=list(cfg.sensitivity_rebalance_frequencies))

        fig, ax = plt.subplots(figsize=(7, 5))
        im = ax.imshow(pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_xlabel("Rebalance frequency")
        ax.set_ylabel("Momentum lookback (days)")
        ax.set_title("Out-of-Sample Sharpe -- Parameter Sensitivity")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                val = pivot.values[i, j]
                if pd.notna(val):
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax, label="OOS Sharpe Ratio")
        fig.tight_layout()
        sensitivity_fig = figures_dir / "sensitivity_grid.png"
        fig.savefig(sensitivity_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)

        # Direct rebalance-frequency comparison at the production lookback:
        # OOS Sharpe alongside annualized turnover, so the cost/reactivity
        # trade-off between e.g. monthly and quarterly rebalancing is explicit.
        freq_at_default_lookback = sensitivity[
            sensitivity["Lookback (days)"] == cfg.lookback_momentum
        ].sort_values("Rebalance Frequency")

        if not freq_at_default_lookback.empty:
            write_table(
                freq_at_default_lookback,
                raw_dir / "rebalance_frequency_comparison.csv",
                float_format="%.4f",
            )

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
            ax1.bar(freq_at_default_lookback["Rebalance Frequency"],
                    freq_at_default_lookback["OOS Sharpe Ratio"], color="#4C72B0")
            ax1.axhline(0, linewidth=1, color="black")
            ax1.set_title(f"OOS Sharpe @ {cfg.lookback_momentum}d lookback")
            ax1.set_ylabel("OOS Sharpe Ratio")

            ax2.bar(freq_at_default_lookback["Rebalance Frequency"],
                    freq_at_default_lookback["Annualized Turnover (x/yr)"], color="#C44E52")
            ax2.set_title("Annualized Turnover")
            ax2.set_ylabel("Turnover (x/year)")

            fig.suptitle("Rebalance Frequency Comparison: Weekly vs Monthly vs Quarterly")
            fig.tight_layout()
            freq_fig = figures_dir / "rebalance_frequency_comparison.png"
            fig.savefig(freq_fig, dpi=200, bbox_inches="tight")
            plt.close(fig)

    # ------------------------------------------------------------------
    # 8d. Passive benchmark comparison
    # ------------------------------------------------------------------
    print(f"[11/14] Computing benchmark comparison "
          f"(buy & hold {cfg.benchmark_ticker}, equal-weight)...")

    ew_r, ew_weights = equal_weight_returns(
        prices,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance_frequency=cfg.rebalance_frequency,
    )

    has_bh_benchmark = cfg.benchmark_ticker in returns.columns
    if has_bh_benchmark:
        bh_r = buy_and_hold_single_asset(returns, cfg.benchmark_ticker)
    else:
        print(f"        Benchmark ticker '{cfg.benchmark_ticker}' not in the "
              f"universe -- skipping the single-asset buy & hold benchmark.")
        bh_r = None

    strategy_series = {
        "Minimum Variance": minvar_r,
        "Risk Parity": rp_r,
        "Momentum": momentum_r,
        "Equal Weight": ew_r,
    }
    if has_bh_benchmark:
        strategy_series[f"Buy & Hold {cfg.benchmark_ticker}"] = bh_r

    benchmark_rows = []
    for name, r in strategy_series.items():
        row = {
            "Strategy": name,
            "Annual Return (%)": annualized_return(r, cfg.trading_days) * 100,
            "Annual Volatility (%)": annualized_volatility(r, cfg.trading_days) * 100,
            "Sharpe Ratio": sharpe_ratio(r, cfg.annual_risk_free_rate, cfg.trading_days),
            "Max Drawdown (%)": max_drawdown(r) * 100,
        }
        if has_bh_benchmark and name != f"Buy & Hold {cfg.benchmark_ticker}":
            stats = excess_return_stats(r, bh_r, periods=cfg.trading_days)
            row[f"Correlation vs {cfg.benchmark_ticker}"] = stats["correlation"]
            row["Tracking Error (%)"] = stats["tracking_error"] * 100
            row["Information Ratio"] = stats["information_ratio"]
            row[f"Excess Ann. Return vs {cfg.benchmark_ticker} (%)"] = (
                row["Annual Return (%)"] - annualized_return(bh_r, cfg.trading_days) * 100
            )
        benchmark_rows.append(row)

    benchmark_comparison = pd.DataFrame(benchmark_rows)
    write_table(
        benchmark_comparison,
        raw_dir / "benchmark_comparison.csv",
        float_format="%.4f",
    )

    # Like-for-like check: how do the benchmarks perform restricted to
    # EXACTLY the same out-of-sample days used for the momentum
    # walk-forward number above? A momentum Sharpe that only looks good
    # over the full sample but not over its own OOS windows would be a
    # red flag; this makes sure the benchmark is judged on the same clock.
    if not oos_returns.empty:
        oos_index = oos_returns.index
        oos_bench_rows = [
            {
                "Strategy": "Momentum (OOS)",
                "Annual Return (%)": annualized_return(oos_returns, cfg.trading_days) * 100,
                "Sharpe Ratio": sharpe_ratio(oos_returns, cfg.annual_risk_free_rate, cfg.trading_days),
            },
            {
                "Strategy": "Equal Weight (same OOS days)",
                "Annual Return (%)": annualized_return(ew_r.reindex(oos_index).dropna(), cfg.trading_days) * 100,
                "Sharpe Ratio": sharpe_ratio(ew_r.reindex(oos_index).dropna(), cfg.annual_risk_free_rate, cfg.trading_days),
            },
        ]
        if has_bh_benchmark:
            oos_bench_rows.append(
                {
                    "Strategy": f"Buy & Hold {cfg.benchmark_ticker} (same OOS days)",
                    "Annual Return (%)": annualized_return(bh_r.reindex(oos_index).dropna(), cfg.trading_days) * 100,
                    "Sharpe Ratio": sharpe_ratio(bh_r.reindex(oos_index).dropna(), cfg.annual_risk_free_rate, cfg.trading_days),
                }
            )
        oos_benchmark_comparison = pd.DataFrame(oos_bench_rows)
        write_table(
            oos_benchmark_comparison,
            raw_dir / "walk_forward_benchmark_comparison.csv",
            float_format="%.4f",
        )
    else:
        oos_benchmark_comparison = pd.DataFrame()

    bench_fig = figures_dir / "benchmark_comparison.png"

    # ------------------------------------------------------------------
    # 8e. Regime-conditioned momentum: PCA absorption ratio -> exposure scaling
    # ------------------------------------------------------------------
    print("[12/14] Building regime-conditioned momentum "
          "(PCA absorption ratio -> exposure scaling)...")
    regime_r, regime_weights, absorption_ratio, exposure_scale = regime_scaled_momentum_returns(
        prices,
        lookback=cfg.lookback_momentum,
        top_fraction=cfg.top_fraction,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance_frequency=cfg.rebalance_frequency,
        ar_window=cfg.absorption_ratio_window,
        ar_n_components=cfg.absorption_ratio_n_components,
        zscore_window=cfg.regime_zscore_window,
        min_scale=cfg.regime_min_exposure,
        z_low=cfg.regime_zscore_low,
        z_high=cfg.regime_zscore_high,
    )

    regime_signal = pd.DataFrame(
        {"Absorption Ratio": absorption_ratio, "Momentum Exposure Scale": exposure_scale}
    )
    regime_signal.index.name = "Date"
    write_table(
        regime_signal.reset_index(),
        raw_dir / "regime_absorption_ratio.csv",
        float_format="%.6f",
    )

    valid_ar = absorption_ratio.dropna()
    valid_scale = exposure_scale.reindex(valid_ar.index)
    regime_summary = pd.DataFrame(
        [
            {"Metric": "Absorption Ratio -- current", "Value": valid_ar.iloc[-1] if len(valid_ar) else np.nan},
            {"Metric": "Absorption Ratio -- historical mean", "Value": valid_ar.mean()},
            {"Metric": "Absorption Ratio -- historical min", "Value": valid_ar.min()},
            {"Metric": "Absorption Ratio -- historical max", "Value": valid_ar.max()},
            {"Metric": "Momentum exposure scale -- current", "Value": valid_scale.iloc[-1] if len(valid_scale) else np.nan},
            {"Metric": "Momentum exposure scale -- historical mean", "Value": valid_scale.mean()},
            {"Metric": "% of days at full exposure (scale = 1.0)", "Value": (valid_scale >= 0.999).mean() * 100},
            {"Metric": "% of days at minimum exposure (scale = min_scale)", "Value": (valid_scale <= cfg.regime_min_exposure + 1e-6).mean() * 100},
        ]
    )

    regime_row = {
        "Strategy": "Momentum (Regime-Scaled)",
        "Annual Return (%)": annualized_return(regime_r, cfg.trading_days) * 100,
        "Annual Volatility (%)": annualized_volatility(regime_r, cfg.trading_days) * 100,
        "Sharpe Ratio": sharpe_ratio(regime_r, cfg.annual_risk_free_rate, cfg.trading_days),
        "Max Drawdown (%)": max_drawdown(regime_r) * 100,
    }
    if has_bh_benchmark:
        stats = excess_return_stats(regime_r, bh_r, periods=cfg.trading_days)
        regime_row[f"Correlation vs {cfg.benchmark_ticker}"] = stats["correlation"]
        regime_row["Tracking Error (%)"] = stats["tracking_error"] * 100
        regime_row["Information Ratio"] = stats["information_ratio"]
        regime_row[f"Excess Ann. Return vs {cfg.benchmark_ticker} (%)"] = (
            regime_row["Annual Return (%)"] - annualized_return(bh_r, cfg.trading_days) * 100
        )
    benchmark_comparison = pd.concat(
        [benchmark_comparison, pd.DataFrame([regime_row])], ignore_index=True
    )
    write_table(
        benchmark_comparison,
        raw_dir / "benchmark_comparison.csv",
        float_format="%.4f",
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = ["#4C72B0", "#55A868", "#C44E52", "#937860", "#8172B2", "#CCB974"]
    ax.bar(benchmark_comparison["Strategy"], benchmark_comparison["Sharpe Ratio"],
           color=colors[: len(benchmark_comparison)])
    ax.axhline(0, linewidth=1, color="black")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title("Strategy vs Passive Benchmark Comparison")
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(bench_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # Same walk-forward schedule as the plain momentum strategy, so the
    # regime overlay is judged on identical out-of-sample windows -- it
    # has to earn its place OOS, not just look better over one full
    # in-sample backtest.
    regime_oos_returns, regime_fold_summary = walk_forward_regime_momentum_validation(
        prices,
        folds,
        lookback=cfg.lookback_momentum,
        top_fraction=cfg.top_fraction,
        transaction_cost_bps=cfg.transaction_cost_bps,
        rebalance_frequency=cfg.rebalance_frequency,
        ar_window=cfg.absorption_ratio_window,
        ar_n_components=cfg.absorption_ratio_n_components,
        zscore_window=cfg.regime_zscore_window,
        min_scale=cfg.regime_min_exposure,
        z_low=cfg.regime_zscore_low,
        z_high=cfg.regime_zscore_high,
        trading_days=cfg.trading_days,
        annual_risk_free_rate=cfg.annual_risk_free_rate,
    )
    if not regime_fold_summary.empty:
        write_table(
            regime_fold_summary,
            raw_dir / "regime_walk_forward_folds.csv",
            float_format="%.4f",
        )
    if not regime_oos_returns.empty and not oos_benchmark_comparison.empty:
        oos_benchmark_comparison = pd.concat(
            [
                oos_benchmark_comparison,
                pd.DataFrame(
                    [
                        {
                            "Strategy": "Momentum (Regime-Scaled, OOS)",
                            "Annual Return (%)": annualized_return(regime_oos_returns, cfg.trading_days) * 100,
                            "Sharpe Ratio": sharpe_ratio(
                                regime_oos_returns, cfg.annual_risk_free_rate, cfg.trading_days
                            ),
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
        write_table(
            oos_benchmark_comparison,
            raw_dir / "walk_forward_benchmark_comparison.csv",
            float_format="%.4f",
        )

    # Two-panel diagnostic: the raw regime signal on top, the resulting
    # momentum exposure multiplier it produces underneath -- this is the
    # chart that makes the PCA <-> momentum link visible and concrete.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
    ax1.plot(absorption_ratio.index, absorption_ratio.values, color="#8172B2", linewidth=1)
    ax1.set_ylabel("Absorption Ratio\n(PC1 share of variance)")
    ax1.set_title("Regime Signal: Rolling PCA Absorption Ratio -> Momentum Exposure")
    ax2.plot(exposure_scale.index, exposure_scale.values, color="#C44E52", linewidth=1)
    ax2.set_ylabel("Momentum\nExposure Scale")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlabel("Date")
    fig.tight_layout()
    regime_fig = figures_dir / "regime_signal.png"
    fig.savefig(regime_fig, dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------------
    # 8. Wealth curves
    # ------------------------------------------------------------------
    wealth = pd.DataFrame(
        {
            "Minimum Variance": cumulative_wealth(minvar_r),
            "Risk Parity": cumulative_wealth(rp_r),
            "Momentum": cumulative_wealth(momentum_r),
            "Momentum (Regime-Scaled)": cumulative_wealth(regime_r),
            "Equal Weight": cumulative_wealth(ew_r),
            **(
                {f"Buy & Hold {cfg.benchmark_ticker}": cumulative_wealth(bh_r)}
                if has_bh_benchmark
                else {}
            ),
        }
    )
    wealth.index.name = "Date"
    wealth.to_csv(
        raw_dir / "wealth_curves.csv",
        encoding="utf-8-sig",
        float_format="%.8f",
    )

    plt.figure(figsize=(10, 5))
    for column in wealth.columns:
        plt.plot(wealth.index, wealth[column], label=column)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Portfolio and Momentum Wealth Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "wealth_curves.png", dpi=180)
    plt.close()

    # ------------------------------------------------------------------
    # 8f. Current model portfolio: risk-adjusted momentum stock screener
    # ------------------------------------------------------------------
    print(f"[13/14] Building current model portfolio screener "
          f"(risk-adjusted momentum on {len(cfg.stock_universe)} companies)...")
    print(f"        {SCREENER_DISCLAIMER}")

    try:
        # use_synthetic_fallback is deliberately False here: a screener
        # that names real companies must never silently fall back to
        # fabricated prices and present the result as if it were real.
        # If live data isn't available, this stage is skipped outright.
        screener_prices = align_assets(
            download_prices(
                cfg.stock_universe,
                cfg.start,
                cfg.end,
                use_synthetic_fallback=False,
            )
        )
        screener_error = None
    except Exception as exc:  # noqa: BLE001
        screener_prices = None
        screener_error = exc

    if screener_prices is None or screener_prices.empty:
        print(f"        Live data unavailable for the company screener "
              f"({screener_error.__class__.__name__ if screener_error else 'empty result'}) "
              f"-- skipping rather than showing a result built on non-real prices.")
        current_picks = pd.DataFrame()
        screener_scores = pd.DataFrame()
    else:
        current_picks = current_model_portfolio(
            screener_prices,
            momentum_lookback=cfg.lookback_momentum,
            vol_window=cfg.screener_vol_window,
            top_n=cfg.screener_top_n,
            max_volatility=cfg.screener_max_volatility,
        )
        write_table(
            current_picks,
            raw_dir / "current_model_portfolio.csv",
            float_format="%.4f",
        )

        screener_scores = pd.DataFrame(
            {
                "Momentum Return (%)": momentum_scores(screener_prices, cfg.lookback_momentum).iloc[-1] * 100,
                "Realized Volatility (%)": (
                    screener_prices.pct_change().rolling(cfg.screener_vol_window).std().iloc[-1]
                    * np.sqrt(cfg.trading_days) * 100
                ),
            }
        ).dropna()
        write_table(
            screener_scores.reset_index().rename(columns={"index": "Ticker"}),
            raw_dir / "screener_universe_snapshot.csv",
            float_format="%.4f",
        )

        selected = set(current_picks["Ticker"]) if not current_picks.empty else set()
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = ["#C44E52" if t in selected else "#4C72B0" for t in screener_scores.index]
        ax.scatter(
            screener_scores["Momentum Return (%)"],
            screener_scores["Realized Volatility (%)"],
            c=colors, s=60, edgecolors="black", linewidths=0.5,
        )
        for ticker, row in screener_scores.iterrows():
            if ticker in selected:
                ax.annotate(ticker, (row["Momentum Return (%)"], row["Realized Volatility (%)"]),
                            fontsize=8, xytext=(4, 4), textcoords="offset points")
        ax.axvline(0, color="black", linewidth=0.8)
        if cfg.screener_max_volatility is not None:
            ax.axhline(cfg.screener_max_volatility * 100, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel(f"{cfg.lookback_momentum}-day Momentum Return (%)")
        ax.set_ylabel(f"{cfg.screener_vol_window}-day Realized Volatility (%, annualized)")
        ax.set_title("Momentum vs Volatility -- Selected Names Highlighted")
        fig.tight_layout()
        screener_fig = figures_dir / "screener_momentum_vs_volatility.png"
        fig.savefig(screener_fig, dpi=200, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # 9. Research-quality figures and Excel report
    # ------------------------------------------------------------------
    print("\n[14/14] Building research report and visualizations...")

    plt.figure(figsize=(9, 7))
    im = plt.imshow(corr.values, vmin=-1, vmax=1, cmap="RdBu_r")
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.index)), corr.index)
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            plt.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center", fontsize=8)
    plt.title("Asset Return Correlation Matrix")
    plt.tight_layout()
    corr_fig = figures_dir / "correlation_heatmap.png"
    plt.savefig(corr_fig, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    x = np.arange(len(pca_report))
    plt.bar(x, pca_report["Explained Variance (%)"], label="Individual")
    plt.plot(x, pca_report["Cumulative Variance (%)"], marker="o", label="Cumulative")
    plt.xticks(x, pca_report["Component"])
    plt.xlabel("Principal component")
    plt.ylabel("Variance explained (%)")
    plt.title("PCA Factor Structure")
    plt.legend()
    plt.tight_layout()
    pca_fig = figures_dir / "pca_explained_variance.png"
    plt.savefig(pca_fig, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(weights))
    width = 0.36
    plt.bar(x - width/2, weights["Min Variance Weight (%)"], width, label="Minimum Variance")
    plt.bar(x + width/2, weights["Risk Parity Weight (%)"], width, label="Risk Parity")
    plt.xticks(x, weights["Asset"])
    plt.ylabel("Portfolio weight (%)")
    plt.title("Portfolio Allocation")
    plt.legend()
    plt.tight_layout()
    weights_fig = figures_dir / "portfolio_weights.png"
    plt.savefig(weights_fig, dpi=220, bbox_inches="tight")
    plt.close()

    rc_plot = pd.DataFrame({"Minimum Variance": rc_minvar * 100, "Risk Parity": rc_rp * 100})
    plt.figure(figsize=(10, 5))
    x = np.arange(len(rc_plot.index))
    width = 0.36
    plt.bar(x - width/2, rc_plot["Minimum Variance"], width, label="Minimum Variance")
    plt.bar(x + width/2, rc_plot["Risk Parity"], width, label="Risk Parity")
    plt.xticks(x, rc_plot.index)
    plt.ylabel("Risk contribution (%)")
    plt.title("Portfolio Volatility Risk Contributions")
    plt.legend()
    plt.tight_layout()
    rc_fig = figures_dir / "risk_contributions.png"
    plt.savefig(rc_fig, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(11, 5.5))
    for column in wealth.columns:
        plt.plot(wealth.index, wealth[column], label=column)
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.title("Portfolio and Momentum Wealth Curves")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    wealth_fig = figures_dir / "wealth_curves.png"
    plt.savefig(wealth_fig, dpi=220, bbox_inches="tight")
    plt.close()

    drawdowns = wealth / wealth.cummax() - 1.0
    plt.figure(figsize=(11, 5.5))
    for column in drawdowns.columns:
        plt.plot(drawdowns.index, drawdowns[column] * 100, label=column)
    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")
    plt.title("Portfolio Drawdowns")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    dd_fig = figures_dir / "drawdowns.png"
    plt.savefig(dd_fig, dpi=220, bbox_inches="tight")
    plt.close()

    vares = risk_table.set_index("Portfolio")[["Historical VaR (%)", "Historical ES (%)", "Gaussian VaR (%)", "Gaussian ES (%)"]]
    plt.figure(figsize=(9, 5))
    vares.plot(kind="bar", ax=plt.gca())
    plt.ylabel("Loss magnitude (%)")
    plt.title(f"VaR and Expected Shortfall at {cfg.confidence:.0%} Confidence")
    plt.xticks(rotation=0)
    plt.tight_layout()
    var_fig = figures_dir / "var_es_comparison.png"
    plt.savefig(var_fig, dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    scenario_plot = scenario_report.set_index("Scenario")["Portfolio Return (%)"]
    scenario_plot.plot(kind="bar", ax=plt.gca())
    plt.axhline(0, linewidth=1)
    plt.ylabel("Portfolio return (%)")
    plt.title("Deterministic Stress Scenarios — Minimum Variance Portfolio")
    plt.xticks(rotation=0)
    plt.tight_layout()
    stress_fig = figures_dir / "stress_scenarios.png"
    plt.savefig(stress_fig, dpi=220, bbox_inches="tight")
    plt.close()

    # A compact executive summary for recruiters/reviewers.
    summary_report = pd.DataFrame([
        {"Metric": "Data period", "Value": f"{start_date} to {end_date}", "Unit": "date"},
        {"Metric": "Universe", "Value": ", ".join(cfg.tickers), "Unit": "tickers"},
        {"Metric": "Observations", "Value": len(prices), "Unit": "trading days"},
        {"Metric": "Confidence level", "Value": cfg.confidence, "Unit": "probability"},
        {"Metric": "Monte Carlo paths", "Value": cfg.monte_carlo_paths, "Unit": "paths"},
        {"Metric": "Momentum lookback", "Value": cfg.lookback_momentum, "Unit": "trading days"},
        {"Metric": "Rebalance frequency", "Value": cfg.rebalance_frequency, "Unit": "pandas offset alias"},
        {"Metric": "Transaction cost", "Value": cfg.transaction_cost_bps, "Unit": "bps"},
        {"Metric": "Walk-forward window", "Value": f"{cfg.walk_forward_train_years}y train / {cfg.walk_forward_test_years}y test", "Unit": "rolling"},
        {"Metric": "Covariance estimator used for optimization", "Value": "Ledoit-Wolf shrinkage", "Unit": "-"},
    ])

    excel_tables = {
        "SUMMARY": summary_report,
        "ASSET PERFORMANCE": asset_report,
        "CORRELATION": corr.reset_index().rename(columns={"index": "Asset"}),
        "COVARIANCE": cov_sample.reset_index().rename(columns={"index": "Asset"}),
        "PCA": pca_report,
        "PCA LOADINGS": pca_loadings,
        "PORTFOLIO": weights,
        "RISK ANALYTICS": risk_table,
        "STRESS TEST": scenario_report,
        "STRESS SHOCKS": scenario_asset_report,
        "MOMENTUM": momentum_summary,
        "MOMENTUM WEIGHTS": avg_momentum_weights,
        "BACKTEST": wealth.reset_index(),
        "BENCHMARK COMPARISON": benchmark_comparison,
    }
    if not fold_summary.empty:
        excel_tables["WALK-FORWARD"] = fold_summary
        excel_tables["WALK-FORWARD SUMMARY"] = wf_aggregate
    if not sensitivity.empty:
        excel_tables["SENSITIVITY GRID"] = sensitivity
    if not freq_at_default_lookback.empty:
        excel_tables["REBALANCE FREQUENCY"] = freq_at_default_lookback
    if not oos_benchmark_comparison.empty:
        excel_tables["OOS BENCHMARK COMPARISON"] = oos_benchmark_comparison
    if not regime_fold_summary.empty:
        excel_tables["REGIME WALK-FORWARD"] = regime_fold_summary
    excel_tables["REGIME SIGNAL"] = regime_summary
    if not current_picks.empty:
        excel_tables["CURRENT MODEL PICKS"] = current_picks
    titles = {
        "SUMMARY": "Quantitative Portfolio Risk Engine — Executive Summary",
        "ASSET PERFORMANCE": "Asset-Level Performance and Risk",
        "CORRELATION": "Asset Return Correlation Matrix",
        "COVARIANCE": "Annualized Sample Covariance Matrix",
        "PCA": "Principal Component Analysis",
        "PCA LOADINGS": "PCA Factor Loadings",
        "PORTFOLIO": "Portfolio Construction and Risk Contributions",
        "RISK ANALYTICS": "Portfolio Risk Analytics — VaR / ES / Performance",
        "STRESS TEST": "Deterministic Stress Test Results",
        "STRESS SHOCKS": "Asset-Level Stress Assumptions",
        "MOMENTUM": "Cross-Sectional Momentum Strategy",
        "MOMENTUM WEIGHTS": "Average Momentum Portfolio Weights",
        "BACKTEST": "Backtest Wealth Curves",
        "BENCHMARK COMPARISON": f"Strategy vs Passive Benchmark (Equal Weight, Buy & Hold {cfg.benchmark_ticker})",
        "WALK-FORWARD": "Walk-Forward Validation — Per-Fold Out-of-Sample Performance",
        "WALK-FORWARD SUMMARY": "Walk-Forward Validation — Aggregate Out-of-Sample Performance",
        "SENSITIVITY GRID": "Parameter Sensitivity — Lookback x Rebalance Frequency (OOS)",
        "REBALANCE FREQUENCY": f"Rebalance Frequency Comparison @ {cfg.lookback_momentum}d Lookback (OOS)",
        "OOS BENCHMARK COMPARISON": "Momentum vs Benchmarks — Same Out-of-Sample Days",
        "REGIME WALK-FORWARD": "Regime-Scaled Momentum — Per-Fold Out-of-Sample Performance",
        "REGIME SIGNAL": "PCA Absorption Ratio -> Momentum Exposure Scaling",
        "CURRENT MODEL PICKS": f"Current Model Portfolio -- Risk-Adjusted Momentum Screen ({as_of}) -- NOT INVESTMENT ADVICE",
    }
    subtitles = {name: f"Generated {as_of} | Universe: {', '.join(cfg.tickers)}" for name in excel_tables}
    if "CURRENT MODEL PICKS" in subtitles:
        subtitles["CURRENT MODEL PICKS"] = (
            f"Generated {as_of} | {len(cfg.stock_universe)}-stock universe | {SCREENER_DISCLAIMER}"
        )
    pct_cols = {
        "ASSET PERFORMANCE": {"Annual Return", "Annual Volatility", "Max Drawdown"},
        "PCA": {"Explained Variance (%)", "Cumulative Variance (%)"},
        "PORTFOLIO": {"Min Variance Weight (%)", "Min Variance Risk Contribution (%)", "Risk Parity Weight (%)", "Risk Parity Risk Contribution (%)"},
        "RISK ANALYTICS": {"Confidence Level (%)", "Historical VaR (%)", "Historical ES (%)", "Gaussian VaR (%)", "Gaussian ES (%)", "Monte Carlo VaR (%)", "Monte Carlo ES (%)", "Annual Return (%)", "Annual Volatility (%)", "Max Drawdown (%)"},
        "STRESS TEST": {"Portfolio Return (%)", "Portfolio Loss (%)"},
        "MOMENTUM": {"Top Fraction (%)", "Annual Return (%)", "Annual Volatility (%)", "Max Drawdown (%)", "Historical VaR (%)", "Historical ES (%)"},
        "MOMENTUM WEIGHTS": {"Average Weight (%)"},
        "BENCHMARK COMPARISON": {
            "Annual Return (%)", "Annual Volatility (%)", "Max Drawdown (%)", "Tracking Error (%)",
            f"Excess Ann. Return vs {cfg.benchmark_ticker} (%)",
        },
        "WALK-FORWARD": {"Annual Return (%)", "Annual Volatility (%)", "Max Drawdown (%)"},
        "SENSITIVITY GRID": {"OOS Annual Return (%)", "OOS Annual Volatility (%)", "OOS Max Drawdown (%)"},
        "REBALANCE FREQUENCY": {"OOS Annual Return (%)", "OOS Annual Volatility (%)", "OOS Max Drawdown (%)"},
        "OOS BENCHMARK COMPARISON": {"Annual Return (%)"},
        "REGIME WALK-FORWARD": {"Annual Return (%)", "Annual Volatility (%)", "Max Drawdown (%)"},
        "CURRENT MODEL PICKS": {"Momentum Return (%)", "Realized Volatility (%)", "Target Weight (%)"},
    }
    dec_cols = {
        "ASSET PERFORMANCE": {"Sharpe"}, "PCA LOADINGS": {c for c in pca_loadings.columns if c != "Asset"},
        "CORRELATION": set(corr.columns), "COVARIANCE": set(cov_sample.columns),
        "RISK ANALYTICS": {"Sharpe Ratio"}, "WALK-FORWARD": {"Sharpe Ratio"},
        "SENSITIVITY GRID": {"OOS Sharpe Ratio", "Annualized Turnover (x/yr)"},
        "REBALANCE FREQUENCY": {"OOS Sharpe Ratio", "Annualized Turnover (x/yr)"},
        "BENCHMARK COMPARISON": {"Sharpe Ratio", f"Correlation vs {cfg.benchmark_ticker}", "Information Ratio"},
        "OOS BENCHMARK COMPARISON": {"Sharpe Ratio"},
        "REGIME WALK-FORWARD": {"Sharpe Ratio"},
        "CURRENT MODEL PICKS": {"Risk-Adjusted Momentum Score"},
    }
    int_cols = {"ASSET PERFORMANCE": {"Observations"}, "RISK ANALYTICS": {"Observations"}, "SUMMARY": {"Observations", "Monte Carlo paths", "Momentum lookback"}, "WALK-FORWARD": {"Fold", "Observations"}, "SENSITIVITY GRID": {"Lookback (days)", "Folds"}, "REBALANCE FREQUENCY": {"Lookback (days)", "Folds"}, "REGIME WALK-FORWARD": {"Fold", "Observations"}}
    figures = {
        "CORRELATION": corr_fig, "PCA": pca_fig, "PORTFOLIO": weights_fig,
        "RISK ANALYTICS": var_fig, "STRESS TEST": stress_fig, "BACKTEST": wealth_fig,
        "BENCHMARK COMPARISON": bench_fig, "REGIME SIGNAL": regime_fig,
    }
    if not current_picks.empty:
        figures["CURRENT MODEL PICKS"] = screener_fig
    if not fold_summary.empty:
        figures["WALK-FORWARD"] = wf_fig
    if not sensitivity.empty:
        figures["SENSITIVITY GRID"] = sensitivity_fig
    if not freq_at_default_lookback.empty:
        figures["REBALANCE FREQUENCY"] = freq_fig
    create_excel_report(
        results_dir / "quantitative_analysis.xlsx",
        excel_tables,
        titles,
        subtitles,
        matrix_sheets={"CORRELATION"},
        percent_columns=pct_cols,
        decimal_columns=dec_cols,
        integer_columns=int_cols,
        figures=figures,
    )

    # ------------------------------------------------------------------
    # Human-readable methodology / output guide
    # ------------------------------------------------------------------
    guide = f"""# Research Output Guide

Generated by `main/main.py`.

## Data
- Universe: {", ".join(cfg.tickers)}
- Period: {start_date} to {end_date}
- Trading days/year: {cfg.trading_days}

## Risk methodology
- Confidence level: {cfg.confidence:.0%}
- EWMA covariance decay: 0.94
- Ledoit-Wolf shrinkage intensity (this run): {shrink_intensity:.3f}
- Monte Carlo paths: {cfg.monte_carlo_paths:,}
- Random seed: {cfg.random_seed}

## Portfolio construction
- Minimum Variance: long-only, weights sum to 100%
- Risk Parity: approximate equal risk contribution, long-only
- Both portfolios are built on the **Ledoit-Wolf shrinkage covariance**
  (more stable out-of-sample than the raw sample covariance when the
  number of return observations is not much larger than the number of
  assets). Sample and EWMA covariance are also computed and reported
  for comparison. Risk contributions are measured against the same
  shrinkage-covariance matrix used for construction.

## Momentum
- Cross-sectional momentum
- Lookback: {cfg.lookback_momentum} trading days
- Top fraction: {cfg.top_fraction:.0%}
- Rebalance frequency: {cfg.rebalance_frequency} (month-end; selection is
  held between rebalances rather than re-evaluated every day, so
  turnover reflects the intended trading frequency)
- Transaction cost: {cfg.transaction_cost_bps:.1f} bps
- Signals are shifted before returns are applied to avoid look-ahead bias.

## Walk-forward validation
- Rolling schedule: {cfg.walk_forward_train_years}y train / {cfg.walk_forward_test_years}y
  test, stepped forward {cfg.walk_forward_step_years}y at a time.
- The momentum rule has no parameters fit on the training window (the
  lookback is fixed by design); the training window only provides
  warm-up history for the lookback and rebalance schedule.
- Performance is measured strictly on each fold's out-of-sample test
  segment, then aggregated, so a rule that only works in one historical
  regime cannot hide behind a single blended in-sample backtest number.
- See `WALK-FORWARD` and `WALK-FORWARD SUMMARY` in the Excel report, or
  `raw/walk_forward_folds.csv` / `raw/walk_forward_aggregate.csv`.

## Parameter sensitivity (lookback x rebalance frequency)
- Sweeps momentum lookback ({", ".join(str(x) for x in cfg.sensitivity_lookbacks)} days)
  against rebalance frequency ({", ".join(cfg.sensitivity_rebalance_frequencies)}: weekly,
  month-end, quarter-end), re-running the full walk-forward schedule for
  every combination.
- Every cell is scored strictly out-of-sample -- this grid is a robustness
  check, not a parameter-selection tool. Picking the best-looking cell and
  reporting only that number would just relocate the overfitting problem
  from "in-sample vs out-of-sample" to "best of many out-of-sample draws".
- Turnover is reported alongside Sharpe for each configuration, since
  trading less often (e.g. quarterly) lowers cost drag but reacts more
  slowly to a broken trend, while trading more often (weekly) is more
  reactive but pays far more in transaction costs for a signal this size.
- See `SENSITIVITY GRID` (full grid) and `REBALANCE FREQUENCY` (frequency
  comparison at the production lookback) in the Excel report, or
  `raw/sensitivity_grid.csv` / `raw/rebalance_frequency_comparison.csv`.

## Benchmark comparison
- Every strategy (Minimum Variance, Risk Parity, Momentum) is compared
  against two passive benchmarks: an Equal Weight (1/N) portfolio on the
  same universe, rebalanced at the same frequency and cost assumption as
  the momentum strategy (so it isolates what the momentum SIGNAL adds
  over simple diversification, separately from transaction costs), and
  a single-asset Buy & Hold {cfg.benchmark_ticker} ("the market").
- Reported alongside return/vol/Sharpe/drawdown: correlation, tracking
  error and information ratio versus Buy & Hold {cfg.benchmark_ticker}.
- A second, stricter comparison restricts the benchmarks to EXACTLY the
  same out-of-sample days used in the momentum walk-forward number above
  (`OOS BENCHMARK COMPARISON` / `raw/walk_forward_benchmark_comparison.csv`),
  so momentum is judged against the benchmarks on the same clock, not
  over a different (and possibly more favorable) sample period.
- See `BENCHMARK COMPARISON` in the Excel report, or
  `raw/benchmark_comparison.csv`.

## Regime-conditioned momentum (PCA <-> momentum link)
- Connects the PCA factor model to the momentum strategy: a rolling
  "Absorption Ratio" (Kritzman, Li, Page & Rigobon, 2010) tracks the
  share of the universe's variance explained by its first principal
  component over a trailing {cfg.absorption_ratio_window}-day window.
  When this rises well above its own trailing norm, the market's moves
  are increasingly driven by one common factor rather than diversified
  idiosyncratic moves -- the textbook "risk-off" signature where
  correlations spike and momentum crashes are more likely.
- The momentum sleeve's exposure is scaled down (as low as
  {cfg.regime_min_exposure:.0%}) as the absorption ratio's rolling
  z-score (versus its own trailing {cfg.regime_zscore_window}-day
  history) rises from {cfg.regime_zscore_low} to {cfg.regime_zscore_high};
  full exposure is kept otherwise. The scale is computed from information
  known strictly before it is applied (same one-day lag convention as
  the momentum signal itself), so there is no look-ahead advantage. It
  is only re-sampled at the same rebalance dates as the momentum
  selection and held in between -- applying a continuously-varying daily
  scale to the traded weights would reintroduce high-frequency trading
  purely to track the exposure dial. The chart in `figures/regime_signal.png`
  still plots the underlying signal continuously, for visibility.
- Reported as its own row (`Momentum (Regime-Scaled)`) in the benchmark
  comparison and wealth-curve chart, and validated on the SAME
  walk-forward out-of-sample windows as the plain momentum strategy
  (`REGIME WALK-FORWARD` sheet / `raw/regime_walk_forward_folds.csv`,
  plus a `Momentum (Regime-Scaled, OOS)` row in `OOS BENCHMARK
  COMPARISON`) -- the regime overlay has to earn its place
  out-of-sample too, not just look better over one full-sample backtest.
- The `REGIME SIGNAL` sheet / `figures/regime_signal.png` chart plots
  the raw absorption ratio against the resulting exposure multiplier
  over time, and `raw/regime_absorption_ratio.csv` has the full daily
  series of both.

## CSV conventions
- Percentages are explicitly labelled `(%)`.
- VaR and ES are reported as positive loss magnitudes.
- Covariance matrices are annualized.
- CSVs use UTF-8 with BOM for clean opening in Excel on Windows.

## Current model portfolio (company screener)
- **NOT INVESTMENT ADVICE.** {SCREENER_DISCLAIMER}
- Ranks a configurable universe of {len(cfg.stock_universe)} individual
  stocks (`Config.stock_universe`) by risk-adjusted momentum as of the
  latest available date: `score = momentum_return / realized_volatility`,
  the same momentum lookback ({cfg.lookback_momentum} days) used
  throughout this project, divided by trailing realized volatility
  ({cfg.screener_vol_window}-day window, annualized). This is a
  "momentum Sharpe" -- it rewards strong trailing returns AND penalizes
  high volatility, rather than chasing whichever name went up the most
  regardless of how bumpy the ride was.
- Only names with genuinely positive momentum ever qualify; an optional
  volatility ceiling (`Config.screener_max_volatility`, default 45%/yr)
  excludes anything above that regardless of score.
- Surviving names are inverse-volatility weighted (same convention used
  for the momentum backtest itself).
- **Deliberately does not fall back to synthetic data**: if live prices
  cannot be downloaded, this stage is skipped outright rather than
  showing a result built on fabricated prices next to real company names.
- See `CURRENT MODEL PICKS` in the Excel report, `raw/current_model_portfolio.csv`,
  the full-universe snapshot in `raw/screener_universe_snapshot.csv`, and
  the momentum-vs-volatility scatter in `figures/screener_momentum_vs_volatility.png`.

## Files
- `quantitative_analysis.xlsx`: recruiter-facing Excel report with formatted tables and embedded figures.
- `figures/`: correlation heatmap, PCA, portfolio weights, risk contributions, wealth curves, drawdowns, VaR/ES, stress, walk-forward, sensitivity-grid, benchmark-comparison, regime-signal and screener charts.
- `raw/`: machine-readable CSV exports of all underlying tables and time series.
"""
    (results_dir / "RESULTS_GUIDE.md").write_text(
        guide, encoding="utf-8"
    )

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETED")
    print(f"Results saved to: {results_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
