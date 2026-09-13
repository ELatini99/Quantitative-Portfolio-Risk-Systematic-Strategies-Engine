# Quantitative Portfolio Risk & Systematic Strategies Engine

A quantitative research framework for portfolio construction, risk management, and systematic investment strategies.

The project combines:

- market-data acquisition and cleaning (with an offline synthetic fallback)
- return and volatility analysis
- PCA-based factor decomposition
- covariance estimation: sample, EWMA, and **Ledoit-Wolf shrinkage**
- portfolio optimization (minimum variance, risk parity)
- Monte Carlo VaR / Expected Shortfall
- historical VaR / Expected Shortfall
- stress testing
- momentum-based systematic strategies with fixed-frequency rebalancing
- genuine **walk-forward validation** (rolling out-of-sample folds)
- **parameter sensitivity grid** (lookback x rebalance frequency, scored out-of-sample)
- **passive benchmark comparison** (equal-weight and buy & hold, in-sample and out-of-sample)
- **regime-conditioned momentum** (PCA absorption ratio scales momentum exposure)
- **current model portfolio screener** (risk-adjusted momentum ranking on individual stocks)
- transaction costs
- risk-adjusted performance analysis

## Project structure

```text
quant_portfolio_risk_engine/
│
├── main.py                    # Research pipeline / analysis
├── LICENSE                    # MIT License
├── PATCH_NOTES.md             # Changelog of every bug found and fixed during development
│
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── returns.py
│   ├── factors.py             # PCA + rolling absorption ratio (regime signal)
│   ├── covariance.py
│   ├── optimization.py
│   ├── risk.py
│   ├── stress.py
│   ├── momentum.py
│   ├── backtest.py
│   ├── validation.py         # rolling train/test walk-forward validation
│   ├── sensitivity.py         # lookback x rebalance-frequency grid, scored OOS
│   ├── benchmarks.py          # equal-weight and buy & hold passive benchmarks
│   ├── regime.py              # PCA absorption ratio -> momentum exposure scaling
│   └── screener.py            # risk-adjusted momentum company screener
│
├── tests/
├── data/
├── results/
├── requirements.txt
└── README.md
```

## Quick start

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the research pipeline:

```bash
python main.py
```

The default universe is:

```text
SPY, QQQ, IWM, EFA, EEM, TLT, GLD
```

The pipeline downloads historical data through `yfinance`, computes returns, estimates risk, performs PCA, constructs an optimized portfolio, evaluates a momentum strategy, runs a walk-forward validation, and writes figures/tables to `results/`.

If live data cannot be downloaded (no internet, Yahoo Finance rate limiting or outage), the pipeline automatically falls back to a reproducible synthetic multi-asset price panel, so it always runs end to end — a `[data] Live data unavailable...` message is printed when this happens. This is clearly synthetic data, only meant to keep the pipeline demoable; disable it via `Config(use_synthetic_fallback=False)` if you want a hard failure instead.

### Walk-forward validation

Momentum performance is also evaluated with a rolling walk-forward
schedule (`src/validation.py`): a 3-year training/warm-up window is
followed by a 1-year out-of-sample test window, stepped forward one year
at a time. The momentum rule has no parameters fit on the training
window (the lookback is fixed by design); the training window only
provides warm-up history for the lookback and the rebalance schedule.
Reported performance is aggregated strictly from each fold's
out-of-sample segment, so a rule that only works in one historical
regime cannot hide behind a single blended in-sample backtest. Results
are in `results/raw/walk_forward_folds.csv`, `walk_forward_aggregate.csv`,
and the `WALK-FORWARD` / `WALK-FORWARD SUMMARY` sheets of the Excel
report.

### Parameter sensitivity: lookback x rebalance frequency

`src/sensitivity.py` sweeps the momentum lookback (63/126/189/252 days)
against the rebalance frequency (weekly / month-end / quarter-end),
re-running the full walk-forward schedule for every combination and
scoring each cell strictly **out-of-sample**. This is a robustness check,
not a way to hand-pick the best-looking configuration — reporting only
the best cell would just move the overfitting problem from "in-sample
vs out-of-sample" to "best of many out-of-sample draws".

Turnover is reported alongside Sharpe for every cell, since that is what
actually drives transaction costs: rebalancing quarterly instead of
monthly roughly halves annualized turnover, at the cost of reacting more
slowly to a broken trend; weekly rebalancing reacts fastest but pays by
far the most in costs for a signal of this size. Results are in
`results/raw/sensitivity_grid.csv` (full grid) and
`results/raw/rebalance_frequency_comparison.csv` (frequency comparison
at the production lookback), or the `SENSITIVITY GRID` / `REBALANCE
FREQUENCY` sheets of the Excel report.

### Benchmark comparison

`src/benchmarks.py` adds two passive benchmarks so every strategy is
judged against "did nothing clever" rather than only against the other
strategies in this project:

- **Equal Weight (1/N)** on the same universe, rebalanced at the same
  frequency and cost assumption as the momentum strategy — this isolates
  what the momentum **signal** adds over simple diversification,
  separately from turnover/cost effects.
- **Buy & Hold `Config.benchmark_ticker`** (default `SPY`) — the classic
  single-asset "the market" reference.

Correlation, tracking error and information ratio versus Buy & Hold are
reported for every strategy in `results/raw/benchmark_comparison.csv`
(`BENCHMARK COMPARISON` sheet). A second, stricter comparison restricts
both benchmarks to *exactly* the same out-of-sample days used in the
momentum walk-forward validation above
(`results/raw/walk_forward_benchmark_comparison.csv`, `OOS BENCHMARK
COMPARISON` sheet), so momentum is judged against a passive alternative
on the same clock, not over a possibly more favorable sample period.

### Regime-conditioned momentum: connecting PCA to the momentum strategy

Until this point, the PCA factor model and the momentum strategy were
two independent modules that never talked to each other. `src/regime.py`
connects them: `src/factors.py` computes a rolling "Absorption Ratio"
(Kritzman, Li, Page & Rigobon, 2010) -- the share of the universe's
total variance explained by its first principal component, in a
trailing window, using only information available as of each date (no
look-ahead). When this rises well above its own trailing norm, the
market's moves are increasingly driven by one common factor rather than
diversified idiosyncratic moves: the textbook "risk-off" signature where
correlations spike and momentum crashes become more likely.

The momentum sleeve's exposure is scaled down (down to a configurable
floor, `Config.regime_min_exposure`) as the absorption ratio's rolling
z-score rises, and kept at full exposure otherwise. Two details matter
for this to be a fair, tradable overlay rather than a backtest artifact:
the scale uses the same one-day information lag as the momentum signal
itself (no look-ahead advantage), and it is only re-sampled at the same
rebalance dates as the momentum selection and held constant in between
-- a continuously-varying daily scale would otherwise reintroduce
high-frequency trading purely to track the exposure dial.

The regime-scaled strategy is reported everywhere the plain momentum
strategy is (`BENCHMARK COMPARISON`, wealth-curve chart), and separately
validated on the exact same walk-forward out-of-sample windows
(`REGIME WALK-FORWARD` sheet / `raw/regime_walk_forward_folds.csv`) --
the overlay has to earn its place out-of-sample too. A two-panel chart
(`figures/regime_signal.png`, `REGIME SIGNAL` sheet) plots the raw
absorption ratio against the resulting exposure multiplier over time.

**This is reported as an honest empirical question, not a guaranteed
improvement**: on synthetic fallback data it actually underperforms
plain momentum both in-sample and out-of-sample. Whether it helps on
real market data is exactly what this comparison is built to check.

### Current model portfolio: risk-adjusted momentum stock screener

`src/screener.py` applies the same momentum methodology to a
configurable universe of individual stocks (`Config.stock_universe`,
default 31 diversified large-cap names) and ranks them by

```text
score = momentum_return / realized_volatility
```

a "momentum Sharpe" that rewards strong trailing returns *and* penalizes
high volatility -- two stocks with identical momentum are not equally
attractive if one got there with twice the volatility. Only names with
genuinely positive momentum ever qualify, and an optional volatility
ceiling (`Config.screener_max_volatility`, default 45%/yr) excludes
anything above that regardless of score. Surviving names are
inverse-volatility weighted, same convention as the momentum backtest.

**This is a model output, not investment advice.** It reflects only
trailing price momentum and realized volatility on the configured
universe -- it has no view on valuation, fundamentals, earnings, or news.

This stage **deliberately does not use the synthetic-data fallback**:
a screener that names real companies must never substitute fabricated
prices and present a ranking as if it reflected real data. If live
prices aren't available, the stage prints why and is skipped outright.
Results (when live data is available) are in
`results/raw/current_model_portfolio.csv`,
`results/raw/screener_universe_snapshot.csv` (full universe, for
context), `results/figures/screener_momentum_vs_volatility.png`, and the
`CURRENT MODEL PICKS` sheet of the Excel report.

## Research philosophy

This is intentionally designed as a research framework rather than a single "profitable strategy".

The main questions are:

1. How should portfolio risk be estimated?
2. How sensitive are risk estimates to the covariance model?
3. How much diversification is actually achieved?
4. How does PCA reveal the effective risk factors?
5. How does an optimized portfolio behave under historical and Monte Carlo stress?
6. Does a simple momentum signal survive out-of-sample testing?
7. How do transaction costs and turnover change the result?
8. How sensitive is that result to two choices most people treat as free
   knobs — the lookback window and the rebalance frequency — when judged
   strictly out-of-sample?
9. Does any of this actually beat just holding the market, or a naive
   1/N portfolio, over the same period?
10. Does connecting the risk model (PCA) back to the strategy (momentum) --
    scaling exposure down in single-driver-dominated regimes -- actually
    help, or is that just a plausible-sounding idea that doesn't survive
    contact with out-of-sample data?
11. Of a broader universe of individual stocks, which ones currently show
    strong momentum *without* excessive volatility -- and how much of
    that is signal versus noise?

The project is educational/research software, not investment advice.

## Rebalancing

The momentum sleeve rebalances at a fixed frequency (`Config.rebalance_frequency`,
default month-end, `"ME"`) rather than re-evaluating the top-fraction
selection on every trading day. The selected assets are held between
rebalance dates and turnover is measured accordingly, so the transaction
cost applied in the backtest reflects the intended trading frequency,
not artificial daily churn.


## Results and reporting

The `results/` directory separates human-facing research outputs from raw machine-readable data.
The main deliverable is `results/quantitative_analysis.xlsx`, which contains formatted tables,
methodology context, and embedded research figures.

Run:

```bash
python main.py
```

Then start with:

```text
results/quantitative_analysis.xlsx
```


## Example results

The `results/` directory in this repository contains example output from
an actual run against live market data. Each run of `python main.py`
overwrites `results/` with fresh output reflecting current prices, so
treat the committed files as a snapshot/demo rather than a live feed —
re-run the pipeline yourself for up-to-date numbers.


## License

Licensed under the MIT License — see [LICENSE](LICENSE) for details.


The output structure is:

- `quantitative_analysis.xlsx`: recruiter-facing Excel report.
- `figures/`: correlation heatmap, PCA, portfolio weights, risk contributions, wealth curves, drawdowns, VaR/ES, stress, walk-forward, sensitivity-grid, benchmark-comparison, regime-signal and screener charts.
- `raw/`: machine-readable CSV exports of the underlying tables and time series.

Percentages and loss measures are explicitly labelled in the Excel report, while covariance matrices
remain in matrix form because that is the natural representation for quantitative analysis.

