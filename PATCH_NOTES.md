# Patch Notes

Fixes applied to `quant_portfolio_risk_engine_v03`, found during code
review (full pipeline run, `pytest`, and manual inspection of every
module).

---

## 10. Matplotlib non-interactive backend (Windows tkinter shutdown errors)

**File:** `main.py`

Running the script on Windows produced a wall of harmless-but-noisy
`Exception ignored in: <function Image.__del__ ...> RuntimeError: main
thread is not in main loop` / `Tcl_AsyncDelete` messages at the end of
the run. Cause: `main.py` never set matplotlib's backend, so it
auto-selected an interactive GUI backend (TkAgg, common as the Windows
default) even though the script only ever saves figures to disk
(`plt.savefig` + `plt.close`) and never shows a window. The many figures
created and closed in one batch run then hit a tkinter cleanup race at
interpreter shutdown. Fixed by setting both `MPLBACKEND=Agg` (env var)
and `matplotlib.use("Agg", force=True)` before `pyplot` is imported --
belt and suspenders, in case some other import path initializes
matplotlib's backend before this module's own call would otherwise run.
The pipeline's actual output (figures, CSVs, the Excel report) was never
affected; this only silences spurious noise at shutdown.

**Related robustness fix, found while chasing a report of missing
screener output**: `download_prices()` (`src/data.py`) used a row-wise
`.dropna()` that required EVERY ticker in the universe to have data on
a given day to keep that day. A single entirely-failed ticker (delisted,
mistyped, or just a transient per-symbol hiccup -- common with a 31-name
universe, far more exposed to this than the original 7-ETF universe)
therefore silently zeroed out every row for every OTHER ticker too,
making the whole company screener stage look like a total data outage
when in fact only one name had failed. Fixed to drop only the tickers
that returned no data at all (printing which ones, so failures are
visible) before requiring a common date range across the survivors.
Verified with a new test that mocks a partially-failed batch download
and confirms the healthy tickers keep their full date range.

---

## 9. Current model portfolio: risk-adjusted momentum company screener (feature addition)

**Files:** new `src/screener.py`, `src/config.py`, `main.py`, `tests/test_core.py`

Applies the project's momentum methodology to a configurable universe of
individual stocks (`Config.stock_universe`, default 31 diversified
large-cap names) to produce a snapshot "current model portfolio" --
**not investment advice**, purely the mechanical output of the same
momentum/volatility framework used throughout the project, applied to
single stocks instead of the ETF universe.

Ranking: `score = momentum_return / realized_volatility` (a "momentum
Sharpe" -- rewards strong trailing returns AND penalizes high
volatility, exactly the momentum/volatility trade-off a raw momentum
score alone ignores). Two correctness points caught during development:

- The first version's ranking could surface names with **negative**
  momentum (the "least negative" score once high-volatility names were
  excluded) -- clearly wrong for a buy screen. Fixed by requiring
  strictly positive momentum before a name can qualify at all (tests
  added for this).
- **Deliberately disables the synthetic-data fallback** for this stage
  specifically (`use_synthetic_fallback=False`): a screener that names
  real companies must never silently substitute fabricated prices and
  present a ranking as if it reflected real market data. If live prices
  aren't available, the stage prints why and is skipped outright, rather
  than showing a result. Verified in the review sandbox (no live market
  access there): the stage correctly detects the failed download and
  skips, while every other pipeline stage still completes normally. The
  screener's own logic was separately verified end-to-end using
  clearly-labelled placeholder tickers (never real company names) so
  test/demo numbers could never be mistaken for a real signal.

Output: `raw/current_model_portfolio.csv` (the qualifying, inverse-vol-
weighted picks), `raw/screener_universe_snapshot.csv` (the full universe,
for context), `figures/screener_momentum_vs_volatility.png` (scatter of
momentum vs. volatility with selected names highlighted), and a
`CURRENT MODEL PICKS` Excel sheet carrying the disclaimer in its
subtitle.

---

## 8. Regime-conditioned momentum: PCA absorption ratio -> exposure scaling (feature addition)

**Files:** new `src/regime.py`, `src/factors.py` (added `rolling_absorption_ratio`), `src/config.py`, `main.py`, `tests/test_core.py`

Connects the previously-independent PCA factor model to the momentum
strategy. `src/factors.py` gained `rolling_absorption_ratio()`: a
walk-forward-safe rolling measure of the fraction of total variance
explained by the universe's first principal component (the
"Absorption Ratio" of Kritzman, Li, Page & Rigobon, 2010) -- a rising
value signals a single-driver, "risk-off" regime where correlations
spike and momentum crashes become more likely.

`src/regime.py` maps this into a momentum exposure multiplier
(`regime_scaling_factor`): full exposure while the absorption ratio is
at or below its own trailing z-score norm, scaled down (as low as a
configurable floor) as it rises above that norm. Two correctness points
worth calling out because they were caught and fixed during
development, not assumed correct on the first pass:

- **Look-ahead**: the scale is computed from the raw absorption ratio
  and shifted by one day, mirroring the same lag already used by the
  momentum signal, so the regime overlay gets no extra information
  advantage.
- **Turnover discipline**: the first working version applied a
  continuously-varying daily scale directly to the traded weights, which
  reintroduced near-daily turnover -- exactly the problem fixed for the
  base momentum strategy in patch #3 above. Fixed by re-sampling the
  scale only at the same rebalance dates as the momentum selection and
  holding it constant in between (verified by test: turnover dropped
  from near-daily back down to the same order of magnitude as the
  rebalance schedule itself).

The regime-scaled strategy is added as its own row/curve everywhere the
plain momentum strategy already appears (`BENCHMARK COMPARISON`,
wealth-curve chart), and separately validated on the exact same
walk-forward out-of-sample windows as the plain momentum strategy
(`REGIME WALK-FORWARD` sheet, `Momentum (Regime-Scaled, OOS)` row in
`OOS BENCHMARK COMPARISON`) -- the overlay has to earn its place
out-of-sample too, not just look better over one full-sample backtest.
A two-panel diagnostic chart (`figures/regime_signal.png`, `REGIME
SIGNAL` sheet) plots the raw absorption ratio against the resulting
exposure multiplier over time, making the PCA-to-momentum link visible.

**Honest result, not cherry-picked**: on the synthetic fallback data
used during development (no live market access in the review sandbox),
the regime overlay actually *underperforms* the plain momentum strategy,
both in-sample (Sharpe 0.18 vs 0.28) and out-of-sample (Sharpe 0.09 vs
0.24). This is reported as-is rather than tuned until it looked better --
whether it helps is an empirical question to check against real market
data, which is the whole point of building the comparison this
rigorously.

---

## 7. Passive benchmark comparison (feature addition)

**Files:** new `src/benchmarks.py`, `src/config.py`, `main.py`, `tests/test_core.py`

Until now nothing in the project answered "ok, but is this better than
just holding the market?" — Minimum Variance, Risk Parity and Momentum
were only ever compared against each other. Added two passive
benchmarks in `src/benchmarks.py`:

- **Equal Weight (1/N)**, rebalanced at the *same* frequency and cost
  assumption as the momentum strategy — this isolates what the momentum
  **signal** adds on top of simple diversification across the same
  universe, separate from transaction costs or rebalancing cadence.
- **Buy & Hold `Config.benchmark_ticker`** (default `SPY`) — the classic
  single-asset "the market" reference, no rebalancing, no ongoing costs.

For every strategy, `main.py` now also reports correlation, tracking
error and information ratio versus the Buy & Hold benchmark
(`raw/benchmark_comparison.csv`, `BENCHMARK COMPARISON` Excel sheet,
`figures/benchmark_comparison.png`), and both benchmarks are added to
the wealth-curve chart and CSV.

A second, stricter comparison (`raw/walk_forward_benchmark_comparison.csv`,
`OOS BENCHMARK COMPARISON` sheet) restricts both benchmarks to **exactly
the same out-of-sample days** used in the momentum walk-forward
validation (#patch 2 above) — so momentum is judged against a passive
alternative on the same clock, not over a different and possibly more
favorable sample period.

---

## 6. Parameter sensitivity grid: lookback x rebalance frequency (feature addition)

**Files:** new `src/sensitivity.py`, `src/config.py`, `main.py`, `tests/test_core.py`

Added as a follow-up to the rebalance-frequency fix (#3 below): rather
than assuming month-end rebalancing is the right choice, `src/sensitivity.py`
sweeps momentum lookback (63/126/189/252 days) against rebalance
frequency (weekly / month-end / quarter-end), re-running the full
walk-forward schedule for every combination and scoring each cell
**strictly out-of-sample** — never in-sample, to avoid just relocating
the overfitting problem to "best of many OOS draws" instead of fixing it.

Outputs: `raw/sensitivity_grid.csv` (full 4x3 grid) and
`raw/rebalance_frequency_comparison.csv` (frequency comparison at the
production lookback, with Sharpe alongside annualized turnover so the
cost/reactivity trade-off is explicit), plus a heatmap
(`figures/sensitivity_grid.png`) and a two-panel Sharpe/turnover
comparison chart (`figures/rebalance_frequency_comparison.png`). Both
tables are also added as Excel sheets (`SENSITIVITY GRID`, `REBALANCE
FREQUENCY`). New test verifies every (lookback, frequency) combination
is scored and that quarterly rebalancing never shows higher turnover
than monthly at the same lookback.

---

## 1. `"M"` → `"ME"` pandas offset alias (crash on pandas ≥ 3.0)

**Files:** `src/config.py`, `src/backtest.py` (function removed, see #2)

`Config.rebalance_frequency` and the old `walk_forward_momentum()`
helper used `resample("M")`. Pandas deprecated the `"M"` alias in 2.2 and
**removed it entirely in 3.0** (`ValueError: Invalid frequency: M`).
Since `requirements.txt` pins `pandas>=2.2`, a fresh `pip install` today
resolves to pandas 3.x and this crashes immediately. Fixed to `"ME"`
(month-end) everywhere.

## 2. "Walk-forward" was advertised but not actually wired in

**Files:** `src/backtest.py`, `src/momentum.py`, new `src/validation.py`, `main/main.py`

The README listed "walk-forward backtesting" as a feature, and
`Config.rebalance_frequency` existed, but `main.py` never called the one
function that used it (`walk_forward_momentum`, which also contained the
bug above) — the momentum backtest instead re-evaluated the top-fraction
selection continuously (see #3). This has been split into two, now both
real:

- **Fixed-frequency rebalancing** is now built directly into
  `cross_sectional_momentum_weights()` in `src/momentum.py`, using an
  actual-trading-day rebalance schedule (`rebalance_schedule()` — see
  #3). `Config.rebalance_frequency` is now genuinely used.
- **Walk-forward *validation*** (rolling train/test out-of-sample folds)
  is implemented in the new `src/validation.py` and wired into `main.py`
  as pipeline stage `[9/10]`. It reports per-fold and aggregate
  out-of-sample Sharpe/return/drawdown, a bar chart, and two new Excel
  sheets (`WALK-FORWARD`, `WALK-FORWARD SUMMARY`).

The old, buggy `walk_forward_momentum()` in `src/backtest.py` was
removed (its correct logic now lives in `momentum.py`); `backtest.py`
keeps only the general-purpose helpers (`buy_and_hold_returns`,
`cumulative_wealth`, `turnover`).

## 3. Momentum turnover was much higher than intended

**File:** `src/momentum.py`

`cross_sectional_momentum_weights()` re-selected the top-fraction assets
on *every trading day* the ranking shifted, not on a fixed schedule —
despite `Config.rebalance_frequency` implying monthly rebalancing.
Measured on a 10-year synthetic panel: turnover changed on 371/2609
days (~24x annualized). This also meant the root cause of bug #1 (the
`"M"` alias) never surfaced in practice, because the intended monthly
rebalancing path was never actually executed.

Fixed: the function now builds a proper rebalance schedule from the
*actual trading days* in the price index (`rebalance_schedule()`), not
from `resample().last().index` calendar labels (which are frequently
**not** trading days — the original root cause: checking
`date in resampled_calendar_labels` would rarely match). Selections are
now held between rebalance dates. Measured after the fix on the same
panel: turnover changed on 67/2609 days (~5x annualized) — in line with
monthly rebalancing.

## 4. No offline fallback for market data

**File:** `src/data.py`

`download_prices()` raised immediately if `yfinance` could not reach
Yahoo Finance (no internet, rate limiting, temporary outage), which
meant the whole pipeline was unusable in that situation — including
during a live demo or interview without a guaranteed connection.

Added a synthetic multi-asset price panel generator
(`_synthetic_prices()`, GBM with a one-factor market structure so the
resulting covariance matrix stays realistic/non-diagonal) used as an
automatic fallback, controlled by `Config.use_synthetic_fallback`
(default `True`). A console message clearly flags when synthetic data is
being used, so results are never silently mistaken for real market data.

## 5. Covariance estimation: added Ledoit-Wolf shrinkage

**File:** `src/covariance.py`, `requirements.txt` (added `scikit-learn`)

Only sample and EWMA covariance were available. Added
`ledoit_wolf_covariance()`, which shrinks the sample covariance towards
a structured target — materially more stable than the raw sample
covariance whenever the number of observations isn't much larger than
the number of assets (the typical case). Portfolio construction
(`minimum_variance_portfolio`, `risk_parity_portfolio`) now uses the
shrinkage covariance by default; sample and EWMA are still computed and
reported for comparison in `raw/covariance_comparison.csv` and the
`COVARIANCE`/`SUMMARY` sheets. The shrinkage intensity (0 = pure sample
covariance, 1 = pure structured target) is printed at runtime and logged
in `RESULTS_GUIDE.md`.

---

## Other small consistency fixes

- `main.py` previously hardcoded `top_fraction=0.30` in the momentum
  call and `"Top Fraction (%)": 30.0` in the report, instead of using
  `Config.top_fraction` (which existed but wasn't consulted). Now
  consistent throughout, and a new `Config.top_fraction` field is the
  single source of truth.
- `main.py` step counters (`[N/8]` / `[9/9]`, inconsistent with each
  other) are now a single consistent `[N/10]` sequence including the new
  walk-forward stage.
- Added unit tests for every fix above (`tests/test_core.py`): Ledoit-Wolf
  shrinkage bounds, correct rebalance-schedule trading days, reduced
  momentum turnover, and non-overlapping walk-forward folds. Test suite
  grew from 3 to 7 tests, all passing.

## Verification

- `pytest tests/` → 7/7 passed.
- Full pipeline (`python main/main.py`) run end to end with the offline
  synthetic fallback (no live internet available in the review sandbox):
  completes all 10 stages, produces the Excel report, all CSVs, and all
  figures including the two new walk-forward outputs, with no errors or
  warnings.
