from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def portfolio_volatility(weights: np.ndarray, covariance: np.ndarray) -> float:
    return float(np.sqrt(weights @ covariance @ weights))


def minimum_variance_portfolio(
    covariance: pd.DataFrame,
    long_only: bool = True,
) -> pd.Series:
    """Global minimum-variance portfolio under optional long-only constraints."""
    cov = covariance.to_numpy()
    n = len(covariance)
    x0 = np.ones(n) / n

    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        lambda w: portfolio_volatility(w, cov),
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 2000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Optimization failed: {result.message}")

    return pd.Series(result.x, index=covariance.index, name="weight")


def risk_parity_portfolio(
    covariance: pd.DataFrame,
    long_only: bool = True,
) -> pd.Series:
    """
    Approximate equal-risk-contribution portfolio.

    This implementation minimizes the dispersion of risk contributions.
    """
    cov = covariance.to_numpy()
    n = len(covariance)
    x0 = np.ones(n) / n
    bounds = [(1e-8, 1.0)] * n if long_only else [(None, None)] * n

    def objective(w: np.ndarray) -> float:
        sigma = portfolio_volatility(w, cov)
        marginal = cov @ w / max(sigma, 1e-16)
        contributions = w * marginal
        target = sigma / n
        return float(np.sum((contributions - target) ** 2))

    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    result = minimize(
        objective,
        x0=x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 3000, "ftol": 1e-12},
    )

    if not result.success:
        raise RuntimeError(f"Risk-parity optimization failed: {result.message}")

    return pd.Series(result.x, index=covariance.index, name="weight")
