from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Historical VaR and Expected Shortfall expressed as positive loss percentages."""
    losses = -returns.dropna()
    var = float(losses.quantile(confidence))
    es = float(losses[losses >= var].mean())
    return var, es


def parametric_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
) -> tuple[float, float]:
    """Gaussian parametric VaR and ES."""
    r = returns.dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)
    z = norm.ppf(confidence)

    var = -(mu + sigma * z)
    pdf = norm.pdf(z)
    es = -(mu + sigma * pdf / (1.0 - confidence))
    return float(var), float(es)


def monte_carlo_var_es(
    returns: pd.Series,
    confidence: float = 0.99,
    n_paths: int = 100_000,
    horizon: int = 1,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Monte Carlo VaR / ES using a Gaussian return model.

    For a research project, this can later be extended to:
    - Student-t innovations
    - volatility clustering
    - filtered historical simulation
    - multivariate factor models
    """
    r = returns.dropna()
    mu = r.mean()
    sigma = r.std(ddof=1)

    rng = np.random.default_rng(seed)
    simulated = rng.normal(
        loc=horizon * mu,
        scale=np.sqrt(horizon) * sigma,
        size=n_paths,
    )

    losses = -simulated
    var = float(np.quantile(losses, confidence))
    es = float(losses[losses >= var].mean())
    return var, es


def portfolio_returns(asset_returns: pd.DataFrame, weights: pd.Series) -> pd.Series:
    aligned = asset_returns.loc[:, weights.index]
    return aligned @ weights
