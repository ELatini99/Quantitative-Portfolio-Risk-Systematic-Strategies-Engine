from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf


def sample_covariance(returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    cov = returns.cov()
    return cov * 252 if annualize else cov


def ewma_covariance(
    returns: pd.DataFrame,
    decay: float = 0.94,
    annualize: bool = True,
) -> pd.DataFrame:
    """
    Exponentially weighted covariance matrix.

    More recent observations receive larger weights.
    """
    x = returns.dropna().to_numpy()
    if len(x) < 2:
        raise ValueError("At least two observations are required.")

    mean = x.mean(axis=0)
    centered = x - mean

    cov = np.zeros((x.shape[1], x.shape[1]))
    weight = 1.0
    total_weight = 0.0

    for row in centered[::-1]:
        cov += weight * np.outer(row, row)
        total_weight += weight
        weight *= decay

    cov /= total_weight
    if annualize:
        cov *= 252

    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    return returns.corr()


def ledoit_wolf_covariance(
    returns: pd.DataFrame, annualize: bool = True
) -> tuple[pd.DataFrame, float]:
    """
    Ledoit-Wolf shrinkage covariance estimator: shrinks the noisy sample
    covariance towards a structured (scaled identity) target. This is
    typically far more stable out-of-sample than the plain sample
    covariance whenever the number of observations T is not much larger
    than the number of assets N -- exactly the regime most real trading
    universes are in.

    Returns (covariance_df, shrinkage_intensity), where shrinkage_intensity
    in [0, 1] measures how much weight was put on the structured target
    (0 = pure sample covariance, 1 = pure structured target).
    """
    x = returns.dropna().to_numpy()
    lw = LedoitWolf().fit(x)
    cov = pd.DataFrame(lw.covariance_, index=returns.columns, columns=returns.columns)
    return (cov * 252 if annualize else cov), float(lw.shrinkage_)
