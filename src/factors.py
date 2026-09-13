from __future__ import annotations

import numpy as np
import pandas as pd


def pca_factors(
    returns: pd.DataFrame,
    n_components: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    PCA decomposition of standardized asset returns.

    Returns:
        factor_returns: principal-component time series
        explained_variance_ratio: variance explained by each component
    """
    x = returns.dropna().copy()
    if n_components is None:
        n_components = min(x.shape)

    x = x - x.mean()
    scales = x.std(ddof=1).replace(0.0, np.nan)
    x = x.div(scales).dropna(axis=1)

    u, s, _ = np.linalg.svd(x.to_numpy(), full_matrices=False)
    components = u[:, :n_components] * s[:n_components]

    explained = s**2 / np.sum(s**2)
    factor_cols = [f"PC{i+1}" for i in range(n_components)]
    factor_returns = pd.DataFrame(
        components,
        index=x.index,
        columns=factor_cols,
    )

    return factor_returns, explained[:n_components]


def factor_loadings(returns: pd.DataFrame, n_components: int = 3) -> pd.DataFrame:
    """Return PCA loadings for the selected number of components."""
    x = returns.dropna()
    standardized = (x - x.mean()) / x.std(ddof=1)

    _, _, vt = np.linalg.svd(standardized.to_numpy(), full_matrices=False)
    n = min(n_components, vt.shape[0])
    return pd.DataFrame(
        vt[:n].T,
        index=x.columns,
        columns=[f"PC{i+1}" for i in range(n)],
    )


def rolling_absorption_ratio(
    returns: pd.DataFrame, window: int = 126, n_components: int = 1
) -> pd.Series:
    """
    Rolling "Absorption Ratio" (Kritzman, Li, Page & Rigobon, 2010): the
    fraction of total variance within a trailing window explained by the
    first `n_components` principal components.

    This is a walk-forward-safe regime indicator: each value uses only
    returns up to and including that date, never future data. When it
    rises, the universe's variance is increasingly driven by a small
    number of common factors (often just one) rather than diversified
    idiosyncratic moves -- the textbook signature of a "risk-off" regime,
    where correlations spike, most assets fall together, and
    diversification stops helping.

    The first `window - 1` values are NaN (not enough history yet).
    """
    values = returns.to_numpy()
    n_obs, n_assets = values.shape
    n_components = min(n_components, n_assets)
    ratios = np.full(n_obs, np.nan)

    for end in range(window, n_obs + 1):
        chunk = values[end - window: end]
        std = chunk.std(axis=0, ddof=1)
        std_safe = np.where(std == 0, np.nan, std)
        standardized = (chunk - chunk.mean(axis=0)) / std_safe
        keep = ~np.isnan(standardized).any(axis=0)
        standardized = standardized[:, keep]
        if standardized.shape[1] == 0:
            continue

        _, s, _ = np.linalg.svd(standardized, full_matrices=False)
        explained = s**2
        total = explained.sum()
        if total > 0:
            ratios[end - 1] = explained[:n_components].sum() / total

    return pd.Series(ratios, index=returns.index, name="Absorption Ratio")
