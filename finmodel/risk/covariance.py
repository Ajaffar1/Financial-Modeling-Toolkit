"""Positive-semidefinite covariance estimators for high-dimensional research."""
import numpy as np
from ..analysis._inputs import array

def oas_covariance(observations):
    """Oracle Approximating Shrinkage (Chen et al.), using Gaussian OAS formula."""
    x = array(observations, 2, "observations")
    n, p = x.shape
    if n < 2:
        raise ValueError("Need at least two observations")
    x = x - x.mean(axis=0)
    empirical = x.T @ x / n
    trace = np.trace(empirical)
    trace_square = np.sum(empirical ** 2)
    denominator = (n + 1 - 2 / p) * (trace_square - trace ** 2 / p)
    shrinkage = 1.0 if denominator <= 0 else min(1.0, ((1 - 2 / p) * trace_square + trace ** 2) / denominator)
    return (1 - shrinkage) * empirical + shrinkage * trace / p * np.eye(p), float(shrinkage)

def ewma_covariance(observations, decay=.94):
    """Normalized exponentially weighted, weighted-mean-centered covariance."""
    x = array(observations, 2, "observations")
    if len(x) < 2 or not 0 < decay < 1:
        raise ValueError("Need >=2 observations and decay in (0,1)")
    weights = decay ** np.arange(len(x)-1, -1, -1)
    weights /= weights.sum()
    centered = x - weights @ x
    return (centered.T * weights) @ centered
