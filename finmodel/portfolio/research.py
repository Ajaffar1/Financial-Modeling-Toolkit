"""Trailing-window estimation and strictly lagged allocation research."""
import numpy as np
from ..analysis._inputs import array, positive_integer
from .optimization import minimum_variance
from .backtest import backtest

def estimate_moments(returns, *, periods_per_year=252, shrinkage=0.1):
    """Sample arithmetic means and covariance shrunk toward its diagonal."""
    r = array(returns, 2, "returns")
    positive_integer(periods_per_year, "periods_per_year")
    if len(r) < 2 or not np.isfinite(shrinkage) or not 0 <= shrinkage <= 1:
        raise ValueError("At least two observations and shrinkage in [0,1] required")
    cov = np.atleast_2d(np.cov(r, rowvar=False, ddof=1)) * periods_per_year
    return r.mean(axis=0) * periods_per_year, (1-shrinkage)*cov + shrinkage*np.diag(np.diag(cov))

def walk_forward_minimum_variance(returns, *, lookback=60, rebalance_every=21,
                                  periods_per_year=252, shrinkage=.1, transaction_cost_bps=5):
    """Estimate through close t, trade t+1. Warm-up is held in zero-yield cash."""
    r = array(returns, 2, "returns")
    positive_integer(lookback, "lookback")
    positive_integer(rebalance_every, "rebalance_every")
    if lookback < 2 or lookback >= len(r):
        raise ValueError("Lookback must leave at least one out-of-sample period")
    targets = np.zeros_like(r)
    current = np.zeros(r.shape[1])
    for t in range(len(r)):
        if t >= lookback-1 and (t-(lookback-1)) % rebalance_every == 0:
            mu, cov = estimate_moments(r[t-lookback+1:t+1], periods_per_year=periods_per_year, shrinkage=shrinkage)
            current = minimum_variance(mu,cov).weights
        targets[t] = current
        # Between rebalances, maintain drifted holdings instead of constant-weight trading.
        if t + 1 < len(r):
            gain = float(current @ r[t+1])
            if gain <= -1:
                raise ValueError("Portfolio equity exhausted")
            current = current * (1+r[t+1]) / (1+gain)
    return backtest(r, targets, transaction_cost_bps=transaction_cost_bps, periods_per_year=periods_per_year)
