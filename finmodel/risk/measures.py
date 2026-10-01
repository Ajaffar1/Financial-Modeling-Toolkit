from dataclasses import dataclass
import numpy as np
from ..analysis._inputs import array, covariance

@dataclass(frozen=True)
class TailRisk:
    confidence: float
    value_at_risk: float
    expected_shortfall: float
    observations: int

def historical_risk(returns, confidence=0.95, portfolio_value=1.0):
    """Loss-positive one-observation-period VaR and ES with fractional tail mass."""
    r = array(returns, 1, "returns")
    if not 0 < confidence < 1 or not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Confidence must be in (0,1) and portfolio value positive")
    losses = np.sort(-r * portfolio_value)[::-1]
    mass = (1 - confidence) * len(losses)
    whole = int(np.floor(mass))
    fractional = mass - whole
    tail_sum = losses[:whole].sum()
    if fractional > 0:
        tail_sum += fractional * losses[whole]
    return TailRisk(confidence, float(np.quantile(losses, confidence)), float(tail_sum / mass), len(r))

def risk_contributions(weights, cov):
    w = array(weights, 1, "weights")
    cov = covariance(cov)
    if cov.shape != (len(w), len(w)):
        raise ValueError("Weights and covariance dimensions differ")
    vol = float(np.sqrt(max(0, w @ cov @ w)))
    if vol == 0:
        raise ValueError("Risk contributions undefined for zero-volatility portfolio")
    return w * (cov @ w) / vol

def stress_pnl(weights, shocks, portfolio_value=1.0):
    """Linear mark-to-market shocks: scenarios by assets; no option convexity."""
    w, s = array(weights, 1, "weights"), array(shocks, 2, "shocks")
    if s.shape[1] != len(w) or not np.isfinite(portfolio_value) or portfolio_value <= 0:
        raise ValueError("Invalid stress dimensions or portfolio value")
    return portfolio_value * (s @ w)
