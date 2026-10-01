from dataclasses import dataclass
import numpy as np
from ._inputs import array, positive_integer

@dataclass(frozen=True)
class Performance:
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe: float | None
    sortino: float | None
    max_drawdown: float

def simple_returns(prices):
    prices = array(prices, name="prices")
    if prices.ndim not in (1, 2) or len(prices) < 2 or np.any(prices <= 0):
        raise ValueError("Prices must be positive with at least two observations")
    return prices[1:] / prices[:-1] - 1

def performance(returns, periods_per_year=252, annual_risk_free=0.0):
    r = array(returns, 1, "returns")
    positive_integer(periods_per_year, "periods_per_year")
    if len(r) < 2 or np.any(r <= -1) or not np.isfinite(annual_risk_free) or annual_risk_free <= -1:
        raise ValueError("Need at least two returns > -100% and a valid risk-free rate")
    wealth = np.r_[1.0, np.cumprod(1 + r)]
    excess = r - ((1 + annual_risk_free) ** (1 / periods_per_year) - 1)
    vol = float(np.std(r, ddof=1) * np.sqrt(periods_per_year))
    downside = float(np.sqrt(np.mean(np.minimum(excess, 0) ** 2)) * np.sqrt(periods_per_year))
    numerator = float(np.mean(excess) * periods_per_year)
    return Performance(float(wealth[-1] - 1), float(wealth[-1] ** (periods_per_year / len(r)) - 1),
                       vol, numerator / vol if vol > 0 else None,
                       numerator / downside if downside > 0 else None,
                       float(np.min(wealth / np.maximum.accumulate(wealth) - 1)))
