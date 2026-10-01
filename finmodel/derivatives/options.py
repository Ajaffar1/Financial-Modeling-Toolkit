from dataclasses import dataclass
import math
from scipy.stats import norm
from scipy.optimize import brentq

@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float

def _validate(spot, strike, maturity, rate, volatility, dividend_yield, kind):
    if not all(math.isfinite(v) for v in (spot, strike, maturity, rate, volatility, dividend_yield)):
        raise ValueError("Option inputs must be finite")
    if spot <= 0 or strike <= 0 or maturity < 0 or volatility < 0 or kind not in ("call", "put"):
        raise ValueError("Invalid spot, strike, maturity, volatility, or option kind")

def black_scholes(spot, strike, maturity, rate, volatility, *, dividend_yield=0.0, kind="call"):
    """European option with continuous rates/dividend yield; maturity in years."""
    _validate(spot, strike, maturity, rate, volatility, dividend_yield, kind)
    sign = 1 if kind == "call" else -1
    s, k = spot * math.exp(-dividend_yield * maturity), strike * math.exp(-rate * maturity)
    if maturity == 0 or volatility == 0:
        return max(0.0, sign * (s - k))
    d1 = (math.log(spot / strike) + (rate - dividend_yield + volatility ** 2 / 2) * maturity) / (volatility * math.sqrt(maturity))
    d2 = d1 - volatility * math.sqrt(maturity)
    return float(sign * (s * norm.cdf(sign * d1) - k * norm.cdf(sign * d2)))

def greeks(spot, strike, maturity, rate, volatility, *, dividend_yield=0.0, kind="call"):
    """Vega/rho per unit change; theta per year, with calendar-time decay sign."""
    _validate(spot, strike, maturity, rate, volatility, dividend_yield, kind)
    if maturity == 0 or volatility == 0:
        raise ValueError("Greeks require positive maturity and volatility")
    sign = 1 if kind == "call" else -1
    root = math.sqrt(maturity)
    d1 = (math.log(spot / strike) + (rate - dividend_yield + volatility ** 2 / 2) * maturity) / (volatility * root)
    d2 = d1 - volatility * root
    dq, dr, pdf = math.exp(-dividend_yield * maturity), math.exp(-rate * maturity), norm.pdf(d1)
    return Greeks(float(sign * dq * norm.cdf(sign * d1)), float(dq * pdf / (spot * volatility * root)),
                  float(spot * dq * pdf * root),
                  float(-spot * dq * pdf * volatility / (2 * root) - sign * rate * strike * dr * norm.cdf(sign*d2) + sign * dividend_yield * spot * dq * norm.cdf(sign*d1)),
                  float(sign * strike * maturity * dr * norm.cdf(sign*d2)))

def implied_volatility(price, spot, strike, maturity, rate, *, dividend_yield=0.0, kind="call"):
    _validate(spot, strike, maturity, rate, 0, dividend_yield, kind)
    if maturity <= 0 or not math.isfinite(price):
        raise ValueError("Implied volatility requires finite price and positive maturity")
    lower = black_scholes(spot, strike, maturity, rate, 0, dividend_yield=dividend_yield, kind=kind)
    upper = (spot * math.exp(-dividend_yield*maturity) if kind == "call" else strike * math.exp(-rate*maturity))
    if price < lower or price >= upper:
        raise ValueError("Price violates finite-volatility no-arbitrage bounds")
    if price == lower:
        return 0.0
    objective = lambda sigma: black_scholes(spot, strike, maturity, rate, sigma, dividend_yield=dividend_yield, kind=kind) - price
    high = 1.0
    while objective(high) < 0 and high < 128:
        high *= 2
    if objective(high) < 0:
        raise ValueError("Could not bracket implied volatility")
    return float(brentq(objective, 0, high, xtol=1e-12))
