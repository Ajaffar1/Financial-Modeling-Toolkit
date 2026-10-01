from dataclasses import dataclass
import math
from scipy.optimize import brentq

@dataclass(frozen=True)
class BondAnalytics:
    price: float
    macaulay_duration: float
    modified_duration: float
    convexity: float

def bond_analytics(face, coupon_rate, years, yield_to_maturity, frequency=2):
    """Bullet bond priced on coupon date; nominal annual yield, no accrued interest."""
    if not all(math.isfinite(v) for v in (face, coupon_rate, years, yield_to_maturity)):
        raise ValueError("Bond inputs must be finite")
    if isinstance(frequency, bool) or not isinstance(frequency, int) or frequency <= 0:
        raise ValueError("Frequency must be a positive integer")
    n = round(years * frequency)
    if face <= 0 or coupon_rate < 0 or n < 1 or abs(n - years * frequency) > 1e-8 or yield_to_maturity <= -frequency:
        raise ValueError("Invalid bond inputs or nonintegral coupon periods")
    base = 1 + yield_to_maturity / frequency
    flows = [face * coupon_rate / frequency] * n
    flows[-1] += face
    pv = [cf / base ** (i+1) for i, cf in enumerate(flows)]
    price = sum(pv)
    duration = sum((i+1)/frequency * v for i,v in enumerate(pv)) / price
    convexity = sum((i+1)*(i+2)*v for i,v in enumerate(pv)) / (price * frequency**2 * base**2)
    return BondAnalytics(price, duration, duration / base, convexity)

def bond_yield(price, face, coupon_rate, years, frequency=2):
    if not math.isfinite(price) or price <= 0:
        raise ValueError("Price must be positive and finite")
    bond_analytics(face, coupon_rate, years, 0, frequency)
    f = lambda y: bond_analytics(face, coupon_rate, years, y, frequency).price - price
    high = 1.0
    while f(high) > 0 and high < 1e6:
        high *= 2
    if f(high) > 0:
        raise ValueError("Could not bracket bond yield")
    return float(brentq(f, -0.99 * frequency, high))
