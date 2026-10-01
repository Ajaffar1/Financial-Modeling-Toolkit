from dataclasses import dataclass
import math

@dataclass(frozen=True)
class Quotes:
    reservation_price: float
    bid: float
    ask: float

def inventory_quotes(mid,inventory,volatility,time_remaining,*,risk_aversion=.1,liquidity=1.5):
    """Avellaneda-Stoikov finite-horizon approximation; arithmetic-price volatility.

    Research-only quotes: no tick sizes, fills, queues, latency, or exchange routing.
    """
    if not all(math.isfinite(x) for x in (mid,inventory,volatility,time_remaining,risk_aversion,liquidity)) or mid<=0 or volatility<0 or time_remaining<0 or risk_aversion<=0 or liquidity<=0:
        raise ValueError("Invalid quote inputs")
    risk=risk_aversion*volatility**2*time_remaining
    reservation=mid-inventory*risk
    half_spread=.5*risk+math.log1p(risk_aversion/liquidity)/risk_aversion
    if reservation-half_spread<=0:
        raise ValueError("Model-implied bid is nonpositive; change inventory/units")
    return Quotes(reservation,reservation-half_spread,reservation+half_spread)
