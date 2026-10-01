"""Discrete quadratic optimal liquidation with linear temporary impact."""
from dataclasses import dataclass
import numpy as np
from scipy.linalg import solve_banded
from ..analysis._inputs import positive_integer

@dataclass(frozen=True)
class ExecutionPlan:
    holdings: np.ndarray
    trades: np.ndarray
    temporary_impact_cost: float
    inventory_pnl_variance: float
    objective: float

def optimal_liquidation(quantity,horizon,*,intervals=20,temporary_impact=.001,volatility=1.0,risk_aversion=.01):
    """Minimize eta/dt*sum(trade²) + lambda*sigma²*dt*sum(post-trade holdings²).

    Time/price/share units must be consistent. Deterministic liquidation, no drift,
    spread, permanent impact, fills, or volume constraints. Positive quantity only.
    """
    positive_integer(intervals,"intervals")
    values=(quantity,horizon,temporary_impact,volatility,risk_aversion)
    if not np.isfinite(values).all() or quantity<=0 or horizon<=0 or temporary_impact<=0 or volatility<0 or risk_aversion<0:
        raise ValueError("Invalid execution parameters")
    dt=horizon/intervals;impact=temporary_impact/dt;risk=risk_aversion*volatility**2*dt
    holdings=np.r_[quantity,np.zeros(intervals)]
    if intervals>1:
        bands=np.zeros((3,intervals-1));bands[1]=2*impact+risk
        bands[0,1:]=-impact;bands[2,:-1]=-impact
        rhs=np.zeros(intervals-1);rhs[0]=impact*quantity
        holdings[1:-1]=solve_banded((1,1),bands,rhs)
    trades=-np.diff(holdings)
    cost=float(impact*np.sum(trades**2))
    variance=float(volatility**2*dt*np.sum(holdings[1:]**2))
    return ExecutionPlan(holdings,trades,cost,variance,cost+risk_aversion*variance)
