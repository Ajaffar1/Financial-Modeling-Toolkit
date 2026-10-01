from dataclasses import dataclass
import numpy as np
from ..analysis._inputs import array
from ..analysis.returns import performance, Performance

@dataclass(frozen=True)
class BacktestResult:
    net_returns: np.ndarray
    gross_returns: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    performance: Performance

def backtest(returns, target_weights, *, transaction_cost_bps=0.0, periods_per_year=252):
    """Target row t is observed AFTER return t and trades BEFORE return t+1.

    Starts in cash. Turnover is gross traded asset notional (one-way sum of
    absolute changes). Costs are applied before market returns. Residual capital
    is zero-yield cash; leverage financing and short borrow are not modeled.
    """
    r, targets = array(returns, 2, "returns"), array(target_weights, 2, "target_weights")
    if r.shape != targets.shape or len(r) < 2 or np.any(r <= -1):
        raise ValueError("Matching returns/targets with >=2 periods and returns > -100% required")
    if not np.isfinite(transaction_cost_bps) or not 0 <= transaction_cost_bps < 10000:
        raise ValueError("Transaction cost must be in [0,10000) basis points")
    holdings = np.zeros(r.shape[1])
    net, gross, turnovers, costs = [], [], [], []
    for t in range(len(r)):
        target = targets[t-1] if t else np.zeros(r.shape[1])
        turnover = float(np.abs(target - holdings).sum())
        cost = turnover * transaction_cost_bps / 10000
        g = float(target @ r[t])
        if cost >= 1 or g <= -1:
            raise ValueError("Trading cost or leveraged loss exhausts portfolio equity")
        net.append((1 - cost) * (1 + g) - 1)
        gross.append(g)
        turnovers.append(turnover)
        costs.append(cost)
        holdings = target * (1 + r[t]) / (1 + g)
    net = np.asarray(net)
    return BacktestResult(net, np.asarray(gross), np.asarray(turnovers), np.asarray(costs), performance(net, periods_per_year))
