from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from ..analysis._inputs import array, covariance

@dataclass(frozen=True)
class Allocation:
    weights: np.ndarray
    annual_return: float
    annual_volatility: float
    gross_exposure: float

def minimum_variance(expected_returns, cov, *, target_return=None, bounds=(0.0, 1.0)):
    """Fully invested constrained allocation; inputs must share annual units."""
    mu, cov = array(expected_returns, 1, "expected_returns"), covariance(cov)
    if cov.shape != (len(mu), len(mu)):
        raise ValueError("Expected-return and covariance dimensions differ")
    lower, upper = bounds
    if not np.isfinite([lower, upper]).all() or lower > upper or len(mu) * lower > 1 or len(mu) * upper < 1:
        raise ValueError("Infeasible weight bounds")
    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if target_return is not None:
        if not np.isfinite(target_return):
            raise ValueError("Target return must be finite")
        constraints.append({"type": "ineq", "fun": lambda w: w @ mu - target_return})
    result = minimize(lambda w: float(w @ cov @ w), np.full(len(mu), 1 / len(mu)),
                      jac=lambda w: 2 * cov @ w, method="SLSQP",
                      bounds=[bounds] * len(mu), constraints=constraints,
                      options={"ftol": 1e-12, "maxiter": 1000})
    w = result.x
    if not result.success or abs(w.sum() - 1) > 1e-7 or (target_return is not None and w @ mu < target_return - 1e-7):
        raise ValueError(f"Allocation failed or target infeasible: {result.message}")
    w.setflags(write=False)
    return Allocation(w, float(w @ mu), float(np.sqrt(max(0, w @ cov @ w))), float(np.abs(w).sum()))
