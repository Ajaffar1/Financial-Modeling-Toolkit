from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize, linprog
from scipy import sparse
from ..analysis._inputs import array, covariance

@dataclass(frozen=True)
class TailAllocation:
    weights: np.ndarray
    expected_shortfall: float
    threshold: float

def risk_parity(cov, budgets=None):
    """Long-only risk budgeting via convex variance-minus-log objective."""
    cov = covariance(cov)
    if np.linalg.eigvalsh(cov).min() <= 0:
        raise ValueError("Risk budgeting requires positive definite covariance")
    n = len(cov)
    b = np.full(n, 1/n) if budgets is None else array(budgets, 1, "budgets")
    if len(b) != n or np.any(b <= 0) or not np.isclose(b.sum(), 1):
        raise ValueError("Positive budgets must sum to one")
    scale = np.diag(cov).mean()
    c = cov / scale
    result = minimize(lambda x: .5*x@c@x - b@np.log(x), np.ones(n),
                      jac=lambda x: c@x-b/x, method="L-BFGS-B",
                      bounds=[(1e-10,None)]*n, options={"ftol":1e-14,"gtol":1e-10,"maxiter":5000})
    x = result.x
    shares = x * (c @ x) / (x @ c @ x)
    if not result.success or not np.allclose(shares,b,atol=1e-6):
        raise ValueError("Risk budgeting failed to converge")
    return x/x.sum()

def black_litterman(cov, market_weights, views, view_returns, view_covariance, *, risk_aversion=2.5, tau=.05):
    """Posterior expected returns from equilibrium prior and independent Gaussian views.

    Returns posterior mean and covariance of the mean (not predictive covariance).
    """
    cov = covariance(cov)
    w = array(market_weights,1,"market_weights")
    p = array(views,2,"views")
    q = array(view_returns,1,"view_returns")
    omega = covariance(view_covariance)
    if len(w) != len(cov) or p.shape != (len(q),len(w)) or omega.shape != (len(q),len(q)):
        raise ValueError("Black-Litterman dimensions differ")
    if not np.isclose(w.sum(),1) or not np.isfinite([tau,risk_aversion]).all() or tau <= 0 or risk_aversion <= 0:
        raise ValueError("Invalid equilibrium weights, tau, or risk aversion")
    prior = risk_aversion * cov @ w
    c = tau * cov
    innovation = p@c@p.T + omega
    if np.linalg.eigvalsh(innovation).min() <= 0:
        raise ValueError("Views require positive definite innovation covariance")
    gain = np.linalg.solve(innovation,p@c).T
    posterior_cov = c - gain@p@c
    return prior + gain@(q-p@prior), (posterior_cov+posterior_cov.T)/2

def minimum_cvar(scenario_returns, *, confidence=.95, target_return=None):
    """Rockafellar-Uryasev empirical CVaR linear program; fully invested long-only."""
    r = array(scenario_returns,2,"scenario_returns")
    samples, n = r.shape
    if samples < 2 or not 0 < confidence < 1:
        raise ValueError("Need >=2 scenarios and confidence in (0,1)")
    # Variables: asset weights, unrestricted loss threshold, nonnegative excess losses.
    objective = np.r_[np.zeros(n),1,np.full(samples,1/((1-confidence)*samples))]
    inequalities = sparse.hstack([sparse.csr_matrix(-r), sparse.csr_matrix(-np.ones((samples,1))), -sparse.eye(samples)], format="csr")
    limits = np.zeros(samples)
    if target_return is not None:
        if not np.isfinite(target_return):
            raise ValueError("Target return must be finite")
        inequalities = sparse.vstack([inequalities,sparse.csr_matrix(np.r_[-r.mean(axis=0),0,np.zeros(samples)][None,:])],format="csr")
        limits = np.r_[limits,-target_return]
    result = linprog(objective,A_ub=inequalities,b_ub=limits,
                     A_eq=[np.r_[np.ones(n),0,np.zeros(samples)]],b_eq=[1],
                     bounds=[(0,1)]*n+[(None,None)]+[(0,None)]*samples,method="highs")
    if not result.success:
        raise ValueError(f"CVaR allocation failed: {result.message}")
    return TailAllocation(result.x[:n],float(result.fun),float(result.x[n]))
