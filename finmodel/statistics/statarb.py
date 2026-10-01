from dataclasses import dataclass
import numpy as np
from ..analysis._inputs import array

@dataclass(frozen=True)
class OUFit:
    mean_reversion: float
    long_run_mean: float
    diffusion: float
    half_life: float
    ar_coefficient: float

def fit_ou(spread, dt=1.0):
    """Exact discrete AR(1) mapping to stationary Ornstein-Uhlenbeck dynamics.

    This is not a cointegration test; callers must establish spread stationarity.
    """
    s = array(spread,1,"spread")
    if len(s) < 4 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("Need >=4 observations and positive time step")
    x = np.column_stack([np.ones(len(s)-1),s[:-1]])
    if np.linalg.matrix_rank(x) < 2:
        raise ValueError("Constant spread cannot identify OU dynamics")
    intercept, phi = np.linalg.lstsq(x,s[1:],rcond=None)[0]
    if not 0 < phi < 1:
        raise ValueError("Estimated AR coefficient is incompatible with stationary continuous OU")
    kappa = -np.log(phi)/dt
    residuals = s[1:] - x @ [intercept,phi]
    variance = residuals@residuals/(len(residuals)-2)
    return OUFit(float(kappa),float(intercept/(1-phi)),float(np.sqrt(variance*2*kappa/(1-phi**2))),float(np.log(2)/kappa),float(phi))

@dataclass(frozen=True)
class HedgeFilter:
    coefficients: np.ndarray
    innovations: np.ndarray
    innovation_variances: np.ndarray

def kalman_hedge(dependent, independent, *, process_variance=1e-5, observation_variance=.01):
    """Causal random-walk intercept/hedge-ratio filter; outputs posterior at t.

    Innovations use the prior and may form causal signals; posteriors trade at t+1.
    """
    y,x = array(dependent,1,"dependent"),array(independent,1,"independent")
    if len(x) != len(y) or not np.isfinite([process_variance,observation_variance]).all() or process_variance < 0 or observation_variance <= 0:
        raise ValueError("Invalid filter alignment or variances")
    beta = np.zeros(2)
    cov = np.eye(2)
    coefficients, innovations, variances = [],[],[]
    for yi,xi in zip(y,x):
        prior = cov + process_variance*np.eye(2)
        h = np.array([1,xi])
        variance = float(h@prior@h+observation_variance)
        innovation = float(yi-h@beta)
        gain = prior@h/variance
        beta = beta+gain*innovation
        a = np.eye(2)-np.outer(gain,h)
        cov = a@prior@a.T+observation_variance*np.outer(gain,gain)  # Joseph form
        coefficients.append(beta.copy());innovations.append(innovation);variances.append(variance)
    return HedgeFilter(np.asarray(coefficients),np.asarray(innovations),np.asarray(variances))
