from dataclasses import dataclass
import numpy as np
from scipy.stats import t as student_t
from ._inputs import array

@dataclass(frozen=True)
class FactorFit:
    alpha: float
    betas: np.ndarray
    r_squared: float
    standard_errors: np.ndarray
    t_statistics: np.ndarray
    p_values: np.ndarray
    residuals: np.ndarray
    observations: int

def fit_factor_model(excess_returns, factors, hac_lags=0):
    """OLS with Bartlett/Newey-West errors; lags=0 gives heteroskedastic HC1.

    Inputs must be aligned, same-frequency excess returns. Alpha is per period.
    """
    y, factors = array(excess_returns, 1, "excess_returns"), array(factors, 2, "factors")
    if len(y) != len(factors) or isinstance(hac_lags, bool) or not isinstance(hac_lags, int) or not 0 <= hac_lags < len(y):
        raise ValueError("Invalid observation alignment or HAC lags")
    x = np.column_stack([np.ones(len(y)), factors])
    n, k = x.shape
    if n <= k or np.linalg.matrix_rank(x) != k:
        raise ValueError("Need more observations than coefficients and full-rank factors")
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    residuals = y - x @ coef
    scores = x * residuals[:, None]
    meat = scores.T @ scores
    for lag in range(1, hac_lags + 1):
        gamma = scores[lag:].T @ scores[:-lag]
        meat += (1 - lag / (hac_lags + 1)) * (gamma + gamma.T)
    bread = np.linalg.inv(x.T @ x)
    se = np.sqrt(np.maximum(0, np.diag(bread @ meat @ bread) * n / (n - k)))
    statistics = np.divide(coef, se, out=np.full(k, np.nan), where=se > 1e-15)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - float(residuals @ residuals) / total if total > 0 else float("nan")
    return FactorFit(float(coef[0]), coef[1:], r2, se, statistics,
                     2 * student_t.sf(np.abs(statistics), n - k), residuals, n)
