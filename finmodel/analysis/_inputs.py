"""Shared numerical validation; rows are observations and columns are assets."""
import numpy as np

def array(values, ndim=None, name="input"):
    data = np.asarray(values, dtype=float)
    if data.size == 0 or not np.all(np.isfinite(data)) or (ndim is not None and data.ndim != ndim):
        raise ValueError(f"{name} must be nonempty, finite, and have dimension {ndim}")
    return data

def positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value

def covariance(values):
    cov = array(values, 2, "covariance")
    if cov.shape[0] != cov.shape[1] or not np.allclose(cov, cov.T, atol=1e-12):
        raise ValueError("Covariance must be square and symmetric")
    if np.linalg.eigvalsh(cov).min() < -1e-10 * max(1, np.linalg.norm(cov)):
        raise ValueError("Covariance must be positive semidefinite")
    return cov
