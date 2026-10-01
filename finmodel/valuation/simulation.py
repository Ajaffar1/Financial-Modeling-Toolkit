"""Seeded correlated assumption simulation; no silent clipping or rejected draws."""
from dataclasses import dataclass
import numpy as np
from ..analysis._inputs import covariance, positive_integer

@dataclass(frozen=True)
class ValuationSimulation:
    enterprise_values: np.ndarray
    mean: float
    standard_deviation: float
    quantiles: dict
    assumptions: tuple
    seed: int | None

def simulate_valuation(model, assumption_names, assumption_covariance, *, samples=1000, seed=0):
    if not model.periods:
        raise ValueError("Forecast the base model before simulation")
    positive_integer(samples, "samples")
    if samples < 2:
        raise ValueError("At least two simulations required")
    names = tuple(assumption_names)
    if not names or len(set(names)) != len(names):
        raise ValueError("Provide unique assumption names")
    try:
        means = np.array([getattr(model.assumptions, name) for name in names])
    except AttributeError as error:
        raise ValueError("Unknown simulation assumption") from error
    cov = covariance(assumption_covariance)
    if cov.shape != (len(names), len(names)):
        raise ValueError("Simulation covariance dimension mismatch")
    draws = np.random.default_rng(seed).multivariate_normal(means, cov, size=samples)
    values = []
    for index, draw in enumerate(draws):
        try:
            values.append(model.scenario(f"Simulation {index}", dict(zip(names, draw))).enterprise_value())
        except ValueError as error:
            raise ValueError(f"Invalid assumption draw {index}; reduce dispersion or change inputs: {error}") from error
    values = np.asarray(values)
    if not np.isfinite(values).all():
        raise ValueError("Simulation produced nonfinite valuations")
    values.setflags(write=False)
    return ValuationSimulation(values, float(values.mean()), float(values.std(ddof=1)),
                               {q: float(np.quantile(values, q)) for q in (.05, .5, .95)}, names, seed)
