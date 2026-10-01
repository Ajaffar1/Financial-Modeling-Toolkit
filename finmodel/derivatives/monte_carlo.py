"""Randomized Sobol arithmetic Asian valuation with geometric control variate."""
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import qmc, norm, t as student_t
from ..analysis._inputs import positive_integer
from .options import _validate

@dataclass(frozen=True)
class MonteCarloPrice:
    price: float
    standard_error: float
    confidence_interval: tuple
    control_coefficient: float
    paths: int
    seed: int

def asian_call(spot,strike,maturity,rate,volatility,*,dividend_yield=0.0,steps=12,power=10,replicates=8,seed=0):
    """Equally spaced monitoring excludes t=0. Error is across independent scrambles.

    A separate 1024-path pilot estimates the control coefficient. Returned path
    count excludes that pilot. Bounds are approximate Student-t intervals.
    """
    _validate(spot,strike,maturity,rate,volatility,dividend_yield,"call")
    positive_integer(steps,"steps");positive_integer(power,"power");positive_integer(replicates,"replicates")
    if maturity<=0 or replicates<2 or power>20:
        raise ValueError("Positive maturity, >=2 scrambles, and power<=20 required")
    positive_integer(seed+1,"seed+1")
    rng=np.random.default_rng(seed)
    dt=maturity/steps;discount=math.exp(-rate*maturity)
    def payoffs(z):
        logs=math.log(spot)+np.cumsum((rate-dividend_yield-.5*volatility**2)*dt+volatility*math.sqrt(dt)*z,axis=1)
        arithmetic=np.exp(logs).mean(axis=1)
        geometric=np.exp(logs.mean(axis=1))
        return discount*np.maximum(arithmetic-strike,0),discount*np.maximum(geometric-strike,0)
    log_mean=math.log(spot)+(rate-dividend_yield-.5*volatility**2)*maturity*(steps+1)/(2*steps)
    log_variance=volatility**2*maturity*(steps+1)*(2*steps+1)/(6*steps**2)
    if log_variance==0:
        geometric_exact=discount*max(math.exp(log_mean)-strike,0)
    else:
        root=math.sqrt(log_variance)
        d2=(log_mean-math.log(strike))/root
        geometric_exact=discount*(math.exp(log_mean+.5*log_variance)*norm.cdf(d2+root)-strike*norm.cdf(d2))
    a,g=payoffs(rng.standard_normal((1024,steps)))
    coefficient=float(np.cov(a,g,ddof=1)[0,1]/np.var(g,ddof=1)) if np.var(g)>0 else 0.0
    estimates=[]
    for _ in range(replicates):
        sampler=qmc.Sobol(steps,scramble=True,seed=int(rng.integers(0,2**32-1)))
        uniforms=np.clip(sampler.random_base2(power),np.finfo(float).eps,1-np.finfo(float).eps)
        a,g=payoffs(norm.ppf(uniforms))
        estimates.append(float(np.mean(a-coefficient*(g-geometric_exact))))
    price=float(np.mean(estimates));error=float(np.std(estimates,ddof=1)/math.sqrt(replicates))
    half=float(student_t.ppf(.975,replicates-1)*error)
    return MonteCarloPrice(price,error,(price-half,price+half),coefficient,replicates*2**power,seed)
