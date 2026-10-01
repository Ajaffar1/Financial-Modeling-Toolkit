"""Event-interval-aware splitting and multiple-testing-aware Sharpe diagnostics."""
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm
from ..analysis._inputs import array, positive_integer

@dataclass(frozen=True)
class PurgedFold:
    train: np.ndarray
    test: np.ndarray

def purged_kfold(event_starts,event_ends,*,folds=5,embargo=0):
    """Purge inclusive event overlaps; embargo is time units after test interval.

    Numeric timestamps must be sorted by start. Training can include future data;
    this is purged cross-validation, not a walk-forward evaluation.
    """
    starts,ends=array(event_starts,1,"event_starts"),array(event_ends,1,"event_ends")
    positive_integer(folds,"folds")
    if len(starts)!=len(ends) or folds<2 or folds>len(starts) or np.any(np.diff(starts)<0) or np.any(ends<starts) or not np.isfinite(embargo) or embargo<0:
        raise ValueError("Invalid event intervals, folds, or embargo")
    result=[]
    for test in np.array_split(np.arange(len(starts)),folds):
        keep=np.ones(len(starts),dtype=bool)
        for index in test:
            keep &= ~((starts<=ends[index]) & (ends>=starts[index]))
        final=ends[test].max()
        keep &= ~((starts>final)&(starts<=final+embargo))
        keep[test]=False
        result.append(PurgedFold(np.flatnonzero(keep),test))
    return result

def probabilistic_sharpe(returns,benchmark=0.0):
    """Probability-like asymptotic Sharpe statistic; benchmark is per period.

    Uses population skewness and raw kurtosis. Assumes IID returns.
    """
    r=array(returns,1,"returns")
    if len(r)<4 or not np.isfinite(benchmark) or r.std(ddof=1)==0:
        raise ValueError("Need >=4 nonconstant observations and finite benchmark")
    sr=float(r.mean()/r.std(ddof=1))
    z=(r-r.mean())/r.std(ddof=0)
    skew=float(np.mean(z**3));kurtosis=float(np.mean(z**4))
    denominator=1-skew*sr+(kurtosis-1)*sr**2/4
    if denominator<=0:
        raise ValueError("Invalid asymptotic Sharpe variance")
    return float(norm.cdf((sr-benchmark)*np.sqrt(len(r)-1)/np.sqrt(denominator)))

def deflated_sharpe(returns,*,trials,sharpe_variance):
    """Selection-adjusted PSR using expected maximum of independent Gaussian trials.

    Supply cross-trial per-period Sharpe variance. Correlated search trials require
    an externally estimated effective trial count; this API does not estimate it.
    """
    positive_integer(trials,"trials")
    if not np.isfinite(sharpe_variance) or sharpe_variance<0:
        raise ValueError("Sharpe variance must be finite and nonnegative")
    if trials==1:
        benchmark=0
    else:
        euler=.5772156649015329
        benchmark=np.sqrt(sharpe_variance)*((1-euler)*norm.ppf(1-1/trials)+euler*norm.ppf(1-1/(trials*np.e)))
    return probabilistic_sharpe(returns,float(benchmark))
