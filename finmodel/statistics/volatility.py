from dataclasses import dataclass
import numpy as np
from scipy.optimize import minimize
from ..analysis._inputs import array, positive_integer

@dataclass(frozen=True)
class GARCHFit:
    mean: float
    omega: float
    alpha: float
    beta: float
    conditional_variances: np.ndarray
    log_likelihood: float
    def forecast(self, steps=1):
        positive_integer(steps,"steps")
        persistence = self.alpha+self.beta
        long_run = self.omega/(1-persistence)
        # conditional_variances includes next-period variance at its final entry.
        return long_run+(self.conditional_variances[-1]-long_run)*persistence**np.arange(steps)

def fit_garch(returns):
    """Gaussian constant-mean GARCH(1,1) QMLE, constrained stationary fit."""
    r = array(returns,1,"returns")
    if len(r) < 30 or r.std() <= 0:
        raise ValueError("Need >=30 nonconstant observations")
    scale = r.std()
    y = (r-r.mean())/scale
    def variances(parameters):
        omega,alpha,beta = parameters
        h = np.empty(len(y)+1);h[0]=np.var(y)
        for t in range(len(y)):
            h[t+1]=omega+alpha*y[t]**2+beta*h[t]
        return h
    def objective(parameters):
        h=variances(parameters)[:-1]
        return .5*np.sum(np.log(2*np.pi)+np.log(h)+y*y/h)
    best=None
    for alpha,beta in ((.05,.9),(.15,.7),(.3,.3)):
        result=minimize(objective,[1-alpha-beta,alpha,beta],method="SLSQP",
                        bounds=[(1e-9,None),(0,.999),(0,.999)],
                        constraints=[{"type":"ineq","fun":lambda p:.999-p[1]-p[2]}],
                        options={"ftol":1e-9,"maxiter":1000})
        if result.success and np.isfinite(result.fun) and result.x[1]+result.x[2]<1 and (best is None or result.fun < best.fun):best=result
    if best is None:
        raise ValueError("GARCH likelihood optimizer did not converge")
    omega,alpha,beta=best.x
    return GARCHFit(float(r.mean()),float(omega*scale**2),float(alpha),float(beta),
                    variances(best.x)*scale**2,float(-best.fun-len(r)*np.log(scale)))
