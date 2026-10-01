"""Univariate Gaussian HMM with log-domain inference and deterministic EM."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from ..analysis._inputs import array, positive_integer

@dataclass(frozen=True)
class GaussianHMM:
    means: np.ndarray
    variances: np.ndarray
    transition: np.ndarray
    initial: np.ndarray
    likelihood_history: tuple
    converged: bool
    def filter(self, observations):
        """Causal probabilities P(state_t | observations through t)."""
        return _infer(array(observations,1,"observations"),self.means,self.variances,self.transition,self.initial)[1]
    def smooth(self, observations):
        """Retrospective probabilities using the entire sequence; not tradable at t."""
        return _infer(array(observations,1,"observations"),self.means,self.variances,self.transition,self.initial)[2]

def _infer(y,means,variances,transition,initial):
    emissions=-.5*(np.log(2*np.pi*variances)[None,:]+(y[:,None]-means)**2/variances)
    log_transition=np.log(transition)
    alpha=np.empty_like(emissions)
    alpha[0]=np.log(initial)+emissions[0]
    for t in range(1,len(y)):
        alpha[t]=emissions[t]+logsumexp(alpha[t-1][:,None]+log_transition,axis=0)
    likelihood=float(logsumexp(alpha[-1]))
    filtered=np.exp(alpha-logsumexp(alpha,axis=1)[:,None])
    beta=np.zeros_like(alpha)
    for t in range(len(y)-2,-1,-1):
        beta[t]=logsumexp(log_transition+emissions[t+1][None,:]+beta[t+1][None,:],axis=1)
    log_gamma=alpha+beta
    gamma=np.exp(log_gamma-logsumexp(log_gamma,axis=1)[:,None])
    counts=np.zeros_like(transition)
    for t in range(len(y)-1):
        log_xi=alpha[t][:,None]+log_transition+emissions[t+1][None,:]+beta[t+1][None,:]
        counts+=np.exp(log_xi-logsumexp(log_xi))
    return likelihood,filtered,gamma,counts

def fit_gaussian_hmm(observations,*,states=2,max_iterations=200,tolerance=1e-6):
    y=array(observations,1,"observations")
    positive_integer(states,"states");positive_integer(max_iterations,"max_iterations")
    if states<2 or len(y)<states*5 or y.var()==0 or not np.isfinite(tolerance) or tolerance<=0:
        raise ValueError("Need nonconstant observations, >=5 per state, states>=2 and positive tolerance")
    means=np.quantile(y,np.linspace(.1,.9,states))
    floor=max(y.var()*1e-6,np.finfo(float).tiny)
    variances=np.full(states,y.var())
    transition=.1*np.ones((states,states))/states+.9*np.eye(states)
    initial=np.full(states,1/states)
    history=[];converged=False
    for iteration in range(max_iterations):
        ll,_,gamma,counts=_infer(y,means,variances,transition,initial)
        history.append(ll)
        if len(history)>1 and ll<history[-2]-1e-6:
            raise ValueError("HMM likelihood decreased; numerical convergence failure")
        if len(history)>1 and abs(ll-history[-2])<tolerance:
            converged=True;break
        if iteration==max_iterations-1:break
        mass=gamma.sum(axis=0)
        if np.any(mass<1e-8):raise ValueError("HMM state collapsed")
        means=(gamma.T@y)/mass
        variances=np.maximum(floor,(gamma*(y[:,None]-means)**2).sum(axis=0)/mass)
        initial=np.maximum(gamma[0],1e-12);initial/=initial.sum()
        transition=np.maximum(counts,1e-12);transition/=transition.sum(axis=1)[:,None]
    order=np.argsort(means)
    return GaussianHMM(means[order],variances[order],transition[np.ix_(order,order)],initial[order],tuple(history),converged)
