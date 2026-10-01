# Advanced algorithms: equations, contracts, and limits

Install `.[all]` and run `python examples/advanced_research.py`. All inputs use
explicit units; arrays do not carry timestamps or market-data provenance.

## Allocation and covariance

**OAS** uses the original Gaussian oracle-approximating shrinkage formula of Chen
et al. The covariance convention is maximum likelihood (divide by n), not n-1.
It shrinks the covariance toward tr(S)/p times identity. It is an estimator under
Gaussian assumptions, not protection against structural breaks or heavy tails.
EWMA centers on the normalized exponentially weighted mean and returns a weighted
population covariance. Neither estimator annualizes inputs automatically.

**Risk budgets** solve the convex objective 0.5*xᵀΣx - Σ bᵢ log(xᵢ), then normalize
x into fully invested weights. Positive definite covariance and strictly positive
budgets summing to one are required. Euler volatility shares must match budgets.

**Black–Litterman** starts from equilibrium π=δΣw and mean uncertainty C=τΣ.
The posterior mean is π+C Pᵀ(P C Pᵀ+Ω)⁻¹(q-Pπ). Returned uncertainty is covariance
of the mean, not a predictive covariance for future returns. View returns and
covariance must have consistent horizons; Ω is analyst-supplied uncertainty.

**CVaR allocation** uses the Rockafellar–Uryasev objective
z + Σ max(-rₜᵀw-z,0)/(n(1-α)), subject to long-only unit-sum weights and optional
sample expected-return target. A sparse HiGHS linear program avoids a dense n×n
identity matrix. Scenario/sample bias remains; historical CVaR is not a forecast.

## Time-series models

**OU** fits an intercept AR(1) and maps φ into κ=-log(φ)/dt, θ=a/(1-φ), and
σ²=innovation_variance*2κ/(1-φ²). Half-life is log(2)/κ. Only 0<φ<1 is compatible
with the implemented stationary continuous process. The estimator does not test
cointegration, structural stationarity, or residual independence.

**Kalman hedge ratios** follow a random walk for intercept and slope. The filter
uses prior innovations and Joseph covariance updates. Posterior coefficient t
uses observation t and is first tradable at t+1. Process/observation variances are
explicit hyperparameters, not automatically calibrated. Price levels can lead to
poor conditioning; transform and scale input series deliberately.

**GARCH(1,1)** fits Gaussian quasi-likelihood on scaled demeaned returns with
ω>0, α≥0, β≥0, α+β<1. Conditional variance evolves as
hₜ₊₁=ω+α εₜ²+β hₜ. Three starting points mitigate local optimizer dependence.
The result includes the next-period variance as its final entry, and forecasts
mean-revert at α+β. No Student-t likelihood, leverage effects, parameter standard
errors, or stationarity confidence intervals are estimated.

**Gaussian HMM** uses log-domain forward/backward recursions and Baum–Welch EM,
with a variance floor. States are sorted by mean. Check `converged` and likelihood
history. EM can converge to a local optimum; labels are not economic truths.
`filter` is causal for fixed parameters; `smooth` uses future observations.
Fit parameters on training data only. Calling filter on a new sequence restarts
from the learned initial distribution; it does not carry the preceding sequence's
terminal state distribution automatically.

## Statistical research validation

**Purged K-fold** removes training events whose inclusive [start,end] label
intervals overlap any test event. An embargo removes starts within the specified
time units after the latest test end. Times are numeric and must be sorted.
This splitter can train on future observations: use it for purged CV, not as a
substitute for chronological out-of-sample testing. Long events can empty training
folds; callers must check fold sizes before fitting a learner.

**Probabilistic Sharpe** uses the skewness/kurtosis-adjusted asymptotic statistic:
(SR-SR*)√(n-1) / √(1-skew*SR+(kurtosis-1)*SR²/4).
Sharpe and its benchmark are per observation, not annualized. Returns are assumed
IID; serial dependence violates the nominal interpretation.

**Deflated Sharpe** sets SR* to an expected maximum of independent Gaussian trial
Sharpes using Euler's constant. Supply cross-trial Sharpe variance and a defensible
trial count. Correlated trials require an independently assessed effective count.
It does not reconstruct a hidden experiment history or certify a strategy.

## Monte Carlo derivatives

Arithmetic-average Asian calls use discrete monitoring dates T/n,...,T (excluding
initial spot), lognormal GBM, randomized Sobol samples, and a geometric-average
control variate with a closed-form expectation. A separate pseudorandom pilot
estimates the control coefficient to avoid fitting it on valuation paths.
Independent scrambles provide replicate means; uncertainty uses their sample
standard deviation divided by √replicates, not IID errors over Sobol paths.
The reported interval is approximate Student-t 95%; it captures numerical sampling
uncertainty, not model risk. The result stores the seed, path count, coefficient,
price, and standard error. Working memory grows as 2^power × monitoring steps.

## Execution research

**Inventory quotes** use the Avellaneda–Stoikov finite-horizon approximation:
reservation price = mid - inventory*γ*σ²*remaining_time;
half spread = γσ²remaining_time/2 + log(1+γ/k)/γ.
σ is arithmetic price volatility per square-root time; γ/k units must match.
Nonpositive bids are rejected. No queue, tick, inventory limits, fills, latency,
exchange rules, or broker connectivity are simulated.

**Optimal liquidation** minimizes η/dt Σ trades² + λσ²dt Σ post-trade holdings²
subject to initial quantity and zero final inventory. A tridiagonal positive
definite linear system gives the deterministic schedule. Risk-neutral execution
is uniform; risk aversion front-loads trades. Temporary impact is linear in trade
rate. Permanent impact, drift, spread, participation limits, and stochastic fills
are absent. This is an execution cost/risk model, not an order router.

## References

- Chen et al. (2010), *Shrinkage Algorithms for MMSE Covariance Estimation*.
- Black and Litterman (1992), *Global Portfolio Optimization*.
- Rockafellar and Uryasev (2000), *Optimization of Conditional Value-at-Risk*.
- Bollerslev (1986), *Generalized Autoregressive Conditional Heteroskedasticity*.
- Rabiner (1989), *A Tutorial on Hidden Markov Models*.
- Bailey and López de Prado (2014), *The Deflated Sharpe Ratio*.
- Kemna and Vorst (1990), *A Pricing Method for Options Based on Average Asset Values*.
- Avellaneda and Stoikov (2008), *High-Frequency Trading in a Limit Order Book*.
- Almgren and Chriss (2001), *Optimal Execution of Portfolio Transactions*.
