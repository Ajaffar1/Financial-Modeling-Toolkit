# Methodology and research controls

## Data contract

Numerical APIs accept arrays, not timestamped market-data tables. Observations are
rows, assets/factors are columns; decimal returns use 0.01 for 1%. Callers must align
calendar dates and currencies, use total-return adjusted data, and handle delistings,
missing observations, corporate actions, and point-in-time universes. Nonfinite
inputs are rejected instead of silently filled. Positive prices are required for
price-to-return conversion. Portfolio units, factor units, and covariance frequency
must match. These checks cannot detect survivorship bias in supplied data.

## Portfolio construction and evaluation

Minimum variance solves a fully invested quadratic objective with SLSQP, common
per-asset bounds, and an optional minimum expected return. Defaults are long-only.
Covariance must be symmetric positive semidefinite. Infeasible or unsuccessful
solutions raise an exception. The sample estimator annualizes arithmetic means and
sample covariance. User-selected shrinkage reduces off-diagonal covariance toward
zero; it is not an automatically estimated Ledoit–Wolf coefficient.

Backtest target row t is available after close t and first participates in return
t+1. Initial capital is cash. Turnover equals sum of absolute asset weight changes,
so selling one unit and buying one unit costs two units of turnover. Costs apply
before returns: net return = (1 - cost) * (1 + gross return) - 1. Holdings drift after
market moves. Residual capital earns zero; financing, margin, stock borrow, market
impact, capacity, intraday execution, and taxes are not modeled.

Walk-forward allocation estimates moments from the trailing window through t and
trades on t+1. Between scheduled rebalances it retains drifted weights. Its
performance includes the cash warm-up, while the example explicitly excludes that
warm-up from tail-risk and factor diagnostics. Changing future data must not change
past P&L. Tests verify that invariant. This does not establish economic performance;
parameter selection and universe construction can still overfit research.

CAGR compounds returns; volatility uses sample standard deviation. Sharpe uses
annualized arithmetic excess return and sample volatility, with a geometrically
converted risk-free rate. Sortino uses the root mean squared negative excess return.
Undefined ratios return None. Drawdown includes initial capital as the first peak.

## Risk and factors

Historical VaR is the interpolated confidence quantile of observed losses.
Expected shortfall averages the worst (1-confidence) fraction of the empirical
sample with fractional weight at its boundary. Loss is positive; all-positive
return samples can produce negative risk estimates rather than being clamped.
Both estimates have the same horizon as input observations; no square-root-of-time
scaling is assumed. Small samples provide unstable tail estimates.

Volatility contributions use Euler allocation w_i*(Sigma*w)_i/sigma and sum to
portfolio volatility. Short positions may produce negative contributions. Stress
P&L is linear in asset return shocks; derivatives need separate full revaluation.

Factor regression includes an intercept. Standard errors use a Bartlett
Newey–West sandwich estimator with n/(n-k) finite-sample correction; zero lags give
HC1 heteroskedastic errors. P-values use a Student-t approximation with n-k degrees
of freedom and are not exact finite-sample inference under serial dependence.
Rank-deficient designs are rejected. Alpha is per observation, not annualized.
Constant dependent returns yield undefined R²; zero uncertainty yields NaN t-stats.

## Options and bonds

Black–Scholes assumes European exercise, constant volatility, continuous rates and
dividend yield, and frictionless lognormal dynamics. Zero-volatility price uses the
discounted deterministic payoff; expiration uses intrinsic value. Greeks require
positive volatility and maturity. Vega/rho are per 1.0 change, so multiply by 0.01
for a one-percentage-point move. Theta is annual calendar decay. Implied volatility
checks discounted no-arbitrage bounds and uses bracketed root finding.

Bonds are fixed-rate bullets on a coupon date; years*frequency must be integral.
Yield is nominal annual compounded at coupon frequency. Prices exclude accrued
interest because valuation is on a coupon date. Modified duration and convexity
are yield derivatives; they are not key-rate sensitivities or credit-spread risk.
No callable bonds, floating-rate notes, default, or yield curves are modeled.

## Corporate-finance scenarios and simulation

Operating drivers replace base assumptions for a single forecast year. Scenario
changes do not displace explicit year overrides. Ratios use closing debt/cash and
annual earnings. Covenant checks are configurable numeric thresholds, not
interpretations of contractual definitions, EBITDA addbacks, or cure rights.

DCF simulation samples correlated normal assumption shocks using a local NumPy
random generator and a stored seed. Covariance is in squared assumption units.
Every draw is reforecast. Any invalid draw stops with its index; no hidden clipping,
rejection sampling, or survivorship of valid draws is applied. This distribution
is an analyst-selected uncertainty model, not a calibrated market pricing model.
Quantiles are simulated valuation percentiles, not guaranteed confidence bounds.

## Verification

Tests cover analytical minimum variance, known Black–Scholes price, put–call parity,
finite-difference Greeks and bond derivatives, yield/volatility inversion, factor
coefficient recovery, cash/statement reconciliation, scenario isolation, Excel
round trips, tail mass, delayed signals, and future-data independence. GitHub Actions
runs tests across Python 3.10–3.13 and validates distributions before a tagged release.
