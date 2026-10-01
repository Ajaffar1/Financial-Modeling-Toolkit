# Financial Modeling Toolkit

A Python library for corporate-finance modeling and quantitative investment research.
Version 0.3 adds advanced statistical research, allocation, and execution models
on top of portfolio construction, backtesting, risk, factors, derivatives,
fixed income, and stochastic valuation alongside integrated financial statements.

## Install

Requires Python 3.10 or later. From this repository:

```sh
python -m pip install '.[all]'
```

For calculations without Excel dependencies, use `python -m pip install .`.
The package is not yet published to PyPI.

## Quick start

```python
from finmodel import DCFModel, OpeningBalance

model = DCFModel(
    revenue=100_000_000,
    ebitda_margin=0.22,
    tax_rate=0.265,
    wacc=0.09,
    terminal_growth=0.025,
    opening=OpeningBalance(
        cash=5_000_000, working_capital=15_000_000,
        ppe=40_000_000, debt=20_000_000,
    ),
).forecast(years=5)

print(model.enterprise_value())
print(model.equity_value())
print(model.audit())
print(model.sensitivity("wacc", "terminal_growth"))

downside = model.scenario("Downside", {"revenue_growth": -0.05})
model.export_excel("Company_Model.xlsx")
```

`FinancialModel` and `DCFModel` expose the same integrated model API. Scenarios
return independent models; overrides replace values rather than add deltas.
`forecast()` replaces the forecast and returns the model. Sensitivity returns
`{(row_value, column_value): enterprise_value}` and accepts explicit
`row_values` and `column_values`. Invalid valuation combinations raise an error.

## Accounting and valuation conventions

- Revenue is the opening/base-year figure. Year 1 applies revenue growth.
- EBITDA uses revenue times EBITDA margin. Depreciation applies to opening PP&E;
  capex is a revenue percentage. No depreciation is charged on current-year capex.
- Working capital is net operating working capital, represented as a single net
  asset. This is a condensed balance sheet, not a gross AR/inventory/AP model.
- Interest is charged on beginning-of-year debt. Repayments are capped at debt
  outstanding. No automatic revolver or cash sweep is assumed.
- Cash taxes equal the tax rate times positive pretax income; losses generate no
  immediate tax benefit. Deferred tax and loss carryforwards are not modeled.
- Opening equity is derived from opening net assets. Net income flows into equity;
  dividends, share issues, acquisitions, and disposals are outside the current model.
- Unlevered cash flow uses EBIT less cash taxes on positive EBIT, plus depreciation,
  less capex and the change in working capital. DCF uses annual end-period
  discounting and Gordon-growth terminal value. WACC must exceed terminal growth.
- Equity value uses **opening-date** cash and debt. All amounts use a single,
  user-selected currency and consistent units; no currency conversion is performed.
- Default opening balances are illustrative. Supply actual opening balances for
  company analysis. Negative cash remains visible as a funding shortfall.

## Controls and Excel

`audit()` returns structured findings with code, severity, message, and year.
It checks balance-sheet equality, cash reconciliation, negative cash/debt/PP&E,
nonpositive enterprise value, and terminal value above 75% of enterprise value.
Warnings require review and do not automatically invalidate the model.

`finmodel.forecasting.scenarios.assumption_changes(base, downside)` reports all
changed assumptions. `finmodel.validation.balance_checks.circular_references`
checks an explicitly supplied dependency graph. It does not scan workbook formulas.
The core forecast is sequential and avoids an interest/cash circularity.

Excel export writes calculated values and input sheets, not live Excel formulas.
`FinancialModel.from_excel("Company_Model.xlsx")` imports only this toolkit's
version 1 workbook and recomputes from the Assumptions and Opening sheets.
Editing output statement cells does not change the imported model. Formula inputs
are rejected. Arbitrary third-party Excel models are unsupported.

## Development

```sh
python -m unittest discover -s tests -v
python examples/basic.py  # after installation
```

Modules are grouped under `statements`, `valuation`, `forecasting`, `debt`,
`analysis`, `excel`, and `validation`. GitHub Actions tests Python 3.10–3.13.

## Roadmap

Further development: detailed working capital, revolver facilities, investment
returns, formula-aware Excel adapters, and a separate Canadian tax extension with
rules versioned by tax year (CCA and project economics).
Detailed working capital, revolvers, investment-return models, formula-aware Excel adapters,
and Canadian tax rules are planned rather than implemented. This toolkit is a
modeling aid; validate assumptions and results before relying on them.

MIT licensed. See LICENSE.

## Automated builds and releases

Every push and pull request runs tests on Python 3.10–3.13. To publish a GitHub
release, update the version in `pyproject.toml`, commit it, then push a matching
tag (for example `v0.1.0`). The release workflow verifies the version, runs tests,
builds and checks a wheel and source distribution, and attaches them to a GitHub
release. It uses GitHub's built-in workflow token; no persistent release secret
is needed. PyPI publishing is not configured.


## Quantitative research API

Install `.[quant]` for NumPy/SciPy analytics or `.[all]` for analytics plus Excel.
The core corporate-finance API remains usable without these optional dependencies.

| Area | Implemented capability |
| --- | --- |
| Portfolio construction | Constrained minimum variance, target return, weight bounds, diagonal covariance shrinkage |
| Backtesting | Delayed targets, drifted holdings, transaction costs, walk-forward allocation |
| Risk | Historical VaR/expected shortfall, volatility contributions, linear stress P&L |
| Factors | OLS alpha/betas, R², HAC/Newey–West uncertainty and approximate p-values |
| Derivatives | European Black–Scholes calls/puts, dividend yield, Greeks, implied volatility |
| Fixed income | Bullet bond pricing, yield inversion, duration, convexity |
| Valuation | Correlated seeded assumption simulation, quantiles, EV/equity bridge, multiples |
| Credit controls | Leverage, coverage, and minimum-cash covenant checks |

```python
import numpy as np
from finmodel.portfolio.optimization import minimum_variance
from finmodel.risk.measures import risk_contributions, stress_pnl
from finmodel.derivatives.options import black_scholes, implied_volatility

covariance = np.array([[0.04, 0.01], [0.01, 0.09]])
allocation = minimum_variance([0.08, 0.12], covariance, target_return=0.09)
print(allocation.weights)
print(risk_contributions(allocation.weights, covariance))
print(stress_pnl(allocation.weights, [[-0.20, -0.35]], portfolio_value=10_000_000))
price = black_scholes(100, 105, 1, 0.04, 0.25)
print(implied_volatility(price, 100, 105, 1, 0.04))
```

Run `python examples/quant_research.py` after installation for a complete, seeded,
synthetic portfolio/factor/risk/options/bond/valuation report. See
[methodology](docs/methodology.md) for units, time alignment, and limitations.

## Year-specific operating forecasts

```python
model.forecast(5, drivers={
    1: {"revenue_growth": 0.02, "capex_ratio": 0.06},
    2: {"ebitda_margin": 0.20},
})
```

Overrides apply only to the specified year and take precedence over base/scenario
assumptions for that year. Unspecified years use base assumptions. Excel round trips
preserve drivers. WACC and terminal growth stay valuation-date assumptions.

## Scope and deployment

This is a research library, not an institutionally certified trading or bank-risk
platform. It is not affiliated with Goldman Sachs or any hedge fund. Production
use would require independent model validation, controlled market-data pipelines,
portfolio/accounting reconciliation, operational monitoring, and execution systems.
There are no live data feeds, broker connections, or automatic trades.


## Advanced quantitative research (v0.3)

| Research area | Algorithms |
| --- | --- |
| Covariance | Gaussian OAS shrinkage, exponentially weighted covariance |
| Allocation | Convex risk budgets, Black–Litterman Bayesian views, sparse CVaR linear programming |
| Statistical arbitrage | OU spread calibration, causal Kalman hedge-ratio filtering |
| Volatility and regimes | Stationary GARCH(1,1), log-domain Gaussian HMM with EM |
| Experiment controls | Event-overlap purging, embargo, probabilistic and deflated Sharpe |
| Derivatives simulation | Randomized Sobol Asian calls with independently calibrated control variates |
| Execution mathematics | Inventory-aware quotes, tridiagonal optimal-liquidation schedules |

```python
from finmodel.risk.covariance import oas_covariance
from finmodel.portfolio.advanced import risk_parity, minimum_cvar
from finmodel.derivatives.monte_carlo import asian_call
from finmodel.execution.optimal_execution import optimal_liquidation

# training_returns: observations × assets, decimal returns, training data only
cov, shrinkage = oas_covariance(training_returns)
weights = risk_parity(cov)
tail_allocation = minimum_cvar(training_returns, confidence=0.95)
option = asian_call(100, 100, 1, 0.04, 0.20, seed=42)
plan = optimal_liquidation(1000, 1, intervals=20)
```

Run `python examples/advanced_research.py` for a reproducible synthetic lab.
Read [advanced methodology](docs/advanced_research.md) before selecting data,
units, thresholds, or interpreting statistical confidence. Algorithms follow
published literature; they do not replicate a proprietary fund's system or imply
comparable investment performance.
