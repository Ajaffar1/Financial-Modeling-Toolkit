"""Reproducible synthetic research demo; no external market-data credentials."""
import json
from dataclasses import asdict
import numpy as np
from finmodel.analysis.factors import fit_factor_model
from finmodel.portfolio.research import walk_forward_minimum_variance
from finmodel.risk.measures import historical_risk
from finmodel.derivatives.options import black_scholes, greeks
from finmodel.fixed_income.bonds import bond_analytics
from finmodel.valuation.simulation import simulate_valuation
from finmodel import FinancialModel

rng = np.random.default_rng(42)
factors = rng.normal([.0002,.0001], [.01,.005], (756,2))
returns = factors @ np.array([[1.0,.8,1.2], [.3,-.2,.1]]) + rng.normal(.0001,.006,(756,3))
strategy = walk_forward_minimum_variance(returns,lookback=126,rebalance_every=21,transaction_cost_bps=5)
fit = fit_factor_model(strategy.net_returns[126:],factors[126:],hac_lags=5)
model = FinancialModel(100_000_000).forecast(5,drivers={1:{'revenue_growth':.03},2:{'capex_ratio':.06}})
simulation = simulate_valuation(model,['revenue_growth','ebitda_margin'],[[.0001,0],[0,.0001]],samples=500,seed=42)
report = {
    'data': 'Synthetic; not an investment recommendation or a live strategy',
    'performance_including_cash_warmup': asdict(strategy.performance),
    'daily_tail_risk_after_warmup': asdict(historical_risk(strategy.net_returns[126:])),
    'factor_alpha_per_day': fit.alpha, 'factor_betas': fit.betas.tolist(),
    'option_price': black_scholes(100,105,1,.04,.25),
    'option_greeks': asdict(greeks(100,105,1,.04,.25)),
    'bond': asdict(bond_analytics(100,.05,10,.045)),
    'dcf_quantiles': simulation.quantiles,
}
print(json.dumps(report,indent=2,allow_nan=False))
