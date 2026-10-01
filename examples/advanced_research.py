"""Advanced synthetic research lab; parameters are illustrative, not trading advice."""
import json
from dataclasses import asdict
import numpy as np
from finmodel.risk.covariance import oas_covariance
from finmodel.portfolio.advanced import risk_parity, black_litterman, minimum_cvar
from finmodel.statistics.regimes import fit_gaussian_hmm
from finmodel.statistics.volatility import fit_garch
from finmodel.statistics.statarb import fit_ou, kalman_hedge
from finmodel.research.validation import purged_kfold, deflated_sharpe
from finmodel.derivatives.monte_carlo import asian_call
from finmodel.execution.optimal_execution import optimal_liquidation
from finmodel.execution.market_making import inventory_quotes

rng=np.random.default_rng(42)
returns=rng.multivariate_normal([.0003,.0002,.0001],[[.0001,.00003,0],[.00003,.00015,.00002],[0,.00002,.00008]],size=1000)
# Fit on training data only; never fit a regime model to the evaluation period.
train,test=returns[:750],returns[750:]
cov,shrinkage=oas_covariance(train)
annual_cov=cov*252
weights=risk_parity(annual_cov)
posterior,_=black_litterman(annual_cov,[1/3]*3,[[1,-1,0]],[.03],[[.001]])
tail=minimum_cvar(train,confidence=.95)
hmm=fit_gaussian_hmm(train[:,0])
vol=fit_garch(train[:,0])
spread=np.zeros(1000)
for i in range(1,len(spread)):spread[i]=.9*spread[i-1]+rng.normal(0,.1)
x=rng.normal(size=1000);hedge=kalman_hedge(1.5*x+2+rng.normal(0,.01,1000),x)
execution=optimal_liquidation(1000,1,temporary_impact=.001,volatility=1,risk_aversion=.01)
report={
    'data':'Synthetic; no live orders or institutional-performance claims',
    'oas_shrinkage':shrinkage,'risk_parity_weights':weights.tolist(),
    'black_litterman_expected_returns':posterior.tolist(),
    'cvar_weights':tail.weights.tolist(),'daily_cvar':tail.expected_shortfall,
    'regime_converged':hmm.converged,'last_filtered_test_state':hmm.filter(test[:,0])[-1].tolist(),
    'next_day_garch_volatility':float(np.sqrt(vol.forecast()[0])),
    'ou_fit':asdict(fit_ou(spread)), 'last_kalman_hedge':hedge.coefficients[-1].tolist(),
    'deflated_sharpe':deflated_sharpe(test@weights,trials=50,sharpe_variance=.002),
    'purged_fold_sizes':[(len(f.train),len(f.test)) for f in purged_kfold(np.arange(1000),np.arange(1000)+5,embargo=3)],
    'asian_option':asdict(asian_call(100,100,1,.04,.2,power=9)),
    'liquidation_first_five_trades':execution.trades[:5].tolist(),
    'liquidation_objective':execution.objective,
    'market_making_quotes':asdict(inventory_quotes(100,1,2,1)),
}
print(json.dumps(report,indent=2,allow_nan=False))
