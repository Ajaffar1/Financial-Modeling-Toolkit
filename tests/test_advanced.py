import unittest
import numpy as np
from finmodel.risk.covariance import oas_covariance, ewma_covariance
from finmodel.portfolio.advanced import risk_parity, black_litterman, minimum_cvar
from finmodel.risk.measures import historical_risk, risk_contributions
from finmodel.statistics.statarb import fit_ou, kalman_hedge
from finmodel.statistics.volatility import fit_garch
from finmodel.statistics.regimes import fit_gaussian_hmm
from finmodel.research.validation import purged_kfold, probabilistic_sharpe, deflated_sharpe
from finmodel.derivatives.monte_carlo import asian_call
from finmodel.derivatives.options import black_scholes
from finmodel.execution.market_making import inventory_quotes

class AdvancedTests(unittest.TestCase):
    def test_oas_high_dimensional_psd_and_scale(self):
        x=np.random.default_rng(4).normal(size=(20,50))
        cov,shrinkage=oas_covariance(x)
        self.assertTrue(0<=shrinkage<=1)
        self.assertGreater(np.linalg.eigvalsh(cov).min(),0)
        np.testing.assert_allclose(oas_covariance(3*x)[0],9*cov)
        np.testing.assert_allclose(oas_covariance(x+10)[0],cov)
        self.assertGreaterEqual(np.linalg.eigvalsh(ewma_covariance(x)).min(),-1e-12)
    def test_risk_budget_analytical_and_custom(self):
        cov=np.diag([.04,.09,.16])
        expected=1/np.sqrt(np.diag(cov));expected/=expected.sum()
        np.testing.assert_allclose(risk_parity(cov),expected,atol=1e-6)
        budgets=np.array([.2,.3,.5]);weights=risk_parity(cov,budgets)
        rc=risk_contributions(weights,cov)
        np.testing.assert_allclose(rc/rc.sum(),budgets,atol=1e-6)
        with self.assertRaises(ValueError):risk_parity([[1,1],[1,1]])
    def test_black_litterman_scalar_bayes(self):
        mu,cov=black_litterman([[.04]],[1],[[1]],[.2],[[.01]],risk_aversion=2,tau=.5)
        self.assertAlmostEqual(mu[0],.08+.02/.03*(.2-.08))
        self.assertAlmostEqual(cov[0,0],.02-.02**2/.03)
    def test_cvar_matches_empirical_tail(self):
        r=np.array([[-.2,.01],[-.1,.01],[.1,.01],[.2,.01]])
        allocation=minimum_cvar(r,confidence=.5)
        np.testing.assert_allclose(allocation.weights,[0,1],atol=1e-8)
        self.assertAlmostEqual(allocation.expected_shortfall,-.01)
        one=minimum_cvar(r[:,:1],confidence=.625)
        self.assertAlmostEqual(one.expected_shortfall,historical_risk(r[:,0],.625).expected_shortfall)
        with self.assertRaises(ValueError):minimum_cvar(r,target_return=.5)
    def test_ou_recovers_synthetic_dynamics(self):
        rng=np.random.default_rng(5);phi=.9;spread=np.zeros(20000)
        for t in range(1,len(spread)):spread[t]=.1+phi*spread[t-1]+rng.normal(0,.1)
        fit=fit_ou(spread)
        self.assertAlmostEqual(fit.ar_coefficient,phi,delta=.01)
        self.assertAlmostEqual(fit.long_run_mean,1,delta=.05)
        self.assertAlmostEqual(fit.half_life,np.log(2)/-np.log(phi),delta=.7)
        with self.assertRaises(ValueError):fit_ou(np.ones(10))
    def test_kalman_causal_and_coefficient_recovery(self):
        x=np.random.default_rng(8).normal(size=500);y=2+1.5*x
        fit=kalman_hedge(y,x,process_variance=0,observation_variance=.001)
        np.testing.assert_allclose(fit.coefficients[-1],[2,1.5],atol=.001)
        changed=y.copy();changed[300:]+=50
        other=kalman_hedge(changed,x,process_variance=0,observation_variance=.001)
        np.testing.assert_array_equal(fit.coefficients[:300],other.coefficients[:300])
        self.assertTrue(np.all(fit.innovation_variances>0))
    def test_garch_stationarity_and_forecast_recursion(self):
        rng=np.random.default_rng(7);r=np.zeros(1000);h=.0001
        for t in range(len(r)):
            r[t]=rng.normal(0,np.sqrt(h));h=.000005+.1*r[t]**2+.85*h
        fit=fit_garch(r)
        self.assertLess(fit.alpha+fit.beta,1)
        self.assertGreater(fit.omega,0)
        f=fit.forecast(3)
        self.assertAlmostEqual(f[0],fit.conditional_variances[-1])
        self.assertAlmostEqual(f[1],fit.omega+(fit.alpha+fit.beta)*f[0])
        self.assertTrue(np.all(fit.conditional_variances>0))
    def test_hmm_likelihood_and_causal_inference(self):
        rng=np.random.default_rng(1)
        y=np.r_[rng.normal(-2,.3,150),rng.normal(2,.4,150),rng.normal(-2,.3,150)]
        model=fit_gaussian_hmm(y)
        self.assertTrue(np.all(np.diff(model.likelihood_history)>=-1e-6))
        self.assertTrue(model.converged)
        np.testing.assert_allclose(model.means,[-2,2],atol=.1)
        filtered=model.filter(y)
        np.testing.assert_allclose(filtered.sum(axis=1),1)
        np.testing.assert_allclose(filtered[:200],model.filter(y[:200]))
        np.testing.assert_allclose(model.smooth(y).sum(axis=1),1)
        self.assertGreater(filtered[200,1],.99)
    def test_purging_and_embargo(self):
        starts=np.arange(20);ends=starts+3
        for fold in purged_kfold(starts,ends,folds=4,embargo=2):
            for i in fold.train:
                for j in fold.test:self.assertFalse(starts[i]<=ends[j] and ends[i]>=starts[j])
                self.assertFalse(ends[fold.test].max()<starts[i]<=ends[fold.test].max()+2)
        with self.assertRaises(ValueError):purged_kfold([2,1],[3,2])
    def test_deflation_penalizes_search(self):
        r=np.random.default_rng(3).normal(.001,.01,500)
        p=probabilistic_sharpe(r)
        self.assertTrue(0<=p<=1)
        self.assertEqual(deflated_sharpe(r,trials=1,sharpe_variance=.01),p)
        self.assertLess(deflated_sharpe(r,trials=100,sharpe_variance=.01),p)
    def test_sobol_asian_benchmark_and_seed(self):
        result=asian_call(100,100,1,.05,.2,steps=1,power=11,replicates=8)
        exact=black_scholes(100,100,1,.05,.2)
        self.assertAlmostEqual(result.price,exact,places=9)
        asian=asian_call(100,100,1,.05,.2,steps=12,power=9)
        self.assertGreater(asian.price,0)
        self.assertLess(asian.price,exact)
        self.assertEqual(asian,asian_call(100,100,1,.05,.2,steps=12,power=9))
        deterministic=asian_call(100,90,1,0,0,steps=12,power=5)
        self.assertAlmostEqual(deterministic.price,10)
        self.assertAlmostEqual(deterministic.standard_error,0)
    def test_inventory_quotes_symmetric_and_inventory_shift(self):
        zero=inventory_quotes(100,0,2,1)
        long=inventory_quotes(100,10,2,1)
        self.assertAlmostEqual((zero.bid+zero.ask)/2,100)
        self.assertLess(long.bid,zero.bid)
        self.assertLess(long.ask,zero.ask)
        self.assertGreater(zero.ask,zero.bid)


class ExecutionTests(unittest.TestCase):
    def test_uniform_risk_neutral_and_front_loaded_risky(self):
        from finmodel.execution.optimal_execution import optimal_liquidation
        neutral=optimal_liquidation(100,1,intervals=10,risk_aversion=0)
        np.testing.assert_allclose(neutral.trades,10)
        risky=optimal_liquidation(100,1,intervals=10,risk_aversion=1)
        self.assertGreater(risky.trades[0],neutral.trades[0])
        self.assertGreater(risky.temporary_impact_cost,neutral.temporary_impact_cost)
        self.assertLess(risky.inventory_pnl_variance,neutral.inventory_pnl_variance)
        self.assertAlmostEqual(risky.trades.sum(),100)
        self.assertTrue(np.all(risky.trades>=0))
        # Compare stationary solution with feasible perturbations of an interior inventory.
        dt=.1;eta=.001;lam=1
        for delta in (-1,1):
            h=risky.holdings.copy();h[5]+=delta
            objective=eta/dt*np.sum(np.diff(h)**2)+lam*dt*np.sum(h[1:]**2)
            self.assertGreater(objective,risky.objective)

if __name__ == "__main__":
    unittest.main()
