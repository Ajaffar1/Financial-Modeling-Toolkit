import unittest
import numpy as np
from finmodel import FinancialModel
from finmodel.analysis.returns import performance, simple_returns
from finmodel.analysis.factors import fit_factor_model
from finmodel.portfolio.optimization import minimum_variance
from finmodel.portfolio.backtest import backtest
from finmodel.risk.measures import historical_risk, risk_contributions, stress_pnl
from finmodel.derivatives.options import black_scholes, greeks, implied_volatility
from finmodel.fixed_income.bonds import bond_analytics, bond_yield
from finmodel.valuation.simulation import simulate_valuation

class QuantTests(unittest.TestCase):
    def test_known_option_price_parity_and_iv(self):
        call = black_scholes(100, 100, 1, .05, .2)
        self.assertAlmostEqual(call, 10.450583572185565)
        put = black_scholes(100, 100, 1, .05, .2, kind='put')
        self.assertAlmostEqual(call-put, 100-100*np.exp(-.05))
        self.assertAlmostEqual(implied_volatility(call, 100,100,1,.05), .2)
        self.assertEqual(black_scholes(100,90,0,.05,.2),10)
        self.assertAlmostEqual(black_scholes(100,100,1,.05,0),100-100*np.exp(-.05))
    def test_greeks_finite_differences(self):
        s,k,t,r,v,q=100,105,1.2,.03,.25,.02
        h=.0001
        for kind in ('call','put'):
            f=lambda spot=s, time=t, rate=r, vol=v: black_scholes(spot,k,time,rate,vol,dividend_yield=q,kind=kind)
            g=greeks(s,k,t,r,v,dividend_yield=q,kind=kind)
            self.assertAlmostEqual(g.delta,(f(spot=s+h)-f(spot=s-h))/(2*h),places=6)
            self.assertAlmostEqual(g.gamma,(f(spot=s+h)-2*f()+f(spot=s-h))/h**2,places=5)
            self.assertAlmostEqual(g.vega,(f(vol=v+h)-f(vol=v-h))/(2*h),places=4)
            self.assertAlmostEqual(g.theta,-(f(time=t+h)-f(time=t-h))/(2*h),places=5)
            self.assertAlmostEqual(g.rho,(f(rate=r+h)-f(rate=r-h))/(2*h),places=4)
    def test_option_invalid_inputs(self):
        for price in (-1, 100, float('nan')):
            with self.assertRaises(ValueError): implied_volatility(price,100,100,1,.05)
        with self.assertRaises(ValueError): greeks(100,100,0,.05,.2)
    def test_bond_par_yield_and_derivatives(self):
        a=bond_analytics(100,.05,5,.05)
        self.assertAlmostEqual(a.price,100)
        self.assertAlmostEqual(bond_yield(a.price,100,.05,5),.05)
        h=.00001
        up=bond_analytics(100,.05,5,.05+h).price
        down=bond_analytics(100,.05,5,.05-h).price
        self.assertAlmostEqual(a.modified_duration,-(up-down)/(2*h*a.price),places=6)
        self.assertAlmostEqual(a.convexity,(up-2*a.price+down)/(h*h*a.price),places=4)
        zero=bond_analytics(100,0,5,.05)
        self.assertAlmostEqual(zero.macaulay_duration,5)
    def test_minimum_variance_analytical(self):
        a=minimum_variance([.1,.2],[[.04,0],[0,.09]])
        np.testing.assert_allclose(a.weights,[.09/.13,.04/.13],atol=1e-5)
        with self.assertRaises(ValueError): minimum_variance([.1,.2],[[.04,0],[0,.09]],target_return=.3)
        with self.assertRaises(ValueError): minimum_variance([.1,.2],[[1,2],[2,1]])
        constrained=minimum_variance([.1,.2],[[.04,0],[0,.09]],target_return=.18)
        self.assertGreaterEqual(constrained.annual_return,.18-1e-7)
    def test_tail_risk_fractional_mass(self):
        risk=historical_risk([-.4,-.3,-.2,-.1],confidence=.625,portfolio_value=100)
        self.assertAlmostEqual(risk.expected_shortfall,(40+15)/1.5)
        self.assertGreaterEqual(risk.expected_shortfall,risk.value_at_risk)
        w=np.array([.4,.6]);cov=np.diag([.04,.09])
        self.assertAlmostEqual(risk_contributions(w,cov).sum(),np.sqrt(w@cov@w))
        np.testing.assert_allclose(stress_pnl(w,[[-.1,-.2]],100),[-16])
    def test_backtest_lag_drift_costs(self):
        r=np.array([[.5,0],[.1,0],[0,.1]])
        weights=np.tile([.5,.5],(3,1))
        result=backtest(r,weights,transaction_cost_bps=10)
        self.assertEqual(result.gross_returns[0],0)
        self.assertAlmostEqual(result.net_returns[1],.999*1.05-1)
        self.assertAlmostEqual(result.turnover[2],abs(.5-.55/1.05)+abs(.5-.5/1.05))
        modified=weights.copy();modified[-1]=[1,0]
        np.testing.assert_allclose(result.net_returns,backtest(r,modified,transaction_cost_bps=10).net_returns)
    def test_performance_includes_initial_peak(self):
        p=performance([-.2,.1],periods_per_year=2)
        self.assertAlmostEqual(p.max_drawdown,-.2)
        self.assertAlmostEqual(p.total_return,-.12)
        np.testing.assert_allclose(simple_returns([100,110,99]),[.1,-.1])
    def test_factor_recovery_and_hac(self):
        rng=np.random.default_rng(42)
        x=rng.normal(0,.02,(500,2));y=.001+x@np.array([1.2,-.4])+rng.normal(0,.0001,500)
        fit=fit_factor_model(y,x,hac_lags=5)
        self.assertAlmostEqual(fit.alpha,.001,delta=.00002)
        np.testing.assert_allclose(fit.betas,[1.2,-.4],atol=.001)
        self.assertGreater(fit.r_squared,.99)
        self.assertTrue(np.all(fit.standard_errors>0))
        with self.assertRaises(ValueError): fit_factor_model(y,np.ones((500,2)))
    def test_simulation_seed_and_zero_covariance(self):
        m=FinancialModel(1000).forecast()
        a=simulate_valuation(m,['revenue_growth'],[[.0001]],samples=100,seed=7)
        b=simulate_valuation(m,['revenue_growth'],[[.0001]],samples=100,seed=7)
        np.testing.assert_array_equal(a.enterprise_values,b.enterprise_values)
        z=simulate_valuation(m,['revenue_growth'],[[0]],samples=10)
        np.testing.assert_allclose(z.enterprise_values,m.enterprise_value())
        self.assertEqual(m.assumptions.revenue_growth,.05)


class ResearchTests(unittest.TestCase):
    def test_covariance_shrinkage_and_no_future_leakage(self):
        from finmodel.portfolio.research import estimate_moments, walk_forward_minimum_variance
        rng=np.random.default_rng(10)
        r=rng.normal(0,.01,(100,3))
        _,cov=estimate_moments(r,shrinkage=1)
        np.testing.assert_allclose(cov,np.diag(np.diag(cov)))
        a=walk_forward_minimum_variance(r,lookback=20,rebalance_every=10)
        changed=r.copy();changed[80:]=.02
        b=walk_forward_minimum_variance(changed,lookback=20,rebalance_every=10)
        np.testing.assert_allclose(a.net_returns[:80],b.net_returns[:80])
        np.testing.assert_allclose(a.net_returns[:20],0)
        # First trade then drift until rebalance: no artificial daily rebalancing.
        np.testing.assert_allclose(a.turnover[21:30],0,atol=1e-14)
        self.assertGreater(a.turnover[20],0)
    def test_covenants(self):
        from finmodel import OpeningBalance
        from finmodel.validation.covenants import check_covenants
        m=FinancialModel(1000,opening=OpeningBalance(cash=100,ppe=400,working_capital=150,debt=500)).forecast()
        codes={f.code for f in check_covenants(m,maximum_debt_to_ebitda=1,minimum_interest_coverage=100,minimum_cash=10000)}
        self.assertEqual(codes,{'LEVERAGE_COVENANT','COVERAGE_COVENANT','LIQUIDITY_COVENANT'})

if __name__ == "__main__":
    unittest.main()
