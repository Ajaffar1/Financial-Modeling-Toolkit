import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from finmodel import FinancialModel, OpeningBalance
from finmodel.valuation.dcf import discounted_cash_flows
from finmodel.validation.balance_checks import circular_references
from finmodel.forecasting.scenarios import assumption_changes

class ModelTests(unittest.TestCase):
    def model(self, **kwargs):
        return FinancialModel(1000, opening=OpeningBalance(cash=100, working_capital=150, ppe=400, debt=200), **kwargs).forecast(5)
    def test_hand_calculated_first_year(self):
        m = self.model(annual_debt_repayment=50)
        p = m.periods[0]
        self.assertAlmostEqual(p.income_statement.revenue, 1050)
        self.assertAlmostEqual(p.income_statement.net_income, (231 - 40 - 10) * .735)
        self.assertAlmostEqual(p.cash_flow.unlevered_free_cash_flow, 191 * .735 + 40 - 42 - 7.5)
        self.assertEqual(p.balance_sheet.debt, 150)
        self.assertAlmostEqual(p.balance_sheet.ppe, 402)
        self.assertAlmostEqual(p.balance_sheet.balance_error, 0)
    def test_reconciliation_and_equity_bridge(self):
        m = self.model(annual_debt_repayment=1000)
        self.assertFalse([f for f in m.audit() if f.severity == "error"])
        self.assertEqual(m.periods[-1].balance_sheet.debt, 0)
        self.assertAlmostEqual(m.equity_value(), m.enterprise_value() - 100)
    def test_dcf_against_manual_discounting(self):
        ev, terminal = discounted_cash_flows([100, 110], .1, .02)
        tv = 110 * 1.02 / .08 / 1.1 ** 2
        self.assertAlmostEqual(terminal, tv)
        self.assertAlmostEqual(ev, 100 / 1.1 + 110 / 1.1 ** 2 + tv)
    def test_scenarios_are_isolated(self):
        base = self.model()
        original = base.periods
        down = base.scenario("Downside", {"revenue_growth": -.05})
        self.assertEqual(base.periods, original)
        self.assertLess(down.enterprise_value(), base.enterprise_value())
        self.assertEqual(assumption_changes(base, down), {"revenue_growth": (.05, -.05)})
        grid = base.sensitivity(row_values=[.09], column_values=[.025])
        self.assertAlmostEqual(grid[(.09, .025)], base.enterprise_value())
    def test_input_validation(self):
        for kwargs in ({"wacc": .02, "terminal_growth": .025}, {"tax_rate": 1.1}, {"revenue_growth": -1}, {"revenue": float("nan")}):
            with self.assertRaises(ValueError):
                FinancialModel(**({"revenue": 1000} | kwargs))
        for years in (0, True, 1.5):
            with self.assertRaises(ValueError):
                self.model().forecast(years)
        with self.assertRaises(ValueError):
            FinancialModel(1000).enterprise_value()
    def test_audit_detects_corruption(self):
        m = self.model()
        p = m.periods[0]
        m.periods = (replace(p, balance_sheet=replace(p.balance_sheet, cash=p.balance_sheet.cash + 10)),) + m.periods[1:]
        codes = {f.code for f in m.audit()}
        self.assertTrue({"BALANCE_SHEET", "CASH_RECONCILIATION"} <= codes)
    def test_funding_shortfall_reported(self):
        m = FinancialModel(1000, ebitda_margin=-.1).forecast()
        self.assertIn("NEGATIVE_CASH", {f.code for f in m.audit()})
    def test_cycle_detection(self):
        self.assertEqual(circular_references({"cash": ["interest"], "interest": ["cash"]}), [("cash", "interest", "cash")])
        self.assertEqual(circular_references({"cash": ["income"]}), [])
    def test_excel_round_trip(self):
        from openpyxl import load_workbook
        with TemporaryDirectory() as d:
            path = Path(d) / "model.xlsx"
            m = self.model()
            m.export_excel(path)
            imported = FinancialModel.from_excel(path)
            self.assertEqual(m.assumptions, imported.assumptions)
            self.assertEqual(m.periods, imported.periods)
            wb = load_workbook(path)
            wb['Assumptions']['B4'] = '=1+1'
            wb.save(path)
            wb.close()
            with self.assertRaises(ValueError):
                FinancialModel.from_excel(path)

if __name__ == "__main__":
    unittest.main()
