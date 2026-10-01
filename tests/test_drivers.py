import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
from finmodel import FinancialModel

class DriverTests(unittest.TestCase):
    def test_year_specific_overrides_do_not_persist(self):
        m=FinancialModel(1000).forecast(3,drivers={1:{'revenue_growth':.1},2:{'revenue_growth':0}})
        self.assertAlmostEqual(m.periods[0].income_statement.revenue,1100)
        self.assertAlmostEqual(m.periods[1].income_statement.revenue,1100)
        self.assertAlmostEqual(m.periods[2].income_statement.revenue,1155)
        self.assertFalse([f for f in m.audit() if f.severity=='error'])
        self.assertEqual(m.scenario('Down',{'ebitda_margin':.1}).drivers,m.drivers)
    def test_excel_preserves_drivers(self):
        with TemporaryDirectory() as d:
            m=FinancialModel(1000).forecast(3,drivers={2:{'ebitda_margin':.1}})
            path=Path(d)/'model.xlsx';m.export_excel(path)
            loaded=FinancialModel.from_excel(path)
            self.assertEqual(loaded.periods,m.periods)
            self.assertEqual(loaded.drivers,m.drivers)
    def test_invalid_drivers(self):
        for drivers in ({0:{'revenue_growth':.1}},{1:{'wacc':.2}},{1:{'capex_ratio':-1}}):
            with self.assertRaises(ValueError): FinancialModel(1000).forecast(drivers=drivers)
