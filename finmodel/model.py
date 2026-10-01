from dataclasses import dataclass, asdict, replace
import math
from .statements.income_statement import IncomeStatement
from .statements.balance_sheet import BalanceSheet
from .statements.cash_flow import CashFlow
from .debt.debt_schedule import debt_period
from .valuation.dcf import discounted_cash_flows

@dataclass(frozen=True)
class Assumptions:
    revenue: float
    ebitda_margin: float = 0.22
    revenue_growth: float = 0.05
    tax_rate: float = 0.265
    wacc: float = 0.09
    terminal_growth: float = 0.025
    working_capital_ratio: float = 0.15
    capex_ratio: float = 0.04
    depreciation_rate: float = 0.10
    interest_rate: float = 0.05
    annual_debt_repayment: float = 0.0
    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("All assumptions must be finite numbers")
        if self.revenue < 0 or self.revenue_growth <= -1:
            raise ValueError("Require nonnegative revenue and growth > -100%")
        for name in ("tax_rate", "working_capital_ratio", "capex_ratio", "depreciation_rate"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.interest_rate < 0 or self.annual_debt_repayment < 0:
            raise ValueError("Interest and debt repayment must be nonnegative")
        if self.terminal_growth <= -1 or self.wacc <= self.terminal_growth:
            raise ValueError("Require terminal growth > -100% and WACC > terminal growth")

@dataclass(frozen=True)
class OpeningBalance:
    cash: float = 0.0
    working_capital: float = 0.0
    ppe: float = 0.0
    debt: float = 0.0
    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in asdict(self).values()):
            raise ValueError("Opening balances must be finite and nonnegative")
    @property
    def equity(self):
        return self.cash + self.working_capital + self.ppe - self.debt

@dataclass(frozen=True)
class Period:
    year: int
    income_statement: IncomeStatement
    balance_sheet: BalanceSheet
    cash_flow: CashFlow

class FinancialModel:
    def __init__(self, revenue=None, *, opening=None, assumptions=None, name="Base", **kwargs):
        if assumptions is not None and (revenue is not None or kwargs):
            raise ValueError("Supply either assumptions or revenue and keyword assumptions")
        self.assumptions = assumptions if assumptions is not None else Assumptions(revenue=revenue, **kwargs)
        a = self.assumptions
        self.opening = opening if opening is not None else OpeningBalance(working_capital=a.revenue * a.working_capital_ratio, ppe=a.revenue * a.capex_ratio / max(a.depreciation_rate, 0.1))
        self.name = name
        self.periods = ()
        self.drivers = {}
    def forecast(self, years=5, *, drivers=None):
        if isinstance(years, bool) or not isinstance(years, int) or years < 1:
            raise ValueError("years must be a positive integer")
        drivers = {} if drivers is None else {year: dict(values) for year, values in drivers.items()}
        allowed = {"revenue_growth", "ebitda_margin", "tax_rate", "working_capital_ratio", "capex_ratio", "depreciation_rate", "interest_rate", "annual_debt_repayment"}
        for year, values in drivers.items():
            if isinstance(year, bool) or not isinstance(year, int) or not 1 <= year <= years or not set(values) <= allowed:
                raise ValueError("Drivers require forecast-year keys and supported operating assumptions")
            replace(self.assumptions, **values)
        a, b, revenue = self.assumptions, self.opening, self.assumptions.revenue
        equity = b.equity
        periods = []
        for year in range(1, years + 1):
            a = replace(self.assumptions, **drivers.get(year, {}))
            revenue *= 1 + a.revenue_growth
            ebitda = revenue * a.ebitda_margin
            depreciation = b.ppe * a.depreciation_rate
            capex = revenue * a.capex_ratio
            nwc = revenue * a.working_capital_ratio
            change_nwc = nwc - b.working_capital
            debt = debt_period(b.debt, a.annual_debt_repayment, a.interest_rate)
            ebit = ebitda - depreciation
            taxes = max(0, ebit - debt.interest) * a.tax_rate
            ni = ebit - debt.interest - taxes
            income = IncomeStatement(revenue, ebitda, depreciation, ebit, debt.interest, taxes, ni)
            cashflow = CashFlow(ni + depreciation - change_nwc, -capex, -debt.repayment,
                               ebit - max(0, ebit) * a.tax_rate + depreciation - capex - change_nwc)
            equity += ni
            b = BalanceSheet(b.cash + cashflow.net_change, nwc, b.ppe + capex - depreciation, debt.closing, equity)
            periods.append(Period(year, income, b, cashflow))
        self.periods = tuple(periods)
        self.drivers = drivers
        return self
    def _valuation(self):
        if not self.periods:
            raise ValueError("Call forecast() first")
        return discounted_cash_flows((p.cash_flow.unlevered_free_cash_flow for p in self.periods), self.assumptions.wacc, self.assumptions.terminal_growth)
    def enterprise_value(self):
        return self._valuation()[0]
    def equity_value(self):
        # Valuation as of the opening date: subtract opening net debt.
        return self.enterprise_value() - self.opening.debt + self.opening.cash
    def scenario(self, name, overrides):
        """Overrides are absolute replacement values, including growth/margins."""
        model = type(self)(assumptions=replace(self.assumptions, **overrides), opening=self.opening, name=name)
        if self.periods:
            model.forecast(len(self.periods), drivers=self.drivers)
        return model
    def sensitivity(self, row="wacc", column="terminal_growth", *, row_values=None, column_values=None):
        if row == column:
            raise ValueError("Sensitivity axes must differ")
        if not self.periods:
            raise ValueError("Call forecast() first")
        fields = asdict(self.assumptions)
        if row not in fields or column not in fields:
            raise ValueError("Unknown sensitivity assumption")
        rows = row_values if row_values is not None else [fields[row] + d for d in (-0.01, 0, 0.01)]
        cols = column_values if column_values is not None else [fields[column] + d for d in (-0.005, 0, 0.005)]
        return {(r, c): self.scenario("Sensitivity", {row: r, column: c}).enterprise_value() for r in rows for c in cols}
    def audit(self, tolerance=1e-6):
        from .validation.model_checks import audit_model
        if tolerance < 0 or not math.isfinite(tolerance):
            raise ValueError("tolerance must be finite and nonnegative")
        return audit_model(self, tolerance)
    def export_excel(self, path):
        from .excel.exporter import export_excel
        return export_excel(self, path)
    @classmethod
    def from_excel(cls, path):
        from .excel.importer import from_excel
        return from_excel(cls, path)

DCFModel = FinancialModel
