from dataclasses import dataclass
@dataclass(frozen=True)
class IncomeStatement:
    revenue: float
    ebitda: float
    depreciation: float
    ebit: float
    interest: float
    taxes: float
    net_income: float
