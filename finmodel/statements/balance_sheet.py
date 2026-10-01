from dataclasses import dataclass
@dataclass(frozen=True)
class BalanceSheet:
    cash: float
    working_capital: float
    ppe: float
    debt: float
    equity: float
    @property
    def balance_error(self):
        return self.cash + self.working_capital + self.ppe - self.debt - self.equity
