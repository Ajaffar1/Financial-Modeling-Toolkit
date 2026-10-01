from dataclasses import dataclass
@dataclass(frozen=True)
class CashFlow:
    operating: float
    investing: float
    financing: float
    unlevered_free_cash_flow: float
    @property
    def net_change(self):
        return self.operating + self.investing + self.financing
