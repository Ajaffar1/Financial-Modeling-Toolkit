from dataclasses import dataclass
@dataclass(frozen=True)
class DebtPeriod:
    opening: float
    repayment: float
    closing: float
    interest: float

def debt_period(opening, repayment, rate):
    if min(opening, repayment, rate) < 0:
        raise ValueError("Debt, repayment, and interest rate must be nonnegative")
    paid = min(opening, repayment)
    # Beginning-of-period interest avoids an implicit financing circularity.
    return DebtPeriod(opening, paid, opening - paid, opening * rate)
