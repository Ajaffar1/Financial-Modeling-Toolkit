import math

def terminal_value(cash_flow, wacc, growth):
    if not all(math.isfinite(v) for v in (cash_flow, wacc, growth)):
        raise ValueError("Valuation inputs must be finite")
    if growth <= -1 or wacc <= growth:
        raise ValueError("Require terminal growth > -100% and WACC > terminal growth")
    return cash_flow * (1 + growth) / (wacc - growth)
